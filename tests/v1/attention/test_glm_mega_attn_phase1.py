# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for GLM MegaAttention Phase-1 fused Q-prep / out-layout.

No GPU required. Triton path is exercised only when CUDA + Triton are present;
reference implementations always run on CPU.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.mla import glm_mega_attn_phase1 as phase1

# Mac CPU runs hit a Torch segfault in the autouse dist cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("tokens", "num_heads", "pad_heads", "kv_lora", "q_head_size"),
    [
        (1, 8, 64, 512, 576),
        (4, 16, 64, 512, 576),
        (3, 64, 64, 512, 576),
        (2, 32, 64, 512, 512),  # no RoPE slot padding
    ],
)
def test_prepare_q_nope_fp8_decode_ref_matches_manual(
    tokens, num_heads, pad_heads, kv_lora, q_head_size
):
    torch.manual_seed(0)
    ql = torch.randn(tokens, num_heads, kv_lora, dtype=torch.bfloat16)
    out = phase1.prepare_q_nope_fp8_decode_ref(
        ql, num_heads, pad_heads, q_head_size, kv_lora
    )
    assert out.shape == (tokens, pad_heads, q_head_size)
    assert torch.equal(out[:, :num_heads, :kv_lora], ql)
    assert torch.count_nonzero(out[:, num_heads:, :]) == 0
    if q_head_size > kv_lora:
        assert torch.count_nonzero(out[:, :num_heads, kv_lora:]) == 0


@pytest.mark.parametrize(
    ("tokens", "num_heads", "pad_heads", "dim"),
    [
        (1, 8, 64, 512),
        (5, 16, 64, 512),
        (2, 64, 64, 256),
    ],
)
def test_layout_attn_out_for_uv_ref(tokens, num_heads, pad_heads, dim):
    torch.manual_seed(1)
    attn = torch.randn(tokens, pad_heads, dim, dtype=torch.bfloat16)
    out = phase1.layout_attn_out_for_uv_ref(attn, num_heads)
    assert out.shape == (tokens, num_heads, dim)
    assert out.is_contiguous()
    assert torch.equal(out, attn[:, :num_heads, :])


def test_pad_q_heads_ref_3d_and_4d():
    q = torch.randn(3, 8, 576, dtype=torch.bfloat16)
    out = phase1.pad_q_heads_ref(q, 64)
    assert out.shape == (3, 64, 576)
    assert torch.equal(out[:, :8], q)
    assert torch.count_nonzero(out[:, 8:]) == 0

    q4 = q.view(1, 3, 8, 576)
    out4 = phase1.pad_q_heads_ref(q4, 64)
    assert out4.shape == (1, 3, 64, 576)


def test_fuse_prepare_matches_ref_on_cpu():
    ql = torch.randn(2, 8, 512, dtype=torch.bfloat16)
    ref = phase1.prepare_q_nope_fp8_decode_ref(ql, 8, 64, 576, 512)
    # Public fuse entry uses the registered custom op; on CPU it falls back
    # to the reference path inside the impl.
    out = phase1.fuse_prepare_q_nope_fp8_decode(ql, 8, 64, 576, 512)
    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def test_fuse_layout_matches_ref_on_cpu():
    attn = torch.randn(4, 64, 512, dtype=torch.bfloat16)
    ref = phase1.layout_attn_out_for_uv_ref(attn, 16)
    out = phase1.fuse_layout_attn_out_for_uv(attn, 16)
    assert torch.equal(out, ref)
    assert out.is_contiguous()


def test_fuse_pad_q_heads_matches_ref_on_cpu():
    q = torch.randn(2, 3, 8, 576, dtype=torch.bfloat16)
    ref = phase1.pad_q_heads_ref(q, 64)
    out = phase1.fuse_pad_q_heads(q, 64)
    assert torch.equal(out, ref)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_GLM_MEGA_ATTN_PHASE1", raising=False)
    # Re-read through envs module attribute (already bound at import via
    # __getattr__). Force a fresh lookup by calling the helper which reads
    # envs.VLLM_GLM_MEGA_ATTN_PHASE1.
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_GLM_MEGA_ATTN_PHASE1", False, raising=False)
    phase1.ensure_glm_mega_attn_phase1_ready.cache_clear()
    assert phase1.is_glm_mega_attn_phase1_requested() is False
    assert phase1.ensure_glm_mega_attn_phase1_ready() is False


def test_flag_fail_closed_non_sm90(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_GLM_MEGA_ATTN_PHASE1", True, raising=False)
    monkeypatch.setattr(phase1, "is_glm_mega_attn_phase1_supported", lambda: False)
    phase1.ensure_glm_mega_attn_phase1_ready.cache_clear()
    with pytest.raises(RuntimeError, match="SM90"):
        phase1.ensure_glm_mega_attn_phase1_ready()
    phase1.ensure_glm_mega_attn_phase1_ready.cache_clear()
