# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM MegaAttention Phase-1 fused Q-prep / out-layout.

CPU reference + fuse fallbacks always run. CUDA+Triton correctness is gated
behind ``torch.cuda.is_available()`` (skipped on Mac / CPU-only hosts).
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.mla import glm_mega_attn_phase1 as phase1

# Mac CPU runs hit a Torch segfault in the autouse dist cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup

_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required")


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


def test_fuse_prepare_out_param_reuses_storage_on_cpu():
    ql = torch.randn(2, 8, 512, dtype=torch.bfloat16)
    buf = torch.empty(2, 64, 576, dtype=torch.bfloat16)
    out = phase1.fuse_prepare_q_nope_fp8_decode(ql, 8, 64, 576, 512, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    ref = phase1.prepare_q_nope_fp8_decode_ref(ql, 8, 64, 576, 512)
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


# ---------------------------------------------------------------------------
# CUDA + Triton correctness (skipped when no GPU)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 7, 32, 129])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("pad_heads", [64, 128])
@pytest.mark.parametrize("q_head_size", [512, 576])
def test_cuda_fuse_prepare_matches_ref(tokens, num_heads, pad_heads, q_head_size):
    if num_heads > pad_heads:
        pytest.skip("num_heads exceeds pad_heads")
    kv_lora = 512
    torch.manual_seed(123 + tokens + num_heads)
    ql = torch.randn(tokens, num_heads, kv_lora, dtype=torch.bfloat16, device="cuda")
    ref = phase1.prepare_q_nope_fp8_decode_ref(
        ql.cpu(), num_heads, pad_heads, q_head_size, kv_lora
    ).to("cuda")
    buf = torch.empty(
        tokens, pad_heads, q_head_size, dtype=torch.bfloat16, device="cuda"
    )
    out = phase1.fuse_prepare_q_nope_fp8_decode(
        ql, num_heads, pad_heads, q_head_size, kv_lora, out=buf
    )
    assert out.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 7, 32, 129])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("pad_heads", [64, 128])
def test_cuda_fuse_layout_matches_ref(tokens, num_heads, pad_heads):
    if num_heads > pad_heads:
        pytest.skip("num_heads exceeds pad_heads")
    dim = 512
    torch.manual_seed(456 + tokens * 17 + num_heads)
    attn = torch.randn(tokens, pad_heads, dim, dtype=torch.bfloat16, device="cuda")
    ref = phase1.layout_attn_out_for_uv_ref(attn.cpu(), num_heads).to("cuda")
    buf = torch.empty(tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
    out = phase1.fuse_layout_attn_out_for_uv(attn, num_heads, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 7, 32])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("pad_heads", [64, 128])
def test_cuda_fuse_pad_matches_ref(tokens, num_heads, pad_heads):
    if num_heads > pad_heads:
        pytest.skip("num_heads exceeds pad_heads")
    dim = 576
    torch.manual_seed(789)
    q = torch.randn(tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
    ref = phase1.pad_q_heads_ref(q.cpu(), pad_heads).to("cuda")
    buf = torch.empty(tokens, pad_heads, dim, dtype=torch.bfloat16, device="cuda")
    out = phase1.fuse_pad_q_heads(q, pad_heads, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@requires_cuda
def test_cuda_fuse_prepare_strided_input():
    """Non-contiguous / strided ql_nope (transpose of [H, T, D])."""
    tokens, num_heads, pad_heads = 7, 16, 64
    kv_lora, q_head_size = 512, 576
    torch.manual_seed(42)
    raw = torch.randn(num_heads, tokens, kv_lora, dtype=torch.bfloat16, device="cuda")
    ql = raw.transpose(0, 1)  # [T, H, D], non-contiguous
    assert not ql.is_contiguous()
    ref = phase1.prepare_q_nope_fp8_decode_ref(
        ql.contiguous().cpu(), num_heads, pad_heads, q_head_size, kv_lora
    ).to("cuda")
    buf = torch.empty(
        tokens, pad_heads, q_head_size, dtype=torch.bfloat16, device="cuda"
    )
    out = phase1.fuse_prepare_q_nope_fp8_decode(
        ql, num_heads, pad_heads, q_head_size, kv_lora, out=buf
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@requires_cuda
def test_cuda_fuse_layout_strided_input():
    tokens, num_heads, pad_heads, dim = 32, 16, 64, 512
    torch.manual_seed(99)
    raw = torch.randn(pad_heads, tokens, dim, dtype=torch.bfloat16, device="cuda")
    attn = raw.transpose(0, 1)  # [T, pad_H, D] strided
    assert not attn.is_contiguous()
    ref = phase1.layout_attn_out_for_uv_ref(attn.contiguous().cpu(), num_heads).to(
        "cuda"
    )
    buf = torch.empty(tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
    out = phase1.fuse_layout_attn_out_for_uv(attn, num_heads, out=buf)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
