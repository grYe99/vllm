# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 MegaAttention-style fuse for GLM sparse MLA decode (SM90 / FP8).

GLM-5.3 sparse MLA (``FLASHMLA_SPARSE`` + ``fp8_ds_mla``) already has a solid
FlashMLA sparse body. Profile data shows decode GPU time is still split across
several small launches around that body:

1. Q layout / pad into the FP8 FlashMLA workspace (RoPE slot + heads → 64/128)
2. FlashMLA sparse attention kernel (unchanged; keep existing FlashMLA sparse)
3. Attention output unpad + contiguous layout for the subsequent ``W_UV`` bmm

True SM100 MegaAttention kernels (``FLASHMLA_MEGA_ATTN_DSV41``) are out of
scope for Hopper. Phase-1 therefore fuses the *pre* and *post* layout steps
around the existing sparse MLA kernel into CUDA-graph-friendly custom ops,
gated by ``VLLM_GLM_MEGA_ATTN_PHASE1`` (default off). Fail closed unless the
device is exactly SM90.

Phase-2 (indexer / top-k fusion) is intentionally deferred.
"""

from __future__ import annotations

from functools import lru_cache

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    eager_bypass_custom_op,
)

logger = init_logger(__name__)

# Hopper / H100 / H200 (exact SM90). Fail closed elsewhere.
_SM90_CAPABILITY = 90


def is_glm_mega_attn_phase1_requested() -> bool:
    """Return True when the Phase-1 opt-in env flag is set."""
    return bool(envs.VLLM_GLM_MEGA_ATTN_PHASE1)


def is_glm_mega_attn_phase1_supported() -> bool:
    """Return True when running on CUDA SM90 (Hopper)."""
    if not current_platform.is_cuda():
        return False
    return bool(current_platform.is_device_capability(_SM90_CAPABILITY))


@lru_cache(maxsize=1)
def ensure_glm_mega_attn_phase1_ready() -> bool:
    """Validate Phase-1 can run. Raises if requested on an unsupported device.

    Returns True when Phase-1 is active (requested + SM90). Returns False when
    the flag is off. Never silently enables a partial path on non-SM90.
    """
    if not is_glm_mega_attn_phase1_requested():
        return False
    if not is_glm_mega_attn_phase1_supported():
        raise RuntimeError(
            "VLLM_GLM_MEGA_ATTN_PHASE1 is set but this device is not CUDA SM90 "
            "(Hopper). Phase-1 MegaAttention-style fusion for GLM sparse MLA "
            "is fail-closed outside SM90. Unset the env var or run on H100/H200."
        )
    logger.info_once(
        "GLM MegaAttention Phase-1 enabled: fused Q layout-pad + FlashMLA "
        "sparse + out layout on SM90 (FP8 KV / fp8_ds_mla)."
    )
    return True


# ---------------------------------------------------------------------------
# Reference (CPU / correctness) implementations
# ---------------------------------------------------------------------------


def prepare_q_nope_fp8_decode_ref(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Reference: pack ``[T, H, kv_lora]`` into ``[T, pad_H, q_head_size]``.

    Copies real heads into ``[:num_q_heads, :kv_lora_rank]`` and zero-fills the
    RoPE slot plus any padded head rows. Matches the historical FlashMLA sparse
    FP8 NoPE decode layout.
    """
    assert ql_nope.dim() == 3
    tokens = ql_nope.shape[0]
    assert ql_nope.shape[1] == num_q_heads
    assert ql_nope.shape[2] == kv_lora_rank
    assert pad_q_heads >= num_q_heads
    assert q_head_size >= kv_lora_rank
    q = ql_nope.new_empty((tokens, pad_q_heads, q_head_size))
    q.zero_()
    q[:, :num_q_heads, :kv_lora_rank].copy_(ql_nope)
    return q


def layout_attn_out_for_uv_ref(
    attn_out: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """Reference: unpad heads and materialize contiguous ``[T, H, D]`` for W_UV."""
    assert attn_out.dim() == 3
    out = attn_out[:, :num_q_heads, :]
    if not out.is_contiguous():
        out = out.contiguous()
    return out


def pad_q_heads_ref(
    q: torch.Tensor,
    pad_q_heads: int,
) -> torch.Tensor:
    """Reference: pad ``[N, H, D]`` (or ``[B, S, H, D]``) heads to ``pad_q_heads``."""
    if q.dim() == 4:
        b, s, h, d = q.shape
        if h == pad_q_heads:
            return q
        out = q.new_zeros((b, s, pad_q_heads, d))
        out[:, :, :h, :].copy_(q)
        return out
    assert q.dim() == 3
    n, h, d = q.shape
    if h == pad_q_heads:
        return q
    out = q.new_zeros((n, pad_q_heads, d))
    out[:, :h, :].copy_(q)
    return out


# ---------------------------------------------------------------------------
# Triton fused kernels (CUDA SM90 decode hot path)
# ---------------------------------------------------------------------------


@triton.jit
def _fuse_prepare_q_nope_fp8_kernel(
    ql_ptr,
    q_ptr,
    tokens,
    num_q_heads,
    pad_q_heads,
    kv_lora_rank,
    q_head_size,
    stride_ql_t,
    stride_ql_h,
    stride_ql_d,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    BLOCK_D: tl.constexpr,
):
    """Zero-fill padded Q workspace and copy real NoPE heads in one launch."""
    pid = tl.program_id(0)
    t = pid // pad_q_heads
    h = pid % pad_q_heads
    if t >= tokens:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < q_head_size
    out_ptrs = q_ptr + t * stride_q_t + h * stride_q_h + offs_d * stride_q_d

    if h < num_q_heads:
        # Real head: copy kv_lora_rank from ql_nope; zero the RoPE slot.
        mask_copy = (offs_d < kv_lora_rank) & mask_d
        in_ptrs = ql_ptr + t * stride_ql_t + h * stride_ql_h + offs_d * stride_ql_d
        vals = tl.load(in_ptrs, mask=mask_copy, other=0.0)
        tl.store(out_ptrs, vals, mask=mask_d)
    else:
        tl.store(out_ptrs, 0.0, mask=mask_d)


@triton.jit
def _fuse_pad_q_heads_kernel(
    q_ptr,
    out_ptr,
    rows,
    num_q_heads,
    pad_q_heads,
    head_size,
    stride_q_r,
    stride_q_h,
    stride_q_d,
    stride_o_r,
    stride_o_h,
    stride_o_d,
    BLOCK_D: tl.constexpr,
):
    """Pad heads ``[rows, H, D] -> [rows, pad_H, D]`` in one launch."""
    pid = tl.program_id(0)
    r = pid // pad_q_heads
    h = pid % pad_q_heads
    if r >= rows:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_size
    out_ptrs = out_ptr + r * stride_o_r + h * stride_o_h + offs_d * stride_o_d
    if h < num_q_heads:
        in_ptrs = q_ptr + r * stride_q_r + h * stride_q_h + offs_d * stride_q_d
        vals = tl.load(in_ptrs, mask=mask_d, other=0.0)
        tl.store(out_ptrs, vals, mask=mask_d)
    else:
        tl.store(out_ptrs, 0.0, mask=mask_d)


@triton.jit
def _fuse_layout_attn_out_kernel(
    attn_ptr,
    out_ptr,
    tokens,
    num_q_heads,
    pad_q_heads,
    head_size,
    stride_a_t,
    stride_a_h,
    stride_a_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    BLOCK_D: tl.constexpr,
):
    """Copy unpadded heads into a contiguous ``[T, H, D]`` buffer in one launch."""
    pid = tl.program_id(0)
    t = pid // num_q_heads
    h = pid % num_q_heads
    if t >= tokens:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_size
    in_ptrs = attn_ptr + t * stride_a_t + h * stride_a_h + offs_d * stride_a_d
    out_ptrs = out_ptr + t * stride_o_t + h * stride_o_h + offs_d * stride_o_d
    vals = tl.load(in_ptrs, mask=mask_d, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_d)


def _next_pow2_ge(n: int, min_n: int = 16) -> int:
    v = max(int(n), min_n)
    return 1 << (v - 1).bit_length()


def _fuse_prepare_q_nope_fp8_decode_impl(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    tokens = ql_nope.shape[0]
    q = ql_nope.new_empty((tokens, pad_q_heads, q_head_size))
    if tokens == 0 or pad_q_heads == 0:
        return q

    if (not ql_nope.is_cuda) or (not HAS_TRITON):
        return prepare_q_nope_fp8_decode_ref(
            ql_nope, num_q_heads, pad_q_heads, q_head_size, kv_lora_rank
        )

    block_d = _next_pow2_ge(q_head_size)
    grid = (tokens * pad_q_heads,)
    _fuse_prepare_q_nope_fp8_kernel[grid](
        ql_nope,
        q,
        tokens,
        num_q_heads,
        pad_q_heads,
        kv_lora_rank,
        q_head_size,
        ql_nope.stride(0),
        ql_nope.stride(1),
        ql_nope.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        BLOCK_D=block_d,
    )
    return q


def _fuse_pad_q_heads_impl(
    q: torch.Tensor,
    pad_q_heads: int,
) -> torch.Tensor:
    """Pad last-head dim of a 3D ``[N, H, D]`` tensor. Callers flatten 4D."""
    assert q.dim() == 3
    rows, num_q_heads, head_size = q.shape
    if num_q_heads == pad_q_heads:
        return q
    if rows == 0:
        return q.new_empty((0, pad_q_heads, head_size))

    if (not q.is_cuda) or (not HAS_TRITON):
        return pad_q_heads_ref(q, pad_q_heads)

    out = q.new_empty((rows, pad_q_heads, head_size))
    block_d = _next_pow2_ge(head_size)
    grid = (rows * pad_q_heads,)
    _fuse_pad_q_heads_kernel[grid](
        q,
        out,
        rows,
        num_q_heads,
        pad_q_heads,
        head_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )
    return out


def _fuse_layout_attn_out_for_uv_impl(
    attn_out: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    tokens, pad_q_heads, head_size = attn_out.shape
    if tokens == 0 or num_q_heads == 0:
        return attn_out.new_empty((tokens, num_q_heads, head_size))

    if pad_q_heads == num_q_heads and attn_out.is_contiguous():
        return attn_out

    if (not attn_out.is_cuda) or (not HAS_TRITON):
        return layout_attn_out_for_uv_ref(attn_out, num_q_heads)

    out = attn_out.new_empty((tokens, num_q_heads, head_size))
    block_d = _next_pow2_ge(head_size)
    grid = (tokens * num_q_heads,)
    _fuse_layout_attn_out_kernel[grid](
        attn_out,
        out,
        tokens,
        num_q_heads,
        pad_q_heads,
        head_size,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )
    return out


def _fuse_prepare_q_nope_fp8_decode_fake(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    return ql_nope.new_empty((ql_nope.shape[0], pad_q_heads, q_head_size))


def _fuse_pad_q_heads_fake(
    q: torch.Tensor,
    pad_q_heads: int,
) -> torch.Tensor:
    return q.new_empty((q.shape[0], pad_q_heads, q.shape[2]))


def _fuse_layout_attn_out_for_uv_fake(
    attn_out: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    tokens, _, head_size = attn_out.shape
    return attn_out.new_empty((tokens, num_q_heads, head_size))


direct_register_custom_op(
    op_name="glm_mega_attn_phase1_prepare_q",
    op_func=_fuse_prepare_q_nope_fp8_decode_impl,
    mutates_args=[],
    fake_impl=_fuse_prepare_q_nope_fp8_decode_fake,
)

direct_register_custom_op(
    op_name="glm_mega_attn_phase1_pad_q_heads",
    op_func=_fuse_pad_q_heads_impl,
    mutates_args=[],
    fake_impl=_fuse_pad_q_heads_fake,
)

direct_register_custom_op(
    op_name="glm_mega_attn_phase1_layout_out",
    op_func=_fuse_layout_attn_out_for_uv_impl,
    mutates_args=[],
    fake_impl=_fuse_layout_attn_out_for_uv_fake,
)

# Eager decode path (``eager_break_during_capture``) skips dispatcher overhead.
_prepare_q_op = eager_bypass_custom_op(
    "glm_mega_attn_phase1_prepare_q", _fuse_prepare_q_nope_fp8_decode_impl
)
_pad_q_heads_op = eager_bypass_custom_op(
    "glm_mega_attn_phase1_pad_q_heads", _fuse_pad_q_heads_impl
)
_layout_out_op = eager_bypass_custom_op(
    "glm_mega_attn_phase1_layout_out", _fuse_layout_attn_out_for_uv_impl
)


def fuse_prepare_q_nope_fp8_decode(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fused NoPE Q pack + head pad for FP8 sparse MLA decode (custom op)."""
    return _prepare_q_op(ql_nope, num_q_heads, pad_q_heads, q_head_size, kv_lora_rank)


def fuse_pad_q_heads(q: torch.Tensor, pad_q_heads: int) -> torch.Tensor:
    """Fused head pad for FP8 sparse MLA decode.

    Accepts ``[N, H, D]`` or ``[B, S, H, D]``; returns the same rank.
    """
    if q.dim() == 4:
        b, s, h, d = q.shape
        if h == pad_q_heads:
            return q
        flat = q.reshape(b * s, h, d)
        padded = _pad_q_heads_op(flat, pad_q_heads)
        return padded.view(b, s, pad_q_heads, d)
    return _pad_q_heads_op(q, pad_q_heads)


def fuse_layout_attn_out_for_uv(
    attn_out: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """Fused attention-out unpad/layout for W_UV absorb (custom op).

    Accepts ``[N, pad_H, D]`` or ``[B, S, pad_H, D]``; returns the same rank
    with ``num_q_heads`` and contiguous storage.
    """
    if attn_out.dim() == 4:
        b, s, h, d = attn_out.shape
        flat = attn_out.reshape(b * s, h, d)
        out = _layout_out_op(flat, num_q_heads)
        return out.view(b, s, num_q_heads, d)
    return _layout_out_op(attn_out, num_q_heads)
