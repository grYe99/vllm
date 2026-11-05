# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 MegaAttention-style fuse for GLM sparse MLA decode (SM90 / FP8).

GLM-5.3 sparse MLA (``FLASHMLA_SPARSE`` + ``fp8_ds_mla``) already has a solid
FlashMLA sparse body. Profile data shows decode GPU time is still split across
several small launches around that body:

1. Q layout / pad into the FP8 FlashMLA workspace (RoPE slot + heads → 64/128)
2. FlashMLA sparse attention kernel (unchanged; keep existing FlashMLA sparse)
3. Attention output unpad + contiguous layout for ``W_UV`` / DCP LSE paths
   that need packed heads (non-DCP decode keeps the baseline free slice)

True SM100 MegaAttention kernels (``FLASHMLA_MEGA_ATTN_DSV41``) are out of
scope for Hopper. Phase-1 therefore fuses the *pre* and *post* layout steps
around the existing sparse MLA kernel into CUDA-graph-friendly custom ops,
gated by ``VLLM_GLM_MEGA_ATTN_PHASE1`` (default off). Fail closed unless the
device is exactly SM90.

Hot-path alloc hygiene: callers that already hold a
``current_workspace_manager`` Q buffer (``q_concat_buffer`` in
``FlashMLASparseImpl.forward_mqa``) should pass it as ``out=`` so the fused
kernels write in-place. Allocating via ``new_empty`` is only a fallback for
tests / paths without a reserved workspace.

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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: pack ``[T, H, kv_lora]`` into ``[T, pad_H, q_head_size]``.

    Copies real heads into ``[:num_q_heads, :kv_lora_rank]`` and zero-fills the
    RoPE slot plus any padded head rows. Matches the historical FlashMLA sparse
    FP8 NoPE decode layout. When ``out`` is provided, writes into it in-place.
    """
    assert ql_nope.dim() == 3
    tokens = ql_nope.shape[0]
    assert ql_nope.shape[1] == num_q_heads
    assert ql_nope.shape[2] == kv_lora_rank
    assert pad_q_heads >= num_q_heads
    assert q_head_size >= kv_lora_rank
    if out is None:
        q = ql_nope.new_empty((tokens, pad_q_heads, q_head_size))
    else:
        assert out.shape == (tokens, pad_q_heads, q_head_size)
        q = out
    q.zero_()
    q[:, :num_q_heads, :kv_lora_rank].copy_(ql_nope)
    return q


def layout_attn_out_for_uv_ref(
    attn_out: torch.Tensor,
    num_q_heads: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: unpad heads and materialize contiguous ``[T, H, D]`` for W_UV."""
    assert attn_out.dim() == 3
    tokens, _, head_size = attn_out.shape
    sliced = attn_out[:, :num_q_heads, :]
    if out is None:
        if sliced.is_contiguous():
            return sliced
        return sliced.contiguous()
    assert out.shape == (tokens, num_q_heads, head_size)
    out.copy_(sliced)
    return out


def pad_q_heads_ref(
    q: torch.Tensor,
    pad_q_heads: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: pad ``[N, H, D]`` (or ``[B, S, H, D]``) heads to ``pad_q_heads``."""
    if q.dim() == 4:
        b, s, h, d = q.shape
        if h == pad_q_heads:
            if out is None:
                return q
            out.copy_(q)
            return out
        if out is None:
            dest = q.new_zeros((b, s, pad_q_heads, d))
        else:
            assert out.shape == (b, s, pad_q_heads, d)
            dest = out
            dest.zero_()
        dest[:, :, :h, :].copy_(q)
        return dest
    assert q.dim() == 3
    n, h, d = q.shape
    if h == pad_q_heads:
        if out is None:
            return q
        out.copy_(q)
        return out
    if out is None:
        dest = q.new_zeros((n, pad_q_heads, d))
    else:
        assert out.shape == (n, pad_q_heads, d)
        dest = out
        dest.zero_()
    dest[:, :h, :].copy_(q)
    return dest


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
    out: torch.Tensor,
) -> None:
    """Write packed NoPE Q into ``out`` (mutates). No per-call alloc."""
    tokens = ql_nope.shape[0]
    assert out.shape == (tokens, pad_q_heads, q_head_size)
    if tokens == 0 or pad_q_heads == 0:
        return

    if (not ql_nope.is_cuda) or (not HAS_TRITON):
        prepare_q_nope_fp8_decode_ref(
            ql_nope,
            num_q_heads,
            pad_q_heads,
            q_head_size,
            kv_lora_rank,
            out=out,
        )
        return

    block_d = _next_pow2_ge(q_head_size)
    grid = (tokens * pad_q_heads,)
    _fuse_prepare_q_nope_fp8_kernel[grid](
        ql_nope,
        out,
        tokens,
        num_q_heads,
        pad_q_heads,
        kv_lora_rank,
        q_head_size,
        ql_nope.stride(0),
        ql_nope.stride(1),
        ql_nope.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )


def _fuse_pad_q_heads_impl(
    q: torch.Tensor,
    pad_q_heads: int,
    out: torch.Tensor,
) -> None:
    """Pad last-head dim of a 3D ``[N, H, D]`` tensor into ``out`` (mutates)."""
    assert q.dim() == 3
    rows, num_q_heads, head_size = q.shape
    assert out.shape == (rows, pad_q_heads, head_size)
    if num_q_heads == pad_q_heads:
        if out.data_ptr() != q.data_ptr():
            out.copy_(q)
        return
    if rows == 0:
        return

    if (not q.is_cuda) or (not HAS_TRITON):
        pad_q_heads_ref(q, pad_q_heads, out=out)
        return

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


def _fuse_layout_attn_out_for_uv_impl(
    attn_out: torch.Tensor,
    num_q_heads: int,
    out: torch.Tensor,
) -> None:
    """Unpad / layout attention out into contiguous ``out`` (mutates)."""
    tokens, pad_q_heads, head_size = attn_out.shape
    assert out.shape == (tokens, num_q_heads, head_size)
    if tokens == 0 or num_q_heads == 0:
        return

    if (
        pad_q_heads == num_q_heads
        and attn_out.is_contiguous()
        and out.data_ptr() == attn_out.data_ptr()
    ):
        return

    if (not attn_out.is_cuda) or (not HAS_TRITON):
        layout_attn_out_for_uv_ref(attn_out, num_q_heads, out=out)
        return

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


def _fuse_prepare_q_nope_fp8_decode_fake(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
    out: torch.Tensor,
) -> None:
    return None


def _fuse_pad_q_heads_fake(
    q: torch.Tensor,
    pad_q_heads: int,
    out: torch.Tensor,
) -> None:
    return None


def _fuse_layout_attn_out_for_uv_fake(
    attn_out: torch.Tensor,
    num_q_heads: int,
    out: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="glm_mega_attn_phase1_prepare_q",
    op_func=_fuse_prepare_q_nope_fp8_decode_impl,
    mutates_args=["out"],
    fake_impl=_fuse_prepare_q_nope_fp8_decode_fake,
)

direct_register_custom_op(
    op_name="glm_mega_attn_phase1_pad_q_heads",
    op_func=_fuse_pad_q_heads_impl,
    mutates_args=["out"],
    fake_impl=_fuse_pad_q_heads_fake,
)

direct_register_custom_op(
    op_name="glm_mega_attn_phase1_layout_out",
    op_func=_fuse_layout_attn_out_for_uv_impl,
    mutates_args=["out"],
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused NoPE Q pack + head pad for FP8 sparse MLA decode (custom op).

    Prefer passing a preallocated ``out`` from ``current_workspace_manager`` /
    ``q_concat_buffer`` so CUDA-graph / eager decode does not allocate scratch
    each step. When ``out`` is omitted, allocates a fresh buffer (tests only).
    """
    if out is None:
        out = ql_nope.new_empty((ql_nope.shape[0], pad_q_heads, q_head_size))
    _prepare_q_op(ql_nope, num_q_heads, pad_q_heads, q_head_size, kv_lora_rank, out)
    return out


def fuse_pad_q_heads(
    q: torch.Tensor,
    pad_q_heads: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused head pad for FP8 sparse MLA decode.

    Accepts ``[N, H, D]`` or ``[B, S, H, D]``; returns the same rank.
    Pass ``out`` to avoid a per-call alloc when a workspace buffer is available.
    """
    if q.dim() == 4:
        b, s, h, d = q.shape
        if h == pad_q_heads:
            if out is None:
                return q
            out.copy_(q)
            return out
        flat = q.reshape(b * s, h, d)
        if out is None:
            padded = flat.new_empty((b * s, pad_q_heads, d))
        else:
            assert out.shape == (b, s, pad_q_heads, d)
            padded = out.reshape(b * s, pad_q_heads, d)
        _pad_q_heads_op(flat, pad_q_heads, padded)
        return padded.view(b, s, pad_q_heads, d)
    rows, h, d = q.shape
    if h == pad_q_heads:
        if out is None:
            return q
        out.copy_(q)
        return out
    if out is None:
        out = q.new_empty((rows, pad_q_heads, d))
    _pad_q_heads_op(q, pad_q_heads, out)
    return out


def fuse_layout_attn_out_for_uv(
    attn_out: torch.Tensor,
    num_q_heads: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused attention-out unpad/layout for W_UV absorb (custom op).

    Accepts ``[N, pad_H, D]`` or ``[B, S, pad_H, D]``; returns the same rank
    with ``num_q_heads`` and contiguous storage. Pass ``out`` to reuse a
    preallocated workspace buffer.
    """
    if attn_out.dim() == 4:
        b, s, h, d = attn_out.shape
        flat = attn_out.reshape(b * s, h, d)
        if out is None:
            dest = flat.new_empty((b * s, num_q_heads, d))
        else:
            assert out.shape == (b, s, num_q_heads, d)
            dest = out.reshape(b * s, num_q_heads, d)
        _layout_out_op(flat, num_q_heads, dest)
        return dest.view(b, s, num_q_heads, d)
    tokens, _, head_size = attn_out.shape
    if out is None and attn_out.shape[1] == num_q_heads and attn_out.is_contiguous():
        return attn_out
    if out is None:
        out = attn_out.new_empty((tokens, num_q_heads, head_size))
    _layout_out_op(attn_out, num_q_heads, out)
    return out
