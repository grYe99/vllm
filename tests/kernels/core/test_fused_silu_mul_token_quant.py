# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.platforms import current_platform

DTYPES = [torch.float16, torch.bfloat16]
QUANT_DTYPES = [current_platform.fp8_dtype()]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, *VEC_HIDDEN_SIZES, 2048, 5120]],
    *[(16, i) for i in [64, *VEC_HIDDEN_SIZES, 5120]],
    *[(128, i) for i in [64, *VEC_HIDDEN_SIZES]],
    *[(512, i) for i in [64, 5120]],
]
SCALE_UBS = [False]
SEEDS = [0]
CUDA_DEVICES = [i for i in range(1 if torch.accelerator.device_count() == 1 else 2)]


def ref_silu_and_mul_per_token_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: unfused SiLU+Mul then per-token dynamic FP8 quant."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)

    # SiLU(gate) * up
    silu_out = F.silu(gate) * up

    # per-token dynamic quantization
    out = torch.empty_like(silu_out, dtype=quant_dtype)
    scales = torch.empty(silu_out.size(0), 1, device=x.device, dtype=torch.float32)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        out,
        silu_out,
        scales,
        None,  # scale_ub=None
    )
    return out, scales


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device_idx", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_token_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device_idx: int,
) -> None:
    """Test SiLU+Mul+Per‑Token Dynamic Quantization kernel correctness."""
    torch.accelerator.set_device_index(device_idx)
    device = f"cuda:{device_idx}"
    torch.random.manual_seed(seed)
    torch.set_default_device(device)

    if has_scale_ub:
        pytest.skip("Scale upper bound not yet supported")

    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device) * scale

    # Reference implementation
    ref_out, ref_scales = ref_silu_and_mul_per_token_quant(x, quant_dtype)

    # Fused kernel implementation
    ops_out, ops_scales = ops.silu_and_mul_per_token_quant(
        x, quant_dtype, scale_ub=None
    )

    # Check for NaN/Inf
    assert not torch.isnan(ops_out.float()).any(), "Kernel output contains NaN"
    assert not torch.isinf(ops_out.float()).any(), "Kernel output contains Inf"
    assert not torch.isnan(ops_scales).any(), "Kernel scales contain NaN"
    assert not torch.isinf(ops_scales).any(), "Kernel scales contain Inf"

    # Check dtypes
    assert ops_out.dtype == quant_dtype
    assert ref_out.dtype == quant_dtype

    # Check scales match
    torch.testing.assert_close(ref_scales, ops_scales, rtol=1e-3, atol=1e-5)

    ref_deq = ref_out.to(dtype=torch.float32) * ref_scales
    ops_deq = ops_out.to(dtype=torch.float32) * ops_scales

    torch.testing.assert_close(ref_deq, ops_deq, atol=5e-2, rtol=5e-2)

    # opcheck
    output = torch.empty(num_tokens, hidden_size, device=device, dtype=quant_dtype)
    scales = torch.empty(num_tokens, device=device, dtype=torch.float32)
    opcheck(
        torch.ops._C.silu_and_mul_per_token_quant,
        (output, x, scales, None),
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_silu_per_token_quant_shapes(
    default_vllm_config,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
):
    """Test that output shapes are correct for per‑token quantization."""
    torch.set_default_device("cuda")
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.silu_and_mul_per_token_quant(
        x,
        current_platform.fp8_dtype(),
        scale_ub=None,
    )
    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, 1)
    assert out.dtype == current_platform.fp8_dtype()
    assert scales.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 14336])
def test_silu_per_token_quant_edge_cases(
    default_vllm_config, dtype: torch.dtype, batch_size: int, hidden_size: int
):
    """Test edge cases: single token, large batch, large hidden size."""
    torch.set_default_device("cuda")
    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.silu_and_mul_per_token_quant(
        x,
        current_platform.fp8_dtype(),
        scale_ub=None,
    )

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == current_platform.fp8_dtype()
    assert scales.dtype == torch.float32
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert not torch.isinf(scales).any()
