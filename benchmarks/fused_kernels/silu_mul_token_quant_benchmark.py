# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

import vllm._custom_ops as ops


@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype

    def description(self):
        return f"N {self.num_tokens} x D {self.hidden_size} x DT {self.dtype}"


def get_bench_params() -> list[bench_params_t]:
    """Test configurations covering common model sizes."""
    NUM_TOKENS = [16, 128, 512, 2048]
    HIDDEN_SIZES = [1024, 2048, 4096, 5120, 14336]  # Common FFN sizes
    DTYPES = [torch.float16, torch.bfloat16]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, DTYPES)
    bench_params = list(map(lambda x: bench_params_t(x[0], x[1], x[2]), combinations))
    return bench_params


# ---------- Reference implementations ----------
def unfused_per_tensor_fp8_impl(x: torch.Tensor, quant_dtype: torch.dtype):
    """
    Unfused: SiLU+Mul then per‑tensor dynamic FP8 quant.
    (Uses the default behaviour of scaled_fp8_quant without scale)
    """
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up

    # per‑tensor dynamic quantization (scale is a scalar)
    silu_out, _ = ops.scaled_fp8_quant(silu_out)


def unfused_per_token_fp8_impl(x: torch.Tensor, quant_dtype: torch.dtype):
    """
    Unfused: SiLU+Mul then per‑token dynamic FP8 quant.
    """
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


def fused_per_token_impl(x: torch.Tensor, quant_dtype: torch.dtype):
    """
    Fused: SiLU+Mul+Per‑Token Quantization in a single CUDA kernel.
    """
    out, _ = ops.silu_and_mul_per_token_quant(
        x,
        quant_dtype,
        scale_ub=None,
    )


# ---------- Benchmark helpers ----------
def bench_fn(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    label: str,
    sub_label: str,
    fn: Callable,
    description: str,
) -> TMeasurement:
    min_run_time = 1
    globals = {
        "x": x,
        "quant_dtype": quant_dtype,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(x, quant_dtype)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(params: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]:
    """Run benchmarks for all implementations."""
    scale = 1 / params.hidden_size
    x = (
        torch.randn(
            params.num_tokens,
            params.hidden_size * 2,  # gate || up layout
            dtype=params.dtype,
            device="cuda",
        )
        * scale
    )

    timers = []

    # Unfused per‑tensor FP8 (for baseline)
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            label,
            sub_label,
            unfused_per_tensor_fp8_impl,
            "unfused_per_tensor_fp8",
        )
    )

    # Unfused per‑token FP8 (accurate reference)
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            label,
            sub_label,
            unfused_per_token_fp8_impl,
            "unfused_per_token_fp8",
        )
    )

    # Fused per‑token FP8
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            label,
            sub_label,
            fused_per_token_impl,
            "fused_per_token_fp8",
        )
    )

    return timers


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    torch.set_default_device("cuda")
    bench_params = get_bench_params()

    print(f"Running {len(bench_params)} benchmark configurations...")
    print(
        f"This will take approximately {len(bench_params) * 3} seconds (1s per variant)"
    )
    print()

    timers = []
    for bp in tqdm(bench_params):
        result_timers = bench(bp, "silu-mul-per-token-quant", bp.description())
        timers.extend(result_timers)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON - ALL RESULTS")
    print("=" * 80)
    print_timers(timers)


if __name__ == "__main__":
    main()
