# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""µs harness: baseline copy_/zero_/slice vs Phase-1 layout fuse (CUDA Graph).

Times the Phase-1 Q-prep / pad / out-layout fused custom ops against the
historical eager baseline (``new_empty`` + ``zero_`` + ``copy_`` / slice).
Uses ``triton.testing.do_bench_cudagraph`` when CUDA is present so numbers
reflect graph-captured launch cost (the decode hot path).

FlashMLA sparse attention itself is **not** timed here — layout fuse only.

CLI
---
::

    # Requires CUDA. Prints a table; does not invent numbers when CUDA is absent.
    PYTHONPATH=. python benchmarks/kernels/benchmark_glm_mega_attn_phase1_layout.py

    PYTHONPATH=. python benchmarks/kernels/benchmark_glm_mega_attn_phase1_layout.py \
        --tokens 1,8,32,128 --pad-heads 64 --heads 8,16 --reps 200

Axes (defaults)
---------------
- tokens ∈ {1, 8, 32, 128}
- pad_heads = 64
- live heads ∈ {8, 16} (plus any passed via ``--heads``)
- q_head_size = 576, kv_lora_rank = 512, dtype = bf16
"""

from __future__ import annotations

import argparse
import sys

import torch


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def baseline_prepare(
    ql_nope: torch.Tensor,
    num_q_heads: int,
    pad_q_heads: int,
    q_head_size: int,
    kv_lora_rank: int,
    out: torch.Tensor,
) -> torch.Tensor:
    out.zero_()
    out[:, :num_q_heads, :kv_lora_rank].copy_(ql_nope)
    return out


def baseline_layout(
    attn_out: torch.Tensor, num_q_heads: int, out: torch.Tensor
) -> torch.Tensor:
    out.copy_(attn_out[:, :num_q_heads, :])
    return out


def baseline_pad(q: torch.Tensor, pad_q_heads: int, out: torch.Tensor) -> torch.Tensor:
    out.zero_()
    out[:, : q.shape[1], :].copy_(q)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tokens", type=str, default="1,8,32,128")
    parser.add_argument("--pad-heads", type=int, default=64)
    parser.add_argument("--heads", type=str, default="8,16")
    parser.add_argument("--q-head-size", type=int, default=576)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument(
        "--ops",
        type=str,
        default="prepare,layout,pad",
        help="Comma list of: prepare, layout, pad",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "CUDA is not available; this µs harness is a no-op on CPU hosts. "
            "Re-run on an SM90 GPU to collect timings.",
            file=sys.stderr,
        )
        return 0

    from vllm.triton_utils import triton
    from vllm.v1.attention.backends.mla import glm_mega_attn_phase1 as phase1

    tokens_list = _parse_int_list(args.tokens)
    heads_list = _parse_int_list(args.heads)
    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    pad_heads = args.pad_heads
    q_head_size = args.q_head_size
    kv_lora = args.kv_lora_rank
    dtype = torch.bfloat16
    device = "cuda"
    quantiles = [0.5, 0.2, 0.8]

    print("=" * 78)
    print("GLM MegaAttn Phase-1 layout fuse µs harness (CUDA Graph)")
    print("=" * 78)
    print(
        f"pad_heads={pad_heads} q_head_size={q_head_size} "
        f"kv_lora={kv_lora} dtype=bf16 reps={args.reps}"
    )
    print(f"tokens={tokens_list} heads={heads_list} ops={sorted(ops)}")
    print()

    header = (
        f"{'op':<10} {'T':>5} {'H':>4} {'pad':>4} | "
        f"{'base us':>10} {'fuse us':>10} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for tokens in tokens_list:
        for num_heads in heads_list:
            if num_heads > pad_heads:
                continue

            if "prepare" in ops:
                ql = torch.randn(tokens, num_heads, kv_lora, dtype=dtype, device=device)
                out_b = torch.empty(
                    tokens, pad_heads, q_head_size, dtype=dtype, device=device
                )
                out_f = torch.empty_like(out_b)

                def run_base(
                    ql=ql,
                    num_heads=num_heads,
                    out_b=out_b,
                ):
                    return baseline_prepare(
                        ql, num_heads, pad_heads, q_head_size, kv_lora, out_b
                    )

                def run_fuse(
                    ql=ql,
                    num_heads=num_heads,
                    out_f=out_f,
                ):
                    return phase1.fuse_prepare_q_nope_fp8_decode(
                        ql,
                        num_heads,
                        pad_heads,
                        q_head_size,
                        kv_lora,
                        out=out_f,
                    )

                # Correctness smoke before timing.
                torch.testing.assert_close(run_base(), run_fuse(), atol=0, rtol=0)

                ms_b, _, _ = triton.testing.do_bench_cudagraph(
                    run_base, quantiles=quantiles, rep=args.reps
                )
                ms_f, _, _ = triton.testing.do_bench_cudagraph(
                    run_fuse, quantiles=quantiles, rep=args.reps
                )
                us_b, us_f = ms_b * 1000, ms_f * 1000
                print(
                    f"{'prepare':<10} {tokens:>5} {num_heads:>4} {pad_heads:>4} | "
                    f"{us_b:>10.2f} {us_f:>10.2f} {us_b / max(us_f, 1e-9):>8.2f}x"
                )

            if "layout" in ops:
                attn = torch.randn(
                    tokens, pad_heads, kv_lora, dtype=dtype, device=device
                )
                out_b = torch.empty(
                    tokens, num_heads, kv_lora, dtype=dtype, device=device
                )
                out_f = torch.empty_like(out_b)

                def run_base_l(
                    attn=attn,
                    num_heads=num_heads,
                    out_b=out_b,
                ):
                    return baseline_layout(attn, num_heads, out_b)

                def run_fuse_l(
                    attn=attn,
                    num_heads=num_heads,
                    out_f=out_f,
                ):
                    return phase1.fuse_layout_attn_out_for_uv(
                        attn, num_heads, out=out_f
                    )

                torch.testing.assert_close(run_base_l(), run_fuse_l(), atol=0, rtol=0)
                ms_b, _, _ = triton.testing.do_bench_cudagraph(
                    run_base_l, quantiles=quantiles, rep=args.reps
                )
                ms_f, _, _ = triton.testing.do_bench_cudagraph(
                    run_fuse_l, quantiles=quantiles, rep=args.reps
                )
                us_b, us_f = ms_b * 1000, ms_f * 1000
                print(
                    f"{'layout':<10} {tokens:>5} {num_heads:>4} {pad_heads:>4} | "
                    f"{us_b:>10.2f} {us_f:>10.2f} {us_b / max(us_f, 1e-9):>8.2f}x"
                )

            if "pad" in ops:
                q = torch.randn(
                    tokens, num_heads, q_head_size, dtype=dtype, device=device
                )
                out_b = torch.empty(
                    tokens, pad_heads, q_head_size, dtype=dtype, device=device
                )
                out_f = torch.empty_like(out_b)

                def run_base_p(q=q, out_b=out_b):
                    return baseline_pad(q, pad_heads, out_b)

                def run_fuse_p(q=q, out_f=out_f):
                    return phase1.fuse_pad_q_heads(q, pad_heads, out=out_f)

                torch.testing.assert_close(run_base_p(), run_fuse_p(), atol=0, rtol=0)
                ms_b, _, _ = triton.testing.do_bench_cudagraph(
                    run_base_p, quantiles=quantiles, rep=args.reps
                )
                ms_f, _, _ = triton.testing.do_bench_cudagraph(
                    run_fuse_p, quantiles=quantiles, rep=args.reps
                )
                us_b, us_f = ms_b * 1000, ms_f * 1000
                print(
                    f"{'pad':<10} {tokens:>5} {num_heads:>4} {pad_heads:>4} | "
                    f"{us_b:>10.2f} {us_f:>10.2f} {us_b / max(us_f, 1e-9):>8.2f}x"
                )

    print()
    print("Done. Attention body (flash_mla_with_kvcache) was not timed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
