# GLM MegaAttention Phase-1 (SM90 / FP8 sparse MLA)

## Summary

Phase-1 adds an **opt-in** MegaAttention-*style* fuse for GLM-5.3 sparse MLA
decode on Hopper (SM90) with `FLASHMLA_SPARSE` + `fp8_ds_mla` (NoPE-512).

It does **not** port the SM100-only `FLASHMLA_MEGA_ATTN_DSV41` kernels. Those
remain DeepSeek V4.1 / Blackwell-specific. Phase-1 keeps the existing FlashMLA
sparse attention body and fuses the surrounding Q layout-pad and attention-out
layout launches that show up in decode profiles when
`_sparse_indexer_and_attn` runs under `eager_break_during_capture`.

## Flag

| Env | Default | Meaning |
| --- | --- | --- |
| `VLLM_GLM_MEGA_ATTN_PHASE1` | `0` (off) | Enable Phase-1 fuse |

Fail closed: if the flag is set on a non-CUDA-SM90 device, vLLM raises at first
use of the FlashMLA sparse decode path.

## What is fused

1. **Q prep (NoPE FP8 decode)** — `ql_nope` `[T, H, kv_lora]` → padded workspace
   `[T, pad_H, q_head_size]` (RoPE slot zero + head pad to 64/128) in one custom
   op / Triton kernel.
2. **Sparse MLA body** — unchanged `flash_mla_with_kvcache` (FlashMLA sparse).
3. **Out layout** — unpad heads + contiguous materialization for `W_UV` absorb
   in one custom op (also used when head pad still happens inside
   `_fp8_flash_mla_kernel`).

## What is deferred (Phase-2+)

- Indexer / top-k fusion
- MoE / NCCL / MTP changes
- Porting `FLASHMLA_MEGA_ATTN_DSV41` / SM100 mega kernels to H200
- NVFP4 attention

## Expected gain

Conservative **+3–8% decode GPU** on the profiled GLM-5.3 NVFP4 + FP8 KV setup.
The fuse candidate pool around sparse MLA is ~13–15% of GPU time; MoE (~50%) and
NCCL (~22–29%) are untouched. Do **not** advertise ~1.45× e2e decode.

## Design choice

**Option B-style**: extend `FlashMLASparseImpl` with optional fused pre/post
around the existing sparse decode kernel, gated by env. No new backend enum.

True single-kernel MegaAttention needs new SM90 kernels (out of scope). Phase-1
is the smallest change that removes real Python/CUDA launches on the GLM decode
hot path and stays CUDA-graph friendly via registered custom ops.

## Workspace reuse (alloc hygiene)

`forward_mqa` already reserves a `q_concat` buffer via
`current_workspace_manager`. For FP8 sparse decode the buffer is sized to the
kernel's padded head count (64/128) so both baseline pad and Phase-1 fuse can
write in-place — `new_zeros` / `new_empty` during CUDA graph capture invalidates
the stream (`cudaErrorStreamCaptureInvalidated`). Phase-1 fused Q-prep /
pad write into that buffer (`out=`).

**Out-layout policy:** the baseline returns a free head slice; `_v_up_proj`
tolerates it. Phase-1 only runs the materializing layout fuse when DCP LSE
merge needs packed contiguous heads (`need_to_return_lse_for_decode`). Always-on
layout fuse was measured ~+100µs eager on H200 microbench (B1 ≈ +119% vs B0).

FlashMLA sparse body (`flash_mla_with_kvcache` / `sparse_decode_fwd`) is
unchanged. If full-forward `do_bench_cudagraph` still fails after pad is
workspace-backed, the residual blocker is inside that body (fresh `out`/`lse`
or scheduler metadata alloc during capture) — not Phase-1 layout ops (those
capture cleanly in the layout µs harness).

## Code map

- `vllm/envs.py` — `VLLM_GLM_MEGA_ATTN_PHASE1`
- `vllm/v1/attention/backends/mla/glm_mega_attn_phase1.py` — gates + fused ops
- `vllm/v1/attention/backends/mla/flashmla_sparse.py` — NoPE Q pack + FP8 pad/unpad
