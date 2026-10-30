# RFC: Colocated Engram Lookup ∥ Multimodal GPU Encoder Overlap

Status: draft, experimental flag (`VLLM_ENGRAM_PREFETCH_BEFORE_ENCODER`, default off)

## Motivation

DeepSeek-V4.1-style Engram layers inject n-gram table lookups into a few
decoder layers. Since [#56512](https://github.com/vllm-project/vllm/pull/56512),
the lookup (hash → gather → offloaded-table read) runs on a side prefetch
stream, started at the language-model forward entry so it overlaps the early
decoder layers.

For multimodal requests the expensive part of the step is not the decoder: the
GPU encoder (ViT + aligner) runs **before** the LM forward entry, in the model
runner's input-preparation phase. Today the engram launch happens only after
that encode has fully consumed the GPU, so the lookup's side-stream window —
the one piece of work that could run concurrently — starts after the best
free GPU window is already gone:

```text
today:     [ GPU encode ][ hash + prepare ][ decoder layers (wait) ]
proposed:  [ hash + prepare starts ][ GPU encode ][ decoder layers (wait only) ]
                                        └── side-stream lookup overlaps encode
```

The window is the encoder itself: CPU preprocessing is out of scope (it cannot
hide device-side UVA reads well, and ids/mask are only final once device
tensors are staged), and no E/P/D disaggregation is considered — colocated
serving only.

## Proposed Change

For colocated steps that **will run a GPU encode** (scheduler-scheduled
encoder inputs that survive the encoder-cache filter), start the single engram
hash+prepare **just before the encoder is submitted**, and make the LM forward
entry adopt it instead of launching again. All other steps — text-only,
decode, encoder-cache hits, graph replays — keep today's entry-time prepare
exactly.

Concretely, at this tree's tip:

- `DeepseekV4Model.prepare_engram_embeddings(...)` (both hardware variants):
  the existing entry-block body (n-gram hash over `input_ids` / image-sentinel
  mask / lookback window, then `EngramBatch.prepare_embeddings` or the
  per-layer `prepare_embeddings`), extracted verbatim into a method callable
  from two sites.
- `DeepseekV41ModelState.maybe_prefetch_engram_before_encoder(...)`: a runner
  hook called by the V2 GPU model runner immediately before
  `prepare_inputs_embeds` submits the encoder. It gathers the lookback window,
  opens a short-lived forward context (the same mechanism the decoder-replay
  sub-forward already uses) carrying this step's attention metadata and DP
  token counts, runs `prepare_engram_embeddings`, and stores the
  `(hashes, mask)` pair on the model.
- `DeepseekV4Model.forward` entry: first consumes the stored pair
  (`_take_early_engram_embeddings`); only when no pair was stored does it call
  `prepare_engram_embeddings` as before.

Fusion / gating / WKV stay at the engram layers — only the lookup's **launch
time** moves. One prepare per step is structural, not disciplinary: the pair
is consumed exactly once and the runner hook clears any leftover before
gating.

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| Hook lives in the model runner (V2), not in `embed_multimodal` | The hash needs runner-owned inputs (SWA attention metadata, `lookback_token_ids`, DP token counts) that are assembled around — not inside — the model's `embed_multimodal`. The V2 runner has all of them on device right before `execute_mm_encoder`. |
| Eager (cg_mode=NONE) steps only | Under cudagraph replay the entry prepare is baked into the captured segment list; a Python-side "already prepared" check would either double-launch or skip a launch the replay expects. Captures never schedule encoder inputs, so the early branch is never baked in; eager MM prefills — the steps that matter — always take the hook. PIECEWISE/breakable-CG MM steps keep entry-time prepare (no overlap there; documented limitation). |
| Private per-replica engram tables only (`get_engram_dp_size() == 1`) | With a DP-sharded table the prepare is collective; ranks can disagree on whether an encoder runs this step, and an early-vs-entry mismatch would deadlock the collective. Private tables make the launch purely local. Extending to sharded tables needs one agreed boolean per step (piggybacked on the existing DP sync) — future work. |
| `will_run_gpu_encoder` = non-empty `mm_kwargs` from the existing filter | Reuses `EncoderRunner.prepare_mm_inputs`: encoder-cache hits, `prompt_embeds` passthrough, and data-less features are already excluded. One boolean for all modalities (currently image; video/audio inherit the same path when added). |
| Flag is an env var, default off | Experimental; keeps it out of config-hash and engine args until the MM TTFT win is measured. An `--engram-prefetch-before-encoder` engine arg is a mechanical follow-up once the flag graduates. |
| Stale-store hardening | The hook clears any stored pair **before** its gates, so a pair left by a step that died between launch and forward can never be adopted by a later step; the entry also refuses adoption outside eager mode. |
| V1 model runner unchanged | MRV1 (`gpu_model_runner.py`) has a parallel but different input pipeline; the entry change is inert there (nothing ever stores a pair). Wiring the V1 hook is follow-up work if MRV1 deployments need it. |

Out of scope (unchanged from the design draft): E/P/D early fetch on a P
instance while E encodes; GPU preprocess / CUDA decode as a hide window;
migrating or extending [#64500](https://github.com/vllm-project/vllm/pull/64500)'s
host-side staging (its behavior is orthogonal — this change only moves *when*
`prepare_embeddings` is invoked).

## What is hidden, and what is not

Hidden: the engram **lookup** — hash kernels, DP-hash gather (no-op here),
offloaded-table UVA reads, staging unpack — while the ViT/aligner occupies the
main stream. Not hidden: the fused gate/WKV at the engram layers (needs hidden
states) and any inline-threshold lookup (very large batches where #56512
already runs the lookup inline on the main stream; for those the reorder is
TTFT-neutral). Expected benefit shape: ΔTTFT ≈ the exposed engram tail that
previously sat after the encode, upper-bounded by lookup cost.

## Related PRs

- [#56512](https://github.com/vllm-project/vllm/pull/56512) — engram side-stream prefetch vs the transformer (this change re-times its launch; per-layer staging semantics untouched).
- [#64500](https://github.com/vllm-project/vllm/pull/64500) — host-side prefetch at the LM eager boundary; neighbor work, not duplicated here.
- [#63068](https://github.com/vllm-project/vllm/pull/63068) — background context for the multimodal encoder runner structure this hooks into.

## Feedback Period

Until the flag is either graduated to an engine arg with measured MM TTFT
numbers or withdrawn. Please comment on: (1) whether PIECEWISE/breakable-CG MM
steps should also overlap (needs a capture-aware scheme), (2) the right
long-term home for the flag (env var vs `EngramConfig` field vs engine arg),
(3) whether MRV1 should get the mirrored hook.

## Any Other Things

- The hook re-runs `prepare_mm_inputs` (cheap CPU scan) so the real
  `execute_mm_encoder` call re-derives the same item list; `prompt_embeds`
  items are cached by the first pass, so no work is duplicated beyond the
  scan.
- When the hook fires, `prepare_inputs` gathers the lookback window a second
  time later in the step (idempotent write of identical values, one extra
  µs-scale Triton launch per MM prefill; marked with a `shortcut:` comment).
- Encoder CUDA graphs: overlap still holds — the side-stream lookup is
  ordered after prior main-stream work and the encoder graph replay is
  submitted after the lookup launch, on the main stream.
- Qwen4Exp PLE layers share `EngramConfig` plumbing but a different n-gram
  module; this feature is DeepSeek-V4.1-only as implemented (the hook checks
  the model's `engram_hash`).
- No @mentions by policy; reviewers will be subscribed via the normal
  CODEOWNERS flow.

## CC List

(none)
