# Hybrid EPD Store delivery test plan

Status: planned, not executed. This document is a lab acceptance plan, not a
record of passing tests. AI assistance was used to prepare it.

**Goal:** establish whether P-side Store consumption improves complete
GLM-5.3-Flash E-first serving relative to the existing cross-Encoder relay.

**Architecture:** two encoder-only instances at TP1 and one combined Prefill /
Decode instance at TP4, sharing a RAM Mooncake Store. Pin the Encoder used for
each request; keep the PD and GPU mapping fixed across comparisons.

**Spec:** [design and source analysis](../specs/2026-11-09-epd-hybrid-store-design.md).

**Feasibility and patch scope:**
[latest upstream report](../../design/epd_hybrid_push_store_feasibility.md).

## Minimal implementation gates

Implementation is not started. The report checks upstream main
`a3c14efbc77c018e8bf579935095a87721349e05` as well as this branch's fork base.
Work through these gates before the full lab matrix below:

1. **Protocol proof:** lookup-only backend, per-item control handoff and bounded
   P reader, actual tensor readiness and source-free E completion. Test that
   the reader runs before any P inference request exists. Use existing
   cross-cache/control/protocol/worker suites with native I/O faked.
2. **Runner/cache closure:** explicit remote satisfaction and scheduler
   nonresident feedback. A second same-image request after Store eviction must
   recheck/encode rather than see a false E-local hit. Cover mixed/repeated
   inputs and preserve real resident tensors and reference/budget accounting.
3. **Real fail-fast PoC:** one receiving worker's Store roundtrip, then the
   agreed full Flash 2E TP1 + PD TP4 topology. Prove cross-E and mixed delivery;
   GET failure must explicitly terminate and safely settle owned buffers.
   This does not establish automatic retry or production fallback.
4. **Production evidence:** add and validate same-request E fallback for safe
   recoverable GET failures before E returns, compatibility/default-off behavior,
   cancellation/shutdown and the paired performance matrix below. Update
   rendered usage docs for the actual shipped option and support limits.

No gate changes Proxy scheduling, adds a Store-only connector or redesigns TP
distribution. Source selection before reservation in the simple PoC can lose
current cold-path overlap; measure it rather than assuming unchanged P2P payload
code means unchanged latency. The fallback assertions below belong to gate 4.

## Constraints and observed environment

Read-only discovery on 2026-11-09 found:

- Rundao namespace `public`, pod **`test-ygr-69cc75df5c-b6s6t`**, node
  `gpu-10-127-107-18`, IP `172.16.11.213`. This is `test-ygr`, not
  `test-ygr-h200`. Re-resolve the live suffix before execution.
- Eight H200 GPUs, each 143771 MiB; all reported 0 MiB used at inspection.
  All GPU pairs reported NV18. This snapshot is not a reservation of those GPUs.
- `/glm-5-3-flash-nfs/config.json`: `Glm5NextForConditionalGeneration`,
  `glm5_next`, 45 text layers, hidden width 4096, FP8 checkpoint with BF16
  vision weights in the source model implementation. Vision: depth 24,
  hidden width 1024, output width 4096, patch 14, spatial merge 2.
- Root checkpoint index reports `total_size=328326771576` bytes, about
  **305.8 GiB**; there are 62 shards. This is tensor storage, not a measured
  GPU memory requirement. Use this exact full Flash checkpoint in all arms,
  with identical E and PD model paths/revision and preprocessing settings.
- `/models/GLM-5.3-Flash` also exists. Do not assume it aliases the NFS root;
  compare config/index checksums before substituting a model path.
- Installed distribution metadata: vLLM `0.32.0`, Mooncake CUDA13 package
  `0.3.13.post1`. `uv`, `vllm`, `mooncake_master`, `mooncake_client` exist.
  The installed wheel is not proof that the pinned fork's MRv2 GLM model and
  #56242 call chain can run. No import or server startup was tested here.

User-confirmed topology: **GPU 0 E0 TP1, GPU 1 E1 TP1, GPUs 2–5 PD TP4**.
GPUs 6–7 stay unused by this experiment. The mapping is a proposed fixed layout,
not a claim that all six processes/ranks have launched successfully. Record
NUMA/NIC placement and keep it fixed; run one later swapped-E placement check
if the measured effect is small. This is an intra-node TP4 experiment, not a
multi-node RDMA performance claim.

No new pod, no killing unrelated processes, and no in-place overwrite of the
shared pod's site-packages. Use a task-local checkout/environment or isolated
wheel with compatible compiled artifacts. Use `uv` and `.venv/bin/python` for
Python commands. Keep the full checkpoint and existing services unchanged.

## 1. Preflight and reproducibility manifest

- [ ] Connect through the Rundao skill's `ygr-dev0` jump and password-file /
  ControlMaster workflow. Use `KUBECONFIG=/home/jovyan/.kube/config` on the jump.
  Resolve `test-ygr` by workload identity, then inspect GPUs and process owners.
- [ ] Allocate an experiment directory on a verified persistent mount with a
  unique run ID, for example
  `/glm-5-3-flash-nfs/benchmarks/epd-hybrid-store/<run-id>/`. Verify actual
  mounts; do not assume `/ygr-sc` is shared or exists.
- [ ] Record source SHA, patch SHA, installed versions, model config/index
  checksums, driver/CUDA, GPU UUIDs, topology, CPU/NUMA affinity, Store version,
  services/ports, protocol, Store replica layout and all launch arguments.
  Redact credentials; record no ambient environment dump.
- [ ] Establish an isolated environment compatible with source
  `625d2d74eab9b1da1a9793d97100d714b885f2e0`. Check `uv` and an existing
  interpreter before installing anything. Confirm actual imports and
  `vllm serve --help` in that environment. Record `vllm.__file__` so a source
  overlay cannot silently use the wrong wheel.
- [ ] Run the Store and connector capability smoke tests before loading the
  306-GiB model. Use a private cache prefix and isolated Store tenant/services;
  never clear a shared Store. Discover free loopback ports rather than assuming
  an existing endpoint is ours. Do not start or replace a shared Store daemon.
- [ ] Start one PD TP4 and the two E TP1 processes. Confirm E logs select
  `MMEncoderModelRunner`, load only the vision-side weights, create no language
  model/KV/sampler, and set encoder-only mode. `ec_producer` automatically
  resolves `mm_encoder_only` at this source; using a pooling/BERT runner is wrong.
- [ ] Confirm full Flash PD reaches health readiness, all four EC receiving
  ranks advertise correct ports, and memory leaves room for KV, the EC pool,
  Store read staging and temporary tensors. Start with `max_model_len=8192`,
  `max_num_seqs=16`, `max_num_batched_tokens=8192`; adjust only before fixing
  the common A/B configuration if memory/profile constraints require it.
- [ ] Initially disable MTP and prefix caching; use identical model/kernel/
  attention/quantization/graph settings in both arms. Start with eager E graphs
  disabled for mechanism checks, then test one fixed graph setting in both
  performance arms. Keep `mm_processor_cache_gb > 0` throughout.
- [ ] Set `--no-async-scheduling` on E in both arms. The implemented experimental
  hybrid mode rejects asynchronous E scheduling to prevent an in-flight
  remote-only output from becoming a false local hit. Keep P's scheduling
  setting identical in both arms; asynchronous P remains supported.

Suggested loopback assignment, subject to free-port checks: E0 HTTP 18100,
E1 HTTP 18101, PD HTTP 18200, PD EC control base 19100 (ranks 19100–19103),
test proxies 18300/18301. Store endpoints are discovered/configured separately;
do not reuse a default port merely because its health check succeeds.

Use `tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh`
as the launch/lifecycle reference, not as a ready-made Flash TP4 harness: its
default baseline and PD are single-GPU and need adaptation. Launch E through
`VLLM_USE_V2_MODEL_RUNNER=1` and `ECMooncakeConnector/ec_producer`; P through
MRv2 and `ECMooncakeConnector/ec_consumer`. Configure `cross_encoder_cache`
only on E; configure the proposed `prefill_store_delivery` on both sides for
the hybrid arm. Give E and P the same approved namespace settings and Store
configuration. Do not enable a proposed flag against unmodified main.

## 2. Deterministic cross-Encoder request routing

Do not add a production routing feature just for the benchmark. The reference
proxy round-robins per multimodal item using a shared cursor, so “send every
other request” is not reliable once health checks, concurrent clients or
multi-image requests interleave.

For single-E attribution, launch two static reference proxy configurations:
one lists only E0 and the fixed PD, the other only E1 and the same PD. Thus the
client's proxy endpoint pins E without changing the E-first flow or inventing
request headers. Verify selected E address and PD control address in logs.

For a mixed-source request, send all three images through the E1-pinned proxy;
their preconditions, not different routing within the request, create the three
delivery sources. Use a normal two-E reference proxy only in the later natural
round-robin workload, with assignment recorded per item.

Maintain a manifest for each occurrence: request ID, item index, raw fixture
checksum, stable input UUID if used, **engine-reported** `mm_hash`, transfer ID,
selected E, PD, TP shard and expected source. Reuse actual bytes/processing
settings; do not make up `mm_hash` from the image UUID. Fresh transfer IDs are
required for new requests. Store codec keys and placeholder metadata must use
the actual reported model contracts.

### Controlled Store-hot / E1-cold / PD-cold workload

1. Choose a bank of distinct image fixtures. Seed them through **E0's real
   encoder-only HTTP endpoint without `consumer_zmq`**, so normal forward and
   asynchronous Store publication run without warming PD. Verify this
   destination-free publication behavior first; if unsupported, seed through
   the real producer with a disposable consumer fixture, then release it before
   the full-model run. Never seed the primary end-to-end comparison with
   handcrafted serialized tensors.
2. Poll publication visibility from a separate observer with a bounded deadline
   and validate a sample object against its `TensorSpec`. E HTTP success does
   not imply Store readiness. Record failed/skipped publications and exclude
   them before generating the timed request manifest.
3. E1 and PD have never seen these image identifiers. Warm their processors,
   allocators, graph shapes and Store buffers with a disjoint fixture bank.
   Each measured image goes to E1 **once per arm**. In relay baseline, the first
   GET makes it E1-local; reusing it later would invalidate the experiment.
4. Before every new arm, reset state by restarting only owned E1/PD processes
   and warming disjoint inputs again. Retain the same E0/Store seed bank or
   reseed an isolated equivalent namespace. Use the exact same ordered bank
   for a paired B/C block. For larger banks/concurrency, keep all objects
   resident in Store and verify this instead of assuming no eviction.
5. Prove the intended E1-local miss and P-local miss from instrumentation before
   timing. Prefix caching stays disabled for this mechanism-isolating workload.
   A request's text nonce alone does not establish that the image's prefix KV
   or P embedding was cold.

### Per-request mixed workload

On E1, prime image L without a PD destination and wait for its tensor to be
locally resident; on E0, seed S only to Store; never seed M. Send `[L, S, M]`
in one request to the E1 proxy and a cold PD. Expected: L P2P, S P Store GET,
M one vision forward + P2P + best-effort PUT. Include repeated occurrences
`[S, S, M]` in a separate correctness case to test shared bytes and independent
occurrence leases. Use `limit_mm_per_prompt` allowing at least three images.

## 3. Behavioral tests before serving measurements

Before adding tests, read `tests/AGENTS.md` and state each test's contract:
this module resolves and delivers immutable encoder outputs; input is image
identity plus `TensorSpec`/destination; output is exactly one valid tensor at
each required P rank or an explicit failure. Guard against wrong source,
false E residency, premature readiness and unsafe memory reuse. Reuse the
existing suites rather than one new file per helper.

### CPU tests with native I/O faked

Extend these existing suites:

- `tests/v1/ec_connector/unit/test_cross_encoder_cache.py`: lookup-only hits
  perform no E payload GET, malformed/header/size errors, partial batch results,
  no format drift, E namespace transport, reader capacity, independent tensors
  and registration reuse. Retain default relay tests.
- `tests/v1/ec_connector/unit/test_control.py` and `test_protocol.py`: old-peer
  negotiation, idempotent descriptors, Store pending versus READY, delayed
  completion queues, cancellation/generation conflicts, single writer with
  same-key Store/P2P followers, and backpressure without a blocked REP loop.
- `tests/v1/ec_connector/unit/test_worker_ec_connector.py` and nearby existing
  worker suites: source-free producer completion, mixed P2P/Store delivery,
  failed Store read followed by same-request relay, four-rank partial success,
  unconfirmed native I/O retaining owners, and shutdown while work is pending.
- `tests/v1/core/test_encoder_cache_manager.py` and
  `tests/v1/worker/test_encoder_runner.py`: remote-only results do not create
  E-local hits; replay a second request after Store eviction and require actual
  recomputation. Cover shared hashes, distinct destinations, repeated positions,
  chunked requests, preemption, cancellation and balanced budget/ref release.
- `tests/models/glm5next/test_epd_metadata.py`: retain metadata-only GLM grid /
  output shape contract. The new feature must not change parser semantics.

Suggested existing unit entry point (extend with the owning worker/core tests
after inspecting their fixtures):

```bash
.venv/bin/python -m pytest --confcutdir=tests/v1/ec_connector/unit -q \
  tests/v1/ec_connector/unit/test_cross_encoder_cache.py \
  tests/v1/ec_connector/unit/test_worker_ec_connector.py \
  tests/v1/ec_connector/unit/test_control.py \
  tests/v1/ec_connector/unit/test_protocol.py
```

Use a meaningful red/green failure for each new behavior. The CPU fake proves
Python lifecycle semantics, not Mooncake's native completion guarantees.

### Real Store component validation

- [ ] FP16, BF16 and FP32 matrix roundtrips remain bit-exact, including header,
  short read, wrong shape/dtype, byte count, over-capacity and chunk boundaries.
- [ ] P CUDA copies complete before READY and staging/pool reuse. Vary the
  default CUDA device to detect accidental GPU0 allocation in the new thread.
- [ ] Concurrent same-key clients on separate ranks produce valid immutable
  tensors. Show bounded pending bytes/items and complete lease/registration
  cleanup on healthy shutdown.
- [ ] Force safe recoverable GET failures after E `batch_exists` and before P
  GET (remove only experiment keys, or use a test-only paused-read barrier).
  Verify same-request E fallback. Repeat with one TP rank failing and the
  others ready; only failed shards accept P2P writes.
- [ ] Fake uncertain I/O only in isolated unit fixtures; do not crash or stall
  shared Mooncake services to create a fault. Confirm owner-retention policy
  separately from safely recoverable missing-object behavior.

### Full Flash end-to-end correctness gates

For each case record actual per-item source and per-rank completion, compare
the tensors entering PD to a cold-encode reference, then compare generated
outputs to the existing relay arm. Exact tensor equality is required for
identical BF16 encoder execution. Establish repeated-reference output stability
before interpreting a generated-token mismatch under parallel GPU execution.

- [ ] E1 local hit: no new ViT/Store GET; original P2P if P is cold.
- [ ] E1 local miss, Store hit: both arms skip ViT; hybrid has **zero E1 payload
  GET / embedding H2D / payload P2P**, and every needed PD rank loads correctly.
- [ ] Both miss: one necessary ViT execution, P2P and normal asynchronous PUT.
- [ ] `[L,S,M]` mixed delivery and `[S,S,M]` repeated occurrences.
- [ ] PD already has the embedding: no unnecessary P GET or overwrite; preserve
  resident handoff. Check prefix-KV-covered input separately with caching on.
- [ ] Store eviction/corruption/read failure between lookup and GET safely
  falls back; no hang, false success, stale event or silent wrong embedding.
- [ ] Partial TP failure, request abort, expired reservations, preemption and
  second request after remote-only completion all terminate correctly.
- [ ] Concurrency 1/4/8/16 with shared and distinct identifiers; reader buffer
  pressure does not corrupt outputs, leak registrations or strand requests.
- [ ] Stream and nonstream transport both work. No benchmark retry or fallback
  is silently excluded from latency/success accounting.

Use existing `test_epd_correctness.py` scaffolding and model/fixture utilities.
Add at least 100 representative fixed image-question prompts with deterministic
sampling (`temperature=0`, fixed output cap, initially 32 tokens), including OCR,
counting, spatial questions, multiple images and repeated conversation images.
Save fixture IDs, prompts and outputs. Full-model relay/hybrid answer agreement
and a labeled quality score where available are separate evidence from tensor
transport equality; do not label a reduced-model smoke run “Flash validated.”

## 4. Instrumentation and attribution

Use temporary structured tracing, separate from uninstrumented performance
runs. Correlate by request/item/transfer/rank/generation, not just timestamp.
Record these stages with monotonic host timings and CUDA events where relevant:

- E local lookup, Store existence lookup, P acquisition submission/ack wait;
- E payload GET calls/bytes and GET elapsed time, vision forwards/items/time;
- E embedding H2D, P2P source staging, E→P bytes and transfer completion;
- P per-rank Store GET calls/bytes/time, H2D, final D2D, READY-to-adoption;
- producer terminalization, fallback reason, queue depth, retained bytes,
  reservation/lease count and pinned CPU / GPU staging peaks;
- client send-to-first-nonempty-token TTFT, completion latency, TPOT, request
  throughput, output throughput, failure/timeout/retry counts and tail latency.

Export actual Store calls, bytes and bandwidth, including metadata/retries,
rather than assuming a win. Validate against the existing TP4 receiving-worker
and completion contract; redesigning TP distribution is outside this experiment.
Successful-path assertions exclude unrelated tokenizer/input H2D from the
“zero E embedding H2D” requirement. Source profiling must identify payloads.

## 5. Paired performance experiment

Arms:

- **A, context only:** main with shared reuse disabled (cold ViT + P2P).
- **B, primary baseline:** pinned main with `cross_encoder_cache=true`; Store
  hit loads through E and then P2P. No hybrid flag.
- **C, treatment:** same source/config plus the hybrid change and proposed flag.
- **C-off, compatibility:** patched code with feature off, checked against B.

The causal result is **B versus C**, not A versus C. B and C must have the
same checkpoint, Store seed bank, selected E, PD TP4 ranks, protocol, cache
state, input order, graph configuration and generation settings. C-off must
preserve B behavior. Do not run B and C simultaneously on the same GPUs.

First run a short functional pilot to validate states and select three image
sizes that yield distinct observed embedding byte sizes. Record actual grids,
token counts and bytes; do not assume input resolution maps linearly to tokens.

Main cells:

- Controlled 100% cross-E Store hits at concurrency 1, 4, 8, 16, across the
  three chosen embedding sizes. Distinct keys keep E1 and PD cold per request.
- Local-hit-only and Store-miss-only controls at the same concurrency levels.
- Mixed-source workloads at cross-hit shares 0%, 25%, 50%, 75%, 100%, using
  measured per-item fractions and `[L,S,M]` requests where applicable.
- A natural two-E round-robin replay and a separate multi-turn replay with
  prefix caching enabled, to report how often cross-E delivery remains needed.

For each cell, run at least six independently reset paired B/C blocks, balanced
as BC/CB. Use at least 100 measured requests per arm per block for the primary
hit cell; expand the fixture bank before timing. Warm on disjoint inputs until
initialization is complete and a stable window is observed. Report p50/p95;
p99 is descriptive unless there are enough samples per block to resolve it.
Record offered load and achieved concurrency; add a fixed-rate, open-loop sweep
near saturation after the closed-loop comparison to expose queueing costs.

Use request-paired TTFT deltas when keys match, but compute confidence intervals
by resampling **run pairs**, not treating every correlated request as an
independent experiment. Report median paired relative change, 95% interval,
per-block results and run-to-run spread. Include errors/retries in the outcome.
Stop or invalidate a block if unrelated GPU/CPU/network work intervenes.

Proposed decision gates, fixed before examining performance results:

1. All correctness/lifecycle gates pass; no silent fallback, hanging requests,
   tensor mismatches or unreclaimed live buffers.
2. Every successful cross-hit eliminates the E payload path and unnecessary
   ViT work remains zero in **both** B and C.
3. Before measurement, choose a primary practical-value criterion: at least 3%
   lower p50 TTFT in the representative cross-hit cell, with the paired-run
   interval excluding zero; or a separately quantified E resource/headroom or
   saturation-throughput benefit without material TTFT regression. Account for
   costs moved to P. This is a proposed gate, not a predicted gain; any resource
   threshold must also be fixed before examining results.
4. Local-hit/cold-miss controls should remain within a predeclared 2% p50
   regression budget; investigate changes above noise and all p95 regressions
   above 5%. Underpowered/noisy comparisons remain inconclusive, not passed.
5. If Store I/O or P-side stalls erase the gain, record that result and
   do not open a production performance PR merely because the path is shorter.

TCP and RDMA are separate experiments. Start with the functioning, explicitly
recorded protocol and run the same protocol in B/C. An all-NVLink, same-node
result cannot establish inter-node RDMA benefit. Do not change protocol or
Store placement within an A/B pair.

## 6. Artifacts and completion

Required output files inside the run directory:

- `manifest.json`: complete source/model/environment/launch identity;
- `requests.jsonl`: fixture/route/cache-state expectations and actual outcomes;
- `trace.jsonl`: item/rank stage attribution from diagnostic runs;
- `correctness.json`: tensor equality, output/quality comparison and failures;
- `serving.csv`: one row per request, arm, block and workload cell;
- `summary.json`: run-level paired statistics, sample counts and decisions;
- `RESULTS.md`: what was actually validated, gains/regressions/uncertainty,
  reproducible commands and the limitations of the topology.

Keep generated data in experiment storage, not committed as model outputs in
`tests/`. Put any reusable performance harness under `benchmarks/`, and extend
the existing integration correctness harness for real-entrypoint behavior.
Track only owned process groups for cleanup; verify they have released GPU
memory/ports and restore any intentionally isolated environment changes.

This branch currently delivers the design and test plan only. Before upstream
submission, repeat duplicate searches, update rendered user docs for the new
flag, run the affected checks and model evaluation, apply `pr-checklist`, and
have the human submitter review every changed line and relevant test results.
