# Hybrid Push / Store Direct GET: feasibility report

Evidence level: **source-review**, 2026-11-09. No product implementation or
runtime benchmark was performed. AI assistance was used to prepare this report.
Input: the user's `vllm_epd_hybrid_push_store_direct_get_handoff.md`.

## Verdict and scope

**Conditional GO for a protocol PoC; not yet GO for a production performance
PR.** The handoff is implementable in principle without changing E-first Proxy
scheduling, but it requires a new connector operation and lifecycle transitions.
Current `ECMooncakeConnector` is push-based; there is no existing P-side Store
pull operation to enable with metadata alone.

The proposal is **E-initiated control handoff, P-executed Store GET** for an
E-local miss / Store hit. E-local hits and newly computed outputs keep their
existing P2P push path. Store publication stays asynchronous and best-effort.
Do not take over #47302, implement a second codec, change TP distribution, or
turn the system into P-first scheduling.

The strongest plausible benefit is lower E GPU allocation, embedding H2D and
payload-forwarding cost. A material TTFT or throughput gain is unproven: P must
still load the payload, and the new control/waiting cost can offset saved work.
Existing cross-Encoder Store hits already skip ViT, so that saving is not new.

## 1. Latest source and duplicate-work check

Both refs were freshly fetched on 2026-11-09:

- Development base `origin/main`, fork `grYe99/vllm`:
  `625d2d74eab9b1da1a9793d97100d714b885f2e0`.
- Latest inspected `upstream/main`, `vllm-project/vllm`:
  `a3c14efbc77c018e8bf579935095a87721349e05`.

The Mooncake worker, producer manager, scheduler, Store backend and MRv2 EC
wrapper are identical between these refs. Latest upstream removed Model Runner
V1 and changed namespace hashing from `compute_hash()` to `hash_config()`; those
changes do not add Direct GET. The compact object protocol remains `v3`.
Verify configuration hashes for the deployed versions rather than assuming
arbitrary mixed-version namespaces match. The development branch was not rebased.

- [#56242](https://github.com/vllm-project/vllm/pull/56242) is merged. Its
  [follow-up discussion](https://github.com/vllm-project/vllm/pull/56242#issuecomment-5619301051)
  explicitly separates output reuse from delivery and leaves Direct P reads,
  per-item completion and read-failure recovery for a follow-up.
- [#47302](https://github.com/vllm-project/vllm/pull/47302) remains open, head
  `0d1f71f5d36c92fa9fbffac72b14e160623e8aa6`. It adds a separate Store connector,
  not this hybrid follow-up. Its 304-byte metadata / structured-key codec is
  incompatible with the merged 24-byte header / `protocol:v3` objects. Borrow
  consumer-lifecycle ideas, not its serialization implementation.
- [#57275](https://github.com/vllm-project/vllm/pull/57275) is a related draft
  about fallback identity and lifecycle; it is not the Direct GET feature.
- [#55639](https://github.com/vllm-project/vllm/issues/55639) concerns P-aware
  suppression of E work. It remains outside this delivery-only proposal.

Searches covered `56242 in:body`, `ECMooncakeConnector`, cross-Encoder/Store and
Prefill/Direct/Store terms using `gh pr list`, plus `gh search prs` for
`ECMooncakeConnector` and `"cross-encoder" "Store"`. No exact active hybrid
implementation was identified. This is a dated search result, not an exhaustive
absence claim. Repeat the project's issue/comment and PR checks before opening
or proposing an upstream PR.

## 2. Actual call chain

Links pin latest upstream; line numbers refer to that version.

1. [Proxy `prepare_for_decode`, L1000](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py#L1000)
   awaits the E primer, rewrites returned placeholder metadata, then forwards to
   P/PD. P's destination control address is already known to E.
2. [EC scheduler, L367](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/scheduler.py#L367)
   prepares push specs. `update_state_after_alloc()` at L454 records image Store
   candidates and expected shapes/bytes from measured encoder output width.
3. [E worker `start_save_caches`, L481](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/worker.py#L481)
   starts asynchronous P2P reservations, binds E-local source tensors, then
   resolves Store candidates. Reservations currently begin before Store lookup.
4. [Backend `resolve_inputs`, L70](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake_store_embedding/backend.py#L70)
   filters actual local tensors, builds namespaced keys, does `batch_exists()`,
   calls `load_tensors()`, and inserts valid tensors in E's `encoder_cache`.
   Recoverable misses/errors leave the item for encoding.
5. [EC wrapper, L49](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/v1/worker/gpu/ec_connector.py#L49)
   runs that resolver before the [encoder-only runner, L167](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/v1/worker/mm_encoder_model_runner.py#L167)
   executes the scheduled image inputs. `EncoderRunner.prepare_mm_inputs()`
   skips actual tensor-cache hits. The wrapper's cache snapshot means loaded
   outputs are not republished as newly computed outputs.
6. [Producer `submit_batches`, L167](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/producer.py#L167)
   requires a source tensor. [Worker `_push_batch`, L648](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/worker.py#L648)
   stages sources and writes writable destinations, then notifies completion.
7. [Control dispatch, L353](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/control.py#L353)
   supports reserve/status/peers/event-port/complete/cancel. Completion publishes
   READY notifications. No operation requests a Store GET.
8. P scheduler waits for readiness, then [P `start_load_caches`, L607](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake/worker.py#L607)
   takes a received pool allocation into P's encoder cache. It does not fetch a
   remote object. Existing TP receiving-worker and readiness semantics remain
   the integration contract; changing distribution is not part of this report.

Two concrete gaps follow. Removing E `load_tensors()` does not skip forward,
because the runner no longer sees a real tensor. Nor can a Store reference
finish an unchanged producer push record, because it is source-driven.

## 3. Integration point that works with E-first

**Do not merely add GET to P's normal `start_load_caches()`.** If E waits for
that GET, Proxy waits for E and never sends P its inference request. In addition,
P scheduler needs an availability event before dispatching a load. That naive
integration either deadlocks or never schedules the GET.

Instead use P's already running control service to accept an E-originated,
per-item Store handoff. A bounded background reader loads before P's inference
request needs to run. Once the actual tensor is ready, publish the existing
READY event and keep P's normal pool-adoption path. This reuses the control
channel, receive memory and readiness consumers; **the handoff command, local
writer ownership and source-free E completion are new semantics**.

For the first PoC, E waits for acquisition to succeed before skipping the
corresponding encoder input and returning its normal metadata. This avoids a
new late recomputation service. It may block the E batch, so it is a candidate
protocol to measure, not a preselected optimal pipeline.

Per-item metadata needs the existing transfer/occurrence identity, destination,
source (`push` or `store`), E-derived namespace/key, `TensorSpec` and reservation
generation. Keep content identity separate from occurrence leases. Prefer
extending existing batch control messages over adding HTTP request fields;
Proxy continues to forward its existing identities and placeholder metadata.

Keep default behavior unchanged and use an opt-in extra-config mode. The name
`prefill_store_delivery=false` is a proposal, not an existing option. P needs
a separate reader-enable path: current `cross_encoder_cache`
validation permits only an encoder-only producer, so enabling that flag on P
does not work.

The simple PoC can classify missing inputs before reserving their destinations.
This may lose current overlap of reservation and Store lookup on cold misses.
Treat cold-path latency as an acceptance gate. An alternative is a source-neutral
reservation/explicit pre-write ownership conversion, but that adds transitions;
do not implement it speculatively before measuring the simple path.

## 4. Store client and object compatibility

Reuse [current `load_tensors`, L263](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/distributed/ec_transfer/ec_connector/mooncake_store_embedding/store_client.py#L263).
It takes contracts and a device without requiring E-side execution. It already
chunks `batch_get_into()` reads through registered CPU staging, validates the
24-byte header and full byte count, creates independent output tensors and
confirms CUDA copy completion before reusing staging.

Initial direct consumption therefore means Store → P CPU → P GPU. Keeping the
existing receive-pool adoption can require a final GPU-to-pool copy. Neither
GPU-direct RDMA nor lower total-system H2D is established by moving GET to P.

Use the same v3 object and exact E namespace; E-only and PD configurations can
have different multimodal hashes. P validates model/revision/prefix/protocol,
shape/dtype/byte count against its approved deployment settings before I/O.
Do not accept arbitrary Store endpoints, pointers or unvalidated object layouts
from user metadata. Serialize use of each reusable staging client, bound queued
items/bytes, and set the correct worker CUDA device/stream in background jobs.

## 5. Completion, concurrency and cancellation

Proposed states are separate from current enum names:

- P2P item: planned → remote-write-owned → data ready → adopted / released.
- Store item: planned → local-GET-owned → data ready → adopted / released.
- Failure/abort: no new adoption; release only after the writer is confirmed
  settled, otherwise retain its owners and report explicit failure.

A Store descriptor or successful enqueue is not READY. Only validated bytes
and completed GPU copies satisfy availability. A Store item must not await E's
P2P `complete_batch`; the local writer drives completion and the control thread
publishes events. Background jobs return completion through a queue and never
operate ZMQ sockets. Blocking Store I/O in the REP loop would also delay P2P
reservations/cancellation and is unacceptable.

Reuse receive-pool storage as appropriate, but do not leave a fictitious remote
P2P writer attached to it. Current reservations assume a remote writer; add
explicit source ownership. A pool allocation is not itself a P2P transfer.
Deduplicate same-key writers while retaining separate request leases; followers
must resolve exactly once. Preserve current TP readiness aggregation.

Late completion must match its reservation generation. Abort or TTL expiry
does not prove native I/O stopped. Never reuse/overwrite a destination while a
Store copy might still be active. Existing poisoned-client owner retention and
fatal-error semantics apply; do not turn an uncertain native completion into a
normal cache miss. Shutdown stops admission, settles owned jobs and then closes
clients/pools in ownership order.

**E cache correctness is another required change.** The scheduler's
[local-cache shortcut, L2198](https://github.com/vllm-project/vllm/blob/a3c14efbc77c018e8bf579935095a87721349e05/vllm/v1/core/sched/scheduler.py#L2198)
can skip later encoding while `free_encoder_input()` retains unreferenced
entries. A remote-only result needs explicit runner skip state plus scheduler
nonresident feedback/invalidation before that shortcut can observe it again.
A check only in a later external-cache lookup is too late. Preserve actual
local tensors, active references and encoder budgets; do not insert fake
tensors or run unscheduled work. This crosses the connector/runner/scheduler
boundary, so the complete fix is not connector-only.

## 6. Failure recovery choices

**Minimal protocol PoC: fail fast.** A failed acquisition causes an explicit
request failure with settled ownership/cleanup, never a wait for nonexistent
push completion. This proves the happy path and reliable failure reporting,
not production availability. Do not claim the existing Proxy retries guarantee
recovery: they cover particular inference failures, not every primer/streaming
failure, and retry can choose the same unreadable Store object again.

**Smallest production candidate: same-request fallback before E returns.** On
a confirmed recoverable GET failure, settle started jobs, preserve targets
already READY, prepare safe replacement generations for failed targets, and
retain the raw E input for the existing resolver/encode + push path. Add this
explicitly; E's current read-error fallback cannot observe P failures by itself.
Successful hybrid hits still perform no E GET/H2D/payload push/ViT.

**Late re-dispatch after E returns:** requires retained/recreated inputs and
new E work scheduling or a proven force-legacy request retry. It is a larger
protocol and is not the minimal plan. If the bounded same-request approach is
too costly or unreliable, stop and reassess rather than widening this PR.

P-resident embedding reuse can avoid duplicate loads. P prefix-KV coverage does
not become an early E decision just because this feature exists. Preserving
E-first does not promise to avoid every E lookup/compute when P already has KV;
that part of the handoff table must remain subject to existing #55639 behavior.

## 7. Minimal implementation plan and PR boundary

1. **Protocol proof:** extend backend lookup-only behavior, internal metadata /
   config, control dispatch, consumer reader and producer source-free completion.
   Fake only native I/O in the existing cross-cache/control/protocol/worker
   tests. Prove the P reader can run before a P inference request exists.
2. **Cache/runner closure:** add explicit remote satisfaction and scheduler
   nonresident handling. Extend existing encoder-cache/runner suites with a
   second same-image request after Store eviction, repeated occurrences,
   preemption and shared hashes. No fake tensor-cache entries.
3. **Real PoC:** one receiving worker's Store roundtrip, then 2E TP1 + one PD
   TP4 on the full GLM-5.3-Flash checkpoint. E0 seeds Store, E1/P are cold; test
   local hit, cross-E Store hit, cold miss and one-request mixed delivery.
4. **Production gate:** same-request recoverable fallback, abort/shutdown and
   default-off/old-peer behavior, then paired B/C serving evidence. Add rendered
   usage docs only for the shipped option and its actual support limits.

Expected modules: `mooncake/{config,metadata,control,worker,producer,reservation}.py`,
`mooncake_store_embedding/backend.py`, the MRv2 EC wrapper/encoder-only runner,
and the scheduler/cache-manager boundary. Prefer the existing Store client
unchanged. No Proxy scheduling, Store-only connector, TP distribution, Store
kernel or CUDA-kernel work belongs in this patch.

## 8. Benchmark and Go/No-Go

Use the [detailed test plan](../superpowers/plans/2026-11-09-epd-hybrid-store-test-plan.md).
The decisive comparison is current **Store-enabled E relay (B)** versus hybrid
(C), not Store-disabled encoding versus hybrid. Pin E1 through a static
single-E reference proxy configuration; E0 seeds distinct image keys through
normal encoder-only execution and publication is observed before measuring.
Measure each key once per arm so B's E1 cache warming does not contaminate
subsequent cross-E hits. Reset E1/P and warm disjoint inputs between paired runs.

Record TTFT p50/p95, throughput, failures/retries, E lookup/GET/ViT/H2D/staging/
push cost, P GET/copy/queue cost and actual bytes/memory. Include first-reader
registration separately from steady state. Preserve the same full model,
GPU mapping, Store, protocol, input bank and cache state. Use repeated BC/CB
blocks and run-pair uncertainty; separate instrumented attribution from timing.
Add local-hit, cold-miss and mixed-source controls at concurrency 1/4/8/16.

GO requires correctness and bounded recovery, measurable E resource or stable
serving benefit, and acceptable P2P fast-path regression. Suggested gates in
the test plan are at least 3% primary-cell p50 TTFT improvement with a paired-run
interval excluding zero, or a separately justified resource benefit; local/cold
p50 regression budget 2%, investigate p95 regressions above 5%. Fix the criterion
before examining results. Noise or insufficient samples are inconclusive.

NO-GO for a production performance PR if coordination/resource migration erases
benefit, reliability requires a broad EPD rewrite, or existing fast paths
regress materially. No model evaluation, GPU resource reduction, TTFT gain or
production-ready fallback has been runtime-validated in this report.
