# Hybrid Mooncake Store delivery for E-first EPD

Status: proposed design; source-review only. No implementation or serving
measurement is included in this branch. AI assistance was used to prepare this
document. Companion: [test plan](../plans/2026-11-09-epd-hybrid-store-test-plan.md).

## Decision and scope

The idea is feasible and worth a bounded experiment. Preserve E-first dispatch,
`ECMooncakeConnector`, local-output P2P delivery, and cold-encode P2P delivery.
For an E-local miss with a shared Store hit, let the selected P/PD worker load
the embedding instead of relaying its payload through E.

The first implementation should confirm P-side acquisition before E skips that
item. Recoverable Store-read failures can then return to the existing E-side
resolver/encode path in the same request. This is a design choice, not an
already implemented capability. It trades additional control coordination for
a smaller failure protocol than accepting a Store reference and recovering
after the E request has finished.

Two qualifications matter:

- Direct **P-side consumption** initially means Store → P pinned CPU staging →
  P GPU tensor → P receive-pool allocation. The last copy is a conservative D2D
  handoff needed to reuse current reservation ownership. It is not GPUDirect,
  zero-copy, or elimination of all H2D traffic.
- This cannot safely be only a `load_tensors()` deletion. E runner input
  selection and scheduler cache accounting currently depend on an actual local
  tensor. Delivery completion must not become a false local cache hit.

The agreed lab layout is **2 E instances at TP1 + 1 PD instance at TP4**, six
GPUs total. “4PD” here means the four ranks of that one PD, not four replicas.

Non-goals: P-first scheduling, deciding Encoder admission from P prefix-KV
coverage, a Store-only connector, CUDA kernels, Mooncake internals, video/audio
reuse, E TP/PP/DP, dynamic LoRA, and GPU-direct Store reads. Keep current image,
MRv2, contiguous 2D FP16/BF16/FP32 and RAM-Store limitations.

## Source baseline and related work

Inspected on 2026-11-09 from freshly fetched **`origin/main`**:
`625d2d74eab9b1da1a9793d97100d714b885f2e0`. Here `origin` is
`grYe99/vllm`, not `vllm-project/vllm`. GLM-5.3-Flash integration in this fork
must be kept separate from the general connector change when preparing an
upstream submission. Do not describe the installed lab wheel as this source.

- [#56242](https://github.com/vllm-project/vllm/pull/56242) was merged on
  2026-09-17. Its
  [direct-P-read discussion](https://github.com/vllm-project/vllm/pull/56242#issuecomment-5619301051)
  explicitly leaves per-item source selection, completion and fallback for a
  follow-up. This design builds on its object format and resolver.
- [#47302](https://github.com/vllm-project/vllm/pull/47302) is open at inspection
  time. Its separate Store connector demonstrates consumer loading, but is not
  the hybrid delivery baseline. Its proposed structured keys/range-read codec
  differ from the current `protocol:v3` embedding objects; do not transplant its
  client or assume the formats interoperate. Reuse the already merged
  `MooncakeEmbeddingStoreClient` instead.
- [RFC #55639](https://github.com/vllm-project/vllm/issues/55639) is open and
  addresses P-aware avoidance of unnecessary E dispatch. This design still
  visits E and optimizes delivery only when P needs an embedding.
- [#57275](https://github.com/vllm-project/vllm/pull/57275) is a related draft
  about raw fallback identity and lifecycle, rather than this delivery source.
  Recheck it before implementation/rebase.

Open-PR searches covered `56242 in:body`, `ECMooncakeConnector`,
`"ECMooncakeConnector" in:title,body`, `"cross-encoder" in:title,body`, and
embedding/Store area terms. No exact active hybrid implementation was identified
in those results. This is a dated search result, not proof none exists. Repeat
the contribution-policy issue/comment and duplicate checks before any PR.

## Current call chain

All paths below are repository-relative at the pinned baseline.

1. `examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py` chooses
   P/PD and its control address before `fanout_encoder_primer()`. It assigns
   items to Encoders and sends positional `ec_items[].transfer_id` with
   `consumer_zmq`. The primer response supplies model placeholder metadata and
   engine-derived identifiers; the proxy rewrites the inference input and then
   forwards to P/PD. No embedding payload travels in that response.
2. `mooncake/scheduler.py::ensure_cache_available()` prepares producer pushes.
   `update_state_after_alloc()` marks the logical E encoder-cache allocation and
   adds image `TensorSpec`s to Store candidates. `_encoder_output_spec()` uses
   the producer's measured output width, not a guessed LM hidden width.
3. `build_connector_meta()` emits `pushes` and `store_candidates`, excluding
   preempted candidates. `store_candidates` is keyed by content identifier;
   `pushes` carries the destination and per-occurrence transfer identity.
4. `vllm/v1/worker/mm_encoder_model_runner.py::execute_model()` enters
   `ActiveECConnector.maybe_get_output()` before executing the vision encoder.
   Dedicated producers use this MRv2 runner; the BERT-style
   `gpu/model_states/encoder_only.py` is unrelated.
5. `mooncake/worker.py::start_save_caches()` first prepares asynchronous P2P
   reservations, binds any existing E-local tensors, then calls
   `MooncakeEmbeddingStoreBackend.resolve_inputs()`.
6. `mooncake_store_embedding/backend.py::resolve_inputs()` excludes actual
   local-cache entries, makes namespaced `protocol:v3` keys, calls
   `batch_exists()`, then `load_tensors(hits, device)`. Valid returned tensors
   enter `encoder_cache`; `start_save_caches()` explicitly binds those tensors
   as P2P sources.
7. `store_client.py::load_tensors()` chunks full objects by a reusable read
   buffer (default 128 MiB). `_load_chunk()` performs `batch_get_into()` into
   registered CPU storage, validates the 24-byte header and full byte count,
   copies into independent destination tensors, and confirms CUDA completion
   before reusing staging. Missing, malformed and recoverably failed objects
   are omitted. Unconfirmed native I/O/copies poison the client and retain
   owners rather than pretending a normal miss occurred.
8. `gpu/mm/encoder_runner.py::prepare_mm_inputs()` skips inputs whose identifiers
   now exist in `encoder_cache.encoder_outputs`. Remaining items execute
   `embed_multimodal()`. The connector wrapper snapshots the cache after Store
   resolution, so it saves/publishes only newly computed outputs afterward.
9. `mooncake/worker.py::_push_batch()` stages a source, writes all required
   consumer ranks, then sends `complete_batch`. The P control server emits
   readiness. `scheduler.py` waits for all TP shards, dispatches load specs,
   and `start_load_caches()` takes pool allocations into P's encoder cache.
   Worker `loaded` results intersect across TP ranks; failures union.

The Store client/backend paths in steps 5–7 live under
`vllm/distributed/ec_transfer/ec_connector/`. Publication remains bounded,
asynchronous and best-effort; E HTTP completion is not a PUT acknowledgement.

## Alternatives considered

1. **Keep the current relay.** Lowest complexity; retain as the default and
   benchmark control. It still spends E read staging, H2D, GPU allocation and
   P2P source bandwidth on cross-E hits.
2. **Confirmed P prefetch — recommended first experiment.** E classifies each
   output and requests P acquisition on Store hits before running its remaining
   encoder inputs. Successful items never become E payloads. Recoverable
   failures remain within the current E request. Control overhead and blocking
   of the E batch are explicit costs to measure.
3. **Unconfirmed Store reference, load after E returns.** Could allow more
   overlap, but requires persistent references, a later E recomputation service
   or a coordinated proxy retry with force-P2P state. The current metadata-only
   P request cannot simply recompute raw media. Defer until option 2 shows
   value; do not silently claim option 2 has that overlap.

## Proposed per-item protocol

Names below are proposed internal interfaces, not existing API fields.
Public request/response schemas and the E-first proxy flow need no new fields
for the first experiment; the existing `transfer_id` connects both halves.

### Selection and metadata

- Keep `ECMooncakePushSpec` as the scheduled item contract and identity. Add an
  internal immutable `StoreDeliverySpec` with the exact E-derived namespace,
  identifier, `TensorSpec`, and delivery protocol version. Associate it with
  each transfer/destination, not only with a request-wide switch.
- Distinguish `p2p`, `store_pending`, and `store_ready` in producer records.
  Existing P-resident/local handoff is an outcome and takes precedence over a
  GET. Keep content identity separate from request occurrence and reservation
  generation. Duplicate image occurrences can share bytes, not request leases.
- Refactor the backend to expose a lookup-only operation returning eligible
  hit descriptors, retaining the existing local filtering, namespace creation,
  publication candidate tracking, and recoverable-lookup behavior. Keep
  `resolve_inputs()` for the default relay and fallback. The successful hybrid
  path must not call E `load_tensors()` or insert anything in E's tensor cache.
- E-local tensor → existing P2P; E-local miss/Store miss → encode + existing
  P2P + asynchronous PUT; E-local miss/Store hit → attempt P acquisition.
- Add proposed `prefill_store_delivery: false` extra config. On E, enabling it
  requires `cross_encoder_cache: true`; on P it enables a Store reader without
  enabling the current producer-only `cross_encoder_cache` flag. Both sides
  require MRv2 for this first implementation. The default remains unchanged.

### Consumer acquisition without blocking the control server

Extend `peers` with an optional `store_delivery_v1` capability. Absence or a
definite unsupported response chooses the legacy relay before delegation;
transport errors are not evidence that an outstanding writer stopped.

Extend `reserve_batch` items with an optional Store descriptor. Unchanged
items retain the exact old path. For an accepted Store item:

1. Validate descriptor, shape, dtype, byte count, namespace policy and capacity;
   acquire the usual consumer allocation/lease. A P-resident hit returns ready
   immediately, without Store I/O.
2. For the unique writer of a cold identifier, enqueue a bounded Store-read job
   and return `store_pending`. Followers attach to that writer, including when
   a Store and P2P request race for the same identifier. Never start two writers
   into the same allocation.
3. Use one serialized read executor/client per receiving rank, an explicit CUDA
   device/stream and the existing registered read buffer. Batch admitted items
   and bound both queued items and retained bytes; reject excess work back to
   relay. Do not call Store I/O under the reservation lock or in the REP loop.
4. Reuse `load_tensors()` with the consumer device and original `TensorSpec`.
   Copy validated independent tensors into the reserved pool allocations;
   confirm the final copies before reporting completion. Bound a job's batch
   bytes to the read budget so returned temporary GPU tensors cannot accumulate
   across an unbounded `load_tensors()` call.
5. Return completed jobs through a thread-safe queue. The control thread drains
   this queue, checks reservation generation and writer ownership, transitions
   through the existing completion/follower logic, and emits the same READY
   events as P2P. No background thread touches a ZMQ socket.

`status` needs explicit `store_pending`, `store_failed_safe` and terminal-error
information alongside existing reservation/ready fields. E waits outside the
control thread for all receiving TP ranks to become READY before declaring a
Store-delivered item successful. P scheduler metadata can remain ordinary
ready/load metadata: by the time it sees READY, the source has materialized in
its existing pool. Store references are not READY events.

### Producer completion and same-request fallback

For all-rank success, terminalize that producer delivery without waiting for a
source tensor. `pending_saves` must include pending Store acknowledgements, then
clear when they terminate. Mixed batches still bind and send P2P tensors for
their other items. Existing `ProducerPushManager` is source-driven, so this
terminalization is an explicit change, not an assumed side effect.

For a confirmed recoverable failure on any rank, settle all started Store jobs
for that transfer before assigning another writer. Preserve successful READY
shards; failed shards get a fresh reservation generation and P2P destination.
Then run the original E resolver for this identifier: GET if still readable,
otherwise encode raw media already held by E, bind the real tensor, and P2P
only to ranks that need it. Ready shards must advertise `write=false`.

Per-item fallback must preserve original request/transfer identifiers and
placeholder metadata. Late Store completions tagged with the old generation
cannot complete a replacement reservation. Same-key followers inherit the
writer's committed result or explicit failure; cancellation of one follower
does not release the writer's allocation.

Cancellation, expiry or an RPC timeout does not prove a native writer is done.
Uncertain I/O retains buffers, fails the affected delivery explicitly and follows
the existing fatal-client policy when applicable. Never use P2P to overwrite an
allocation that might still be receiving a Store copy. An unresolved timeout
can fail the request; only confirmed recoverable failures promise in-request
fallback. An ordinary timeout is not a bounded cancellation API for Mooncake.

### E runner and cache-accounting contract

Add an explicit connector result for the current step's externally satisfied
producer inputs. The MRv2 encoder-only runner excludes only those inputs whose
destination deliveries have all completed. It must never insert `None`, an
empty tensor, or a marker into `dict[str, Tensor]` to trick input selection.
All other runners/connectors retain their present behavior.

E scheduler's `EncoderCacheManager.allocate()` currently assumes every
scheduled output becomes locally cached, and `free_encoder_input()` retains
unreferenced entries as reusable. Therefore add explicit nonresident-output
feedback from the worker to scheduler, with these rules:

- A remotely delivered input is satisfied for its current producer step and
  destinations; it is not evidence of an E-local output for a later request.
- Preserve budget/ref accounting until all active occurrences in that step are
  accounted for. Release their references exactly once; discard an unreferenced
  nonresident entry instead of retaining it in the local LRU. An entry still
  referenced by another request must be marked nonresident so its subsequent
  scheduling rechecks delivery rather than treating it as a local hit.
- Before reusing scheduler-local cache state in a subsequent step, consume that
  feedback and ensure missing tensors remain schedulable under the ordinary
  encoder budget. Do not repair this by injecting unscheduled work on the worker.
- Distinguish a genuinely computed resident output from a remote-only result
  when duplicate hashes share a batch; retain a real local output if one exists.
- Preemption, chunked scheduling, cancellation and repeated occurrences must
  clean up nonresident records as well as the connector's `_local_cache` mirror.

The exact small cache-manager API is an implementation decision, but these
semantics and the multi-request tests are mandatory before a serving PoC. This
is the main reason review is medium-to-high complexity rather than a small
connector-only patch. Keep the change encode-only and opt-in; do not redesign
the general cache manager.

## Compatibility, resources and trust boundaries

- Store lookup proves existence only. Full contract and byte validation remain
  at P GET. Do not trust the key alone or skip the existing object codec.
- Carry E's namespace verbatim after checking it against deployment-approved
  model/revision/prefix/protocol settings. E-only and PD multimodal config hashes
  may differ; independently rebuilding the key from P config can create misses.
  TP4 ranks receive the full embedding, not quarter-width output shards.
- Store descriptors originate in the trusted internal connector, not arbitrary
  user `ec_transfer_params`. Reject shape/size overflow, unsupported dtype,
  namespace/protocol mismatch and conflicting transfer identities before I/O.
  No arbitrary server URL or memory address is accepted as a Store descriptor.
- P receiving ranks need their own Store clients and CPU buffers. At TP4, four
  rank-local GETs are the conservative design; they may multiply Store egress
  versus one E GET. Measure this explicitly. Rank-0 GET plus TP broadcast is a
  separate possible optimization, not part of this PoC.
- Current receive-pool resident leases, TTLs and tombstones still own published
  tensors. Backpressure, shutdown and unknown-I/O retention apply equally to
  Store and P2P writers. Stop job admission, settle jobs, drain completion queues,
  then close clients/pools in ownership order.
- Deploy control/Store services on the existing isolated internal network as
  described in `docs/usage/security.md`; this adds no authentication model.

## Expected benefit and falsifiable risks

The gain is removal of E-side read staging, E H2D/allocation and E payload
forwarding on successful cross-E hits. E preprocessing, Store existence lookup,
control requests, encoder budget reservation and waits remain. The conservative
P implementation adds D2D and four independent TP-rank reads. Large Store RTT,
bandwidth saturation, or blocking E batches on P may erase or reverse the gain.

Baseline #56242 already skips ViT on a Store hit. Counting fewer ViT forwards
against a Store-disabled baseline does not establish this proposal's value.
The decisive comparison is existing Store-enabled E relay versus P consumption,
with identical local/P/Store states and the actual TP4 topology.

Use the companion test plan for correctness, controlled cross-E requests,
transfer attribution, and paired serving measurements. No speedup is claimed.
Proceed to a PR only if the end-to-end benefit is stable and worth the lifecycle
complexity; otherwise retain the experiment result without a production PR.

## Implementation sequence

1. Extend existing CPU codec/resolver and reservation/control suites to express
   lookup-only hits, Store writer ownership, source-free completion and fallback.
2. Implement the internal descriptor/capability and bounded consumer reader,
   retaining existing `protocol:v3` codec and P2P behavior by default.
3. Implement E input filtering plus nonresident scheduler feedback as one
   correctness change, with repeated/chunked/concurrent request tests.
4. Run a real Store transport/tensor roundtrip, then full Flash 2E+TP4-PD
   correctness cases. Only then run the uninstrumented paired benchmark.
5. Update rendered connector/cache documentation for any shipped configuration,
   recheck duplicates and upstream compatibility, run `pr-checklist`, and have
   the human submitter review the change and relevant tests before a PR.
