// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    CacheHitSourceLabels, EngineLabels, EnginePositionLabels, F64Gauge, Family, HistogramMetric,
    LoraAdapterNames, LoraInfoLabels, MooncakeOperationCounterFamily,
    MooncakeOperationHistogramFamily, MooncakeOperationLabels, RedhareEngineLabels, RedhareMetrics,
    RedhareOutcomeLabels, RedharePathLabels, RequestMetrics, SchedulerLogStatsAccumulator,
    SchedulerMetrics, U64Counter, U64Gauge, WaitingReasonLabels,
};

use crate::protocol::stats::{
    CachedTokensBySource, KvConnectorStats, MooncakeStats, MooncakeTransferStats,
    MultiConnectorStats, NixlStats, RedhareStats, SchedulerStats,
};
use crate::transport::ConnectedEngine;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

static REDHARE_KV_CLUSTER: OnceLock<String> = OnceLock::new();

/// Record the Redhare `cluster` label for this process.
///
/// Python's `RedhareKVConnectorPromMetrics` reads it from
/// `kv_connector_extra_config["cluster"]`, which is not on the scheduler
/// stats payload. The frontend sets it once from `--args-json`. A later call
/// is ignored. Unset means `"default"`, matching `RedhareExtraConfig`.
pub fn set_redhare_kv_cluster(cluster: impl Into<String>) {
    let _ = REDHARE_KV_CLUSTER.set(cluster.into());
}

fn redhare_kv_cluster() -> String {
    REDHARE_KV_CLUSTER.get().cloned().unwrap_or_else(|| "default".to_string())
}

/// Cached output-batch metric handles for one model and engine index.
pub(crate) struct IterationMetricHandles {
    pub iteration_tokens_total: HistogramMetric,
}

impl IterationMetricHandles {
    pub(crate) fn new(metrics: &RequestMetrics, model_name: &str, engine: u32) -> Self {
        let labels = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };
        Self {
            iteration_tokens_total: metrics.iteration_tokens_total.get_or_create_owned(&labels),
        }
    }
}

/// Cached scheduler-stats metric handles for all engines connected to one
/// frontend client.
pub(crate) struct SchedulerStatsRecorder {
    engines: BTreeMap<u32, SchedulerStatsHandles>,
}

/// Per-engine cached metric handles used while recording `SchedulerStats`.
struct SchedulerStatsHandles {
    // Base labels reused for dynamic child labels.
    labels: EngineLabels,

    // Scheduler state gauges.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    scheduler_waiting_capacity: U64Gauge,
    scheduler_waiting_deferred: U64Gauge,
    kv_cache_usage: F64Gauge,

    // Prefix-cache counters, including the connector-backed external cache path.
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,
    external_prefix_cache_queries: U64Counter,
    external_prefix_cache_hits: U64Counter,
    /// One handle per `CachedTokensBySource::SOURCES` entry.
    prompt_tokens_cached_by_source: [U64Counter; 5],

    // Speculative decoding counters.
    spec_decode_num_drafts: U64Counter,
    spec_decode_num_draft_tokens: U64Counter,
    spec_decode_num_accepted_tokens: U64Counter,
    spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,

    // Per-engine performance / MFU counters.
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,

    // Sampled KV-cache residency histograms.
    kv_block_lifetime_seconds: HistogramMetric,
    kv_block_idle_before_evict_seconds: HistogramMetric,
    kv_block_reuse_gap_seconds: HistogramMetric,

    // Mooncake store connector telemetry, decoded from `kv_connector_stats`.
    // Kept as `Family` (not pre-resolved) because `operation`/`status` are
    // dynamic per-record labels.
    mooncake_operation_time_seconds: MooncakeOperationHistogramFamily,
    mooncake_operation_total: MooncakeOperationCounterFamily,
    mooncake_operation_keys_total: MooncakeOperationCounterFamily,
    mooncake_operation_bytes_total: MooncakeOperationCounterFamily,
    mooncake_operation_failed_keys_total: MooncakeOperationCounterFamily,

    // Mooncake connector transfer telemetry, decoded from `kv_connector_stats`.
    mooncake_xfer_time_seconds: HistogramMetric,
    mooncake_bytes_transferred: HistogramMetric,
    mooncake_num_descriptors: HistogramMetric,
    mooncake_num_failed_transfers: U64Counter,
    mooncake_num_failed_recvs: U64Counter,
    mooncake_num_kv_expired_reqs: U64Counter,

    // NIXL connector telemetry, decoded from `kv_connector_stats`.
    nixl_xfer_time_seconds: HistogramMetric,
    nixl_post_time_seconds: HistogramMetric,
    nixl_bytes_transferred: HistogramMetric,
    nixl_num_descriptors: HistogramMetric,
    nixl_num_failed_transfers: U64Counter,
    nixl_num_failed_notifications: U64Counter,
    nixl_num_kv_expired_reqs: U64Counter,

    // Redhare connector telemetry, decoded from `kv_connector_stats`.
    redhare: RedhareMetrics,

    // Non-Prometheus interval accumulator for periodic text-log helpers.
    log_stats: SchedulerLogStatsAccumulator,
}

impl SchedulerStatsRecorder {
    /// Resolve the fixed-label metric handles for the connected engines.
    pub(crate) fn new(
        metrics: &SchedulerMetrics,
        model_name: &str,
        engines: &[ConnectedEngine],
    ) -> Self {
        let engines = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                Some((
                    engine,
                    resolve_scheduler_stats_handles(metrics, model_name, engine),
                ))
            })
            .collect();

        Self { engines }
    }

    /// Record one scheduler-stats payload for the given engine index.
    pub(crate) fn record(&self, engine_index: u32, stats: &SchedulerStats) {
        if let Some(handles) = self.engines.get(&engine_index) {
            record_scheduler_stats_with_handles(handles, stats);
        }
    }
}

/// Resolve all fixed-label scheduler metrics for one engine.
fn resolve_scheduler_stats_handles(
    metrics: &SchedulerMetrics,
    model_name: &str,
    engine: u32,
) -> SchedulerStatsHandles {
    let labels = EngineLabels {
        model_name: model_name.to_string(),
        engine,
    };
    let capacity = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_CAPACITY,
    };
    let deferred = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_DEFERRED,
    };

    SchedulerStatsHandles {
        scheduler_running: metrics.scheduler_running.get_or_create_owned(&labels),
        scheduler_waiting: metrics.scheduler_waiting.get_or_create_owned(&labels),
        scheduler_waiting_capacity: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&capacity),
        scheduler_waiting_deferred: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&deferred),
        kv_cache_usage: metrics.kv_cache_usage.get_or_create_owned(&labels),
        prefix_cache_queries: metrics.prefix_cache_queries.get_or_create_owned(&labels),
        prefix_cache_hits: metrics.prefix_cache_hits.get_or_create_owned(&labels),
        external_prefix_cache_queries: metrics
            .external_prefix_cache_queries
            .get_or_create_owned(&labels),
        external_prefix_cache_hits: metrics.external_prefix_cache_hits.get_or_create_owned(&labels),
        prompt_tokens_cached_by_source: CachedTokensBySource::SOURCES.map(|source| {
            metrics
                .prompt_tokens_cached_by_source
                .get_or_create_owned(&CacheHitSourceLabels {
                    model_name: model_name.to_string(),
                    engine,
                    source,
                })
        }),
        spec_decode_num_drafts: metrics.spec_decode_num_drafts.get_or_create_owned(&labels),
        spec_decode_num_draft_tokens: metrics
            .spec_decode_num_draft_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens: metrics
            .spec_decode_num_accepted_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens_per_pos: metrics
            .spec_decode_num_accepted_tokens_per_pos
            .clone(),
        log_stats: metrics.log_stats.get_or_create_owned(&labels),
        estimated_flops_per_gpu: metrics.estimated_flops_per_gpu.get_or_create_owned(&labels),
        estimated_read_bytes_per_gpu: metrics
            .estimated_read_bytes_per_gpu
            .get_or_create_owned(&labels),
        estimated_write_bytes_per_gpu: metrics
            .estimated_write_bytes_per_gpu
            .get_or_create_owned(&labels),
        kv_block_lifetime_seconds: metrics.kv_block_lifetime_seconds.get_or_create_owned(&labels),
        kv_block_idle_before_evict_seconds: metrics
            .kv_block_idle_before_evict_seconds
            .get_or_create_owned(&labels),
        kv_block_reuse_gap_seconds: metrics.kv_block_reuse_gap_seconds.get_or_create_owned(&labels),
        mooncake_operation_time_seconds: metrics.mooncake_operation_time_seconds.clone(),
        mooncake_operation_total: metrics.mooncake_operation_total.clone(),
        mooncake_operation_keys_total: metrics.mooncake_operation_keys_total.clone(),
        mooncake_operation_bytes_total: metrics.mooncake_operation_bytes_total.clone(),
        mooncake_operation_failed_keys_total: metrics.mooncake_operation_failed_keys_total.clone(),
        mooncake_xfer_time_seconds: metrics.mooncake_xfer_time_seconds.get_or_create_owned(&labels),
        mooncake_bytes_transferred: metrics.mooncake_bytes_transferred.get_or_create_owned(&labels),
        mooncake_num_descriptors: metrics.mooncake_num_descriptors.get_or_create_owned(&labels),
        mooncake_num_failed_transfers: metrics
            .mooncake_num_failed_transfers
            .get_or_create_owned(&labels),
        mooncake_num_failed_recvs: metrics.mooncake_num_failed_recvs.get_or_create_owned(&labels),
        mooncake_num_kv_expired_reqs: metrics
            .mooncake_num_kv_expired_reqs
            .get_or_create_owned(&labels),
        nixl_xfer_time_seconds: metrics.nixl_xfer_time_seconds.get_or_create_owned(&labels),
        nixl_post_time_seconds: metrics.nixl_post_time_seconds.get_or_create_owned(&labels),
        nixl_bytes_transferred: metrics.nixl_bytes_transferred.get_or_create_owned(&labels),
        nixl_num_descriptors: metrics.nixl_num_descriptors.get_or_create_owned(&labels),
        nixl_num_failed_transfers: metrics.nixl_num_failed_transfers.get_or_create_owned(&labels),
        nixl_num_failed_notifications: metrics
            .nixl_num_failed_notifications
            .get_or_create_owned(&labels),
        nixl_num_kv_expired_reqs: metrics.nixl_num_kv_expired_reqs.get_or_create_owned(&labels),
        redhare: metrics.redhare.clone(),
        labels,
    }
}

/// Record scheduler-stats values through pre-resolved metric handles.
fn record_scheduler_stats_with_handles(handles: &SchedulerStatsHandles, stats: &SchedulerStats) {
    // Scheduler state gauges.
    handles.scheduler_running.set(stats.num_running_reqs);
    handles
        .scheduler_waiting
        .set(stats.num_waiting_reqs + stats.num_skipped_waiting_reqs);
    handles.scheduler_waiting_capacity.set(stats.num_waiting_reqs);
    handles.scheduler_waiting_deferred.set(stats.num_skipped_waiting_reqs);
    handles.kv_cache_usage.set(stats.kv_cache_usage);

    // Prefix-cache counters, including the connector-backed external cache path.
    handles.prefix_cache_queries.inc_by(stats.prefix_cache_stats.base.queries);
    handles.prefix_cache_hits.inc_by(stats.prefix_cache_stats.base.hits);
    // Local prefix-cache hits are the `device` tier.
    handles.prompt_tokens_cached_by_source[0].inc_by(stats.prefix_cache_stats.base.hits);

    if let Some(connector_prefix_cache_stats) = &stats.connector_prefix_cache_stats {
        handles
            .external_prefix_cache_queries
            .inc_by(connector_prefix_cache_stats.base.queries);
        handles
            .external_prefix_cache_hits
            .inc_by(connector_prefix_cache_stats.base.hits);
        let counts = connector_prefix_cache_stats.hits_by_source.counts();
        for (handle, count) in handles.prompt_tokens_cached_by_source.iter().zip(counts) {
            handle.inc_by(count);
        }
    }

    // Speculative decoding counters.
    if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
        handles.spec_decode_num_drafts.inc_by(spec_decoding_stats.num_drafts);
        handles
            .spec_decode_num_draft_tokens
            .inc_by(spec_decoding_stats.num_draft_tokens);
        handles
            .spec_decode_num_accepted_tokens
            .inc_by(spec_decoding_stats.num_accepted_tokens);
        handles.log_stats.observe_spec_decode(
            spec_decoding_stats.num_drafts,
            &spec_decoding_stats.num_accepted_tokens_per_pos,
        );

        for (position, accepted_tokens) in
            spec_decoding_stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
        {
            handles
                .spec_decode_num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: handles.labels.model_name.clone(),
                    engine: handles.labels.engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
    }

    // Per-engine performance / MFU counters.
    if let Some(perf_stats) = &stats.perf_stats
        && (perf_stats.num_flops_per_gpu != 0
            || perf_stats.num_read_bytes_per_gpu != 0
            || perf_stats.num_write_bytes_per_gpu != 0)
    {
        handles.estimated_flops_per_gpu.inc_by(perf_stats.num_flops_per_gpu);
        handles.estimated_read_bytes_per_gpu.inc_by(perf_stats.num_read_bytes_per_gpu);
        handles.estimated_write_bytes_per_gpu.inc_by(perf_stats.num_write_bytes_per_gpu);
    }

    if let Some(cudagraph_stats) = &stats.cudagraph_stats {
        handles.log_stats.observe_cudagraph(
            cudagraph_stats.num_unpadded_tokens,
            cudagraph_stats.num_padded_tokens,
            cudagraph_stats.num_paddings,
            &cudagraph_stats.runtime_mode,
        );
    }

    // Sampled KV-cache residency histograms.
    if !stats.kv_cache_eviction_events.is_empty() {
        for event in &stats.kv_cache_eviction_events {
            handles.kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            handles.kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                handles.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }

    // Connector-specific KV transfer stats. A bare connector reports its own
    // flat payload; MultiConnector reports connector class name -> flat child
    // payload.
    if let Some(kv_connector_stats) = &stats.kv_connector_stats {
        match kv_connector_stats {
            KvConnectorStats::Nixl(stats) => record_nixl_stats(handles, stats),
            KvConnectorStats::MooncakeTransfer(stats) => {
                record_mooncake_transfer_stats(handles, stats)
            }
            KvConnectorStats::Mooncake(stats) => record_mooncake_stats(handles, stats),
            KvConnectorStats::Redhare(stats) => record_redhare_stats(handles, stats),
            KvConnectorStats::Multi(stats) => record_multi_connector_stats(handles, stats),
            KvConnectorStats::Other(_) => {}
        }
    }
}

fn record_multi_connector_stats(handles: &SchedulerStatsHandles, stats: &MultiConnectorStats) {
    for nixl in [&stats.nixl, &stats.nixl_pull, &stats.nixl_push].into_iter().flatten() {
        record_nixl_stats(handles, nixl);
    }
    if let Some(mooncake_transfer) = &stats.mooncake_transfer {
        record_mooncake_transfer_stats(handles, mooncake_transfer);
    }
    if let Some(mooncake) = &stats.mooncake {
        record_mooncake_stats(handles, mooncake);
    }
    if let Some(redhare) = &stats.redhare {
        record_redhare_stats(handles, redhare);
    }
}

fn record_mooncake_transfer_stats(handles: &SchedulerStatsHandles, stats: &MooncakeTransferStats) {
    for value in &stats.transfer_duration {
        handles.mooncake_xfer_time_seconds.observe(*value);
    }
    for value in &stats.bytes_transferred {
        handles.mooncake_bytes_transferred.observe(*value as f64);
    }
    for value in &stats.num_descriptors {
        handles.mooncake_num_descriptors.observe(*value as f64);
    }
    handles
        .mooncake_num_failed_transfers
        .inc_by(stats.num_failed_transfers.iter().sum());
    handles.mooncake_num_failed_recvs.inc_by(stats.num_failed_recvs.iter().sum());
    handles
        .mooncake_num_kv_expired_reqs
        .inc_by(stats.num_kv_expired_reqs.iter().sum());
}

fn record_mooncake_stats(handles: &SchedulerStatsHandles, stats: &MooncakeStats) {
    for (operation, records) in &stats.0 {
        for record in records {
            let labels = MooncakeOperationLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                operation: operation.as_str().to_string(),
                status: record.status.as_str().to_string(),
            };
            handles
                .mooncake_operation_time_seconds
                .get_or_create(&labels)
                .observe(record.duration_seconds);
            handles.mooncake_operation_total.get_or_create(&labels).inc();
            handles
                .mooncake_operation_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_keys);
            handles
                .mooncake_operation_bytes_total
                .get_or_create(&labels)
                .inc_by(record.num_bytes);
            handles
                .mooncake_operation_failed_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_failed_keys);
        }
    }
}

fn record_nixl_stats(handles: &SchedulerStatsHandles, stats: &NixlStats) {
    for value in &stats.transfer_duration {
        handles.nixl_xfer_time_seconds.observe(*value);
    }
    for value in &stats.post_duration {
        handles.nixl_post_time_seconds.observe(*value);
    }
    for value in &stats.bytes_transferred {
        handles.nixl_bytes_transferred.observe(*value as f64);
    }
    for value in &stats.num_descriptors {
        handles.nixl_num_descriptors.observe(*value as f64);
    }
    handles
        .nixl_num_failed_transfers
        .inc_by(stats.num_failed_transfers.iter().sum());
    handles
        .nixl_num_failed_notifications
        .inc_by(stats.num_failed_notifications.iter().sum());
    handles.nixl_num_kv_expired_reqs.inc_by(stats.num_kv_expired_reqs.iter().sum());
}

fn redhare_engine_labels(labels: &EngineLabels) -> RedhareEngineLabels {
    RedhareEngineLabels {
        model_name: labels.model_name.clone(),
        engine: labels.engine,
        kv_cluster: redhare_kv_cluster(),
    }
}

fn inc_redhare(
    counter: &Family<RedhareEngineLabels, U64Counter>,
    labels: &EngineLabels,
    value: u64,
) {
    if value != 0 {
        counter.get_or_create(&redhare_engine_labels(labels)).inc_by(value);
    }
}

fn inc_redhare_path(
    counter: &Family<RedharePathLabels, U64Counter>,
    labels: &EngineLabels,
    path: &'static str,
    value: u64,
) {
    if value != 0 {
        counter
            .get_or_create(&RedharePathLabels {
                model_name: labels.model_name.clone(),
                engine: labels.engine,
                kv_cluster: redhare_kv_cluster(),
                path,
            })
            .inc_by(value);
    }
}

fn observe_redhare_us(
    histogram: &Family<RedhareEngineLabels, HistogramMetric, fn() -> HistogramMetric>,
    labels: &EngineLabels,
    samples_us: &[u64],
) {
    if samples_us.is_empty() {
        return;
    }
    let histogram = histogram.get_or_create(&redhare_engine_labels(labels));
    for sample_us in samples_us {
        histogram.observe(*sample_us as f64 / 1e6);
    }
}

fn set_redhare_gauge(
    gauge: &Family<RedhareEngineLabels, U64Gauge>,
    labels: &EngineLabels,
    value: Option<u64>,
) {
    if let Some(value) = value {
        gauge.get_or_create(&redhare_engine_labels(labels)).set(value);
    }
}

/// Record one Redhare stats payload. Latencies arrive in microseconds and are
/// stored in seconds, matching `RedhareKVConnectorPromMetrics.observe`.
fn record_redhare_stats(handles: &SchedulerStatsHandles, stats: &RedhareStats) {
    let labels = &handles.labels;
    let metrics = &handles.redhare;
    inc_redhare(&metrics.put, labels, stats.put_total);
    inc_redhare(&metrics.put_failed, labels, stats.put_failed);
    inc_redhare(
        &metrics.put_fingerprint_mismatch,
        labels,
        stats.put_fingerprint_mismatch,
    );
    inc_redhare(&metrics.get, labels, stats.get_total);
    inc_redhare(&metrics.get_hit_local, labels, stats.get_hit_local);
    inc_redhare(&metrics.get_hit_remote, labels, stats.get_hit_remote);
    inc_redhare(&metrics.get_hit_shared_fs, labels, stats.get_hit_shared_fs);
    inc_redhare(
        &metrics.get_hit_local_disk,
        labels,
        stats.get_hit_local_disk,
    );
    inc_redhare(&metrics.get_miss, labels, stats.get_miss);
    inc_redhare(
        &metrics.get_fingerprint_mismatch,
        labels,
        stats.get_fingerprint_mismatch,
    );
    inc_redhare(&metrics.bytes_put, labels, stats.bytes_put);
    inc_redhare(&metrics.bytes_get, labels, stats.bytes_get);
    inc_redhare(&metrics.local_evictions, labels, stats.local_evictions);
    inc_redhare(
        &metrics.local_tinylfu_protected,
        labels,
        stats.local_tinylfu_protected,
    );
    inc_redhare(
        &metrics.shared_fs_demote,
        labels,
        stats.shared_fs_demote_total,
    );
    inc_redhare(
        &metrics.shared_fs_demote_failed,
        labels,
        stats.shared_fs_demote_failed,
    );
    inc_redhare(
        &metrics.local_disk_demote,
        labels,
        stats.local_disk_demote_total,
    );
    inc_redhare(
        &metrics.local_disk_demote_failed,
        labels,
        stats.local_disk_demote_failed,
    );
    inc_redhare(
        &metrics.remote_dram_reserve,
        labels,
        stats.remote_dram_reserve_total,
    );
    inc_redhare(
        &metrics.remote_dram_reserve_no_space,
        labels,
        stats.remote_dram_reserve_no_space,
    );
    inc_redhare(
        &metrics.remote_dram_reserve_failed,
        labels,
        stats.remote_dram_reserve_failed,
    );
    inc_redhare(
        &metrics.remote_dram_bind,
        labels,
        stats.remote_dram_bind_total,
    );
    inc_redhare(
        &metrics.remote_dram_bind_failed,
        labels,
        stats.remote_dram_bind_failed,
    );
    inc_redhare(
        &metrics.remote_dram_free,
        labels,
        stats.remote_dram_free_total,
    );
    inc_redhare(
        &metrics.remote_dram_free_failed,
        labels,
        stats.remote_dram_free_failed,
    );
    inc_redhare(
        &metrics.remote_dram_reserve_reaped,
        labels,
        stats.remote_dram_reserve_reaped,
    );
    inc_redhare(
        &metrics.remote_dram_evictions,
        labels,
        stats.remote_dram_evictions,
    );
    inc_redhare(
        &metrics.remote_dram_window_evictions,
        labels,
        stats.remote_dram_window_evictions,
    );
    inc_redhare(
        &metrics.remote_dram_main_evictions,
        labels,
        stats.remote_dram_main_evictions,
    );
    inc_redhare(
        &metrics.remote_dram_window_promotions,
        labels,
        stats.remote_dram_window_promotions,
    );
    inc_redhare(
        &metrics.remote_dram_tinylfu_protected,
        labels,
        stats.remote_dram_tinylfu_protected,
    );
    inc_redhare(
        &metrics.remote_dram_evicted_bytes,
        labels,
        stats.remote_dram_evicted_bytes,
    );
    inc_redhare(
        &metrics.remote_dram_write_bytes,
        labels,
        stats.remote_dram_write_bytes,
    );
    inc_redhare(
        &metrics.connector_save_failed,
        labels,
        stats.connector_save_failed,
    );
    inc_redhare(
        &metrics.connector_load_failed,
        labels,
        stats.connector_load_failed,
    );
    inc_redhare(
        &metrics.connector_chunks_dedup_skipped,
        labels,
        stats.connector_chunks_dedup_skipped,
    );
    inc_redhare(
        &metrics.connector_chunks_skipped_unbacked,
        labels,
        stats.connector_chunks_skipped_unbacked,
    );
    inc_redhare(
        &metrics.connector_chunks_skipped_retention,
        labels,
        stats.connector_chunks_skipped_retention,
    );
    inc_redhare(
        &metrics.connector_save_skipped_behind,
        labels,
        stats.connector_save_skipped_behind,
    );
    inc_redhare(
        &metrics.connector_save_cancelled,
        labels,
        stats.connector_save_cancelled,
    );
    inc_redhare(
        &metrics.connector_load_cancelled,
        labels,
        stats.connector_load_cancelled,
    );
    inc_redhare(
        &metrics.connector_save_unpublished,
        labels,
        stats.connector_save_unpublished,
    );
    inc_redhare(
        &metrics.connector_preempt_fence_timeout,
        labels,
        stats.connector_preempt_fence_timeout,
    );
    inc_redhare(
        &metrics.connector_ssm_fp_checked,
        labels,
        stats.connector_ssm_fp_checked,
    );
    inc_redhare(
        &metrics.connector_ssm_fp_mismatch,
        labels,
        stats.connector_ssm_fp_mismatch,
    );
    inc_redhare(
        &metrics.connector_ssm_fp_unsampled,
        labels,
        stats.connector_ssm_fp_unsampled,
    );
    inc_redhare(
        &metrics.external_cache_lookup,
        labels,
        stats.external_cache_lookup,
    );
    inc_redhare(
        &metrics.external_cache_eligible_tokens,
        labels,
        stats.external_cache_eligible_tokens,
    );
    inc_redhare(
        &metrics.external_cache_hit_tokens,
        labels,
        stats.external_cache_hit_tokens,
    );

    inc_redhare_path(&metrics.put_path, labels, "local", stats.put_path_local);
    inc_redhare_path(&metrics.put_path, labels, "remote", stats.put_path_remote);
    inc_redhare_path(
        &metrics.put_path,
        labels,
        "shared_fs",
        stats.put_path_shared_fs,
    );
    inc_redhare_path(
        &metrics.put_path,
        labels,
        "local_disk",
        stats.put_path_local_disk,
    );
    inc_redhare_path(&metrics.put_bytes, labels, "local", stats.put_bytes_local);
    inc_redhare_path(&metrics.put_bytes, labels, "remote", stats.put_bytes_remote);
    inc_redhare_path(
        &metrics.put_bytes,
        labels,
        "shared_fs",
        stats.put_bytes_shared_fs,
    );
    inc_redhare_path(
        &metrics.put_bytes,
        labels,
        "local_disk",
        stats.put_bytes_local_disk,
    );
    inc_redhare_path(&metrics.get_bytes, labels, "local", stats.get_bytes_local);
    inc_redhare_path(&metrics.get_bytes, labels, "remote", stats.get_bytes_remote);
    inc_redhare_path(
        &metrics.get_bytes,
        labels,
        "shared_fs",
        stats.get_bytes_shared_fs,
    );
    inc_redhare_path(
        &metrics.get_bytes,
        labels,
        "local_disk",
        stats.get_bytes_local_disk,
    );
    inc_redhare_path(
        &metrics.local_admission_bypass,
        labels,
        "remote",
        stats.local_admission_bypass_remote,
    );
    inc_redhare_path(
        &metrics.local_admission_bypass,
        labels,
        "shared_fs",
        stats.local_admission_bypass_shared_fs,
    );
    inc_redhare_path(
        &metrics.local_admission_bypass,
        labels,
        "local_disk",
        stats.local_admission_bypass_local_disk,
    );

    set_redhare_gauge(
        &metrics.local_objects_current,
        labels,
        stats.local_objects_current,
    );
    set_redhare_gauge(
        &metrics.local_bytes_current,
        labels,
        stats.local_bytes_current,
    );
    set_redhare_gauge(
        &metrics.local_capacity_bytes,
        labels,
        stats.local_capacity_bytes,
    );
    set_redhare_gauge(&metrics.local_free_bytes, labels, stats.local_free_bytes);
    set_redhare_gauge(
        &metrics.kvcache_capacity_bytes,
        labels,
        stats.kvcache_capacity_bytes,
    );
    set_redhare_gauge(
        &metrics.remote_dram_slots_current,
        labels,
        stats.remote_dram_slots_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_bytes_current,
        labels,
        stats.remote_dram_bytes_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_window_percent_bps,
        labels,
        stats.remote_dram_window_percent_bps,
    );
    set_redhare_gauge(
        &metrics.remote_dram_window_slots_current,
        labels,
        stats.remote_dram_window_slots_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_window_bytes_current,
        labels,
        stats.remote_dram_window_bytes_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_main_slots_current,
        labels,
        stats.remote_dram_main_slots_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_main_bytes_current,
        labels,
        stats.remote_dram_main_bytes_current,
    );
    set_redhare_gauge(
        &metrics.remote_dram_capacity_bytes,
        labels,
        stats.remote_dram_capacity_bytes,
    );
    set_redhare_gauge(
        &metrics.remote_dram_free_bytes,
        labels,
        stats.remote_dram_free_bytes,
    );

    observe_redhare_us(&metrics.put_latency_seconds, labels, &stats.put_latency_us);
    observe_redhare_us(
        &metrics.transport_read_latency_seconds,
        labels,
        &stats.transport_read_latency_us,
    );
    observe_redhare_us(
        &metrics.remove_latency_seconds,
        labels,
        &stats.remove_latency_us,
    );
    for (outcome, samples) in [
        ("local", &stats.get_latency_us_local),
        ("remote", &stats.get_latency_us_remote),
        ("shared_fs", &stats.get_latency_us_shared_fs),
        ("local_disk", &stats.get_latency_us_local_disk),
        ("miss", &stats.get_latency_us_miss),
    ] {
        if samples.is_empty() {
            continue;
        }
        let histogram = metrics.get_latency_seconds.get_or_create(&RedhareOutcomeLabels {
            model_name: labels.model_name.clone(),
            engine: labels.engine,
            kv_cluster: redhare_kv_cluster(),
            outcome,
        });
        for sample_us in samples {
            histogram.observe(*sample_us as f64 / 1e6);
        }
    }
}

/// Exports `vllm:lora_requests_info` as a single series covering all LoRA
/// requests tracked by this client across every engine in the replica.
///
/// The engine's `SchedulerStats` never carries adapter names: the Python
/// frontend fills them in from per-request lifecycle events tracked by
/// `LoRARequestStates` in `vllm/v1/engine/output_processor.py`. The Rust
/// frontend mirrors that, deriving the sets from the request registry.
#[derive(Default)]
pub(crate) struct LoraInfoExporter {
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub(crate) fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
            running_lora_adapters: LoraAdapterNames(running),
            waiting_lora_adapters: LoraAdapterNames(waiting),
        });

        if self.current != next
            && let Some(prev) = &self.current
        {
            metrics.lora_info.remove(prev);
        }

        // Python sets this gauge to the current time on every record.
        if let Some(labels) = &next {
            metrics.lora_info.get_or_create(labels).set(now_unix_secs());
        }

        self.current = next;
    }
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use expect_test::expect;
    use vllm_metrics::Metrics;

    use crate::metrics::LoraInfoExporter;
    use crate::protocol::stats::{
        KvConnectorStats, MooncakeOperation, MooncakeRecord, MooncakeStats, MooncakeStatus,
        MooncakeTransferStats, MultiConnectorStats, NixlStats, SchedulerStats,
    };

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
    }

    /// The `lora_requests_info` series with the non-deterministic timestamp
    /// value replaced by `<ts>`, one line per series.
    fn lora_series(rendered: &str) -> String {
        rendered
            .lines()
            .filter(|l| l.starts_with("vllm:lora_requests_info{"))
            .map(|l| match l.rsplit_once("} ") {
                Some((labels, _value)) => format!("{labels}}} <ts>"),
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn lora_info_emits_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::default();

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }

    fn nixl_stats() -> NixlStats {
        NixlStats {
            transfer_duration: vec![0.01, 0.02],
            post_duration: vec![0.001, 0.002],
            bytes_transferred: vec![4096, 8192],
            num_descriptors: vec![2, 4],
            num_failed_transfers: vec![],
            num_failed_notifications: vec![],
            num_kv_expired_reqs: vec![1],
        }
    }

    fn mooncake_stats() -> MooncakeStats {
        MooncakeStats(BTreeMap::from([(
            MooncakeOperation::LoadGet,
            vec![MooncakeRecord {
                duration_seconds: 0.05,
                num_keys: 3,
                num_bytes: 1024,
                status: MooncakeStatus::Ok,
                num_failed_keys: 0,
            }],
        )]))
    }

    /// Records a bare `MooncakeConnector` payload, in the shape Python's
    /// `MooncakeKVConnectorStats` sends, into the `vllm:mooncake_*` transfer
    /// metrics.
    #[test]
    fn mooncake_transfer_stats_are_recorded_into_mooncake_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);

        let kv_connector_stats: KvConnectorStats = serde_json::from_value(serde_json::json!({
            "transfer_duration": [0.01, 0.02],
            "bytes_transferred": [4096, 8192],
            "num_descriptors": [2, 4],
            "num_failed_transfers": [1, 1],
            "num_failed_recvs": [1],
            "num_kv_expired_reqs": [],
        }))
        .unwrap();
        let stats = SchedulerStats {
            kv_connector_stats: Some(kv_connector_stats),
            ..Default::default()
        };

        super::record_scheduler_stats_with_handles(&handles, &stats);

        let rendered = metrics.render().unwrap();
        let mooncake_transfer_series = rendered
            .lines()
            .filter(|line| {
                line.starts_with("vllm:mooncake_")
                    && !line.starts_with("vllm:mooncake_store_")
                    && !line.contains("_bucket{")
            })
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            vllm:mooncake_xfer_time_seconds_sum{model_name="model",engine="0"} 0.03
            vllm:mooncake_xfer_time_seconds_count{model_name="model",engine="0"} 2
            vllm:mooncake_bytes_transferred_sum{model_name="model",engine="0"} 12288.0
            vllm:mooncake_bytes_transferred_count{model_name="model",engine="0"} 2
            vllm:mooncake_num_descriptors_sum{model_name="model",engine="0"} 6.0
            vllm:mooncake_num_descriptors_count{model_name="model",engine="0"} 2
            vllm:mooncake_num_failed_transfers_total{model_name="model",engine="0"} 2
            vllm:mooncake_num_failed_recvs_total{model_name="model",engine="0"} 1
            vllm:mooncake_num_kv_expired_reqs_total{model_name="model",engine="0"} 0"#]]
        .assert_eq(&mooncake_transfer_series);
    }

    /// Records one MultiConnector payload into Mooncake and NIXL metrics.
    #[test]
    fn kv_connector_stats_are_recorded_into_mooncake_and_nixl_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);

        let stats = SchedulerStats {
            kv_connector_stats: Some(KvConnectorStats::Multi(Box::new(MultiConnectorStats {
                nixl: Some(nixl_stats()),
                mooncake_transfer: Some(MooncakeTransferStats {
                    transfer_duration: vec![0.03],
                    bytes_transferred: vec![2048],
                    num_descriptors: vec![1],
                    num_failed_recvs: vec![1],
                    ..Default::default()
                }),
                mooncake: Some(mooncake_stats()),
                ..Default::default()
            }))),
            ..Default::default()
        };

        super::record_scheduler_stats_with_handles(&handles, &stats);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_keys_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 3"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_bytes_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1024"
        ));
        assert!(
            rendered.contains(
                "vllm:nixl_num_kv_expired_reqs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(
            rendered
                .contains("vllm:nixl_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 2")
        );
        assert!(rendered.contains(
            "vllm:mooncake_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(
            rendered.contains(
                "vllm:mooncake_num_failed_recvs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
    }

    /// A flat Redhare payload is not swallowed by `Multi`, and microseconds
    /// become seconds on the same metric names the Python frontend registers.
    #[test]
    fn redhare_stats_are_recorded_into_redhare_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let kv_connector_stats: KvConnectorStats = serde_json::from_value(serde_json::json!({
            "put_total": 2,
            "get_hit_local": 1,
            "put_path_remote": 4,
            "get_bytes_local": 50,
            "local_admission_bypass_shared_fs": 3,
            "local_objects_current": 7,
            "put_latency_us": [1000, 3000],
            "get_latency_us_remote": [2000],
            "future_counter": 9,
            "_n_steps": 1,
        }))
        .unwrap();
        assert!(matches!(kv_connector_stats, KvConnectorStats::Redhare(_)));
        super::record_scheduler_stats_with_handles(
            &handles,
            &SchedulerStats {
                kv_connector_stats: Some(kv_connector_stats),
                ..Default::default()
            },
        );

        let rendered = metrics.render().unwrap();
        let series = rendered
            .lines()
            .filter(|line| line.starts_with("redhare_") && !line.contains("_bucket{"))
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            redhare_put_path_total{model_name="model",engine="0",kv_cluster="default",path="remote"} 4
            redhare_get_bytes_total{model_name="model",engine="0",kv_cluster="default",path="local"} 50
            redhare_local_admission_bypass_total{model_name="model",engine="0",kv_cluster="default",path="shared_fs"} 3
            redhare_get_latency_seconds_sum{model_name="model",engine="0",kv_cluster="default",outcome="remote"} 0.002
            redhare_get_latency_seconds_count{model_name="model",engine="0",kv_cluster="default",outcome="remote"} 1
            redhare_put_total{model_name="model",engine="0",kv_cluster="default"} 2
            redhare_get_hit_local_total{model_name="model",engine="0",kv_cluster="default"} 1
            redhare_put_latency_seconds_sum{model_name="model",engine="0",kv_cluster="default"} 0.004
            redhare_put_latency_seconds_count{model_name="model",engine="0",kv_cluster="default"} 2
            redhare_local_objects_current{model_name="model",engine="0",kv_cluster="default"} 7"#]]
        .assert_eq(&series);
    }
}
