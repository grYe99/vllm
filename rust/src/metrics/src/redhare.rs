// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prometheus families for Redhare connector telemetry.
//!
//! Names and buckets match `RedhareKVConnectorPromMetrics` in the Redhare
//! Python package. `kv_cluster` is the connector extra-config `cluster`
//! (Python default `"default"`), copied onto every series at record time.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::{Histogram, U64Counter, U64Gauge};

/// Buckets copied from `redhare.vllm_connector.stats._LAT_BUCKETS_S`.
const REDHARE_LATENCY_BUCKETS_S: [f64; 18] = [
    50e-6, 100e-6, 250e-6, 500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3,
    1.0, 2.5, 5.0, 10.0, 30.0,
];

fn redhare_latency_histogram() -> Histogram {
    Histogram::new(REDHARE_LATENCY_BUCKETS_S.iter().copied())
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RedhareEngineLabels {
    pub model_name: String,
    pub engine: u32,
    pub kv_cluster: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RedhareOutcomeLabels {
    pub model_name: String,
    pub engine: u32,
    pub kv_cluster: String,
    pub outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RedharePathLabels {
    pub model_name: String,
    pub engine: u32,
    pub kv_cluster: String,
    pub path: &'static str,
}

pub type RedhareLatencyFamily = Family<RedhareEngineLabels, Histogram, fn() -> Histogram>;
pub type RedhareOutcomeLatencyFamily = Family<RedhareOutcomeLabels, Histogram, fn() -> Histogram>;

/// Families mirroring `RedhareKVConnectorPromMetrics`.
#[derive(Clone)]
pub struct RedhareMetrics {
    pub put: Family<RedhareEngineLabels, U64Counter>,
    pub put_failed: Family<RedhareEngineLabels, U64Counter>,
    pub put_fingerprint_mismatch: Family<RedhareEngineLabels, U64Counter>,
    pub get: Family<RedhareEngineLabels, U64Counter>,
    pub get_hit_local: Family<RedhareEngineLabels, U64Counter>,
    pub get_hit_remote: Family<RedhareEngineLabels, U64Counter>,
    pub get_hit_shared_fs: Family<RedhareEngineLabels, U64Counter>,
    pub get_hit_local_disk: Family<RedhareEngineLabels, U64Counter>,
    pub get_miss: Family<RedhareEngineLabels, U64Counter>,
    pub get_fingerprint_mismatch: Family<RedhareEngineLabels, U64Counter>,
    pub bytes_put: Family<RedhareEngineLabels, U64Counter>,
    pub bytes_get: Family<RedhareEngineLabels, U64Counter>,
    pub local_evictions: Family<RedhareEngineLabels, U64Counter>,
    pub local_tinylfu_protected: Family<RedhareEngineLabels, U64Counter>,
    pub shared_fs_demote: Family<RedhareEngineLabels, U64Counter>,
    pub shared_fs_demote_failed: Family<RedhareEngineLabels, U64Counter>,
    pub local_disk_demote: Family<RedhareEngineLabels, U64Counter>,
    pub local_disk_demote_failed: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_reserve: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_reserve_no_space: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_reserve_failed: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_bind: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_bind_failed: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_free: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_free_failed: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_reserve_reaped: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_evictions: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_window_evictions: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_main_evictions: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_window_promotions: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_tinylfu_protected: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_evicted_bytes: Family<RedhareEngineLabels, U64Counter>,
    pub remote_dram_write_bytes: Family<RedhareEngineLabels, U64Counter>,
    pub connector_save_failed: Family<RedhareEngineLabels, U64Counter>,
    pub connector_load_failed: Family<RedhareEngineLabels, U64Counter>,
    pub connector_chunks_dedup_skipped: Family<RedhareEngineLabels, U64Counter>,
    pub connector_chunks_skipped_unbacked: Family<RedhareEngineLabels, U64Counter>,
    pub connector_chunks_skipped_retention: Family<RedhareEngineLabels, U64Counter>,
    pub connector_save_skipped_behind: Family<RedhareEngineLabels, U64Counter>,
    pub connector_save_cancelled: Family<RedhareEngineLabels, U64Counter>,
    pub connector_load_cancelled: Family<RedhareEngineLabels, U64Counter>,
    pub connector_save_unpublished: Family<RedhareEngineLabels, U64Counter>,
    pub connector_preempt_fence_timeout: Family<RedhareEngineLabels, U64Counter>,
    pub connector_ssm_fp_checked: Family<RedhareEngineLabels, U64Counter>,
    pub connector_ssm_fp_mismatch: Family<RedhareEngineLabels, U64Counter>,
    pub connector_ssm_fp_unsampled: Family<RedhareEngineLabels, U64Counter>,
    pub external_cache_lookup: Family<RedhareEngineLabels, U64Counter>,
    pub external_cache_eligible_tokens: Family<RedhareEngineLabels, U64Counter>,
    pub external_cache_hit_tokens: Family<RedhareEngineLabels, U64Counter>,
    pub put_path: Family<RedharePathLabels, U64Counter>,
    pub put_bytes: Family<RedharePathLabels, U64Counter>,
    pub get_bytes: Family<RedharePathLabels, U64Counter>,
    pub local_admission_bypass: Family<RedharePathLabels, U64Counter>,
    pub put_latency_seconds: RedhareLatencyFamily,
    pub transport_read_latency_seconds: RedhareLatencyFamily,
    pub remove_latency_seconds: RedhareLatencyFamily,
    pub get_latency_seconds: RedhareOutcomeLatencyFamily,
    pub local_objects_current: Family<RedhareEngineLabels, U64Gauge>,
    pub local_bytes_current: Family<RedhareEngineLabels, U64Gauge>,
    pub local_capacity_bytes: Family<RedhareEngineLabels, U64Gauge>,
    pub local_free_bytes: Family<RedhareEngineLabels, U64Gauge>,
    pub kvcache_capacity_bytes: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_slots_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_bytes_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_window_percent_bps: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_window_slots_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_window_bytes_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_main_slots_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_main_bytes_current: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_capacity_bytes: Family<RedhareEngineLabels, U64Gauge>,
    pub remote_dram_free_bytes: Family<RedhareEngineLabels, U64Gauge>,
}

fn register_counter(
    registry: &mut Registry,
    name: &'static str,
    help: &'static str,
) -> Family<RedhareEngineLabels, U64Counter> {
    let family = Family::default();
    registry.register(name, help, family.clone());
    family
}

fn register_gauge(
    registry: &mut Registry,
    name: &'static str,
    help: &'static str,
) -> Family<RedhareEngineLabels, U64Gauge> {
    let family = Family::default();
    registry.register(name, help, family.clone());
    family
}

fn register_latency(
    registry: &mut Registry,
    name: &'static str,
    help: &'static str,
) -> RedhareLatencyFamily {
    let family = Family::new_with_constructor(redhare_latency_histogram as fn() -> Histogram);
    registry.register(name, help, family.clone());
    family
}

impl RedhareMetrics {
    pub(crate) fn register(registry: &mut Registry) -> Self {
        let put_path = Family::default();
        registry.register(
            "redhare_put_path",
            "Successful put commits by storage path.",
            put_path.clone(),
        );
        let put_bytes = Family::default();
        registry.register(
            "redhare_put_bytes",
            "Bytes committed by put path.",
            put_bytes.clone(),
        );
        let get_bytes = Family::default();
        registry.register(
            "redhare_get_bytes",
            "Bytes returned by get path.",
            get_bytes.clone(),
        );
        let local_admission_bypass = Family::default();
        registry.register(
            "redhare_local_admission_bypass",
            "Fresh puts bypassed local DRAM under admission pressure, by destination path.",
            local_admission_bypass.clone(),
        );
        let get_latency_seconds =
            Family::new_with_constructor(redhare_latency_histogram as fn() -> Histogram);
        registry.register(
            "redhare_get_latency_seconds",
            "Redhare get latency by outcome.",
            get_latency_seconds.clone(),
        );

        Self {
            put: register_counter(registry, "redhare_put", "Redhare put attempts."),
            put_failed: register_counter(registry, "redhare_put_failed", "Redhare put failures."),
            put_fingerprint_mismatch: register_counter(
                registry,
                "redhare_put_fingerprint_mismatch",
                "Remote puts rejected before metadata commit because stored bytes did not match the writer content fingerprint.",
            ),
            get: register_counter(registry, "redhare_get", "Redhare get attempts."),
            get_hit_local: register_counter(
                registry,
                "redhare_get_hit_local",
                "Get served from local DRAM.",
            ),
            get_hit_remote: register_counter(
                registry,
                "redhare_get_hit_remote",
                "Get served from a peer via data plane.",
            ),
            get_hit_shared_fs: register_counter(
                registry,
                "redhare_get_hit_shared_fs",
                "Get served from the SharedFs cold tier.",
            ),
            get_hit_local_disk: register_counter(
                registry,
                "redhare_get_hit_local_disk",
                "Get served from the LocalDisk cold tier.",
            ),
            get_miss: register_counter(registry, "redhare_get_miss", "Get missed all tiers."),
            get_fingerprint_mismatch: register_counter(
                registry,
                "redhare_get_fingerprint_mismatch",
                "Reads rejected because the content fingerprint did not match metadata.",
            ),
            bytes_put: register_counter(registry, "redhare_bytes_put", "Bytes committed by put."),
            bytes_get: register_counter(registry, "redhare_bytes_get", "Bytes returned by get."),
            local_evictions: register_counter(
                registry,
                "redhare_local_evictions",
                "Local DRAM objects dropped to make room for new puts.",
            ),
            local_tinylfu_protected: register_counter(
                registry,
                "redhare_local_tinylfu_protected",
                "Local DRAM residents protected from colder write candidates by W-TinyLFU.",
            ),
            shared_fs_demote: register_counter(
                registry,
                "redhare_shared_fs_demote",
                "Successful DRAM→SharedFs demotions.",
            ),
            shared_fs_demote_failed: register_counter(
                registry,
                "redhare_shared_fs_demote_failed",
                "Failed DRAM→SharedFs demotions.",
            ),
            local_disk_demote: register_counter(
                registry,
                "redhare_local_disk_demote",
                "Successful DRAM→LocalDisk demotions.",
            ),
            local_disk_demote_failed: register_counter(
                registry,
                "redhare_local_disk_demote_failed",
                "Failed DRAM→LocalDisk demotions.",
            ),
            remote_dram_reserve: register_counter(
                registry,
                "redhare_remote_dram_reserve",
                "Successful remote-DRAM cache-node reservations.",
            ),
            remote_dram_reserve_no_space: register_counter(
                registry,
                "redhare_remote_dram_reserve_no_space",
                "Remote-DRAM reservations rejected because the cache-node had no space.",
            ),
            remote_dram_reserve_failed: register_counter(
                registry,
                "redhare_remote_dram_reserve_failed",
                "Remote-DRAM reservation errors.",
            ),
            remote_dram_bind: register_counter(
                registry,
                "redhare_remote_dram_bind",
                "Successful remote-DRAM object binds.",
            ),
            remote_dram_bind_failed: register_counter(
                registry,
                "redhare_remote_dram_bind_failed",
                "Failed remote-DRAM object binds.",
            ),
            remote_dram_free: register_counter(
                registry,
                "redhare_remote_dram_free",
                "Successful remote-DRAM slot frees.",
            ),
            remote_dram_free_failed: register_counter(
                registry,
                "redhare_remote_dram_free_failed",
                "Failed remote-DRAM slot frees.",
            ),
            remote_dram_reserve_reaped: register_counter(
                registry,
                "redhare_remote_dram_reserve_reaped",
                "Unfinished remote-DRAM reservations reclaimed by the cache-node TTL reaper.",
            ),
            remote_dram_evictions: register_counter(
                registry,
                "redhare_remote_dram_evictions",
                "Cache-node hosted remote-DRAM slot evictions.",
            ),
            remote_dram_window_evictions: register_counter(
                registry,
                "redhare_remote_dram_window_evictions",
                "Hosted remote-DRAM Window slots evicted before admission to Main.",
            ),
            remote_dram_main_evictions: register_counter(
                registry,
                "redhare_remote_dram_main_evictions",
                "Hosted remote-DRAM Main slots evicted after W-TinyLFU admission.",
            ),
            remote_dram_window_promotions: register_counter(
                registry,
                "redhare_remote_dram_window_promotions",
                "Hosted remote-DRAM Window candidates promoted to Main.",
            ),
            remote_dram_tinylfu_protected: register_counter(
                registry,
                "redhare_remote_dram_tinylfu_protected",
                "Hosted remote-DRAM Main victims protected from colder candidates by TinyLFU.",
            ),
            remote_dram_evicted_bytes: register_counter(
                registry,
                "redhare_remote_dram_evicted_bytes",
                "Bytes evicted from hosted remote-DRAM slots.",
            ),
            remote_dram_write_bytes: register_counter(
                registry,
                "redhare_remote_dram_write_bytes",
                "Bytes written into hosted remote-DRAM slots.",
            ),
            connector_save_failed: register_counter(
                registry,
                "redhare_connector_save_failed",
                "Connector-side save failures.",
            ),
            connector_load_failed: register_counter(
                registry,
                "redhare_connector_load_failed",
                "Connector-side load failures.",
            ),
            connector_chunks_dedup_skipped: register_counter(
                registry,
                "redhare_connector_chunks_dedup_skipped",
                "Chunks skipped because they already existed in the store.",
            ),
            connector_chunks_skipped_unbacked: register_counter(
                registry,
                "redhare_connector_chunks_skipped_unbacked",
                "Chunks skipped because a backing block was the null block or not allocated yet.",
            ),
            connector_chunks_skipped_retention: register_counter(
                registry,
                "redhare_connector_chunks_skipped_retention",
                "Chunks dropped by prefix-cache retention relative to dense spec reachability.",
            ),
            connector_save_skipped_behind: register_counter(
                registry,
                "redhare_connector_save_skipped_behind",
                "Steps where a hybrid model skipped its saves because the send queue was too deep.",
            ),
            connector_save_cancelled: register_counter(
                registry,
                "redhare_connector_save_cancelled",
                "Chunks dropped because their request was preempted while the store was still queued.",
            ),
            connector_load_cancelled: register_counter(
                registry,
                "redhare_connector_load_cancelled",
                "Loads dropped because their request was preempted while the scatter was still queued.",
            ),
            connector_save_unpublished: register_counter(
                registry,
                "redhare_connector_save_unpublished",
                "Objects deleted after being written by a store that was still running when its request was preempted.",
            ),
            connector_preempt_fence_timeout: register_counter(
                registry,
                "redhare_connector_preempt_fence_timeout",
                "Preemption fences that gave up waiting for an in-flight store.",
            ),
            connector_ssm_fp_checked: register_counter(
                registry,
                "redhare_connector_ssm_fp_checked",
                "SSM chunks whose source block was re-verified immediately before the store.",
            ),
            connector_ssm_fp_mismatch: register_counter(
                registry,
                "redhare_connector_ssm_fp_mismatch",
                "SSM chunks dropped because their source block was rewritten between the walk and the store.",
            ),
            connector_ssm_fp_unsampled: register_counter(
                registry,
                "redhare_connector_ssm_fp_unsampled",
                "SSM chunks dropped because no baseline fingerprint could be sampled.",
            ),
            external_cache_lookup: register_counter(
                registry,
                "redhare_external_cache_lookup",
                "External cache prefix lookup requests.",
            ),
            external_cache_eligible_tokens: register_counter(
                registry,
                "redhare_external_cache_eligible_tokens",
                "Tokens eligible for external cache hits after subtracting vLLM-computed prefix.",
            ),
            external_cache_hit_tokens: register_counter(
                registry,
                "redhare_external_cache_hit_tokens",
                "Externally hit tokens after subtracting vLLM-computed prefix.",
            ),
            put_path,
            put_bytes,
            get_bytes,
            local_admission_bypass,
            put_latency_seconds: register_latency(
                registry,
                "redhare_put_latency_seconds",
                "Redhare put latency.",
            ),
            transport_read_latency_seconds: register_latency(
                registry,
                "redhare_transport_read_latency_seconds",
                "Redhare data-plane (NIXL/TCP) read latency.",
            ),
            remove_latency_seconds: register_latency(
                registry,
                "redhare_remove_latency_seconds",
                "Redhare remove latency.",
            ),
            get_latency_seconds,
            local_objects_current: register_gauge(
                registry,
                "redhare_local_objects_current",
                "Current number of objects resident in local DRAM.",
            ),
            local_bytes_current: register_gauge(
                registry,
                "redhare_local_bytes_current",
                "Current bytes resident in local DRAM.",
            ),
            local_capacity_bytes: register_gauge(
                registry,
                "redhare_local_capacity_bytes",
                "Configured local DRAM arena capacity in bytes.",
            ),
            local_free_bytes: register_gauge(
                registry,
                "redhare_local_free_bytes",
                "Current local DRAM arena free bytes.",
            ),
            kvcache_capacity_bytes: register_gauge(
                registry,
                "redhare_kvcache_capacity_bytes",
                "Configured KVCache arena capacity contributed by this Redhare process.",
            ),
            remote_dram_slots_current: register_gauge(
                registry,
                "redhare_remote_dram_slots_current",
                "Current bound remote-DRAM slots hosted by this cache-node.",
            ),
            remote_dram_bytes_current: register_gauge(
                registry,
                "redhare_remote_dram_bytes_current",
                "Current bytes in hosted remote-DRAM slots.",
            ),
            remote_dram_window_percent_bps: register_gauge(
                registry,
                "redhare_remote_dram_window_percent_bps",
                "Configured hosted remote-DRAM W-TinyLFU Window size in basis points.",
            ),
            remote_dram_window_slots_current: register_gauge(
                registry,
                "redhare_remote_dram_window_slots_current",
                "Current hosted remote-DRAM slots in the Window segment.",
            ),
            remote_dram_window_bytes_current: register_gauge(
                registry,
                "redhare_remote_dram_window_bytes_current",
                "Current hosted remote-DRAM bytes in the Window segment.",
            ),
            remote_dram_main_slots_current: register_gauge(
                registry,
                "redhare_remote_dram_main_slots_current",
                "Current hosted remote-DRAM slots in the Main segment.",
            ),
            remote_dram_main_bytes_current: register_gauge(
                registry,
                "redhare_remote_dram_main_bytes_current",
                "Current hosted remote-DRAM bytes in the Main segment.",
            ),
            remote_dram_capacity_bytes: register_gauge(
                registry,
                "redhare_remote_dram_capacity_bytes",
                "Configured DRAM arena capacity available to this cache-node role.",
            ),
            remote_dram_free_bytes: register_gauge(
                registry,
                "redhare_remote_dram_free_bytes",
                "Current free bytes in the cache-node DRAM arena.",
            ),
        }
    }
}
