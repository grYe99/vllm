// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use parking_lot::RwLock;
use rmpv::Value;
use tracing::warn;
use vllm_metrics::{
    ConnectorMetricLabels, F64Counter, F64Gauge, Family, Histogram, MetricConstructor, Metrics,
    U64Counter, U64Gauge, connector_metric_labels,
};

use super::schema::{
    METRICS_SCHEMA_KEY, MetricDef, MetricType, MetricsSchemaV1, SCHEMA_VERSION_V1, SampleKind,
};

const ENV_SCHEMA_PATHS: &str = "VLLM_KV_CONNECTOR_METRICS_SCHEMA";
const ENV_DEFAULT_CONNECTOR_ID: &str = "VLLM_KV_CONNECTOR_METRICS_DEFAULT_ID";

#[derive(Clone)]
struct HistogramBuckets(Arc<Vec<f64>>);

impl MetricConstructor<Histogram> for HistogramBuckets {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.0.iter().copied())
    }
}

enum DynamicFamily {
    Counter(Family<ConnectorMetricLabels, U64Counter>),
    CounterF64(Family<ConnectorMetricLabels, F64Counter>),
    GaugeU64(Family<ConnectorMetricLabels, U64Gauge>),
    GaugeF64(Family<ConnectorMetricLabels, F64Gauge>),
    Histogram(Family<ConnectorMetricLabels, Histogram, HistogramBuckets>),
}

impl DynamicFamily {
    fn clone_family(&self) -> Result<Self, String> {
        Ok(match self {
            Self::Counter(f) => Self::Counter(f.clone()),
            Self::CounterF64(f) => Self::CounterF64(f.clone()),
            Self::GaugeU64(f) => Self::GaugeU64(f.clone()),
            Self::GaugeF64(f) => Self::GaugeF64(f.clone()),
            Self::Histogram(f) => Self::Histogram(f.clone()),
        })
    }
}

struct BoundMetric {
    def: MetricDef,
    family: DynamicFamily,
}

struct SchemaInstance {
    metrics: Vec<BoundMetric>,
}

/// Registers and observes schema-driven connector metrics.
pub(crate) struct SchemaDrivenAdapter {
    metrics: &'static Metrics,
    instances: RwLock<BTreeMap<String, SchemaInstance>>,
    /// Connector id used when the wire payload is a flat stats dict.
    default_connector_id: Option<String>,
    warned_missing: Mutex<BTreeSet<String>>,
    warned_bad_schema: Mutex<BTreeSet<String>>,
}

impl SchemaDrivenAdapter {
    /// Register in-tree builtin schemas, then env paths (env overrides by id).
    pub(crate) fn from_env(metrics: &'static Metrics) -> Self {
        let default_connector_id = std::env::var(ENV_DEFAULT_CONNECTOR_ID)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let adapter = Self {
            metrics,
            instances: RwLock::new(BTreeMap::new()),
            default_connector_id,
            warned_missing: Mutex::new(BTreeSet::new()),
            warned_bad_schema: Mutex::new(BTreeSet::new()),
        };

        let mut schemas: BTreeMap<String, MetricsSchemaV1> = BTreeMap::new();
        for schema in builtin_schemas() {
            schemas.insert(schema.connector_id.clone(), schema);
        }
        if let Ok(paths) = std::env::var(ENV_SCHEMA_PATHS) {
            for path in paths.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                match load_schema_file(path) {
                    Ok(schema) => {
                        schemas.insert(schema.connector_id.clone(), schema);
                    }
                    Err(err) => {
                        warn!(
                            path,
                            error = %err,
                            "failed to load KV connector metrics schema"
                        );
                    }
                }
            }
        }
        for schema in schemas.into_values() {
            if let Err(err) = adapter.ensure_registered(schema) {
                warn!(
                    error = %err,
                    "failed to register KV connector metrics schema"
                );
            }
        }
        adapter
    }

    /// Construct an empty adapter (tests) — no builtins, no env.
    #[cfg(test)]
    pub(crate) fn empty(metrics: &'static Metrics) -> Self {
        Self {
            metrics,
            instances: RwLock::new(BTreeMap::new()),
            default_connector_id: None,
            warned_missing: Mutex::new(BTreeSet::new()),
            warned_bad_schema: Mutex::new(BTreeSet::new()),
        }
    }

    /// Register only in-tree builtins (tests / zero-env Offloading path).
    #[cfg(test)]
    pub(crate) fn with_builtins(metrics: &'static Metrics) -> Self {
        let adapter = Self::empty(metrics);
        for schema in builtin_schemas() {
            adapter.ensure_registered(schema).expect("builtin schema must validate");
        }
        adapter
    }

    #[cfg(test)]
    pub(crate) fn with_default_id(mut self, connector_id: impl Into<String>) -> Self {
        self.default_connector_id = Some(connector_id.into());
        self
    }

    pub(crate) fn default_connector_id(&self) -> Option<&str> {
        self.default_connector_id.as_deref()
    }

    pub(crate) fn registered_ids(&self) -> Vec<String> {
        self.instances.read().keys().cloned().collect()
    }

    /// Validate and register one schema document (idempotent per connector_id).
    pub(crate) fn ensure_registered(&self, schema: MetricsSchemaV1) -> Result<(), String> {
        schema.validate()?;
        {
            let instances = self.instances.read();
            if instances.contains_key(&schema.connector_id) {
                return Ok(());
            }
        }
        let instance = self.build_instance(&schema)?;
        let mut instances = self.instances.write();
        instances.entry(schema.connector_id).or_insert(instance);
        Ok(())
    }

    /// Observe one opaque connector stats object for `connector_id`.
    pub(crate) fn observe(
        &self,
        connector_id: &str,
        model_name: &str,
        engine: u32,
        payload: &Value,
    ) {
        let mut payload = payload.clone();
        if let Some(schema_value) = map_remove(&mut payload, METRICS_SCHEMA_KEY) {
            match rmpv_to_schema(&schema_value) {
                Ok(schema) => {
                    if let Err(err) = self.ensure_registered(schema) {
                        self.warn_bad_schema(connector_id, &err);
                    }
                }
                Err(err) => self.warn_bad_schema(connector_id, &err),
            }
        }
        let _ = map_remove(&mut payload, "_n_steps");

        let instances = self.instances.read();
        let Some(instance) = instances.get(connector_id) else {
            drop(instances);
            self.warn_missing(connector_id);
            return;
        };

        for bound in &instance.metrics {
            observe_metric(bound, model_name, engine, &payload);
        }
    }

    fn build_instance(&self, schema: &MetricsSchemaV1) -> Result<SchemaInstance, String> {
        // Multiple MetricDefs may share one Prom name (e.g. path/outcome
        // const_labels). Register each unique name once and reuse the Family.
        let mut families: BTreeMap<String, DynamicFamily> = BTreeMap::new();
        let mut metrics = Vec::with_capacity(schema.metrics.len());
        for def in &schema.metrics {
            let reg_name = strip_counter_total_suffix(&def.name, &def.metric_type).to_string();
            if !families.contains_key(&reg_name) {
                let family = self.register_family(def, &reg_name)?;
                families.insert(reg_name.clone(), family);
            }
            let family = families.get(&reg_name).expect("family just inserted").clone_family()?;
            metrics.push(BoundMetric {
                def: def.clone(),
                family,
            });
        }
        Ok(SchemaInstance { metrics })
    }

    fn register_family(&self, def: &MetricDef, name: &str) -> Result<DynamicFamily, String> {
        let help = if def.documentation.is_empty() {
            format!("KV connector metric {}", def.name)
        } else {
            def.documentation.clone()
        };

        Ok(match def.metric_type {
            MetricType::Counter => match def.sample_kind {
                SampleKind::IncByF64 => {
                    let family: Family<ConnectorMetricLabels, F64Counter> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::CounterF64(family)
                }
                SampleKind::IncByU64 | SampleKind::IncBySumU64 => {
                    let family: Family<ConnectorMetricLabels, U64Counter> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::Counter(family)
                }
                other => {
                    return Err(format!(
                        "counter metric '{}' has unsupported sample_kind {other:?}",
                        def.name
                    ));
                }
            },
            MetricType::Gauge => match def.sample_kind {
                SampleKind::SetU64 => {
                    let family: Family<ConnectorMetricLabels, U64Gauge> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::GaugeU64(family)
                }
                SampleKind::SetF64 => {
                    let family: Family<ConnectorMetricLabels, F64Gauge> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::GaugeF64(family)
                }
                other => {
                    return Err(format!(
                        "gauge metric '{}' has unsupported sample_kind {other:?}",
                        def.name
                    ));
                }
            },
            MetricType::Histogram => {
                let buckets = def
                    .buckets
                    .clone()
                    .ok_or_else(|| format!("histogram '{}' missing buckets", def.name))?;
                let family =
                    Family::<ConnectorMetricLabels, Histogram, HistogramBuckets>::new_with_constructor(
                        HistogramBuckets(Arc::new(buckets)),
                    );
                self.metrics.with_dynamic_registry(|registry| {
                    registry.register(name, help, family.clone());
                });
                DynamicFamily::Histogram(family)
            }
        })
    }

    fn warn_missing(&self, connector_id: &str) {
        let mut warned = self.warned_missing.lock().expect("warn set");
        if warned.insert(connector_id.to_string()) {
            warn!(
                connector_id,
                "KV connector stats collected but no metrics schema is registered; \
                 set {ENV_SCHEMA_PATHS} or publish schema via handshake"
            );
        }
    }

    fn warn_bad_schema(&self, connector_id: &str, err: &str) {
        let mut warned = self.warned_bad_schema.lock().expect("warn set");
        if warned.insert(connector_id.to_string()) {
            warn!(
                connector_id,
                error = err,
                "ignoring invalid KV connector metrics schema"
            );
        }
    }
}

fn strip_counter_total_suffix<'a>(name: &'a str, metric_type: &MetricType) -> &'a str {
    if *metric_type == MetricType::Counter {
        name.strip_suffix("_total").unwrap_or(name)
    } else {
        name
    }
}

fn load_schema_file(path: &str) -> Result<MetricsSchemaV1, String> {
    let raw =
        std::fs::read_to_string(Path::new(path)).map_err(|err| format!("read {path}: {err}"))?;
    parse_schema_json(&raw, path)
}

fn parse_schema_json(raw: &str, source: &str) -> Result<MetricsSchemaV1, String> {
    let schema: MetricsSchemaV1 =
        serde_json::from_str(raw).map_err(|err| format!("parse {source}: {err}"))?;
    if schema.schema_version != SCHEMA_VERSION_V1 {
        return Err(format!(
            "unsupported schema_version {} in {source}",
            schema.schema_version
        ));
    }
    Ok(schema)
}

/// In-tree connector schemas auto-registered without env.
fn builtin_schemas() -> Vec<MetricsSchemaV1> {
    const BUILTINS: &[(&str, &str)] = &[
        (
            "offloading_connector_metrics_v1.json",
            include_str!("builtins/offloading_connector_metrics_v1.json"),
        ),
        (
            "hf3fs_connector_metrics_v1.json",
            include_str!("builtins/hf3fs_connector_metrics_v1.json"),
        ),
        (
            "hisparse_connector_metrics_v1.json",
            include_str!("builtins/hisparse_connector_metrics_v1.json"),
        ),
    ];
    let mut out = Vec::with_capacity(BUILTINS.len());
    for (name, raw) in BUILTINS {
        match parse_schema_json(raw, name) {
            Ok(schema) => out.push(schema),
            Err(err) => warn!(schema = *name, error = %err, "invalid builtin KV connector schema"),
        }
    }
    out
}

fn rmpv_to_schema(value: &Value) -> Result<MetricsSchemaV1, String> {
    // Round-trip via JSON so schema parsing stays serde_json-based.
    let json = serde_json::to_value(value).map_err(|err| err.to_string())?;
    serde_json::from_value(json).map_err(|err| err.to_string())
}

fn map_remove(payload: &mut Value, key: &str) -> Option<Value> {
    let Value::Map(entries) = payload else {
        return None;
    };
    let idx = entries.iter().position(|(k, _)| match k {
        Value::String(s) => s.as_str().is_some_and(|s| s == key),
        _ => false,
    })?;
    Some(entries.remove(idx).1)
}

fn observe_metric(bound: &BoundMetric, model_name: &str, engine: u32, payload: &Value) {
    let const_labels: Vec<(String, String)> =
        bound.def.const_labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let labels = connector_metric_labels(model_name, engine, &const_labels);
    let Some(sample) = value_at_path(payload, &bound.def.samples_path) else {
        return;
    };
    // Offloading (and similar) nest values under label-tuple map keys;
    // msgspec encodes `()` as an empty MessagePack array.
    let sample = unwrap_label_tuple_map(sample);
    let scale = bound.def.scale;

    match (&bound.family, &bound.def.sample_kind) {
        (DynamicFamily::Counter(family), SampleKind::IncByU64) => {
            if let Some(v) = as_u64(sample)
                && v != 0
            {
                family.get_or_create(&labels).inc_by(v);
            }
        }
        (DynamicFamily::Counter(family), SampleKind::IncBySumU64) => {
            let sum = sum_u64(sample);
            if sum != 0 {
                family.get_or_create(&labels).inc_by(sum);
            }
        }
        (DynamicFamily::CounterF64(family), SampleKind::IncByF64) => {
            if let Some(v) = as_f64(sample)
                && v != 0.0
            {
                family.get_or_create(&labels).inc_by(v);
            }
        }
        (DynamicFamily::GaugeU64(family), SampleKind::SetU64) => {
            if let Some(v) = as_u64(sample) {
                family.get_or_create(&labels).set(v);
            }
        }
        (DynamicFamily::GaugeF64(family), SampleKind::SetF64) => {
            if let Some(v) = as_f64(sample) {
                family.get_or_create(&labels).set(v);
            }
        }
        (DynamicFamily::Histogram(family), SampleKind::ObserveEachF64) => {
            for v in as_f64_list(sample) {
                family.get_or_create(&labels).observe(v * scale);
            }
        }
        (DynamicFamily::Histogram(family), SampleKind::ObserveEachU64AsF64) => {
            for v in as_u64_list(sample) {
                family.get_or_create(&labels).observe(v as f64 * scale);
            }
        }
        _ => {}
    }
}

/// Unwrap Offloading-style `{label_tuple: value}` maps to the unlabeled value.
///
/// Prefer the empty-tuple key (`Array([])`). If absent, use the sole entry or
/// leave the map unchanged (callers then no-op on type mismatch).
fn unwrap_label_tuple_map(sample: &Value) -> &Value {
    let Value::Map(entries) = sample else {
        return sample;
    };
    if entries.is_empty() {
        return sample;
    }
    let all_array_keys = entries.iter().all(|(k, _)| matches!(k, Value::Array(_)));
    if !all_array_keys {
        return sample;
    }
    if let Some((_, v)) = entries.iter().find(|(k, _)| matches!(k, Value::Array(a) if a.is_empty()))
    {
        return v;
    }
    if entries.len() == 1 {
        return &entries[0].1;
    }
    sample
}

fn value_at_path<'a>(payload: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let mut cur = payload;
    for part in path.split('.') {
        cur = map_get(cur, part)?;
    }
    Some(cur)
}

fn map_get<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
    match payload {
        Value::Map(entries) => entries.iter().find_map(|(k, v)| match k {
            Value::String(s) if s.as_str().is_some_and(|s| s == key) => Some(v),
            _ => None,
        }),
        _ => None,
    }
}

fn as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Integer(i) => i.as_u64().or_else(|| i.as_i64().and_then(|v| u64::try_from(v).ok())),
        Value::F64(v) => Some(*v as u64),
        Value::F32(v) => Some(*v as u64),
        _ => None,
    }
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::F64(v) => Some(*v),
        Value::F32(v) => Some(f64::from(*v)),
        Value::Integer(i) => i.as_i64().map(|v| v as f64).or_else(|| i.as_u64().map(|v| v as f64)),
        _ => None,
    }
}

fn sum_u64(value: &Value) -> u64 {
    match value {
        Value::Array(items) => items.iter().filter_map(as_u64).sum(),
        other => as_u64(other).unwrap_or(0),
    }
}

fn as_f64_list(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().filter_map(as_f64).collect(),
        other => as_f64(other).into_iter().collect(),
    }
}

fn as_u64_list(value: &Value) -> Vec<u64> {
    match value {
        Value::Array(items) => items.iter().filter_map(as_u64).collect(),
        other => as_u64(other).into_iter().collect(),
    }
}
