// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use rmpv::Value;

use super::adapter::{SchemaDrivenAdapter, connector_id_from_payload_schema};
use super::schema::METRICS_SCHEMA_KEY;

/// Builtin connector class names claimed by typed adapters (not generic).
const BUILTIN_CONNECTOR_IDS: &[&str] = &[
    "NixlConnector",
    "NixlPullConnector",
    "NixlPushConnector",
    "MooncakeStoreConnector",
];

/// Observe opaque third-party connector stats (`Multi.other` / `Other`).
pub(crate) fn observe_opaque_connector_stats(
    generic: &SchemaDrivenAdapter,
    model_name: &str,
    engine: u32,
    map: &BTreeMap<String, Value>,
) {
    if map.is_empty() {
        return;
    }

    let child_keys: Vec<&String> =
        map.keys().filter(|k| looks_like_connector_class_name(k)).collect();

    if !child_keys.is_empty() {
        for key in child_keys {
            if is_builtin_connector_id(key) {
                continue;
            }
            if let Some(value) = map.get(key) {
                generic.observe(key, model_name, engine, value);
            }
        }
        return;
    }

    // Flat single-connector payload. Prefer ``_metrics_schema.connector_id``,
    // else the sole already-registered schema (data-only ticks after one-shot).
    let payload = Value::Map(
        map.iter().map(|(k, v)| (Value::String(k.as_str().into()), v.clone())).collect(),
    );
    match resolve_flat_connector_id(generic, map) {
        Some(connector_id) => generic.observe(&connector_id, model_name, engine, &payload),
        None => generic.observe("<unknown>", model_name, engine, &payload),
    }
}

fn resolve_flat_connector_id(
    generic: &SchemaDrivenAdapter,
    map: &BTreeMap<String, Value>,
) -> Option<String> {
    // 1) Payload carries schema → use its connector_id.
    if map.contains_key(METRICS_SCHEMA_KEY) {
        return connector_id_from_payload_schema(map);
    }
    // 2) Data-only after one-shot: exactly one schema already registered.
    let ids = generic.registered_ids();
    if ids.len() == 1 {
        return ids.into_iter().next();
    }
    None
}

fn is_builtin_connector_id(id: &str) -> bool {
    BUILTIN_CONNECTOR_IDS.iter().any(|b| *b == id)
}

/// Heuristic: Multi child keys are Python ``__class__.__name__`` strings.
fn looks_like_connector_class_name(key: &str) -> bool {
    let Some(first) = key.chars().next() else {
        return false;
    };
    if !first.is_uppercase() {
        return false;
    }
    key.contains("Connector") || key.ends_with("Store")
}
