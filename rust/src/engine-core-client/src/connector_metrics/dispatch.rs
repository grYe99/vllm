// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use rmpv::Value;

use super::adapter::SchemaDrivenAdapter;

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

    // Flat single-connector payload (e.g. Redhare `to_dict()` = stats data,
    // or Offloading `{types,data}`).
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
    // 1) Exactly one schema loaded from env → bind flat maps to that id.
    if let Some(id) = generic.sole_env_connector_id() {
        return Some(id.to_string());
    }
    let ids = generic.registered_ids();
    // 2) Only one schema registered overall (e.g. tests without builtins).
    if ids.len() == 1 {
        return ids.into_iter().next();
    }
    // 3) Distinctive in-tree payload shapes against registered schemas.
    if let Some(id) = infer_connector_id_from_payload(map) {
        if ids.iter().any(|registered| registered == &id) {
            return Some(id);
        }
    }
    // 4) Ambiguous → caller warns via observe("<unknown>").
    None
}

fn infer_connector_id_from_payload(map: &BTreeMap<String, Value>) -> Option<String> {
    if map.contains_key("types") && map.contains_key("data") {
        return Some("OffloadingConnector".to_string());
    }
    if map.contains_key("save_duration")
        || map.contains_key("load_duration")
        || map.contains_key("num_failed_save")
        || map.contains_key("num_failed_load")
    {
        return Some("HF3FSKVConnector".to_string());
    }
    if map.contains_key("cache_hits")
        || map.contains_key("cache_misses")
        || map.contains_key("host_to_device_bytes")
    {
        return Some("HiSparseConnector".to_string());
    }
    None
}

fn is_builtin_connector_id(id: &str) -> bool {
    BUILTIN_CONNECTOR_IDS.iter().any(|b| *b == id)
}

/// Heuristic: Multi child keys are Python `__class__.__name__` strings.
fn looks_like_connector_class_name(key: &str) -> bool {
    let Some(first) = key.chars().next() else {
        return false;
    };
    if !first.is_uppercase() {
        return false;
    }
    key.contains("Connector") || key.ends_with("Store")
}
