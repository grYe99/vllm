// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Schema-driven KV connector metrics for third-party and in-tree connectors.
//!
//! First-party Nixl / Mooncake keep typed DTOs and observe paths. Other
//! payloads (`Multi.other` / `Other`) are recorded here when a metrics schema
//! is available:
//! - **builtins** (Offloading / HF3FS / HiSparse) auto-registered at init;
//! - env `VLLM_KV_CONNECTOR_METRICS_SCHEMA` (escape hatch / override);
//! - optional `_metrics_schema` in the payload;
//! - later: handshake channel.

mod adapter;
mod dispatch;
pub(crate) mod schema;

pub(crate) use adapter::SchemaDrivenAdapter;
pub(crate) use dispatch::observe_opaque_connector_stats;
