// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Schema-driven KV connector metrics for third-party and in-tree connectors.
//!
//! First-party Nixl / Mooncake keep typed DTOs and observe paths. Other
//! payloads (`Multi.other` / `Other`) are recorded here when a metrics schema
//! is available via the reserved stats key ``_metrics_schema`` (connectors
//! emit it once under ``VLLM_USE_RUST_FRONTEND``). Later: handshake channel.

mod adapter;
mod dispatch;
pub(crate) mod schema;

pub(crate) use adapter::SchemaDrivenAdapter;
pub(crate) use dispatch::observe_opaque_connector_stats;
