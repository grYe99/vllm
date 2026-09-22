// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Schema-driven KV connector metrics for third-party connectors.
//!
//! First-party Nixl / Mooncake keep typed DTOs and observe paths. Unknown
//! connector payloads (`Multi.other` / `Other`) are recorded here when a
//! metrics schema is available (env file, optional `_metrics_schema` in the
//! payload, or a future handshake channel).

mod adapter;
mod dispatch;
pub(crate) mod schema;

pub(crate) use adapter::SchemaDrivenAdapter;
pub(crate) use dispatch::observe_opaque_connector_stats;
