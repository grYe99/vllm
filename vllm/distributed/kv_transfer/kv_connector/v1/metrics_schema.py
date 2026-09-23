# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for emitting MetricsSchemaV1 on the KV connector stats wire.

Rust frontend cannot call ``build_prom_metrics()``. Connectors that need
schema-driven Prometheus series emit ``_metrics_schema`` once in their
``to_dict()`` payload when ``VLLM_USE_RUST_FRONTEND`` is enabled. Python
frontend scrapes leave the payload untouched.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

import vllm.envs as envs

METRICS_SCHEMA_KEY = "_metrics_schema"

# Process-local: first to_dict per connector_id carries the schema.
_emitted_connector_ids: set[str] = set()


@lru_cache(maxsize=16)
def load_metrics_schema(
    package: str, resource_name: str = "metrics_schema_v1.json"
) -> dict[str, Any]:
    """Load a MetricsSchemaV1 JSON document shipped beside a connector package."""
    raw = files(package).joinpath(resource_name).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(raw)
    return schema


def reset_metrics_schema_emission_for_tests() -> None:
    """Clear one-shot emission state (unit tests only)."""
    _emitted_connector_ids.clear()
    load_metrics_schema.cache_clear()


def strip_metrics_schema(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop ``_metrics_schema`` when reconstructing stats from a wire dict."""
    if data is None or METRICS_SCHEMA_KEY not in data:
        return data
    out = dict(data)
    out.pop(METRICS_SCHEMA_KEY)
    return out


def maybe_attach_metrics_schema(
    payload: dict[str, Any],
    *,
    connector_id: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Attach ``_metrics_schema`` once when the Rust frontend is enabled.

    Args:
        payload: Connector stats dict (must not already be shared mutably
            with callers that must not see the schema key).
        connector_id: Class name / schema ``connector_id`` for one-shot tracking.
        schema: MetricsSchemaV1 document.

    Returns:
        ``payload`` unchanged for Python frontend, or a shallow copy with
        ``_metrics_schema`` on the first Rust-frontend emit for this id.

    """
    if not envs.VLLM_USE_RUST_FRONTEND:
        return payload
    if connector_id in _emitted_connector_ids:
        return payload
    _emitted_connector_ids.add(connector_id)
    out = dict(payload)
    out[METRICS_SCHEMA_KEY] = schema
    return out
