# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Rust-frontend-gated ``_metrics_schema`` emission."""

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    HiSparseKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema import (
    METRICS_SCHEMA_KEY,
    reset_metrics_schema_emission_for_tests,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)


def test_metrics_schema_not_emitted_for_python_frontend(monkeypatch):
    reset_metrics_schema_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema.envs."
        "VLLM_USE_RUST_FRONTEND",
        False,
    )
    stats = OffloadingConnectorStats()
    stats.increase_counter("vllm:kv_offload_store_bytes", 1)
    payload = stats.to_dict()
    assert METRICS_SCHEMA_KEY not in payload
    assert "data" in payload


def test_metrics_schema_emitted_once_for_rust_frontend(monkeypatch):
    reset_metrics_schema_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = OffloadingConnectorStats()
    stats.increase_counter("vllm:kv_offload_store_bytes", 1)
    first = stats.to_dict()
    assert METRICS_SCHEMA_KEY in first
    schema = first[METRICS_SCHEMA_KEY]
    assert schema["connector_id"] == "OffloadingConnector"
    assert schema["schema_version"] == 1
    assert isinstance(schema["metrics"], list)
    assert len(schema["metrics"]) > 0

    second = stats.to_dict()
    assert METRICS_SCHEMA_KEY not in second
    assert second["data"]["vllm:kv_offload_store_bytes"][()] == 1


def test_hisparse_zero_snapshot_emits_metrics_schema_once(monkeypatch):
    """Empty/zero HiSparse snapshot still carries schema under Rust frontend."""
    reset_metrics_schema_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(0, 0, 0)
    first = stats.to_dict()
    assert METRICS_SCHEMA_KEY in first
    assert first[METRICS_SCHEMA_KEY]["connector_id"] == "HiSparseConnector"
    assert first["cache_hits"] == [0]

    second = stats.to_dict()
    assert METRICS_SCHEMA_KEY not in second
    assert second["cache_hits"] == [0]
