# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Rust-frontend-gated ``_metrics_schema`` emission."""

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    _HF3FS_METRIC_DEFS,
    HF3FSKVConnectorStats,
    build_hf3fs_metrics_schema,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    _HISPARSE_COUNTERS,
    HiSparseKVConnectorStats,
    build_hisparse_metrics_schema,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema import (
    INC_BY_F64,
    INC_BY_SUM_U64,
    INC_BY_U64,
    METRICS_SCHEMA_KEY,
    OBSERVE_EACH_F64,
    SET_F64,
    reset_metrics_schema_emission_for_tests,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _FLOAT_COUNTER_NAMES,
    OffloadingConnectorStats,
    build_offloading_metrics_schema,
    get_connector_metric_definitions,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


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


def test_offloading_metrics_schema_derived_from_metadata():
    """Schema names/kinds match OffloadingMetricMetadata (no JSON file)."""
    schema = build_offloading_metrics_schema()
    assert schema["schema_version"] == 1
    assert schema["connector_id"] == "OffloadingConnector"

    by_name = {m["name"]: m for m in schema["metrics"]}
    expected = {
        **CPUOffloadingSpec.build_metric_definitions({"store_threshold": 2}),
        **get_connector_metric_definitions(),
    }
    assert set(by_name) == set(expected)

    for name, metadata in expected.items():
        entry = by_name[name]
        assert entry["samples_path"] == f"data.{name}"
        if isinstance(metadata, OffloadingHistogramMetadata):
            assert entry["type"] == "histogram"
            assert entry["sample_kind"] == OBSERVE_EACH_F64
            assert entry["buckets"] == list(metadata.buckets or ())
        elif isinstance(metadata, OffloadingGaugeMetadata):
            assert entry["type"] == "gauge"
            assert entry["sample_kind"] == SET_F64
        elif isinstance(metadata, OffloadingCounterMetadata):
            assert entry["type"] == "counter"
            want = INC_BY_F64 if name in _FLOAT_COUNTER_NAMES else INC_BY_U64
            assert entry["sample_kind"] == want
        else:
            raise AssertionError(f"unexpected metadata: {metadata}")


def test_hf3fs_metrics_schema_derived_from_metric_defs():
    """Schema matches ``_HF3FS_METRIC_DEFS`` used by Prom (no JSON file)."""
    schema = build_hf3fs_metrics_schema()
    assert schema["connector_id"] == "HF3FSKVConnector"
    assert len(schema["metrics"]) == len(_HF3FS_METRIC_DEFS)
    for entry, defn in zip(schema["metrics"], _HF3FS_METRIC_DEFS, strict=True):
        assert entry["name"] == defn.name
        assert entry["type"] == defn.metric_type
        assert entry["samples_path"] == defn.wire_key
        assert entry["sample_kind"] == defn.sample_kind
        if defn.buckets is not None:
            assert entry["buckets"] == list(defn.buckets)


def test_hisparse_metrics_schema_derived_from_counters():
    """Schema matches ``_HISPARSE_COUNTERS`` used by Prom (no JSON file)."""
    schema = build_hisparse_metrics_schema()
    assert schema["connector_id"] == "HiSparseConnector"
    assert len(schema["metrics"]) == len(_HISPARSE_COUNTERS)
    for entry, (wire_key, documentation) in zip(
        schema["metrics"], _HISPARSE_COUNTERS, strict=True
    ):
        assert entry["name"] == f"vllm:hisparse_{wire_key}"
        assert entry["type"] == "counter"
        assert entry["samples_path"] == wire_key
        assert entry["sample_kind"] == INC_BY_SUM_U64
        assert entry["documentation"] == documentation


def test_hf3fs_emits_derived_schema_once(monkeypatch):
    reset_metrics_schema_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_schema.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = HF3FSKVConnectorStats()
    stats.record_success_task_duration("Saved", 0.01)
    first = stats.to_dict()
    assert METRICS_SCHEMA_KEY in first
    assert first[METRICS_SCHEMA_KEY] == build_hf3fs_metrics_schema()
    second = stats.to_dict()
    assert METRICS_SCHEMA_KEY not in second
