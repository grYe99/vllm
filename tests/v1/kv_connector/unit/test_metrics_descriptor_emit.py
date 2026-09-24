# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Rust-frontend-gated ``_metrics_descriptor`` emission."""

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnectorStats,
    build_hf3fs_metrics_descriptor,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    _HISPARSE_COUNTERS,
    HiSparseKVConnectorStats,
    build_hisparse_metrics_descriptor,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor import (
    INC_BY_F64,
    INC_BY_SUM_U64,
    INC_BY_U64,
    METRICS_DESCRIPTOR_KEY,
    OBSERVE_EACH_F64,
    SET_F64,
    reset_metrics_descriptor_emission_for_tests,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _FLOAT_COUNTER_NAMES,
    OffloadingConnectorStats,
    build_offloading_metrics_descriptor,
    get_connector_metric_definitions,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def test_metrics_descriptor_not_emitted_for_python_frontend(monkeypatch):
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        False,
    )
    stats = OffloadingConnectorStats()
    stats.increase_counter("vllm:kv_offload_store_bytes", 1)
    payload = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY not in payload
    assert "data" in payload


def test_metrics_descriptor_emitted_once_for_rust_frontend(monkeypatch):
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = OffloadingConnectorStats()
    stats.increase_counter("vllm:kv_offload_store_bytes", 1)
    first = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY in first
    descriptor = first[METRICS_DESCRIPTOR_KEY]
    assert descriptor["connector_id"] == "OffloadingConnector"
    assert descriptor["descriptor_version"] == 1
    assert isinstance(descriptor["metrics"], list)
    assert len(descriptor["metrics"]) > 0

    second = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY not in second
    assert second["data"]["vllm:kv_offload_store_bytes"][()] == 1


def test_hisparse_zero_snapshot_emits_metrics_descriptor_once(monkeypatch):
    """Empty/zero HiSparse snapshot still carries descriptor under Rust frontend."""
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(0, 0, 0)
    first = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY in first
    assert first[METRICS_DESCRIPTOR_KEY]["connector_id"] == "HiSparseConnector"
    assert first["cache_hits"] == [0]

    second = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY not in second
    assert second["cache_hits"] == [0]


def test_offloading_metrics_descriptor_derived_from_metadata():
    """Descriptor names/kinds match OffloadingMetricMetadata (no JSON file)."""
    descriptor = build_offloading_metrics_descriptor()
    assert descriptor["descriptor_version"] == 1
    assert descriptor["connector_id"] == "OffloadingConnector"

    by_name = {m["name"]: m for m in descriptor["metrics"]}
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


def test_hf3fs_metrics_descriptor_matches_prom_names():
    """Descriptor names/kinds/buckets match HF3FSPromMetrics (no JSON / table)."""
    descriptor = build_hf3fs_metrics_descriptor()
    assert descriptor["connector_id"] == "HF3FSKVConnector"
    by_name = {m["name"]: m for m in descriptor["metrics"]}
    assert set(by_name) == {
        "vllm:hf3fs_save_duration_seconds",
        "vllm:hf3fs_load_duration_seconds",
        "vllm:hf3fs_num_failed_save",
        "vllm:hf3fs_num_failed_load",
    }
    for hist_name, wire_key in (
        ("vllm:hf3fs_save_duration_seconds", "save_duration"),
        ("vllm:hf3fs_load_duration_seconds", "load_duration"),
    ):
        entry = by_name[hist_name]
        assert entry["type"] == "histogram"
        assert entry["samples_path"] == wire_key
        assert entry["sample_kind"] == OBSERVE_EACH_F64
        assert entry["buckets"] == [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
    for counter_name, wire_key in (
        ("vllm:hf3fs_num_failed_save", "num_failed_save"),
        ("vllm:hf3fs_num_failed_load", "num_failed_load"),
    ):
        entry = by_name[counter_name]
        assert entry["type"] == "counter"
        assert entry["samples_path"] == wire_key
        assert entry["sample_kind"] == INC_BY_U64


def test_hisparse_metrics_descriptor_derived_from_counters():
    """Schema matches ``_HISPARSE_COUNTERS`` used by Prom (no JSON file)."""
    descriptor = build_hisparse_metrics_descriptor()
    assert descriptor["connector_id"] == "HiSparseConnector"
    assert len(descriptor["metrics"]) == len(_HISPARSE_COUNTERS)
    for entry, (wire_key, documentation) in zip(
        descriptor["metrics"], _HISPARSE_COUNTERS, strict=True
    ):
        assert entry["name"] == f"vllm:hisparse_{wire_key}"
        assert entry["type"] == "counter"
        assert entry["samples_path"] == wire_key
        assert entry["sample_kind"] == INC_BY_SUM_U64
        assert entry["documentation"] == documentation


def test_hf3fs_emits_derived_descriptor_once(monkeypatch):
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = HF3FSKVConnectorStats()
    stats.record_success_task_duration("Saved", 0.01)
    first = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY in first
    assert first[METRICS_DESCRIPTOR_KEY] == build_hf3fs_metrics_descriptor()
    second = stats.to_dict()
    assert METRICS_DESCRIPTOR_KEY not in second
