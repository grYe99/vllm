# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-occurrence embedding demand classification for P-first EPD dispatch."""

from types import SimpleNamespace

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.embedding_demand import (
    EmbeddingDemandReason,
    classify_embedding_demand,
    is_embedding_demand_probe,
)


def feature(offset: int, length: int, mm_hash: str, modality: str = "image"):
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def classify(
    features,
    num_computed_tokens,
    *,
    cached=None,
    external=None,
    num_tokens=None,
):
    cached = cached or {}
    external = external or set()
    num_tokens = (
        num_tokens
        if num_tokens is not None
        else (
            features[-1].mm_position.offset + features[-1].mm_position.length
            if features
            else 0
        )
    )
    return classify_embedding_demand(
        "req",
        features,
        num_computed_tokens=num_computed_tokens,
        num_tokens=num_tokens,
        cached_num_embeds=cached.get,
        has_external_item=lambda mm_hash: mm_hash in external,
    )


def test_cold_request_needs_e_for_every_occurrence():
    report = classify([feature(0, 100, "a"), feature(100, 50, "b")], 0)
    assert [d.reason for d in report.demands] == [
        EmbeddingDemandReason.NEED_FROM_E,
        EmbeddingDemandReason.NEED_FROM_E,
    ]
    assert [o.index for o in report.need_from_e] == [0, 1]


def test_prefix_kv_covered_occurrences_do_not_need_e():
    # Turn-2 history image inside the reused prefix; the new image is cold.
    report = classify(
        [feature(0, 100, "hist"), feature(150, 50, "new")],
        num_computed_tokens=140,
        num_tokens=200,
    )
    assert report.demands[0].reason is EmbeddingDemandReason.KV_COVERED
    assert report.demands[1].reason is EmbeddingDemandReason.NEED_FROM_E
    assert [o.index for o in report.need_from_e] == [1]


def test_consumer_local_and_external_cache_are_split_from_kv():
    report = classify(
        [feature(0, 100, "cached"), feature(100, 100, "external")],
        0,
        cached={"cached": 100},
        external={"external"},
    )
    assert report.demands[0].reason is EmbeddingDemandReason.P_LOCAL_EMBEDDING
    assert report.demands[1].reason is EmbeddingDemandReason.P_EXTERNAL_EMBEDDING


def test_embed_count_mismatch_demotes_local_hit_to_need_from_e():
    # A reused uuid pointing at different content must not silently reuse the
    # wrong embedding; the scheduler rejects it, so the probe must too.
    report = classify([feature(0, 100, "reused")], 0, cached={"reused": 3})
    assert report.demands[0].reason is EmbeddingDemandReason.NEED_FROM_E


def test_encoder_decoder_classifies_all_occurrences():
    report = classify_embedding_demand(
        "req",
        [feature(0, 100, "a")],
        num_computed_tokens=100,
        num_tokens=100,
        is_encoder_decoder=True,
        cached_num_embeds=lambda mm_hash: None,
        has_external_item=lambda mm_hash: False,
    )
    # Encoder-decoder inputs sit at start_pos=0 and stay schedulable even
    # after num_computed_tokens advances.
    assert report.demands[0].reason is EmbeddingDemandReason.NEED_FROM_E


def test_report_dict_roundtrip_keeps_occurrence_order():
    # "hist" ends at 50 and num_computed_tokens covers it fully; "new" is cold.
    report = classify(
        [feature(0, 50, "hist"), feature(100, 100, "new")],
        num_computed_tokens=50,
        num_tokens=200,
    )
    payload = report.to_dict()
    assert payload["request_id"] == "req"
    assert [d["index"] for d in payload["demands"]] == [0, 1]
    assert payload["demands"][0]["reason"] == "kv_covered"
    assert payload["demands"][1]["reason"] == "need_from_e"


def test_probe_flag_reads_extra_args_only():
    from vllm.sampling_params import SamplingParams

    plain = SamplingParams(max_tokens=1)
    probe = SamplingParams(max_tokens=1, extra_args={"embedding_demand_probe": True})
    assert not is_embedding_demand_probe(plain)
    assert is_embedding_demand_probe(probe)
    assert not is_embedding_demand_probe(None)


def test_probe_report_on_scheduler_kv_hit_skips_encoder():
    """A probe request finishes with a report and never reaches the model."""
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        PlaceholderRange,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.is_encoder_decoder = False
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    features = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="hist",
            mm_position=PlaceholderRange(offset=0, length=4),
        )
    ]
    request = SimpleNamespace(
        request_id="r1",
        client_index=0,
        has_encoder_inputs=True,
        mm_features=features,
        num_tokens=8,
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"embedding_demand_probe": True}
        ),
    )
    report = scheduler.embedding_demand_report(request, 0)
    assert [d.reason for d in report.demands] == [EmbeddingDemandReason.NEED_FROM_E]
    assert is_embedding_demand_probe(request.sampling_params)
