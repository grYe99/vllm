# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-occurrence encoder demand for disaggregated EPD orchestration.

The engine classifies each multimodal occurrence by how the consumer (P/PD)
can satisfy its embedding, so an orchestrator only sends encoder work that is
still missing (see vllm#55639): an occurrence already covered by reusable
prefix KV needs no embedding at all, and one cached on the consumer needs no
encoder forward. The report is advisory: the scheduler re-validates on
admission, so a stale report degrades to redundant work, never a wrong one.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import get_mm_features_in_window

# extra_args key that marks a request as a demand probe. A probe is answered
# with a report on the response's ec_transfer_params and never scheduled.
EMBEDDING_DEMAND_PROBE_KEY = "embedding_demand_probe"


class EmbeddingDemandReason(str, Enum):
    KV_COVERED = "kv_covered"
    P_LOCAL_EMBEDDING = "p_local_embedding"
    P_EXTERNAL_EMBEDDING = "p_external_embedding"
    NEED_FROM_E = "need_from_e"


@dataclass(frozen=True)
class MmOccurrenceRef:
    """One multimodal occurrence, addressable by its request-local index."""

    index: int
    mm_hash: str
    modality: str
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class EmbeddingDemand:
    occurrence: MmOccurrenceRef
    reason: EmbeddingDemandReason


@dataclass(frozen=True)
class PrefillEmbeddingDemandReport:
    request_id: str
    demands: tuple[EmbeddingDemand, ...]

    @property
    def need_from_e(self) -> list[MmOccurrenceRef]:
        return [
            d.occurrence
            for d in self.demands
            if d.reason is EmbeddingDemandReason.NEED_FROM_E
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "demands": [
                {
                    "index": d.occurrence.index,
                    "mm_hash": d.occurrence.mm_hash,
                    "modality": d.occurrence.modality,
                    "start_pos": d.occurrence.start_pos,
                    "end_pos": d.occurrence.end_pos,
                    "reason": d.reason.value,
                }
                for d in self.demands
            ],
        }


def classify_embedding_demand(
    request_id: str,
    mm_features: list[MultiModalFeatureSpec],
    *,
    num_computed_tokens: int,
    num_tokens: int,
    is_encoder_decoder: bool = False,
    cached_num_embeds: Callable[[str], int | None],
    has_external_item: Callable[[str], bool],
) -> PrefillEmbeddingDemandReport:
    """Classify every occurrence the way `_try_schedule_encoder_inputs` would.

    Occurrences fully inside `num_computed_tokens` are never scheduled (their
    KV is reused); the rest are checked against the consumer's local encoder
    cache and the EC connector before falling back to needing E. An embed-count
    mismatch demotes a local hit to NEED_FROM_E, mirroring the scheduler's
    rejection of a reused identifier with different content.
    """
    demands = []
    lo, hi = get_mm_features_in_window(mm_features, num_computed_tokens, num_tokens)
    if is_encoder_decoder:
        lo = 0
    for i, feature in enumerate(mm_features):
        mm_hash = feature.identifier
        position = feature.mm_position
        if i < lo or i >= hi:
            reason = EmbeddingDemandReason.KV_COVERED
        elif (
            cached := cached_num_embeds(mm_hash)
        ) is not None and cached == position.get_num_embeds():
            reason = EmbeddingDemandReason.P_LOCAL_EMBEDDING
        elif has_external_item(mm_hash):
            reason = EmbeddingDemandReason.P_EXTERNAL_EMBEDDING
        else:
            reason = EmbeddingDemandReason.NEED_FROM_E
        demands.append(
            EmbeddingDemand(
                occurrence=MmOccurrenceRef(
                    index=i,
                    mm_hash=mm_hash,
                    modality=feature.modality,
                    start_pos=position.offset,
                    end_pos=position.offset + position.length,
                ),
                reason=reason,
            )
        )
    return PrefillEmbeddingDemandReport(request_id=request_id, demands=tuple(demands))


def is_embedding_demand_probe(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None)
    return bool(extra_args and extra_args.get(EMBEDDING_DEMAND_PROBE_KEY))


def _no_external_item(mm_hash: str) -> bool:
    return False
