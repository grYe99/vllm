# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content identities and output contracts for the shared Encoder Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EMBEDDING_CACHE_KEY_VERSION = "v3"


def make_embedding_key(namespace: str, identifier: str) -> str:
    return f"{namespace}@id:{identifier}"


@dataclass(frozen=True)
class TensorSpec:
    """Contiguous output contract used by the Encoder and Store codec."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def build_embedding_namespace(vllm_config: VllmConfig) -> str:
    model_config = vllm_config.model_config
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None
    extra_config = ec_config.ec_connector_extra_config

    multimodal_config = model_config.multimodal_config
    encoder_hash = (
        multimodal_config.compute_hash()
        if multimodal_config is not None
        else "encoder:default"
    )
    cache_prefix = str(extra_config.get("embedding_cache_prefix", ""))
    model_name = str(extra_config.get("embedding_model_identity", model_config.model))
    prefix = f"{cache_prefix}@" if cache_prefix else ""
    return (
        f"{prefix}embedding"
        f"@model:{model_name}"
        f"@revision:{model_config.revision or 'default'}"
        f"@encoder:{encoder_hash}"
        f"@dtype:{model_config.dtype}"
        f"@protocol:{EMBEDDING_CACHE_KEY_VERSION}"
    )


STORE_DELIVERY_CAPABILITY = "store_delivery_v1"


def parse_embedding_namespace(namespace: str) -> dict[str, str] | None:
    """Split a ``protocol:v3`` embedding namespace into its published fields."""
    marker = "embedding@model:"
    if marker in namespace:
        prefix, rest = namespace.split(marker, 1)
        if prefix.endswith("@"):
            prefix = prefix[:-1]
        rest = marker + rest
    else:
        return None
    fields: dict[str, str] = {"prefix": prefix}
    # rest = embedding@model:X@revision:Y@encoder:Z@dtype:W@protocol:V
    expected = ("model", "revision", "encoder", "dtype", "protocol")
    cursor = rest[len("embedding@") :]
    parts = cursor.split("@")
    if len(parts) != len(expected):
        return None
    for part, name in zip(parts, expected, strict=True):
        key, sep, value = part.partition(":")
        if sep != ":" or key != name or not value:
            return None
        fields[name] = value
    return fields


def store_namespace_is_approved(received: str, approved: str) -> bool:
    """Accept E's namespace if model/revision/prefix/protocol match this deployment.

    Encoder multimodal hashes may differ between E-only and PD instances, so
    the received encoder field is not compared. The key is used verbatim after
    this check; it is not rebuilt from the consumer config.
    """
    rec = parse_embedding_namespace(received)
    appr = parse_embedding_namespace(approved)
    if rec is None or appr is None:
        return False
    return (
        rec["prefix"] == appr["prefix"]
        and rec["model"] == appr["model"]
        and rec["revision"] == appr["revision"]
        and rec["protocol"] == appr["protocol"] == EMBEDDING_CACHE_KEY_VERSION
        and rec["dtype"] == appr["dtype"]
    )
