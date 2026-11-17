# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder output widths for sizing EC transfers.

An encoder can emit features wider than the LM embedding (DeepStack packs
several layers on the last dim) by an amount only the model knows. Producers
size transfers before the first real item is encoded, so the engine measures
the width once at startup by encoding a dummy item per modality.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def measure_encoder_output_widths(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> int | None:
    """Measure encoder output layouts once, before the KV cache is allocated.

    EC producers still record per-modality widths for transfer sizing. When
    `--encoder-cache-bytes` is set, the same dummy encode also supplies the
    bytes per embedding used to resolve the token capacity. Ranks without an
    encoder contribute nothing. Producer widths must agree across measuring ranks;
    byte cost uses the maximum so one token budget covers every layout.

    Returns:
        Bytes per embedding when a byte budget is configured, otherwise None.

    Raises:
        ValueError: A required measurement is missing, widths disagree, or the
            output is not a 2D embedding tensor.

    """
    ec_config = vllm_config.ec_transfer_config
    need_widths = ec_config is not None and ec_config.is_ec_producer
    need_bytes = vllm_config.scheduler_config.encoder_cache_bytes is not None
    if not need_widths and not need_bytes:
        return None
    reports = [layouts for layouts in collective_rpc("measure_encoder_output_layouts") if layouts]
    if not reports:
        if need_widths:
            raise ValueError(
                "EC producer could not measure the encoder output width: the "
                "model encoded no dummy multimodal input."
            )
        raise ValueError(
            "--encoder-cache-bytes requires a local encoder tower so startup "
            "can measure the encoder output layout. This process encoded no "
            "dummy multimodal input. Instances that drop the encoder tower and "
            "have no local layout measurement are not supported."
        )
    width_maps = [
        {modality: width for modality, (width, _) in report.items()}
        for report in reports
    ]
    if need_widths and any(widths != width_maps[0] for widths in width_maps):
        raise ValueError(f"Workers disagree on the encoder output widths: {width_maps}")
    if need_widths:
        assert ec_config is not None
        logger.info("EC producer encoder output widths: %s", width_maps[0])
        ec_config.mm_encoder_output_widths = width_maps[0]
    if not need_bytes:
        return None
    bytes_per_embedding = max(
        width * element_size
        for report in reports
        for width, element_size in report.values()
    )
    if bytes_per_embedding <= 0:
        raise ValueError(
            "Encoder output measurement did not produce a usable embedding size."
        )
    logger.info(
        "Encoder embedding size for the cache budget: %d bytes.",
        bytes_per_embedding,
    )
    return bytes_per_embedding


def get_encoder_output_width(vllm_config: "VllmConfig", modality: str) -> int:
    """Encoder output width for `modality`, as measured at producer startup.

    Raises:
        ValueError: No width was measured for `modality`, so a transfer
            cannot be sized.

    """
    assert vllm_config.ec_transfer_config is not None
    widths = vllm_config.ec_transfer_config.mm_encoder_output_widths
    if modality not in widths:
        raise ValueError(
            f"No encoder output width was measured for modality {modality!r}; "
            f"measured: {sorted(widths)}"
        )
    return widths[modality]
