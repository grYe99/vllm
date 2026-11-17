# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup measurement of the encoder output width EC producers size by."""

import pytest

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector import encoder_output_width as eow

pytestmark = pytest.mark.cpu_test


def test_producer_records_the_widths_its_encoder_ranks_agree_on():
    config = create_ec_vllm_config(ec_role="ec_producer", encoder_output_widths={})
    reports = [{"image": (64, 2)}, {}, {"image": (64, 2)}]
    eow.measure_encoder_output_widths(config, lambda method: reports)
    assert config.ec_transfer_config.mm_encoder_output_widths == {"image": 64}
    assert eow.get_encoder_output_width(config, "image") == 64


@pytest.mark.parametrize(
    "reports, match",
    [
        ([{}, {}], "could not measure"),
        ([{"image": (64, 2)}, {"image": (16, 2)}], "disagree"),
    ],
    ids=["no-encoder", "disagree"],
)
def test_producer_refuses_to_start_without_a_trustworthy_width(reports, match):
    config = create_ec_vllm_config(ec_role="ec_producer", encoder_output_widths={})
    with pytest.raises(ValueError, match=match):
        eow.measure_encoder_output_widths(config, lambda fn: reports)


def test_byte_budget_uses_the_largest_embedding_and_ignores_empty_ranks():
    config = create_ec_vllm_config(ec_role="ec_consumer")
    config.scheduler_config.encoder_cache_bytes = 2 << 30
    reports = [
        {"image": (8192, 2), "audio": (2048, 2)},
        {},
        {"image": (8192, 2), "audio": (2048, 2)},
    ]
    assert eow.measure_encoder_output_widths(config, lambda fn: reports) == 16384


def test_byte_budget_without_a_local_encoder_is_rejected():
    config = create_ec_vllm_config(ec_role="ec_consumer")
    config.scheduler_config.encoder_cache_bytes = 1 << 30
    with pytest.raises(ValueError, match="local encoder tower"):
        eow.measure_encoder_output_widths(config, lambda fn: [{}, {}])
