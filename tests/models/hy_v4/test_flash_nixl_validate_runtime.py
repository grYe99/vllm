# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.kv_transfer import KVTransferConfig
from vllm.models.hy_v4_flash.nvidia.model import validate_runtime


def _minimal_vllm_config(*, kv_transfer_config=None, lora_config=None):
    cfg = MagicMock()
    text_config = cfg.model_config.hf_text_config
    text_config.validate_flash_config = MagicMock()
    text_config.verify_parallel_config = MagicMock()
    text_config.index_topk = 128
    text_config.sliding_window = 128
    text_config.engram_layer_ids = []
    text_config.enable_ihc = True
    cfg.model_config.dtype = torch.bfloat16
    cfg.model_config.quantization = None
    cfg.kv_transfer_config = kv_transfer_config
    cfg.lora_config = lora_config
    cfg.speculative_config = None
    cfg.cache_config.cache_dtype = "auto"
    cfg.parallel_config = MagicMock()
    return cfg


@patch(
    "vllm.models.hy_v4_flash.nvidia.model.current_platform.is_cuda", return_value=True
)
def test_validate_runtime_without_kv_transfer(_mock_cuda):
    validate_runtime(_minimal_vllm_config())


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    ("connector", "extra_config"),
    [
        ("NixlConnector", {}),
        ("NixlConnector", {"transfer_mode": "pull"}),
        ("NixlPullConnector", {}),
    ],
)
@patch(
    "vllm.models.hy_v4_flash.nvidia.model.current_platform.is_cuda", return_value=True
)
def test_validate_runtime_allows_nixl_pull(_mock_cuda, connector, extra_config, kv_role):
    kv_transfer = KVTransferConfig(
        kv_connector=connector,
        kv_connector_extra_config=extra_config,
        kv_role=kv_role,
    )
    validate_runtime(_minimal_vllm_config(kv_transfer_config=kv_transfer))


@pytest.mark.parametrize(
    ("connector", "extra_config", "module_path"),
    [
        ("NixlConnector", {"transfer_mode": "push"}, None),
        ("NixlConnector", {"transfer_mode": "invalid"}, None),
        ("NixlPushConnector", {}, None),
        ("MooncakeConnector", {}, None),
        ("DecodeBenchConnector", {}, None),
        ("MultiConnector", {}, None),
        ("NixlConnector", {}, "external_connector"),
        ("NixlPullConnector", {}, "external_connector"),
    ],
)
@patch(
    "vllm.models.hy_v4_flash.nvidia.model.current_platform.is_cuda", return_value=True
)
def test_validate_runtime_rejects_other_kv_connectors(
    _mock_cuda, connector, extra_config, module_path
):
    kv_transfer = KVTransferConfig(
        kv_connector=connector,
        kv_connector_extra_config=extra_config,
        kv_connector_module_path=module_path,
        kv_role="kv_producer",
    )
    with pytest.raises(ValueError, match="supports only NIXL pull KV transfer"):
        validate_runtime(_minimal_vllm_config(kv_transfer_config=kv_transfer))


@patch(
    "vllm.models.hy_v4_flash.nvidia.model.current_platform.is_cuda", return_value=True
)
def test_validate_runtime_still_rejects_lora(_mock_cuda):
    cfg = _minimal_vllm_config(lora_config=MagicMock())
    with pytest.raises(ValueError, match="does not support LoRA"):
        validate_runtime(cfg)
