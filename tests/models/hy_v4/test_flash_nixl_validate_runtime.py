# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

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
def test_validate_runtime_allows_nixl_kv_transfer(_mock_cuda):
    cfg = _minimal_vllm_config(kv_transfer_config=MagicMock())
    validate_runtime(cfg)


@patch(
    "vllm.models.hy_v4_flash.nvidia.model.current_platform.is_cuda", return_value=True
)
def test_validate_runtime_still_rejects_lora(_mock_cuda):
    cfg = _minimal_vllm_config(lora_config=MagicMock())
    with pytest.raises(ValueError, match="does not support LoRA"):
        validate_runtime(cfg)
