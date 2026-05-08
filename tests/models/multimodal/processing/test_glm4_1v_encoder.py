# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from transformers import Glm4vForConditionalGeneration as HFGlm4v

from vllm import LLM


@pytest.fixture(scope="module")
def vllm_visual():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("Loading vLLM model ...")
    llm = LLM(
        model="cyankiwi/GLM-4.6V-Flash-AWQ-4bit",
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
    )
    engine_core = llm.llm_engine.engine_core.engine_core
    model_runner = engine_core.model_executor.driver_worker.worker.model_runner
    vllm_model = model_runner.model
    # vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    return vllm_model.visual


@pytest.fixture(scope="module")
def hf_visual():
    print("Loading HF model ...")
    hf_model = HFGlm4v.from_pretrained(
        "cyankiwi/GLM-4.6V-Flash-AWQ-4bit",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    return hf_model.model.visual


@pytest.mark.parametrize(
    "grid_thw, total_patches",
    [
        # image
        ([[1, 26, 46]], 1196),
        ([[1, 40, 60]], 2400),
        # video
        ([[8, 26, 46]], 9568),
        ([[4, 30, 50]], 6000),
    ],
)
def test_vision_encoder_equivalence(
    vllm_visual,
    hf_visual,
    grid_thw,
    total_patches,
):
    device = next(vllm_visual.parameters()).device
    dtype = vllm_visual.dtype

    # temporal_patch_size=2, patch_size=14, C=3 => 3*2*14*14 = 1176
    pixel_values = torch.randn(total_patches, 1176, dtype=dtype, device=device)

    grid_thw_tensor = torch.tensor(grid_thw, device=device)

    with torch.no_grad():
        hf_out = hf_visual(pixel_values, grid_thw_tensor)
        vllm_out = vllm_visual(pixel_values, grid_thw=grid_thw_tensor)

    assert hf_out.shape == vllm_out.shape, (
        f"Shape mismatch: HF {hf_out.shape} vs vLLM {vllm_out.shape}"
    )

    max_diff = (hf_out.float() - vllm_out.float()).abs().max().item()
    assert max_diff < 1e-3, f"Outputs differ! Max absolute difference: {max_diff:.6e}"
    print(f"max diff = {max_diff:.2e}")
