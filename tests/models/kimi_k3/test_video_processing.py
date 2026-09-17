# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Kimi-K3 additive video chunk preprocessing."""

from PIL import Image

from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3MultiModalProcessor,
    make_video_chunk_prompt,
    timestamp_as_str,
)
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)


class _FakeInfo:
    def __init__(self, image_processor):
        self.image_processor = image_processor


def _make_processor(media_proc_cfg: dict) -> KimiK3MultiModalProcessor:
    vision = KimiK25FusedVisionProcessor(media_proc_cfg=media_proc_cfg)
    proc = object.__new__(KimiK3MultiModalProcessor)
    proc.info = _FakeInfo(vision)
    return proc


def test_timestamp_and_chunk_prompt_format():
    assert timestamp_as_str(3661.5, "hh:mm:ss.fff") == "01:01:01.500"
    prompt = make_video_chunk_prompt("00:00:01.000")
    assert prompt.startswith("00:00:01.000<|media_begin|>video")
    assert "<|media_pad|>" in prompt


def test_split_video_chunks_four_frame_packs():
    frames = [Image.new("RGB", (28, 28), color=(i, i, i)) for i in range(8)]
    proc = _make_processor(
        {
            "patch_size": 14,
            "merge_kernel_size": 2,
            "in_patch_limit": 16,
            "in_patch_limit_each_frame": 16,
            "in_patch_limit_video": 80,
            "patch_limit_on_one_side": 512,
            "fixed_output_tokens": None,
            "temporal_merge_kernel_size": 4,
            "timestamp_mode": "hh:mm:ss.fff",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )
    chunks = proc.split_video_chunks(frames)
    assert [len(c["video_chunk"]) for c in chunks] == [4, 4]
    assert all(c["type"] == "video_chunk" for c in chunks)
    assert all("<|media_begin|>video" in c["prompt"] for c in chunks)
    # Global video budget 80 / 8 frames => per-frame limit 10
    assert all(c["in_patch_limit"] == 10 for c in chunks)


def test_video_chunk_preserves_temporal_grid_dimension():
    frames = [Image.new("RGB", (28, 28), color=0) for _ in range(4)]
    vision = KimiK25FusedVisionProcessor(
        media_proc_cfg={
            "patch_size": 14,
            "merge_kernel_size": 2,
            "in_patch_limit": 16,
            "in_patch_limit_each_frame": 16,
            "in_patch_limit_video": 80,
            "patch_limit_on_one_side": 512,
            "fixed_output_tokens": None,
            "temporal_merge_kernel_size": 4,
            "timestamp_mode": "hh:mm:ss.fff",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )
    out = vision.preprocess(
        [
            {
                "type": "video_chunk",
                "video_chunk": frames,
                "in_patch_limit": 16,
            }
        ],
        return_tensors="pt",
    )
    assert out["grid_thws"].shape == (1, 3)
    assert int(out["grid_thws"][0, 0]) == 4
    assert out["pixel_values"].shape[0] == int(out["grid_thws"].prod(-1)[0])
