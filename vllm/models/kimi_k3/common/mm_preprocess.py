# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor
from vllm.utils.import_utils import is_numba_available

logger = init_logger(__name__)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def timestamp_as_str(timestamp: float, mode: str = "hh:mm:ss.fff") -> str:
    """Format a video timestamp for chunk prompts (HF discussion #172)."""
    if timestamp < 0:
        timestamp = 0.0
    total_ms = int(round(timestamp * 1000.0))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    if mode == "hh:mm:ss":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if mode == "mm:ss.fff":
        return f"{minutes + hours * 60:02d}:{seconds:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def make_video_chunk_prompt(timestamp_text: str) -> str:
    """Build one timestamped K3 video-chunk placeholder."""
    return (
        f"{timestamp_text}<|media_begin|>video<|media_content|>"
        f"<|media_pad|><|media_end|>"
    )


def frames_from_video_data(video_data: Any) -> tuple[list[Image.Image], list[float]]:
    """Normalize connector / ndarray / PIL video payloads into RGB frames."""
    meta: dict[str, Any] = {}
    if hasattr(video_data, "media"):
        video_data = video_data.media

    if isinstance(video_data, tuple) and len(video_data) >= 1:
        frames_data = video_data[0]
        if len(video_data) >= 2 and isinstance(video_data[1], dict):
            meta = video_data[1]
    else:
        frames_data = video_data

    if isinstance(frames_data, torch.Tensor):
        frames_data = frames_data.detach().cpu().numpy()

    frames: list[Image.Image] = []
    if isinstance(frames_data, np.ndarray):
        if frames_data.ndim == 3:
            frames_data = frames_data[None, ...]
        if frames_data.ndim != 4:
            raise ValueError(
                f"Expected video frames with shape (T,H,W,C), got {frames_data.shape}"
            )
        for frame in frames_data:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr).convert("RGB"))
    elif isinstance(frames_data, (list, tuple)):
        for frame in frames_data:
            if isinstance(frame, Image.Image):
                frames.append(frame.convert("RGB"))
            elif isinstance(frame, np.ndarray):
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(arr).convert("RGB"))
            else:
                raise ValueError(f"Unsupported video frame type: {type(frame)}")
    else:
        raise ValueError(f"Unsupported video data type: {type(frames_data)}")

    if not frames:
        raise ValueError("Video sampling did not return any frames.")

    timestamps: list[float] = []
    frame_indices = meta.get("frames_indices") or meta.get("frame_indices")
    fps = float(meta.get("fps") or meta.get("avg_fps") or 0.0)
    if frame_indices is not None and fps > 0:
        timestamps = [float(idx) / fps for idx in frame_indices]
    elif fps > 0:
        timestamps = [i / fps for i in range(len(frames))]
    else:
        timestamps = [float(i) for i in range(len(frames))]

    if len(timestamps) < len(frames):
        timestamps.extend(float(i) for i in range(len(timestamps), len(frames)))
    return frames, timestamps[: len(frames)]


class KimiK3VideoChunkItems(ProcessorBatchItems[dict[str, Any]]):
    """Processor items for already-split K3 video chunks under modality video."""

    def __init__(self, data: Sequence[dict[str, Any]]) -> None:
        super().__init__(data, "video")

    def get_processor_data(self) -> Mapping[str, object]:
        return {"videos": list(self.data)}


class KimiK3DataParser(MultiModalDataParser):
    """Like the default parser, but accepts pre-split video_chunk dicts."""

    def _parse_video_data(self, data) -> ProcessorBatchItems | None:
        if data is None:
            return None
        if self.is_embeddings(data):
            return super()._parse_video_data(data)
        items = data if isinstance(data, list) else [data]
        if items and isinstance(items[0], dict) and items[0].get("type") == "video_chunk":
            return KimiK3VideoChunkItems(items)
        return super()._parse_video_data(data)


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K3.

    Images keep the original ``image`` modality path. Video is an optional
    additive ``video`` modality that consumes pre-split MoonViT3d chunks.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor

        # Video uses the fused MoonViT preprocess when numba is available so we
        # do not depend on unmerged HF remote video code for the image path.
        self.video_processor: Any = image_processor
        if is_numba_available():
            try:
                self.video_processor = KimiK25FusedVisionProcessor(
                    media_proc_cfg=dict(image_processor.media_proc_cfg)
                )
            except Exception:
                logger.warning_once(
                    "Failed to build fused video processor for Kimi-K3; "
                    "video inputs will require a video-capable image processor."
                )
                self.video_processor = image_processor

        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=self.video_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_data_parser(self) -> MultiModalDataParser:
        return KimiK3DataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited. Image path unchanged; video is additive.
        return {"image": None, "video": None}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds dummy inputs for K3 profiling.

    Image dummy path is unchanged. When the profiler asks for video, emit
    already-split 4-frame chunks under the ``video`` key.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Match post-renderer video prompts: each item is already a chunk
        # string containing one <|media_pad|> for PromptReplacement to expand.
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        hf_config = self.info.get_hf_config()
        return (
            hf_config.image_placeholder * num_images
            + make_video_chunk_prompt("00:00:00.000") * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }

        num_videos = mm_counts.get("video", 0)
        if num_videos:
            num_frames = int(media_proc_cfg.get("temporal_merge_kernel_size", 4))
            per_frame_limit = int(
                media_proc_cfg.get("in_patch_limit_each_frame")
                or media_proc_cfg["in_patch_limit"]
            )
            chunk = {
                "type": "video_chunk",
                "video_chunk": self._get_dummy_images(
                    width=max_size.width,
                    height=max_size.height,
                    num_images=num_frames,
                ),
                "prompt": make_video_chunk_prompt("00:00:00.000"),
                "in_patch_limit": per_frame_limit,
            }
            data["video"] = [dict(chunk) for _ in range(num_videos)]
        return data


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Kimi-K3 processor: original image path + additive video chunk path."""

    def split_video_chunks(self, video_data: Any) -> list[dict[str, Any]]:
        """Split decoded video frames into timestamped MoonViT3d chunks.

        ``in_patch_limit_video`` is treated as a global patch budget shared by
        all sampled frames (HF discussion #172 / K2.5 contract).
        """
        frames, timestamps = frames_from_video_data(video_data)
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        num_frames_per_chunk = int(media_proc_cfg.get("temporal_merge_kernel_size", 4))
        per_frame_limit = int(
            media_proc_cfg.get("in_patch_limit_each_frame")
            or media_proc_cfg["in_patch_limit"]
        )
        total_patch_limit = media_proc_cfg.get("in_patch_limit_video")
        if total_patch_limit is not None:
            per_frame_limit = min(
                per_frame_limit,
                max(1, round(int(total_patch_limit) / len(frames))),
            )

        timestamp_mode = media_proc_cfg.get("timestamp_mode", "hh:mm:ss.fff")
        chunks: list[dict[str, Any]] = []
        for start in range(0, len(frames), num_frames_per_chunk):
            chunk_frames = frames[start : start + num_frames_per_chunk]
            timestamp = timestamp_as_str(timestamps[start], timestamp_mode)
            chunks.append(
                {
                    "type": "video_chunk",
                    "video_chunk": chunk_frames,
                    "prompt": make_video_chunk_prompt(timestamp),
                    "in_patch_limit": per_frame_limit,
                }
            )
        return chunks

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice image and video patch tensors into per-item fields.

        Image keeps ``pixel_values`` / ``grid_thws``. Video uses separate
        ``video_pixel_values`` / ``video_grid_thws`` keys.
        """
        fields: dict[str, MultiModalFieldConfig] = {}

        grid_thws = hf_inputs.get("grid_thws")
        if grid_thws is not None:
            grid_sizes = grid_thws.prod(-1)
            fields["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", grid_sizes
            )
            fields["grid_thws"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )

        video_grid_thws = hf_inputs.get("video_grid_thws")
        if video_grid_thws is not None:
            video_grid_sizes = video_grid_thws.prod(-1)
            fields["video_pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            )
            fields["video_grid_thws"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Image WxH expansion (unchanged) + video media_pad expansion."""
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder
        tokenizer = self.info.get_tokenizer()
        updates: list[PromptUpdate] = []

        if "image" in mm_items:

            def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
                images = mm_items.get_items("image", ImageProcessorItems)
                image = images.get(item_idx)
                if image is None:
                    raise ValueError(f"Missing image data at index {item_idx}")

                num_media_token = self.info.media_tokens_calculator(
                    {"type": "image", "image": image}
                )
                pads = media_token * num_media_token
                width, height = images.get_image_size(item_idx)
                full = (
                    f"<|media_begin|>image {width}x{height}<|media_content|>"
                    f"{pads}<|media_end|>"
                )
                return PromptUpdateDetails.select_token_id(
                    cached_encode(tokenizer, full, add_special_tokens=False),
                    media_token_id,
                )

            updates.append(
                PromptReplacement(
                    modality="image",
                    target=cached_encode(
                        tokenizer, image_placeholder, add_special_tokens=False
                    ),
                    replacement=get_image_replacement,
                )
            )

        if "video" in mm_items:
            video_calc = (
                getattr(self.info.video_processor, "media_tokens_calculator", None)
                or self.info.media_tokens_calculator
            )

            def get_video_replacement(item_idx: int) -> list[int]:
                videos = mm_items.get_items("video", KimiK3VideoChunkItems)
                chunk = videos[item_idx]
                num_media_token = video_calc(chunk)
                return [media_token_id] * num_media_token

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=[media_token_id],
                    replacement=get_video_replacement,
                )
            )

        return updates
