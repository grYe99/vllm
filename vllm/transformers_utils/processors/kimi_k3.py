# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.hf import HfTokenizer


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for Kimi-K3 images and optional videos.

    The image path is unchanged: ``images=[PIL, ...]`` is adapted into
    ``{"type": "image", "image": PIL}`` media dicts for the checkpoint image
    processor and returns ``pixel_values`` / ``grid_thws``.

    Video is an additive side path. Callers pass already-split MoonViT3d
    ``video_chunk`` media dicts via ``videos=``. Those are preprocessed with the
    fused/local video-capable processor and returned under separate keys
    (``video_pixel_values`` / ``video_grid_thws``) so the image tensors stay
    untouched.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
        video_processor: BaseImageProcessor | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # Optional video-capable processor (typically fused MoonViT preprocess).
        # Falls back to image_processor when it already understands video_chunk.
        self.video_processor = video_processor or image_processor

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        videos: list[dict] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        mm_inputs: dict = {}

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias = [{"type": "image", "image": image} for image in images]
            image_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
            )
            mm_inputs.update(dict(image_inputs))

        if videos is not None:
            video_inputs = self.video_processor.preprocess(
                videos,
                return_tensors=return_tensors,
            )
            # Keep video tensors on a separate key namespace so image fields
            # (pixel_values / grid_thws) are never overwritten or reshaped.
            if "pixel_values" in video_inputs:
                mm_inputs["video_pixel_values"] = video_inputs["pixel_values"]
            if "grid_thws" in video_inputs:
                mm_inputs["video_grid_thws"] = video_inputs["grid_thws"]

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **mm_inputs},
            tensor_type=return_tensors,
        )
