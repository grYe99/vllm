# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

# Keep the original image mode (including the alpha channel) for K3 instead of
# flattening images onto a background color. Server-level (--media-io-kwargs)
# and request-level media_io_kwargs still take precedence over this default.
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}
_K3_THINKING_EFFORTS = ("low", "high", "max")


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _dump_k3_template_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    return value


def _apply_k3_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking", enable_thinking)

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)

    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
        supported = ", ".join(_K3_THINKING_EFFORTS)
        raise VLLMValidationError(
            f"Kimi K3 supports thinking_effort values: {supported}",
            parameter="thinking_effort",
            value=thinking_effort,
        )


def _normalize_k3_tool_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, Any]]:
    """Reorder tool-result messages to match assistant tool_call order.

    Supports matching by ``tool_call_id`` or by the synthetic
    ``"{tool}:{zero_based_index}"`` alias. When any tool message in a
    block cannot be resolved, the whole block is left in its original
    order (graceful fallback).

    Returns a new list; caller-owned message dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    i = 0

    while i < len(conversation):
        message = conversation[i]
        normalized.append(dict(message))
        i += 1

        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        # Build the lookup table from the assistant's tool_calls.
        targets_by_id: dict[str, tuple[int, str]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_target = (position, name)
            aliases = [f"{name}:{position}"]
            if tool_call_id := tool_call.get("id"):
                aliases.insert(0, str(tool_call_id))
            for alias in aliases:
                targets_by_id.setdefault(alias, call_target)

        # Collect consecutive tool messages.
        block_start = i
        while i < len(conversation) and conversation[i].get("role") == "tool":
            i += 1
        if i == block_start:
            continue

        # Try to resolve every tool message in the block.
        resolved: list[tuple[int, int, dict[str, Any]]] = []
        for order, tool_message in enumerate(conversation[block_start:i]):
            tool_call_id = tool_message.get("tool_call_id")
            resolved_target = (
                targets_by_id.get(str(tool_call_id))
                if tool_call_id is not None
                else None
            )
            if resolved_target is None:
                normalized.extend(dict(item) for item in conversation[block_start:i])
                break

            position, name = resolved_target
            enriched = dict(tool_message)
            enriched["tool"] = name
            enriched["index"] = position + 1
            resolved.append((position, order, enriched))
        else:
            resolved.sort(key=lambda item: (item[0], item[1]))
            normalized.extend(item for _, _, item in resolved)

    return normalized


def _replace_video_placeholders_in_conversation(
    conversation: list[dict[str, Any]],
    video_placeholder: str,
    video_prompts: list[str],
) -> list[dict[str, Any]]:
    """Replace video placeholders in chat content before tokenization."""
    if not video_placeholder or not video_prompts:
        return conversation

    remaining = list(video_prompts)

    def _replace_in_value(value: Any) -> Any:
        if isinstance(value, str) and video_placeholder in value and remaining:
            parts = value.split(video_placeholder)
            needed = len(parts) - 1
            if needed > len(remaining):
                return value
            out: list[str] = []
            for part in parts[:-1]:
                out.append(part)
                out.append(remaining.pop(0))
            out.append(parts[-1])
            return "".join(out)
        if isinstance(value, list):
            return [_replace_in_value(v) for v in value]
        if isinstance(value, dict):
            return {k: _replace_in_value(v) for k, v in value.items()}
        return value

    return [_replace_in_value(message) for message in conversation]


def _split_k3_videos_in_mm_data(
    model_config,
    conversation: list[dict[str, Any]],
    mm_data,
    mm_uuids,
):
    """Split raw videos into MoonViT3d chunks without touching the image path.

    Unlike K2.5's unified ``vision_chunk`` modality, K3 keeps ``image`` intact
    and only rewrites the additive ``video`` modality here.
    """
    if not mm_data or "video" not in mm_data:
        return conversation, mm_data, mm_uuids

    videos = mm_data["video"]
    if not videos:
        return conversation, mm_data, mm_uuids

    # Already split into video_chunk dicts (e.g. dummy / offline path).
    if isinstance(videos[0], dict) and videos[0].get("type") == "video_chunk":
        video_prompts = []
        # Group by video_idx if present; otherwise each item is its own video.
        by_idx: dict[int, list[str]] = {}
        for item in videos:
            idx = int(item.get("video_idx", 0))
            by_idx.setdefault(idx, []).append(item.get("prompt", ""))
        for idx in sorted(by_idx):
            video_prompts.append("".join(by_idx[idx]))
        video_placeholder = getattr(
            model_config.hf_config, "video_placeholder", None
        )
        if video_placeholder and video_prompts:
            conversation = _replace_video_placeholders_in_conversation(
                conversation, video_placeholder, video_prompts
            )
        return conversation, mm_data, mm_uuids

    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    if not hasattr(mm_processor, "split_video_chunks"):
        return conversation, mm_data, mm_uuids

    raw_uuids = (mm_uuids or {}).get("video") or [None] * len(videos)
    split_chunks: list[dict[str, Any]] = []
    split_uuids: list[str] = []
    video_prompts: list[str] = []

    for video_idx, (video, uuid) in enumerate(zip(videos, raw_uuids)):
        chunks = mm_processor.split_video_chunks(video)
        video_prompts.append("".join(chunk["prompt"] for chunk in chunks))
        # Client-provided UUIDs are used as mm_hash / prefix-cache identifiers.
        # Without one, NEVER invent a content-independent id like "video-0":
        # different videos would collide in encoder + prefix cache and reuse
        # the wrong KV / ViT features. Match K2.5: fall back to random_uuid().
        base_uuid = uuid or random_uuid()
        for i, chunk in enumerate(chunks):
            chunk = dict(chunk)
            chunk["video_idx"] = video_idx
            split_chunks.append(chunk)
            split_uuids.append(f"{base_uuid}-{i}")

    mm_data = dict(mm_data)
    mm_data["video"] = split_chunks
    mm_uuids = dict(mm_uuids or {})
    mm_uuids["video"] = split_uuids

    video_placeholder = getattr(model_config.hf_config, "video_placeholder", None)
    if video_placeholder and video_prompts:
        conversation = _replace_video_placeholders_in_conversation(
            conversation, video_placeholder, video_prompts
        )
    return conversation, mm_data, mm_uuids


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Render chat prompts with Kimi K3's Python XTML encoding.

    K3 ships no Jinja chat template; its tokenizer renders messages through
    ``encoding_k3`` instead. We tokenize eagerly so the structural markers keep
    their special-token ids while user- and tool-supplied text stays ordinary.
    """

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        params: ChatParams,
    ) -> list[int]:
        # Tokenize eagerly: K3 encodes structural markers as special tokens and
        # user/tool text as ordinary tokens, so we cannot defer to a plain
        # re-tokenization of the rendered string downstream.
        kwargs = params.get_apply_chat_template_kwargs()
        _apply_k3_thinking_kwargs(kwargs)
        if params.tool_choice not in (None, "auto"):
            kwargs["tool_choice"] = _dump_k3_template_value(params.tool_choice)
        if params.response_format is not None:
            kwargs["response_format"] = _dump_k3_template_value(params.response_format)
        kwargs["tokenize"] = True
        return self.get_tokenizer().apply_chat_template(conversation, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        rendered_conversation, mm_data, mm_uuids = _split_k3_videos_in_mm_data(
            self.model_config, rendered_conversation, mm_data, mm_uuids
        )
        prompt = parse_dec_only_prompt(
            self._apply_chat_template(rendered_conversation, params)
        )
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        rendered_conversation, mm_data, mm_uuids = _split_k3_videos_in_mm_data(
            self.model_config, rendered_conversation, mm_data, mm_uuids
        )
        token_ids = await self._apply_chat_template_async(rendered_conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt
