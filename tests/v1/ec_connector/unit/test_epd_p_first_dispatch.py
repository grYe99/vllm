# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demand-gated encoder fan-out in the EPD proxy (e-first vs p-first).

Exercises the REAL helpers loaded from the ``examples/`` proxy, so a future
change to them is what these tests catch.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

PROXY_REL = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("disagg_epd_proxy_pfirst", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def image_body(urls: list[str]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}} for url in urls
                ],
            }
        ]
    }


class _EncoderResponse:
    def __init__(self, params):
        self.status = 200
        self._params = params

    async def read(self):
        return msgspec.json.encode({"ec_transfer_params": self._params})

    async def text(self):
        return ""


class _EncoderSession:
    """Record every encoder sub-request; answer with per-image metadata."""

    def __init__(self):
        self.requests: list[list[str]] = []

    async def post(self, url, data=None, headers=None):
        body = msgspec.json.decode(data)
        urls = [item["image_url"]["url"] for item in body["messages"][0]["content"]]
        self.requests.append(urls)
        params = {
            f"{url}-processed": {
                "metadata": {"image_grid_thw": [1, 2, 3]},
                "item_indices": [i],
                "peer_port": 1234,
            }
            for i, url in enumerate(urls)
        }
        return _EncoderResponse(params)


class _ProbeResponse:
    def __init__(self, body):
        self.status = 200
        self._body = body

    async def read(self):
        return msgspec.json.encode(self._body)

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ProbeSession:
    """Answer demand probes; aiohttp's post() is sync and returns a context."""

    def __init__(self, need_indices):
        self.need_indices = need_indices
        self.probed: list[dict] = []

    def post(self, url, data=None, headers=None):
        body = msgspec.json.decode(data)
        assert body["vllm_xargs"]["embedding_demand_probe"]
        assert body["stream"] is False
        self.probed.append(body)
        demands = [
            {
                "index": i,
                "reason": "need_from_e" if i in self.need_indices else "kv_covered",
            }
            for i in range(3)
        ]
        return _ProbeResponse(
            {"ec_transfer_params": {"embedding_demand": {"demands": demands}}}
        )


def test_e_first_encodes_every_item(proxy, monkeypatch):
    encoder = _EncoderSession()
    monkeypatch.setattr(proxy, "encode_session", encoder)
    monkeypatch.setattr(proxy, "NO_REWRITE", False)

    body = image_body(["A", "B", "C"])
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body, "r", ["http://encoder"], "", None, encoder_dispatch="e-first"
        )
    )
    assert len(encoder.requests) == 1
    assert encoder.requests[0] == ["A", "B", "C"]
    # Every item was rewritten to a metadata reference.
    content = prepared["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_embeds"] * 3


def test_p_first_skips_items_the_consumer_covers(proxy, monkeypatch):
    encoder = _EncoderSession()
    probe = _ProbeSession(need_indices={1})
    monkeypatch.setattr(proxy, "encode_session", encoder)
    monkeypatch.setattr(proxy, "decode_session", probe)
    monkeypatch.setattr(proxy, "NO_REWRITE", False)

    body = image_body(["A", "B", "C"])
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body,
            "r",
            ["http://encoder"],
            "",
            None,
            encoder_dispatch="p-first",
            consumer_url="http://consumer",
        )
    )
    # The probe saw the full media list; the encoder saw only the gap.
    assert len(probe.probed) == 1
    assert len(probe.probed[0]["messages"][0]["content"]) == 3
    assert encoder.requests == [["B"]]
    # Untouched items keep their raw media; the encoded one is a reference.
    content = prepared["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "image_embeds"
    assert content[2]["type"] == "image_url"


def test_p_first_with_no_gaps_encodes_nothing(proxy, monkeypatch):
    encoder = _EncoderSession()
    probe = _ProbeSession(need_indices=set())
    monkeypatch.setattr(proxy, "encode_session", encoder)
    monkeypatch.setattr(proxy, "decode_session", probe)
    monkeypatch.setattr(proxy, "NO_REWRITE", False)

    body = image_body(["A", "B", "C"])
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body,
            "r",
            ["http://encoder"],
            "",
            None,
            encoder_dispatch="p-first",
            consumer_url="http://consumer",
        )
    )
    assert encoder.requests == []
    # Nothing was rewritten, so the original body passes through.
    assert prepared["messages"][0]["content"][0]["type"] == "image_url"


def test_failed_probe_falls_back_to_encoding_everything(proxy, monkeypatch):
    """Fail-open: an unanswerable probe must not skip encoder work."""

    class _BrokenProbe:
        async def post(self, url, data=None, headers=None):
            raise ConnectionError("consumer down")

    encoder = _EncoderSession()
    monkeypatch.setattr(proxy, "encode_session", encoder)
    monkeypatch.setattr(proxy, "decode_session", _BrokenProbe())
    monkeypatch.setattr(proxy, "NO_REWRITE", False)

    body = image_body(["A", "B", "C"])
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body,
            "r",
            ["http://encoder"],
            "",
            None,
            encoder_dispatch="p-first",
            consumer_url="http://consumer",
        )
    )
    assert encoder.requests == [["A", "B", "C"]]
    content = prepared["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_embeds"] * 3


def test_p_first_without_text_media_skips_the_probe(proxy, monkeypatch):
    """A text-only request has no demand to ask about."""

    class _NoCalls:
        async def post(self, *args, **kwargs):
            raise AssertionError("no probe or encode should happen")

    monkeypatch.setattr(proxy, "encode_session", _NoCalls())
    monkeypatch.setattr(proxy, "decode_session", _NoCalls())
    monkeypatch.setattr(proxy, "NO_REWRITE", False)

    body = {"messages": [{"role": "user", "content": "hello"}]}
    prepared, _, _ = asyncio.run(
        proxy.prepare_for_decode(
            body,
            "r",
            ["http://encoder"],
            "",
            None,
            encoder_dispatch="p-first",
            consumer_url="http://consumer",
        )
    )
    assert prepared == body
