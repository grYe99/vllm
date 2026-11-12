# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec
import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
    compute_ec_compatibility_hash,
)

pytestmark = pytest.mark.cpu_test


def test_xferreq_roundtrip():
    req = XferReq(mm_hash="h1", compatibility_hash="c1", session_id="s1")
    data = msgspec.msgpack.encode(req)
    out = msgspec.msgpack.decode(data, type=XferReq)
    assert out.mm_hash == "h1"
    assert out.compatibility_hash == "c1"
    assert out.connector_version == EC_CONNECTOR_VERSION


def test_xferack_roundtrip_ok():
    ack = XferAck(
        mm_hash="h1",
        status=XferStatus.OK,
        src_block_indices=[1, 2],
        agent_metadata=b"meta",
        mem_descriptor=b"desc",
    )
    out = msgspec.msgpack.decode(msgspec.msgpack.encode(ack), type=XferAck)
    assert out.status == XferStatus.OK
    assert out.src_block_indices == [1, 2]


def test_compat_hash_deterministic_and_sensitive():
    a = compute_ec_compatibility_hash("0.1", "m", "float16", 64)
    b = compute_ec_compatibility_hash("0.1", "m", "float16", 64)
    c = compute_ec_compatibility_hash("0.1", "m", "float16", 128)
    assert a == b
    assert a != c


def test_store_delivery_capability_constant_and_namespace_approval():
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
        StoreDeliverySpec,
    )
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (  # noqa: E501
        STORE_DELIVERY_CAPABILITY,
        TensorSpec,
        store_namespace_is_approved,
    )

    assert STORE_DELIVERY_CAPABILITY == "store_delivery_v1"
    namespace = (
        "p@embedding@model:m@revision:r@encoder:e@dtype:torch.float32@protocol:v3"
    )
    spec = StoreDeliverySpec(
        namespace=namespace,
        identifier="img",
        tensor_spec=TensorSpec((2, 2), "float32", 16),
    )
    assert spec.protocol == "v3"
    assert store_namespace_is_approved(
        namespace.replace("@encoder:e", "@encoder:other"), namespace
    )


def test_old_peer_without_capability_is_not_store_delivery():
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (  # noqa: E501
        STORE_DELIVERY_CAPABILITY,
    )

    old_peer: dict[str, object] = {"ports": [19100]}
    new_peer: dict[str, object] = {
        "ports": [19100, 19101],
        "capabilities": [STORE_DELIVERY_CAPABILITY],
    }
    old_caps = old_peer.get("capabilities", [])
    assert isinstance(old_caps, list)
    assert STORE_DELIVERY_CAPABILITY not in old_caps
    new_caps = new_peer["capabilities"]
    assert isinstance(new_caps, list)
    assert STORE_DELIVERY_CAPABILITY in new_caps
