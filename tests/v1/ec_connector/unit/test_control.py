# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for control-layer ABCs, ZmqClientConnection, and ECSession."""

import socket
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import zmq

from vllm.config import DeviceConfig, ECTransferConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.control.base import (
    ControlConnection,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    MemoryAllocation,
    ResidentLease,
    _ResidentEntry,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakePushSpec,
    StoreDeliverySpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.store_reader import (
    StoreDeliveryReader,
    StoreReadRequest,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    STORE_DELIVERY_CAPABILITY,
    TensorSpec,
    make_embedding_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (  # noqa: E501
    EmbeddingStoreError,
    MooncakeEmbeddingStoreClient,
)

pytestmark = pytest.mark.cpu_test


class _ConcreteConn(ControlConnection):
    """Minimal concrete subclass for ABC-instantiation tests."""

    def __init__(self):
        self._alive = True
        self._sent = []
        self._inbox = []

    @property
    def alive(self) -> bool:
        return self._alive

    def send(self, msg: bytes) -> None:
        self._sent.append(msg)

    def recv(self) -> list[bytes]:
        msgs, self._inbox = self._inbox, []
        return msgs

    def mark_dead(self) -> None:
        self._alive = False

    def close(self) -> None:
        self._alive = False


def test_control_connection_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ControlConnection()  # type: ignore[abstract]


def test_concrete_connection_send_recv():
    conn = _ConcreteConn()
    assert conn.alive
    conn.send(b"hello")
    conn._inbox.append(b"world")
    assert conn.recv() == [b"world"]
    assert conn._sent == [b"hello"]


def test_mark_dead_sets_alive_false():
    conn = _ConcreteConn()
    conn.mark_dead()
    assert not conn.alive


def test_close_sets_alive_false():
    conn = _ConcreteConn()
    conn.close()
    assert not conn.alive


# ── ZmqClientConnection ───────────────────────────────────────────────────────


from vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq import (  # noqa: E402
    ZmqClientConnection,
    ZmqClientTransport,
    ZmqServerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr  # noqa: E402


def _conn(messages: list[list[bytes]], monitor=None) -> ZmqClientConnection:
    """Build a ZmqClientConnection whose dealer yields the given frame-lists."""
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = [*messages, zmq.Again()]
    return ZmqClientConnection(dealer=dealer, monitor=monitor)


def test_zmq_client_connection_recv_extracts_payloads_from_envelopes():
    conn = _conn([[b"", b"msg1"], [b"", b"msg2"]])
    assert conn.recv() == [b"msg1", b"msg2"]


def test_zmq_client_connection_recv_skips_frames_without_empty_delimiter():
    # A frame that does not start with b"" is malformed and must be dropped.
    conn = _conn([[b"", b"ok"], [b"no-delimiter", b"bad"]])
    assert conn.recv() == [b"ok"]


def test_zmq_client_connection_recv_breaks_on_non_again_exception():
    # A socket error should stop the drain and return whatever was collected.
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = [[b"", b"first"], OSError("broken")]
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    assert conn.recv() == [b"first"]


def test_zmq_client_connection_send_prefixes_empty_delimiter():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.send(b"payload")
    dealer.send_multipart.assert_called_once_with([b"", b"payload"])


def test_zmq_client_connection_mark_dead_does_not_close_socket():
    # mark_dead signals liveness only; resource teardown is close()'s job.
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.mark_dead()
    assert not conn.alive
    dealer.close.assert_not_called()


def test_zmq_client_connection_close_without_monitor():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.close()
    dealer.close.assert_called_once_with(linger=0)
    assert not conn.alive


def test_zmq_client_connection_close_with_monitor_disables_then_closes_both():
    # The disable_monitor → monitor.close → dealer.close sequence must be
    # respected: closing the dealer before disabling the monitor can cause
    # a zmq assertion in some versions.
    dealer = MagicMock()
    monitor = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=monitor)
    conn.close()
    dealer.disable_monitor.assert_called_once()
    monitor.close.assert_called_once_with(linger=0)
    dealer.close.assert_called_once_with(linger=0)
    assert not conn.alive


def test_zmq_client_connection_close_is_idempotent():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.close()
    conn.close()  # must not raise
    assert not conn.alive


# ── ZmqClientTransport ────────────────────────────────────────────────────────


def _make_transport() -> tuple[ZmqClientTransport, MagicMock]:
    """Return a ZmqClientTransport with a mocked ZMQ context."""
    ctx = MagicMock()
    ctx.socket.return_value = MagicMock()
    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.zmq.Context",
            return_value=ctx,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_socket",
            return_value=MagicMock(),
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_path",
            return_value="tcp://host:0",
        ),
    ):
        transport = ZmqClientTransport()
    transport._ctx = ctx
    return transport, ctx


def _inject_conn(transport: ZmqClientTransport, addr: PeerAddr) -> ZmqClientConnection:
    """Inject a ZmqClientConnection directly into the transport pool."""
    conn = ZmqClientConnection(dealer=MagicMock(), monitor=MagicMock())
    transport._connections[addr] = conn
    return conn


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_socket")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_path")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.zmq.Context")
def test_zmq_client_transport_connect_caches_connection(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx_cls.return_value = MagicMock()
    mock_ctx_cls.return_value.socket.return_value = MagicMock()
    mock_sock.return_value = MagicMock()
    transport = ZmqClientTransport()
    addr: PeerAddr = ("host", 1234)
    assert transport.connect(addr) is transport.connect(addr)


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_socket")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_path")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.zmq.Context")
def test_zmq_client_transport_connect_creates_independent_conns_per_addr(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx_cls.return_value = MagicMock()
    mock_ctx_cls.return_value.socket.return_value = MagicMock()
    mock_sock.return_value = MagicMock()
    transport = ZmqClientTransport()
    conn_a = transport.connect(("host-a", 1111))
    conn_b = transport.connect(("host-b", 2222))
    assert conn_a is not conn_b


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.recv_monitor_message")
def test_zmq_client_transport_poll_dead_detects_disconnected_peer(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 1234)
    conn = _inject_conn(transport, addr)
    mock_recv_mon.side_effect = [{"event": zmq.EVENT_DISCONNECTED}, zmq.Again()]

    dead = transport.poll_dead()

    assert addr in dead
    assert not conn.alive


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.recv_monitor_message")
def test_zmq_client_transport_poll_dead_skips_conn_without_monitor(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 9999)
    conn = ZmqClientConnection(dealer=MagicMock(), monitor=None)
    transport._connections[addr] = conn

    dead = transport.poll_dead()

    assert dead == []
    mock_recv_mon.assert_not_called()


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.recv_monitor_message")
def test_zmq_client_transport_poll_dead_ignores_non_disconnect_events(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 1234)
    conn = _inject_conn(transport, addr)
    # A CONNECTED event should not trigger a dead notification.
    mock_recv_mon.side_effect = [{"event": zmq.EVENT_CONNECTED}, zmq.Again()]

    dead = transport.poll_dead()

    assert dead == []
    assert conn.alive


def test_zmq_client_transport_remove_closes_and_evicts_conn():
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 5678)
    conn = _inject_conn(transport, addr)

    removed = transport.remove(addr)

    assert removed is conn
    assert addr not in transport._connections
    conn.dealer.close.assert_called()


def test_zmq_client_transport_remove_returns_none_for_unknown_addr():
    transport, _ = _make_transport()
    assert transport.remove(("ghost", 0)) is None


@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_socket")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.make_zmq_path")
@patch("vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq.zmq.Context")
def test_zmq_client_transport_close_closes_all_connections(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx = MagicMock()
    mock_ctx_cls.return_value = mock_ctx
    mock_ctx.socket.return_value = MagicMock()
    dealer_a, dealer_b = MagicMock(), MagicMock()
    mock_sock.side_effect = [dealer_a, dealer_b]
    transport = ZmqClientTransport()
    transport.connect(("a", 1))
    transport.connect(("b", 2))
    transport.close()
    dealer_a.close.assert_called()
    dealer_b.close.assert_called()
    assert transport._connections == {}


def test_zmq_servers_allocate_distinct_reachable_ports():
    """Simultaneous producers must publish usable ports without configuration."""
    servers = []
    client = ZmqClientTransport()
    try:
        for _ in range(2):
            servers.append(ZmqServerTransport(host="127.0.0.1"))
        assert len({server.port for server in servers}) == 2
        for server in servers:
            assert 0 < server.port <= 65535
            conn = client.connect(("127.0.0.1", server.port))
            conn.send(b"request")
            messages = server.poll(timeout_ms=5000)
            assert len(messages) == 1
            identity, payload = messages[0]
            assert payload == b"request"
            server.send(identity, b"reply")
            assert conn.dealer.poll(timeout=5000)
            assert conn.recv() == [b"reply"]
    finally:
        client.close()
        for server in servers:
            server.close()


def test_zmq_server_rejects_an_occupied_explicit_port():
    server = ZmqServerTransport(host="127.0.0.1")
    try:
        with pytest.raises(zmq.ZMQError) as exc:
            ZmqServerTransport(host="127.0.0.1", port=server.port)
        assert exc.value.errno == zmq.EADDRINUSE
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Mooncake hybrid Store-delivery control / reservation protocol
# ---------------------------------------------------------------------------


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeConsumerPool:
    def __init__(self, capacity: int = 1 << 20) -> None:
        self.lock = __import__("threading").RLock()
        self._storage = __import__("torch").zeros(
            capacity, dtype=__import__("torch").uint8
        )
        self._used = 0
        self._cached: dict[str, MemoryAllocation] = {}

    def try_allocate(self, nbytes, shape, dtype):
        if self._used + nbytes > self._storage.numel():
            return None
        tensor = self._storage.narrow(0, self._used, nbytes).view(dtype).view(shape)
        allocation = MemoryAllocation(self._used, nbytes, tensor)
        self._used += nbytes
        return allocation

    def reclaim_and_allocate(self, nbytes, shape, dtype):
        return self.try_allocate(nbytes, shape, dtype)

    def acquire_cached(self, mm_hash, shape, dtype):
        allocation = self._cached.get(mm_hash)
        if allocation is None:
            return None
        if (
            tuple(allocation.tensor.shape) != tuple(shape)
            or allocation.tensor.dtype != dtype
        ):
            return None
        return ResidentLease(mm_hash, _ResidentEntry(allocation))

    def publish(self, mm_hash, allocation, lease=None, pin=False):
        self._cached[mm_hash] = allocation
        return allocation

    def release_cached(self, lease):
        return None

    def free(self, allocation):
        return None

    def retire_stale(self, encoder_cache, freed=None):
        return None


def _store_client_with(objects: dict[str, "object"] | None = None):
    from tests.v1.ec_connector.unit.test_cross_encoder_cache import MemoryStore

    native = MemoryStore()
    if objects:
        native.objects.update(objects)
    return MooncakeEmbeddingStoreClient(native, read_buffer_bytes=1024), native


@dataclass
class _StoreHarness:
    reservations: ConsumerReservationManager
    reader: StoreDeliveryReader
    server: ConsumerControlServer
    client: ControlClient
    addr: str
    pool: _FakeConsumerPool
    store: object
    namespace: str

    def close(self) -> None:
        self.reader.shutdown(wait=True)
        self.server.close()
        self.client.close()


def _make_store_control_harness(*, max_pending_items: int = 8) -> _StoreHarness:
    import torch

    namespace = (
        "test@embedding@model:model@revision:revision"
        "@encoder:encoder@dtype:torch.float32@protocol:v3"
    )
    pool = _FakeConsumerPool()
    reservations = ConsumerReservationManager(pool, lease_ttl=30, tombstone_limit=16)
    client, native = _store_client_with()
    reader = StoreDeliveryReader(
        client,
        device="cpu",
        max_pending_items=max_pending_items,
        max_pending_bytes=1 << 20,
        read_buffer_bytes=1024,
    )

    def reserve(payload):
        transfer_id = str(payload["transfer_id"])
        mm_hash = str(payload["mm_hash"])
        nbytes = int(payload["nbytes"])
        shape = tuple(int(v) for v in payload["shape"])
        dtype_name = str(payload["dtype"]).removeprefix("torch.")
        dtype = getattr(torch, dtype_name)
        source = "store" if payload.get("store") and reader.can_admit(nbytes) else "p2p"
        record, should_write = reservations.reserve(
            transfer_id, mm_hash, nbytes, shape, dtype_name, dtype, source=source
        )
        if record is None:
            raise RuntimeError("pool full")
        if record.allocation is None:
            raise RuntimeError("reservation has no allocation")
        store_pending = False
        if (
            should_write
            and source == "store"
            and record.state is ConsumerReservationState.WRITING
        ):
            store = payload["store"]
            admitted = reader.enqueue(
                StoreReadRequest(
                    transfer_id=transfer_id,
                    reservation_id=record.reservation_id,
                    mm_hash=mm_hash,
                    namespace=str(store["namespace"]),
                    tensor_spec=TensorSpec(
                        tuple(int(v) for v in store["shape"]),
                        (
                            str(store["dtype"])
                            if str(store["dtype"]).startswith("torch.")
                            else "torch." + str(store["dtype"])
                        ),
                        int(store["nbytes"]),
                    ),
                    destination=record.allocation.tensor,
                )
            )
            store_pending = admitted
            if not admitted:
                record.source = "p2p"
        elif (
            record.source == "store"
            and record.state is ConsumerReservationState.WRITING
        ):
            store_pending = record.store_error is None
        return {
            "reservation_id": record.reservation_id,
            "dst_session": "local",
            "dst_ptr": record.allocation.tensor.data_ptr(),
            "nbytes": record.allocation.tensor.nbytes,
            "write": should_write and record.source != "store",
            "ready": record.state is ConsumerReservationState.READY,
            "cached": record.lease is not None,
            "store_pending": store_pending,
        }

    def status(transfer_id: str):
        record = reservations.status(transfer_id)
        if record is None:
            return None
        return {
            "mm_hash": record.mm_hash,
            "ready": record.state is ConsumerReservationState.READY,
            "reservation_id": record.reservation_id,
            "store_pending": (
                record.source == "store"
                and record.state is ConsumerReservationState.WRITING
                and record.store_error is None
            ),
            "store_failed_safe": record.state is ConsumerReservationState.STORE_FAILED,
            "error": record.store_error,
        }

    def poll_store():
        ready = []
        for completion in reader.drain():
            accepted, became_ready = reservations.complete_store(
                completion.transfer_id,
                completion.reservation_id,
                ok=completion.ok,
                safe=completion.safe,
                error=completion.error,
            )
            if became_ready:
                ready.append(completion.transfer_id)
            ready.extend(reservations.drain_ready())
        return ready

    port = _free_loopback_port()
    server = ConsumerControlServer(
        "127.0.0.1",
        port,
        reserve,
        status,
        reservations.complete,
        reservations.cancel,
        reservations.expire,
        peer_ports=[port],
        device=None,
        drain_ready=reservations.drain_ready,
        poll_store=poll_store,
        capabilities=[STORE_DELIVERY_CAPABILITY],
    )
    server.start()
    control = ControlClient(timeout_ms=2000)
    addr = f"tcp://127.0.0.1:{port}"
    return _StoreHarness(
        reservations, reader, server, control, addr, pool, native, namespace
    )


def _store_item(harness: _StoreHarness, transfer_id: str, mm_hash: str = "img"):
    return {
        "transfer_id": transfer_id,
        "mm_hash": mm_hash,
        "nbytes": 16,
        "shape": [2, 2],
        "dtype": "float32",
        "store": {
            "namespace": harness.namespace,
            "identifier": mm_hash,
            "protocol": "v3",
            "shape": [2, 2],
            "dtype": "torch.float32",
            "nbytes": 16,
        },
    }


def _wait_ready(harness: _StoreHarness, transfer_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = harness.client.request(
            harness.addr, {"op": "status", "transfer_id": transfer_id}
        )
        if status and status.get("ready"):
            return status
        if status and status.get("store_failed_safe"):
            return status
        if status and status.get("error") and not status.get("store_pending"):
            return status
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {transfer_id}")


def test_store_delivery_reader_runs_before_inference():
    """The P reader can finish a Store GET before any inference request exists."""
    import torch

    harness = _make_store_control_harness()
    try:
        peers = harness.client.request(harness.addr, {"op": "peers"})
        assert STORE_DELIVERY_CAPABILITY in peers["capabilities"]
        tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        client = harness.reader._client
        client.put_tensor(make_embedding_key(harness.namespace, "img"), tensor)
        reserved = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "t1")}
        )
        assert reserved["store_pending"] is True
        assert reserved["ready"] is False
        assert reserved["write"] is False
        status = _wait_ready(harness, "t1")
        assert status["ready"] is True
        record = harness.reservations.status("t1")
        assert record is not None
        assert record.allocation is not None
        assert torch.equal(record.allocation.tensor.cpu(), tensor)
    finally:
        harness.close()


def test_old_peer_peers_payload_omits_capability_when_disabled():
    port = _free_loopback_port()
    server = ConsumerControlServer(
        "127.0.0.1",
        port,
        lambda payload: {"reservation_id": "x", "ready": False, "write": True},
        lambda transfer_id: None,
        lambda transfer_id, reservation_id: (False, False),
        lambda *args: True,
        lambda: 0,
        peer_ports=[port],
    )
    client = ControlClient(timeout_ms=2000)
    try:
        server.start()
        addr = f"tcp://127.0.0.1:{port}"
        peers = client.request(addr, {"op": "peers"})
        assert "capabilities" not in peers
        assert client.consumer_capabilities(addr) == frozenset()
    finally:
        server.close()
        client.close()


def test_store_pending_then_ready_and_late_generation_is_ignored():
    import torch

    harness = _make_store_control_harness()
    try:
        tensor = torch.ones(2, 2)
        harness.reader._client.put_tensor(
            make_embedding_key(harness.namespace, "img"), tensor
        )
        first = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "t-gen")}
        )
        first_id = first["reservation_id"]
        status = _wait_ready(harness, "t-gen")
        assert status["ready"] is True
        # A late completion tagged with a stale reservation id must not apply.
        accepted, became_ready = harness.reservations.complete_store(
            "t-gen", "stale-generation", ok=True, safe=True
        )
        assert accepted is False
        assert became_ready is False
        later = harness.reservations.status("t-gen")
        assert later is not None
        assert first_id == later.reservation_id
    finally:
        harness.close()


def test_same_key_store_and_p2p_share_a_single_writer():
    import torch

    harness = _make_store_control_harness()
    try:
        tensor = torch.full((2, 2), 5.0)
        harness.reader._client.put_tensor(
            make_embedding_key(harness.namespace, "img"), tensor
        )
        writer = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "writer")}
        )
        follower = harness.client.request(
            harness.addr,
            {
                "op": "reserve",
                "transfer_id": "follower",
                "mm_hash": "img",
                "nbytes": 16,
                "shape": [2, 2],
                "dtype": "float32",
            },
        )
        assert writer["write"] is False
        assert follower["write"] is False
        # The follower either joins the in-flight Store writer or hits the
        # cache if that writer already completed.
        if follower.get("ready"):
            assert follower["store_pending"] is not True
        else:
            assert follower["store_pending"] is True
            assert _wait_ready(harness, "follower")["ready"] is True
        assert _wait_ready(harness, "writer")["ready"] is True
        # A later same-hash reserve must be a cache hit, not a new P2P writer.
        later = harness.client.request(
            harness.addr,
            {
                "op": "reserve",
                "transfer_id": "later",
                "mm_hash": "img",
                "nbytes": 16,
                "shape": [2, 2],
                "dtype": "float32",
            },
        )
        assert later["write"] is False
        assert later["ready"] is True
    finally:
        harness.close()


def test_store_reader_backpressure_rejects_excess_work_without_blocking_rep():
    import torch

    harness = _make_store_control_harness(max_pending_items=1)
    try:
        # Pause the reader so the first job occupies the only slot.
        started = __import__("threading").Event()
        release = __import__("threading").Event()
        original = harness.reader._run

        def blocked(request):
            started.set()
            assert release.wait(timeout=5)
            return original(request)

        harness.reader._run = blocked  # type: ignore[method-assign]
        harness.reader._client.put_tensor(
            make_embedding_key(harness.namespace, "img"), torch.ones(2, 2)
        )
        first = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "t1", "img")}
        )
        assert first["store_pending"] is True
        assert started.wait(timeout=5)
        # A second unique key cannot be admitted as a Store writer. The REP
        # loop must still answer immediately with a normal P2P reservation.
        second = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "t2", "other")}
        )
        assert second["store_pending"] is False
        assert second["write"] is True
    finally:
        release.set()
        harness.close()


def test_recoverable_store_failure_marks_store_failed_safe():
    harness = _make_store_control_harness()
    try:
        reserved = harness.client.request(
            harness.addr, {"op": "reserve", **_store_item(harness, "missing")}
        )
        assert reserved["store_pending"] is True
        status = _wait_ready(harness, "missing")
        assert status["store_failed_safe"] is True
        assert status["ready"] is not True
        # The same transfer id can be reserved again for P2P fallback.
        retry = harness.client.request(
            harness.addr,
            {
                "op": "reserve",
                "transfer_id": "missing",
                "mm_hash": "img",
                "nbytes": 16,
                "shape": [2, 2],
                "dtype": "float32",
            },
        )
        assert retry["reservation_id"]
        assert retry["reservation_id"] != reserved["reservation_id"]
        assert retry["write"] is True
    finally:
        harness.close()


def test_unconfirmed_store_io_retains_owners():
    harness = _make_store_control_harness()
    try:
        record, should_write = harness.reservations.reserve(
            "unsafe",
            "img",
            16,
            (2, 2),
            "float32",
            __import__("torch").float32,
            source="store",
        )
        assert should_write
        assert record is not None
        assert record.allocation is not None
        allocation = record.allocation
        accepted, became_ready = harness.reservations.complete_store(
            "unsafe",
            record.reservation_id,
            ok=False,
            safe=False,
            error="native i/o still running",
        )
        assert accepted is True
        assert became_ready is False
        assert record.state is ConsumerReservationState.WRITING
        assert record.store_io_retained is True
        assert record.allocation is allocation
        status = harness.reservations.status("unsafe")
        assert status is not None
        assert status.store_error
    finally:
        harness.close()


def test_store_reader_enqueue_does_not_deadlock_when_job_already_finished():
    """A finished future must not re-enter the admission lock from enqueue."""
    from concurrent.futures import Future

    import torch

    client, _native = _store_client_with()
    namespace = (
        "test@embedding@model:model@revision:revision"
        "@encoder:encoder@dtype:torch.float32@protocol:v3"
    )
    reader = StoreDeliveryReader(
        client,
        device="cpu",
        max_pending_items=8,
        max_pending_bytes=1 << 20,
        read_buffer_bytes=1024,
    )
    original = reader._executor

    class _ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            future: Future[None] = Future()
            try:
                fn(*args, **kwargs)
                future.set_result(None)
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            return None

    try:
        client.put_tensor(make_embedding_key(namespace, "img"), torch.ones(2, 2))
        reader._executor = _ImmediateExecutor()
        destination = torch.empty(2, 2)
        admitted = reader.enqueue(
            StoreReadRequest(
                transfer_id="sync",
                reservation_id="rid",
                mm_hash="img",
                namespace=namespace,
                tensor_spec=TensorSpec((2, 2), "torch.float32", 16),
                destination=destination,
            )
        )
        assert admitted is True
        completions = reader.drain()
        assert len(completions) == 1
        assert completions[0].ok is True
        assert torch.equal(destination, torch.ones(2, 2))
    finally:
        reader._executor = original
        reader.shutdown(wait=True)


@pytest.fixture
def native_free_workers(monkeypatch):
    """Use real workers and pools; replace only the native Mooncake transport."""
    from vllm.distributed.ec_transfer.ec_connector.mooncake import worker

    class NativeTransport:
        def __init__(self, *args):
            pass

        def register_memory(self, tensor):
            return 0

        def unregister_memory(self, tensor):
            return True

        def local_session(self):
            return "test-session"

        def write(self, session, sources, destinations, sizes):
            import ctypes

            for src, dst, size in zip(sources, destinations, sizes, strict=True):
                ctypes.memmove(dst, src, size)

        def release_sources(self, addresses):
            pass

        def close(self):
            pass

    monkeypatch.setattr(worker, "ensure_mooncake_available", lambda: None)
    monkeypatch.setattr(worker, "MooncakeTransfer", NativeTransport)
    workers = []

    def create():
        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            ec_transfer_config=ECTransferConfig(
                ec_connector="ECMooncakeConnector",
                ec_role="ec_consumer",
                ec_buffer_device="cpu",
                ec_buffer_size=1024,
                ec_port=_free_loopback_port(),
                ec_connector_extra_config={"control_timeout_s": 1},
            ),
        )
        instance = worker.ECMooncakeWorker(config)
        instance.start_services()
        workers.append(instance)
        return instance

    yield create
    for instance in reversed(workers):
        instance.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("unconfirmed", [False, True])
def test_store_writer_cannot_be_abandoned_by_remote_sender(
    native_free_workers, refresh, unconfirmed
):
    """Remote cancellation cannot recycle a destination owned by local GET."""
    import torch

    consumer = native_free_workers()
    manager = consumer._reservations
    record, write = manager.reserve(
        "store", "img", 16, (2, 2), "float32", torch.float32, source="store"
    )
    assert write and record is not None
    allocation = record.allocation
    if unconfirmed:
        manager.complete_store(
            "store", record.reservation_id, ok=False, safe=False, error="unconfirmed"
        )
    manager.cancel("store", record.reservation_id, abandon=True, refresh=refresh)
    assert record.allocation is allocation
    assert not manager.wait_for_writers(timeout=0)
    manager.complete_store("store", record.reservation_id, ok=False, safe=True)
    assert manager.wait_for_writers(timeout=0)


@pytest.mark.cpu_test
def test_partial_store_rejection_keeps_p2p_fallback_writable(native_free_workers):
    """A resident rank plus a P2P-only rank must not wait for a nonexistent GET."""
    from concurrent.futures import Future

    coordinator = native_free_workers()
    consumers = [native_free_workers(), native_free_workers()]
    first = consumers[0]._reserve_push_destination(
        dict(
            transfer_id="mixed", mm_hash="img", nbytes=16, shape=[2, 2], dtype="float32"
        )
    )
    consumers[0]._reservations.complete("mixed", first["reservation_id"])
    spec = ECMooncakePushSpec(
        "img",
        16,
        (2, 2),
        "float32",
        f"tcp://127.0.0.1:{consumers[0]._control_port}",
        "mixed",
    )
    reservations = [
        coordinator._reserve_one(f"tcp://127.0.0.1:{consumer._control_port}", spec)
        for consumer in consumers
    ]
    future: Future[list[dict[str, Any]]] = Future()
    future.set_result(reservations)
    record, _ = coordinator._producer_pushes.reserve(spec, lambda: future)
    outcome = coordinator._poll_store_shards(
        record, reservations, time.monotonic() + 0.05
    )
    assert outcome == "failed_safe"
    assert coordinator._re_reserve_failed_store_shards(record, reservations)
    assert record.reservations[0]["ready"]
    assert record.reservations[1]["write"]
    assert not record.reservations[1].get("cancelled", False)


@pytest.mark.cpu_test
def test_remote_completion_cannot_publish_store_writer(native_free_workers):
    """Only the local reader can declare a Store destination ready."""
    import torch

    consumer = native_free_workers()
    manager = consumer._reservations
    record, _ = manager.reserve(
        "store", "img", 16, (2, 2), "float32", torch.float32, source="store"
    )
    assert record is not None
    try:
        assert manager.complete("store", record.reservation_id) == (False, False)
        assert record.state is ConsumerReservationState.WRITING
    finally:
        manager.complete_store("store", record.reservation_id, ok=False, safe=True)


def test_safe_store_failure_pushes_to_replacement_generation(native_free_workers):
    """A safe failed GET must deliver real bytes through its new P2P reservation."""
    from concurrent.futures import Future

    import torch

    coordinator = native_free_workers()
    consumer = native_free_workers()
    manager = consumer._reservations
    original, _ = manager.reserve(
        "retry", "img", 16, (2, 2), "float32", torch.float32, source="store"
    )
    assert original is not None
    spec = ECMooncakePushSpec(
        "img",
        16,
        (2, 2),
        "float32",
        f"tcp://127.0.0.1:{consumer._control_port}",
        "retry",
        store=StoreDeliverySpec(
            "namespace", "img", TensorSpec((2, 2), "torch.float32", 16)
        ),
    )
    reservations = [coordinator._reserve_one(spec.consumer_zmq, spec)]
    future: Future[list[dict[str, Any]]] = Future()
    future.set_result(reservations)
    record, _ = coordinator._producer_pushes.reserve(spec, lambda: future)
    manager.complete_store("retry", original.reservation_id, ok=False, safe=True)
    assert (
        coordinator._poll_store_shards(record, reservations, time.monotonic() + 1)
        == "failed_safe"
    )
    assert coordinator._re_reserve_failed_store_shards(record, reservations)
    replacement = manager.status("retry")
    assert replacement is not None
    assert replacement.reservation_id != original.reservation_id

    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    coordinator._producer_pushes.bind_source("img", tensor, None)
    coordinator._push_batch([record])
    assert replacement.state is ConsumerReservationState.READY
    assert replacement.allocation is not None
    assert torch.equal(replacement.allocation.tensor, tensor)
    assert not coordinator._producer_pushes.pending
    assert record.spec.store is None  # Later refreshes must keep the P2P choice.


@pytest.mark.cpu_test
def test_unconfirmed_reader_retains_destination_owner(monkeypatch):
    """An unsafe read/copy completion must keep its destination alive."""
    import gc
    import weakref

    import torch

    from tests.v1.ec_connector.unit.test_cross_encoder_cache import MemoryStore
    from vllm.distributed.ec_transfer.ec_connector.mooncake import store_reader

    # Captured exception arguments would themselves retain the reader frame.
    monkeypatch.setattr(store_reader.logger, "disabled", True)

    class UnconfirmedStore(MemoryStore):
        def batch_get_into(self, *args):
            raise RuntimeError("native completion is unknown")

    client = MooncakeEmbeddingStoreClient(UnconfirmedStore(), read_buffer_bytes=1024)
    reader = StoreDeliveryReader(
        client,
        device="cpu",
        max_pending_items=1,
        max_pending_bytes=1024,
        read_buffer_bytes=1024,
    )
    destination = torch.empty(2, 2)
    owner = weakref.ref(destination)
    request = StoreReadRequest(
        "unsafe",
        "generation",
        "img",
        "namespace",
        TensorSpec((2, 2), "torch.float32", 16),
        destination,
    )
    assert reader.enqueue(request)
    reader.shutdown(wait=True)
    [completion] = reader.drain()
    assert not completion.ok and not completion.safe
    del request, destination
    gc.collect()
    assert owner() is not None


@pytest.mark.cpu_test
def test_poisoned_reader_degrades_reserve_to_p2p(native_free_workers):
    """After unconfirmed I/O poisons the client, the consumer must refuse new
    Store admissions and answer with a normal P2P reservation instead of
    hard-failing every later store delivery until a restart."""
    from tests.v1.ec_connector.unit.test_cross_encoder_cache import MemoryStore

    consumer = native_free_workers()
    client = MooncakeEmbeddingStoreClient(MemoryStore(), read_buffer_bytes=1024)
    reader = StoreDeliveryReader(
        client,
        device="cpu",
        max_pending_items=2,
        max_pending_bytes=1024,
        read_buffer_bytes=1024,
    )
    consumer._store_reader = reader
    namespace = (
        "lab@embedding@model:glm-test@revision:default@encoder:any"
        "@dtype:torch.bfloat16@protocol:v3"
    )
    consumer._store_namespace = namespace

    def reserve_payload(transfer_id: str, mm_hash: str) -> dict[str, object]:
        return {
            "transfer_id": transfer_id,
            "mm_hash": mm_hash,
            "nbytes": 16,
            "shape": [2, 2],
            "dtype": "float32",
            "store": {
                "namespace": namespace,
                "identifier": mm_hash,
                "protocol": "v3",
                "shape": [2, 2],
                "dtype": "torch.float32",
                "nbytes": 16,
            },
        }

    # Healthy wiring check: the descriptor is admitted and stays Store-sourced.
    assert reader.can_admit(16)
    healthy = consumer._reserve_push_destination(reserve_payload("deg-0", "img-a"))
    assert healthy["store_pending"] is True
    assert healthy["write"] is False

    # An unconfirmed native completion poisons the client permanently.
    client._poison([object()])
    assert not reader.can_admit(16)

    degraded = consumer._reserve_push_destination(reserve_payload("deg-1", "img-b"))
    assert degraded["store_pending"] is False
    assert degraded["write"] is True
    assert degraded["reservation_id"]
    record = consumer._reservations.status("deg-1")
    assert record is not None and record.source == "p2p"

    # Teardown hygiene: settle both records and detach the poisoned reader
    # (a poisoned client refuses close by design — "worker must exit").
    accepted, _ = consumer._reservations.complete(
        "deg-1", str(degraded["reservation_id"])
    )
    assert accepted
    reader.shutdown(wait=True)
    # Settle the healthy call's record the way the control loop would (the
    # fixture's control server was wired before the reader was injected).
    for completion in reader.drain():
        consumer._reservations.complete_store(
            completion.transfer_id,
            completion.reservation_id,
            ok=completion.ok,
            safe=completion.safe,
        )
    import contextlib

    with contextlib.suppress(EmbeddingStoreError):
        client.close()
    consumer._store_reader = None


def test_poisoned_store_writer_does_not_capture_same_key_p2p_retry(native_free_workers):
    """A new request must not follow a poisoned writer that can never finish."""
    import torch

    consumer = native_free_workers()
    manager = consumer._reservations
    old, _ = manager.reserve(
        "unsafe", "img", 16, (2, 2), "float32", torch.float32, source="store"
    )
    assert old is not None and old.allocation is not None
    allocation = old.allocation
    allocation.tensor.fill_(-1)
    manager.complete_store(
        "unsafe", old.reservation_id, ok=False, safe=False, error="unconfirmed"
    )
    try:
        new, write = manager.reserve(
            "retry", "img", 16, (2, 2), "float32", torch.float32, source="p2p"
        )
        assert new is not None and write
        assert not new.writer_id
        assert new.allocation is not allocation
        assert new.allocation is not None
        new.allocation.tensor.fill_(5)
        assert manager.complete("retry", new.reservation_id) == (True, True)
        assert torch.equal(manager.take("retry", "img").tensor, torch.full((2, 2), 5.0))
        assert old.allocation is allocation
        assert torch.equal(allocation.tensor, torch.full((2, 2), -1.0))
    finally:
        # Only this test's simulated native writer is now confirmed settled.
        manager.complete_store("unsafe", old.reservation_id, ok=False, safe=True)


def test_queued_read_after_poison_is_a_safe_failure():
    """A queued job that never entered native I/O must permit safe fallback."""
    import threading

    import torch

    from tests.v1.ec_connector.unit.test_cross_encoder_cache import MemoryStore

    started = threading.Event()
    release = threading.Event()

    class BlockedStore(MemoryStore):
        def batch_get_into(self, *args):
            started.set()
            assert release.wait(timeout=5)
            raise RuntimeError("unconfirmed native completion")

    client = MooncakeEmbeddingStoreClient(BlockedStore(), read_buffer_bytes=1024)
    reader = StoreDeliveryReader(
        client,
        device="cpu",
        max_pending_items=2,
        max_pending_bytes=1024,
        read_buffer_bytes=1024,
    )
    try:
        for transfer_id in ("first", "queued"):
            assert reader.enqueue(
                StoreReadRequest(
                    transfer_id,
                    transfer_id,
                    transfer_id,
                    "namespace",
                    TensorSpec((2, 2), "torch.float32", 16),
                    torch.empty(2, 2),
                )
            )
            if transfer_id == "first":
                assert started.wait(timeout=5)
    finally:
        release.set()
        reader.shutdown(wait=True)
    results = {completion.transfer_id: completion for completion in reader.drain()}
    assert not results["first"].safe
    assert not results["queued"].ok
    assert results["queued"].safe
    assert not reader.can_admit(16)
