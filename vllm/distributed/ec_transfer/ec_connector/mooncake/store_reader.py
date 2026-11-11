# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded consumer-side Mooncake Store reader for hybrid EPD delivery.

Store I/O runs on a single serialized worker thread and never touches the
control server's ZMQ sockets. Completions are returned through a queue for
the control thread to apply, and late results are matched by reservation id.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
    make_embedding_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (  # noqa: E501
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class StoreReadRequest:
    transfer_id: str
    reservation_id: str
    mm_hash: str
    namespace: str
    tensor_spec: TensorSpec
    destination: torch.Tensor


@dataclass(frozen=True)
class StoreReadCompletion:
    transfer_id: str
    reservation_id: str
    mm_hash: str
    ok: bool
    safe: bool
    error: str | None = None


class StoreDeliveryReader:
    """Admit, bound, and execute consumer Store GET jobs."""

    def __init__(
        self,
        store_client: MooncakeEmbeddingStoreClient,
        *,
        device: torch.device | str,
        max_pending_items: int,
        max_pending_bytes: int,
        read_buffer_bytes: int,
    ) -> None:
        self._client = store_client
        self._device = torch.device(device)
        self._max_pending_items = max_pending_items
        self._max_pending_bytes = max_pending_bytes
        self._read_buffer_bytes = read_buffer_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ec-mooncake-store-reader"
        )
        self._completions: queue.Queue[StoreReadCompletion] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: dict[str, Future[None]] = {}
        self._pending_bytes = 0
        self._closed = False

    def can_admit(self, nbytes: int) -> bool:
        with self._lock:
            return self._can_admit_locked(nbytes)

    def _can_admit_locked(self, nbytes: int) -> bool:
        if self._closed:
            return False
        if len(self._pending) >= self._max_pending_items:
            return False
        return self._pending_bytes + nbytes <= self._max_pending_bytes

    def enqueue(self, request: StoreReadRequest) -> bool:
        nbytes = request.tensor_spec.nbytes
        with self._lock:
            if request.transfer_id in self._pending or not self._can_admit_locked(
                nbytes
            ):
                return False
            try:
                future = self._executor.submit(self._run, request)
            except RuntimeError:
                return False
            self._pending[request.transfer_id] = future
            self._pending_bytes += nbytes

        def _done(done: Future[None], transfer_id: str = request.transfer_id) -> None:
            with self._lock:
                self._pending.pop(transfer_id, None)
                self._pending_bytes = max(0, self._pending_bytes - nbytes)

        # Register the callback outside the admission lock. A finished future
        # invokes it immediately, and the lock is not reentrant.
        future.add_done_callback(_done)
        return True

    def drain(self) -> list[StoreReadCompletion]:
        completions: list[StoreReadCompletion] = []
        while True:
            try:
                completions.append(self._completions.get_nowait())
            except queue.Empty:
                return completions

    def wait_for(self, transfer_ids: set[str], timeout: float) -> bool:
        """Wait until listed jobs are no longer running. Does not drain."""
        futures: list[Future[None]] = []
        with self._lock:
            for transfer_id in transfer_ids:
                future = self._pending.get(transfer_id)
                if future is not None:
                    futures.append(future)
        for future in futures:
            with contextlib.suppress(Exception):
                future.result(timeout=max(0.0, timeout))
        with self._lock:
            return not any(transfer_id in self._pending for transfer_id in transfer_ids)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run(self, request: StoreReadRequest) -> None:
        tensor: torch.Tensor | None = None
        try:
            if self._device.type == "cuda":
                torch.accelerator.set_device_index(self._device.index or 0)
            pool_key = make_embedding_key(request.namespace, request.mm_hash)
            loaded = self._client.load_tensors(
                {pool_key: request.tensor_spec}, self._device
            )
            tensor = loaded.get(pool_key)
            if tensor is None:
                self._completions.put(
                    StoreReadCompletion(
                        request.transfer_id,
                        request.reservation_id,
                        request.mm_hash,
                        ok=False,
                        safe=True,
                        error="store object missing or invalid",
                    )
                )
                return
            destination = request.destination
            if tuple(tensor.shape) != tuple(destination.shape) or tensor.dtype != (
                destination.dtype
            ):
                self._completions.put(
                    StoreReadCompletion(
                        request.transfer_id,
                        request.reservation_id,
                        request.mm_hash,
                        ok=False,
                        safe=True,
                        error="store tensor does not match the reserved allocation",
                    )
                )
                return
            destination.copy_(tensor)
            if destination.device.type == "cuda":
                torch.accelerator.current_stream(destination.device).synchronize()
            self._completions.put(
                StoreReadCompletion(
                    request.transfer_id,
                    request.reservation_id,
                    request.mm_hash,
                    ok=True,
                    safe=True,
                )
            )
        except (EmbeddingStoreOperationError, OSError) as exc:
            logger.warning(
                "Mooncake Store GET failed for transfer_id=%s: %s",
                request.transfer_id,
                exc,
            )
            self._completions.put(
                StoreReadCompletion(
                    request.transfer_id,
                    request.reservation_id,
                    request.mm_hash,
                    ok=False,
                    safe=True,
                    error=str(exc),
                )
            )
        except EmbeddingStoreError as exc:
            # Includes a poisoned client: native I/O may still be in flight.
            self._client._poison([request, tensor])
            logger.error(
                "Unconfirmed Mooncake Store GET for transfer_id=%s: %s",
                request.transfer_id,
                exc,
            )
            self._completions.put(
                StoreReadCompletion(
                    request.transfer_id,
                    request.reservation_id,
                    request.mm_hash,
                    ok=False,
                    safe=False,
                    error=str(exc),
                )
            )
        except BaseException as exc:
            # The final destination copy has its own lifetime after Store GET.
            self._client._poison([request, tensor])
            logger.exception(
                "Mooncake Store reader failed for transfer_id=%s",
                request.transfer_id,
            )
            self._completions.put(
                StoreReadCompletion(
                    request.transfer_id,
                    request.reservation_id,
                    request.mm_hash,
                    ok=False,
                    safe=False,
                    error=str(exc),
                )
            )
