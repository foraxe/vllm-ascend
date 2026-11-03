# SPDX-License-Identifier: Apache-2.0
"""MemFabric SHARED Offload storage; consumers retain metadata and NPU pointers only."""

from __future__ import annotations

import ctypes
import threading
import weakref
from typing import Any, ClassVar

import torch

GIB = 1 << 30
CHUNK_ROWS = 1 << 22
ALLOCATION_ALIGNMENT = 64


class MemfabricRuntime:
    """One fixed-rank Offload session shared by all Engram layers.

    The process must not also initialize sparse KV Offload: the native Offload
    manager is a singleton and silently ignores a second initialization.
    """

    _lock = threading.RLock()
    _active: ClassVar[MemfabricRuntime | None] = None
    mf: Any
    offload: Any
    spec: tuple
    device: torch.device
    references: int
    objects: set[int]
    rank: int

    @classmethod
    def acquire(cls, generation, store_url, world_size, device, *, owner, capacity, rank):
        device = torch.device(device)
        if device.type != "npu" or device.index is None:
            raise ValueError("An explicit NPU device index is required")
        if capacity <= 0 or capacity % GIB or not 0 <= rank < world_size or owner != (rank == 0):
            raise ValueError("Invalid fixed-rank Offload pool configuration")
        spec = (str(generation), store_url, world_size, device, owner, capacity, rank)
        with cls._lock:
            if cls._active is not None:
                if cls._active.spec != spec:
                    raise RuntimeError("A worker may attach to only one MemFabric Engram generation/device")
                cls._active.references += 1
                return cls._active
            import memfabric_hybrid as mf
            from memfabric_hybrid import offload

            runtime = cls()
            runtime.mf, runtime.offload, runtime.spec = mf, offload, spec
            runtime.device, runtime.references = device, 1
            runtime.objects = set()
            runtime.check(mf.initialize(), "initialize")
            config = offload.OffloadConfig()
            config.device_id = device.index
            config.reserve_size = capacity
            config.alloc_size = capacity if owner else 0
            config.world_size = world_size
            config.rank_id = rank
            config.scene = offload.Scene.SHARED
            config.store_url = store_url
            # This adapter currently supports a single host, like sparse KV.
            config.local_world_size = world_size
            with torch.npu.device(device):
                runtime.check(offload.initialize(config), "Offload initialize")
            runtime.rank = rank
            runtime.capacity = capacity
            runtime.alloc_size = config.alloc_size
            cls._active = runtime
            return runtime

    def check(self, result, operation):
        if result:
            raise RuntimeError(f"Engram MemFabric {operation}: {result}: {self.mf.get_last_err_msg()}")

    def close(self):
        with self._lock:
            if self.references > 1:
                self.references -= 1
                return
            if self.objects:
                raise RuntimeError("Close Engram mappings before their MemFabric session")
            with torch.npu.device(self.device):
                self.offload.uninitialize()
            self.references = 0
            type(self)._active = None
            # mf.uninitialize() is not refcounted and would also disable other
            # MemFabric users, including an independently owned TransferEngine.


class MappedHostBuffer:
    """A suballocation of rank 0's DRAM, mapped at a shared VMM address.

    Only the owner calls malloc/free. Consumers keep metadata and NPU pointer
    tensors. SHARED's native pool fingerprint validates identical GVA bases.
    """

    def __init__(self, runtime, object_id, rows, width, dtype, *, owner, address=None):
        if rows <= 0 or width <= 0 or dtype not in (torch.int8, torch.float32):
            raise ValueError("Shared Engram requires positive INT8/FP32 dimensions")
        if object_id < 0 or object_id in runtime.objects:
            raise ValueError("Duplicate or invalid MemFabric Engram object id")
        self.runtime, self.object_id = runtime, object_id
        self.rows, self.width, self.dtype = rows, width, dtype
        self.owner, self.device = owner, runtime.device
        self.row_bytes = width * torch.empty((), dtype=dtype).element_size()
        self.size = ((rows * self.row_bytes + ALLOCATION_ALIGNMENT - 1) // ALLOCATION_ALIGNMENT) * ALLOCATION_ALIGNMENT
        self.closed = False
        self._views: weakref.WeakSet[HostTableSlice] = weakref.WeakSet()
        with torch.npu.device(self.device):
            if owner:
                self.gva = runtime.offload.malloc(self.size)
                if not self.gva:
                    raise RuntimeError("Engram Offload pool allocation failed")
                # get_dva accepts local allocations only. SHARED VMM must be
                # identity-mapped for the same address to reach the native op.
                if runtime.offload.get_dva(self.gva) != self.gva:
                    runtime.offload.free(self.gva)
                    raise RuntimeError("Engram requires identity-mapped SHARED VMM DRAM")
            else:
                if not isinstance(address, int) or address <= 0 or address % ALLOCATION_ALIGNMENT:
                    raise ValueError("Invalid shared Offload address")
                self.gva = address
        runtime.objects.add(object_id)

    def device_pointer(self, offset, size=1):
        if self.closed or offset < 0 or size <= 0 or offset + size > self.rows * self.row_bytes:
            raise ValueError("Engram pointer is outside the live table")
        return self.gva + offset

    def publish(self, start, source):
        if self.closed or not self.owner:
            raise RuntimeError("Only a live owner may publish table rows")
        if source.device.type != "cpu" or source.dtype != self.dtype:
            raise ValueError("Publication requires a matching CPU dtype")
        if source.ndim != 2 or source.shape[1] != self.width or start < 0 or start + len(source) > self.rows:
            raise ValueError("Publication is outside the table shape")
        source = source.contiguous()
        ctypes.memmove(self.gva + start * self.row_bytes, source.data_ptr(), source.numel() * source.element_size())

    def view(self, start=0, end=None):
        end = self.rows if end is None else end
        if self.closed or not 0 <= start < end <= self.rows:
            raise ValueError("Invalid or closed shared Engram slice")
        view = HostTableSlice(self, start, end)
        self._views.add(view)
        return view

    def close(self):
        if self.closed:
            return
        if self._views:
            raise RuntimeError("Release all Engram slices before closing their mapping")
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            if self.owner:
                self.runtime.offload.free(self.gva)
        self.runtime.objects.remove(self.object_id)
        self.closed = True


class HostTableSlice:
    """Existing lookup ABI: metadata-only tensor plus stable NPU chunk pointers."""

    def __init__(self, mapping, start, end):
        self._mapping = mapping
        self.tensor = torch.empty((end - start, mapping.width), dtype=mapping.dtype, device="meta")
        self.ptrs = torch.tensor(
            [
                mapping.device_pointer(row * mapping.row_bytes, min(CHUNK_ROWS, end - row) * mapping.row_bytes)
                for row in range(start, end, CHUNK_ROWS)
            ],
            dtype=torch.int64,
            device=mapping.device,
        )

    def close(self):
        if self._mapping is None:
            return
        with torch.npu.device(self._mapping.device):
            torch.npu.synchronize()
        self._mapping._views.discard(self)
        self.ptrs = self.tensor = None
        self._mapping = None
