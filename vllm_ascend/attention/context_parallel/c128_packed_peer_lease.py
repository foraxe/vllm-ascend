# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Startup-only VMM peer aliases for packed C128 arenas.

The owner arena remains responsible for its local physical allocation.  This
module exports that allocation once, imports every requested peer allocation
once over a CPU control group, and binds non-owning Torch-NPU tensor aliases.
The request path only dereferences those aliases; it never imports or maps.

Teardown is deliberately collective.  Every rank fences local work, drops its
non-owning aliases, unmaps and closes imported handles, and then acknowledges
that imports are gone.  Only after that acknowledgement may each rank invoke
its owner-release callback.  This keeps exporter physical memory alive longer
than every imported mapping.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Protocol

from .c128_packed_acl_backend import (
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
    AclPhysicalMemoryHandle,
    AclRuntimeCallError,
    AclVmmRollbackError,
    PackedArenaAdapterUnavailable,
    _AclrtMemAccessDesc,
    _AclrtMemLocation,
    _configure_acl_signatures,
)
from .c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
    PackedArenaExportPin,
    PackedArenaLease,
    PackedArenaTensorBinding,
    PackedArenaTensorFactory,
)

ACL_SUCCESS = 0
ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT = 0x1
ACL_RT_VMM_EXPORT_FLAG_DEFAULT = 0
ACL_V2_SHAREABLE_HANDLE_BYTES = ctypes.sizeof(ctypes.c_uint64)

_REQUIRED_PEER_ACL_SYMBOLS = (
    "aclrtSetDevice",
    "aclrtDeviceGetBareTgid",
    "aclrtDeviceCanAccessPeer",
    "aclrtDeviceEnablePeerAccess",
    "aclrtDeviceDisablePeerAccess",
    "aclrtMemGetAllocationGranularity",
    "aclrtMemExportToShareableHandleV2",
    "aclrtMemSetPidToShareableHandleV2",
    "aclrtMemImportFromShareableHandleV2",
    "aclrtReserveMemAddress",
    "aclrtReleaseMemAddress",
    "aclrtMallocPhysical",
    "aclrtFreePhysical",
    "aclrtMapMem",
    "aclrtUnmapMem",
    "aclrtMemSetAccess",
    "aclrtMemset",
)


_AclrtMemDefaultHandle = ctypes.c_uint64


def _configure_peer_acl_signatures(library: object) -> None:
    """Configure the CANN 9.0 V2 ABI after the base VMM signatures."""
    _configure_acl_signatures(library)
    signatures: dict[str, tuple[list[object], object]] = {
        "aclrtDeviceGetBareTgid": (
            [ctypes.POINTER(ctypes.c_int32)],
            ctypes.c_int,
        ),
        "aclrtDeviceCanAccessPeer": (
            [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
                ctypes.c_int32,
            ],
            ctypes.c_int,
        ),
        "aclrtDeviceEnablePeerAccess": (
            [ctypes.c_int32, ctypes.c_uint32],
            ctypes.c_int,
        ),
        "aclrtDeviceDisablePeerAccess": (
            [ctypes.c_int32],
            ctypes.c_int,
        ),
        "aclrtMemExportToShareableHandleV2": (
            [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_void_p,
            ],
            ctypes.c_int,
        ),
        "aclrtMemSetPidToShareableHandleV2": (
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
            ],
            ctypes.c_int,
        ),
        "aclrtMemImportFromShareableHandleV2": (
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            ctypes.c_int,
        ),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype


def _load_peer_acl_library(
    library_path: str,
    loader: Callable[..., object],
) -> object:
    try:
        library = loader(library_path)
    except OSError as error:
        raise PackedArenaAdapterUnavailable(f"cannot load AscendCL library {library_path!r}: {error}") from error
    missing = [name for name in _REQUIRED_PEER_ACL_SYMBOLS if not hasattr(library, name)]
    if missing:
        raise PackedArenaAdapterUnavailable("AscendCL runtime is missing V2 peer-memory symbols: " + ", ".join(missing))
    _configure_peer_acl_signatures(library)
    return library


@dataclass(frozen=True)
class AclV2ShareableHandle:
    """Opaque CANN V2 handle safe to exchange through a CPU object group."""

    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.payload) != ACL_V2_SHAREABLE_HANDLE_BYTES:
            raise ValueError("V2 shareable handle must contain exactly " f"{ACL_V2_SHAREABLE_HANDLE_BYTES} bytes")

    def __repr__(self) -> str:
        return "AclV2ShareableHandle(" f"payload=<redacted {ACL_V2_SHAREABLE_HANDLE_BYTES} bytes>)"


@dataclass(frozen=True)
class AclImportedPhysicalMemoryHandle:
    """An imported ACL physical handle that must be closed by the importer."""

    value: int
    size_bytes: int
    device_index: int


class AclPeerAccessLease:
    """Refcounted worker-scoped peer-access capability lease."""

    def __init__(
        self,
        *,
        backend: AscendAclPackedPeerBackend,
        device_pairs: tuple[tuple[int, int], ...],
    ) -> None:
        self._backend: AscendAclPackedPeerBackend | None = backend
        self._remaining_pairs = list(reversed(device_pairs))

    @property
    def released(self) -> bool:
        return not self._remaining_pairs

    def release(self) -> None:
        if self._backend is None:
            return
        backend = self._backend
        failures: list[tuple[tuple[int, int], BaseException]] = []
        for pair in tuple(self._remaining_pairs):
            try:
                backend._release_peer_access_pair(pair)
                self._remaining_pairs.remove(pair)
            except BaseException as error:
                failures.append((pair, error))
        if failures:
            raise AclPeerAccessReleaseError(tuple(failures))
        self._backend = None


class AclPeerAccessReleaseError(RuntimeError):
    """One or more directed peer-access pairs could not be disabled."""

    def __init__(
        self,
        failures: tuple[tuple[tuple[int, int], BaseException], ...],
    ) -> None:
        self.failures = failures
        detail = "; ".join(f"{pair[0]}->{pair[1]}: {error}" for pair, error in failures)
        super().__init__(f"peer-access release failed: {detail}")


class AclPeerAccessAcquireError(RuntimeError):
    """Peer-access enable failed and its rollback remains retryable."""

    def __init__(
        self,
        *,
        cause: BaseException,
        rollback_error: AclPeerAccessReleaseError,
        lease: AclPeerAccessLease,
    ) -> None:
        self.cause = cause
        self.rollback_error = rollback_error
        self.lease = lease
        super().__init__(f"peer-access enable failed: {cause}; {rollback_error}")


class PeerAccessLease(Protocol):
    def release(self) -> None: ...


class PeerVmmBackend(Protocol):
    """Low-level V2 operations used by :class:`PackedVmmPeerLease`."""

    def bare_tgid(self, *, device_index: int) -> int: ...

    def acquire_peer_access(
        self,
        *,
        device_index: int,
        peer_device_indices: Sequence[int],
    ) -> PeerAccessLease: ...

    def export_v2(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> AclV2ShareableHandle: ...

    def authorize_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        bare_tgids: Sequence[int],
        device_index: int,
    ) -> None: ...

    def import_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        size_bytes: int,
        device_index: int,
    ) -> object: ...

    def reserve_alias(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int: ...

    def map_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        imported_handle: object,
        device_index: int,
    ) -> None: ...

    def unmap_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None: ...

    def close_imported(
        self,
        *,
        imported_handle: object,
        device_index: int,
    ) -> None: ...

    def release_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None: ...


class AscendAclPackedPeerBackend:
    """Concrete CANN 9.0 V2 backend for startup-only peer mappings.

    Constructing this adapter is the feature boundary: importing the module
    alone does not load AscendCL or select a device.
    """

    def __init__(
        self,
        *,
        library: object | None = None,
        library_path: str = "libascendcl.so",
        loader: Callable[..., object] = ctypes.CDLL,
    ) -> None:
        if library is None:
            self._library = _load_peer_acl_library(library_path, loader)
        else:
            missing = [name for name in _REQUIRED_PEER_ACL_SYMBOLS if not hasattr(library, name)]
            if missing:
                raise PackedArenaAdapterUnavailable(
                    "AscendCL runtime is missing V2 peer-memory symbols: " + ", ".join(missing)
                )
            _configure_peer_acl_signatures(library)
            self._library = library
        self._imported_handles: dict[int, AclImportedPhysicalMemoryHandle] = {}
        self._reserved_aliases: dict[int, tuple[int, int]] = {}
        self._mapped_aliases: dict[int, AclImportedPhysicalMemoryHandle] = {}
        # This backend is worker-scoped.  Reusing it across peer leases makes
        # each directed local->peer pair enable exactly once per process.
        self._peer_access_lock = threading.Lock()
        self._peer_access_refcounts: dict[tuple[int, int], int] = {}

    def _call(self, operation: str, *args: Any) -> None:
        result = getattr(self._library, operation)(*args)
        raw_status = getattr(result, "value", result)
        try:
            status = int(raw_status)
        except (TypeError, ValueError) as error:
            raise PackedArenaAdapterUnavailable(
                f"{operation} returned a non-integral ACL status: {result!r}"
            ) from error
        if status != ACL_SUCCESS:
            raise AclRuntimeCallError(operation, status)

    def _set_device(self, device_index: int) -> None:
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        self._call("aclrtSetDevice", device_index)

    @staticmethod
    def _validate_range(*, base_address: int, size_bytes: int) -> None:
        if size_bytes <= 0 or size_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("size_bytes must be a positive measured 2-MiB CANN " "granularity multiple")
        if base_address <= 0 or base_address % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("base_address must be aligned to the measured 2-MiB CANN " "VMM granularity")

    @staticmethod
    def _handle_buffer(
        shareable_handle: AclV2ShareableHandle,
    ) -> _AclrtMemDefaultHandle:
        return _AclrtMemDefaultHandle.from_buffer_copy(shareable_handle.payload)

    def bare_tgid(self, *, device_index: int) -> int:
        self._set_device(device_index)
        bare_tgid = ctypes.c_int32()
        self._call("aclrtDeviceGetBareTgid", ctypes.byref(bare_tgid))
        if bare_tgid.value <= 0:
            raise PackedArenaAdapterUnavailable("aclrtDeviceGetBareTgid returned a non-positive process id")
        return int(bare_tgid.value)

    def acquire_peer_access(
        self,
        *,
        device_index: int,
        peer_device_indices: Sequence[int],
    ) -> AclPeerAccessLease:
        peers = tuple(
            sorted({int(peer_device) for peer_device in peer_device_indices if int(peer_device) != device_index})
        )
        if device_index < 0 or any(peer_device < 0 for peer_device in peers):
            raise ValueError("peer-access device indices must be non-negative")
        acquired: list[tuple[int, int]] = []
        with self._peer_access_lock:
            try:
                for peer_device in peers:
                    pair = (device_index, peer_device)
                    refcount = self._peer_access_refcounts.get(pair, 0)
                    if refcount == 0:
                        self._set_device(device_index)
                        can_access = ctypes.c_int32()
                        self._call(
                            "aclrtDeviceCanAccessPeer",
                            ctypes.byref(can_access),
                            device_index,
                            peer_device,
                        )
                        if can_access.value != 1:
                            raise PackedArenaAdapterUnavailable(
                                "aclrtDeviceCanAccessPeer rejected directed "
                                f"device pair {device_index}->{peer_device}"
                            )
                        self._call(
                            "aclrtDeviceEnablePeerAccess",
                            peer_device,
                            0,
                        )
                    self._peer_access_refcounts[pair] = refcount + 1
                    acquired.append(pair)
            except BaseException as operation_error:
                rollback_failures: list[tuple[tuple[int, int], BaseException]] = []
                for pair in reversed(acquired):
                    try:
                        self._release_peer_access_pair_unlocked(pair)
                    except BaseException as rollback_error:
                        rollback_failures.append((pair, rollback_error))
                if rollback_failures:
                    retry_lease = AclPeerAccessLease(
                        backend=self,
                        device_pairs=tuple(reversed(tuple(pair for pair, _ in rollback_failures))),
                    )
                    raise AclPeerAccessAcquireError(
                        cause=operation_error,
                        rollback_error=AclPeerAccessReleaseError(tuple(rollback_failures)),
                        lease=retry_lease,
                    ) from operation_error
                raise
        return AclPeerAccessLease(
            backend=self,
            device_pairs=tuple(acquired),
        )

    def _release_peer_access_pair_unlocked(
        self,
        pair: tuple[int, int],
    ) -> None:
        local_device, peer_device = pair
        refcount = self._peer_access_refcounts.get(pair, 0)
        if refcount <= 0:
            raise RuntimeError(f"peer-access refcount underflow for {pair}")
        if refcount == 1:
            self._set_device(local_device)
            self._call("aclrtDeviceDisablePeerAccess", peer_device)
            del self._peer_access_refcounts[pair]
        else:
            self._peer_access_refcounts[pair] = refcount - 1

    def _release_peer_access_pair(
        self,
        pair: tuple[int, int],
    ) -> None:
        with self._peer_access_lock:
            self._release_peer_access_pair_unlocked(pair)

    def export_v2(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> AclV2ShareableHandle:
        if not isinstance(physical_handle, AclPhysicalMemoryHandle):
            raise TypeError("physical_handle must be an AclPhysicalMemoryHandle")
        if physical_handle.device_index != device_index:
            raise ValueError("physical handle belongs to another device")
        self._set_device(device_index)
        output = _AclrtMemDefaultHandle()
        self._call(
            "aclrtMemExportToShareableHandleV2",
            ctypes.c_void_p(physical_handle.value),
            ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            ctypes.byref(output),
        )
        return AclV2ShareableHandle(ctypes.string_at(ctypes.byref(output), ACL_V2_SHAREABLE_HANDLE_BYTES))

    def authorize_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        bare_tgids: Sequence[int],
        device_index: int,
    ) -> None:
        trusted = tuple(dict.fromkeys(int(pid) for pid in bare_tgids))
        if not trusted:
            return
        if any(pid <= 0 for pid in trusted):
            raise ValueError("bare TGIDs must be positive")
        self._set_device(device_index)
        handle_buffer = self._handle_buffer(shareable_handle)
        pid_buffer = (ctypes.c_int32 * len(trusted))(*trusted)
        self._call(
            "aclrtMemSetPidToShareableHandleV2",
            ctypes.byref(handle_buffer),
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            pid_buffer,
            len(trusted),
        )

    def import_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        size_bytes: int,
        device_index: int,
    ) -> AclImportedPhysicalMemoryHandle:
        self._validate_range(
            base_address=CANN_VMM_GRANULARITY_BYTES,
            size_bytes=size_bytes,
        )
        self._set_device(device_index)
        handle_buffer = self._handle_buffer(shareable_handle)
        imported = ctypes.c_void_p()
        self._call(
            "aclrtMemImportFromShareableHandleV2",
            ctypes.byref(handle_buffer),
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            0,
            ctypes.byref(imported),
        )
        if not imported.value:
            raise PackedArenaAdapterUnavailable("aclrtMemImportFromShareableHandleV2 returned a null handle")
        result = AclImportedPhysicalMemoryHandle(
            value=int(imported.value),
            size_bytes=size_bytes,
            device_index=device_index,
        )
        if result.value in self._imported_handles:
            raise PackedArenaAdapterUnavailable(
                "aclrtMemImportFromShareableHandleV2 returned an already " f"tracked handle: {result.value:#x}"
            )
        self._imported_handles[result.value] = result
        return result

    def reserve_alias(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        if alignment_bytes != CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("peer aliases require measured 2-MiB alignment")
        self._validate_range(
            base_address=alignment_bytes,
            size_bytes=size_bytes,
        )
        self._set_device(device_index)
        virtual_address = ctypes.c_void_p()
        self._call(
            "aclrtReserveMemAddress",
            ctypes.byref(virtual_address),
            size_bytes,
            0,
            None,
            0,
        )
        base_address = int(virtual_address.value or 0)
        if base_address <= 0 or base_address % alignment_bytes:
            invalid_address = ValueError(
                "aclrtReserveMemAddress returned a null or unaligned peer " f"address: {base_address:#x}"
            )
            if base_address:
                try:
                    self._call(
                        "aclrtReleaseMemAddress",
                        ctypes.c_void_p(base_address),
                    )
                except BaseException as rollback_error:
                    raise AclVmmRollbackError(
                        operation_error=invalid_address,
                        rollback_error=rollback_error,
                    ) from invalid_address
            raise invalid_address
        if base_address in self._reserved_aliases:
            raise PackedArenaAdapterUnavailable(
                "aclrtReserveMemAddress returned an already tracked peer " f"address: {base_address:#x}"
            )
        self._reserved_aliases[base_address] = (size_bytes, device_index)
        return base_address

    def map_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        imported_handle: object,
        device_index: int,
    ) -> None:
        self._validate_range(
            base_address=base_address,
            size_bytes=size_bytes,
        )
        if not isinstance(imported_handle, AclImportedPhysicalMemoryHandle):
            raise TypeError("imported_handle must be an AclImportedPhysicalMemoryHandle")
        if self._imported_handles.get(imported_handle.value) != imported_handle:
            raise ValueError("imported handle is not owned by this backend")
        if imported_handle.size_bytes != size_bytes or imported_handle.device_index != device_index:
            raise ValueError("imported handle size/device does not match alias")
        if self._reserved_aliases.get(base_address) != (
            size_bytes,
            device_index,
        ):
            raise ValueError("peer alias address is not reserved by this backend")
        if base_address in self._mapped_aliases:
            raise ValueError("peer alias address is already mapped")
        self._set_device(device_index)
        self._call(
            "aclrtMapMem",
            ctypes.c_void_p(base_address),
            size_bytes,
            0,
            ctypes.c_void_p(imported_handle.value),
            0,
        )
        self._mapped_aliases[base_address] = imported_handle
        access = _AclrtMemAccessDesc(
            flags=ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
            location=_AclrtMemLocation(
                id=device_index,
                type=ACL_MEM_LOCATION_TYPE_DEVICE,
            ),
        )
        try:
            self._call(
                "aclrtMemSetAccess",
                ctypes.c_void_p(base_address),
                size_bytes,
                ctypes.byref(access),
                1,
            )
        except BaseException as operation_error:
            try:
                self._call("aclrtUnmapMem", ctypes.c_void_p(base_address))
                del self._mapped_aliases[base_address]
            except BaseException as rollback_error:
                raise AclVmmRollbackError(
                    operation_error=operation_error,
                    rollback_error=rollback_error,
                ) from operation_error
            raise

    def unmap_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self._validate_range(
            base_address=base_address,
            size_bytes=size_bytes,
        )
        imported = self._mapped_aliases.get(base_address)
        if imported is None:
            raise ValueError("peer alias is not mapped")
        if imported.size_bytes != size_bytes or imported.device_index != device_index:
            raise ValueError("mapped peer alias size/device does not match")
        self._set_device(device_index)
        self._call("aclrtUnmapMem", ctypes.c_void_p(base_address))
        del self._mapped_aliases[base_address]

    def close_imported(
        self,
        *,
        imported_handle: object,
        device_index: int,
    ) -> None:
        if not isinstance(imported_handle, AclImportedPhysicalMemoryHandle):
            raise TypeError("imported_handle must be an AclImportedPhysicalMemoryHandle")
        if imported_handle.device_index != device_index:
            raise ValueError("imported handle belongs to another device")
        if self._imported_handles.get(imported_handle.value) != imported_handle:
            raise ValueError("imported handle is not owned by this backend")
        if imported_handle in self._mapped_aliases.values():
            raise RuntimeError("cannot close an imported handle that is mapped")
        self._set_device(device_index)
        self._call(
            "aclrtFreePhysical",
            ctypes.c_void_p(imported_handle.value),
        )
        del self._imported_handles[imported_handle.value]

    def release_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self._validate_range(
            base_address=base_address,
            size_bytes=size_bytes,
        )
        if self._reserved_aliases.get(base_address) != (
            size_bytes,
            device_index,
        ):
            raise ValueError("peer alias is not reserved by this backend")
        if base_address in self._mapped_aliases:
            raise RuntimeError("cannot release a mapped peer alias")
        self._set_device(device_index)
        self._call(
            "aclrtReleaseMemAddress",
            ctypes.c_void_p(base_address),
        )
        del self._reserved_aliases[base_address]


class PeerControlGroup(Protocol):
    """Bounded CPU object collectives used outside the request path."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_gather_object(self, value: object) -> tuple[object, ...]: ...

    def barrier(self, *, stage: str) -> None: ...


class TorchDistributedCpuControlGroup:
    """Adapter for an existing Gloo process group.

    The process group itself must be created with a bounded timeout.  The
    explicit monitored barriers add a bounded lifecycle acknowledgement and
    include failed ranks in the resulting PyTorch error.
    """

    def __init__(
        self,
        *,
        group: object,
        timeout_seconds: float = 30.0,
        dist_module: object | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._group = group
        self._timeout_seconds = timeout_seconds
        self._dist_module = dist_module
        self._validated = False

    def _dist(self) -> object:
        if self._dist_module is None:
            try:
                self._dist_module = importlib.import_module("torch.distributed")
            except ImportError as error:
                raise PackedArenaAdapterUnavailable(f"torch.distributed is unavailable: {error}") from error
        if not self._validated:
            backend = str(self._dist_module.get_backend(self._group)).lower()  # type: ignore[attr-defined]
            if "gloo" not in backend:
                raise ValueError(
                    "VMM peer-handle exchange requires the existing Gloo " f"CPU group, got backend {backend!r}"
                )
            if not callable(getattr(self._dist_module, "monitored_barrier", None)):
                raise PackedArenaAdapterUnavailable(
                    "torch.distributed.monitored_barrier is required for " "bounded peer-lease teardown"
                )
            self._validated = True
        return self._dist_module

    @property
    def rank(self) -> int:
        dist = self._dist()
        return int(dist.get_rank(self._group))  # type: ignore[attr-defined]

    @property
    def world_size(self) -> int:
        dist = self._dist()
        return int(dist.get_world_size(self._group))  # type: ignore[attr-defined]

    def all_gather_object(self, value: object) -> tuple[object, ...]:
        dist = self._dist()
        gathered: list[object | None] = [None] * self.world_size
        dist.all_gather_object(gathered, value, group=self._group)  # type: ignore[attr-defined]
        return tuple(gathered)

    def barrier(self, *, stage: str) -> None:
        if not stage:
            raise ValueError("barrier stage must be non-empty")
        dist = self._dist()
        dist.monitored_barrier(  # type: ignore[attr-defined]
            group=self._group,
            timeout=timedelta(seconds=self._timeout_seconds),
            wait_all_ranks=True,
        )


PACKED_VMM_PEER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PeerRankEndpoint:
    rank: int
    device_index: int
    bare_tgid: int
    lease_id: str
    schema_version: int
    allocation_schema: tuple[tuple[str, int], ...]
    metadata_fingerprint: str


@dataclass(frozen=True)
class PeerOwnedAllocation:
    """Owner allocation derived from one active arena export pin."""

    key: str
    size_bytes: int
    physical_handle: object = field(repr=False)


@dataclass(frozen=True)
class PeerHandleDescriptor:
    owner_rank: int
    owner_device_index: int
    key: str
    size_bytes: int
    shareable_handle: AclV2ShareableHandle = field(repr=False)

    def __repr__(self) -> str:
        return (
            "PeerHandleDescriptor("
            f"owner_rank={self.owner_rank}, "
            f"owner_device_index={self.owner_device_index}, "
            f"key={self.key!r}, size_bytes={self.size_bytes}, "
            "shareable_handle=<redacted>)"
        )


@dataclass(frozen=True)
class PeerAlias:
    owner_rank: int
    key: str
    base_address: int
    size_bytes: int


class PackedVmmPeerLeaseState(str, Enum):
    STARTING = "starting"
    OPEN = "open"
    CLOSING = "closing"
    CLEANUP_FAILED = "cleanup_failed"
    CLOSED = "closed"


class PackedVmmPeerLeaseClosedError(RuntimeError):
    """Raised when a non-open peer lease is dereferenced."""


@dataclass(frozen=True)
class PeerStartupStageResult:
    stage: str
    rank: int
    ok: bool
    payload: object = field(default=None, repr=False)
    error_type: str = ""
    error_message: str = ""


class PackedVmmPeerStartupError(RuntimeError):
    """A startup stage failed on at least one rank and was globally observed."""

    def __init__(
        self,
        *,
        stage: str,
        failures: tuple[PeerStartupStageResult, ...],
    ) -> None:
        self.stage = stage
        self.failures = failures
        detail = "; ".join(f"rank{failure.rank}:{failure.error_type}: {failure.error_message}" for failure in failures)
        super().__init__(f"packed VMM peer startup stage {stage!r} failed: {detail}")


class _PackedVmmPeerControlError(RuntimeError):
    def __init__(self, *, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"peer control exchange {stage!r} failed: {cause}")


@dataclass(frozen=True)
class PackedVmmPeerCleanupFailure:
    operation: str
    alias: str
    error: BaseException


class PackedVmmPeerCleanupError(RuntimeError):
    def __init__(
        self,
        failures: tuple[PackedVmmPeerCleanupFailure, ...],
    ) -> None:
        self.failures = failures
        detail = "; ".join(f"{failure.alias}:{failure.operation}: {failure.error}" for failure in failures)
        super().__init__(f"packed VMM peer cleanup failed: {detail}")


class PackedVmmPeerOpenError(RuntimeError):
    def __init__(
        self,
        *,
        cause: BaseException,
        cleanup_error: PackedVmmPeerCleanupError | None,
        lease: PackedVmmPeerLease,
    ) -> None:
        self.cause = cause
        self.cleanup_error = cleanup_error
        self.lease = lease
        detail = f"; {cleanup_error}" if cleanup_error is not None else ""
        super().__init__("packed VMM peer startup failed and coordinated teardown is " f"incomplete: {cause}{detail}")


@dataclass(frozen=True)
class PackedVmmPeerCounters:
    startup_control_exchange_calls: int
    startup_peer_access_acquire_calls: int
    startup_export_calls: int
    startup_authorize_calls: int
    startup_import_calls: int
    startup_reserve_calls: int
    startup_map_calls: int
    startup_bind_calls: int
    request_tensor_calls: int
    request_import_calls: int
    request_map_calls: int
    request_control_collective_calls: int
    request_fence_calls: int
    teardown_unmap_calls: int
    teardown_close_import_calls: int
    teardown_release_alias_calls: int
    teardown_peer_access_release_calls: int


@dataclass
class _PeerAliasAllocation:
    descriptor: PeerHandleDescriptor
    imported_handle: object | None = None
    base_address: int | None = None
    binding: PackedArenaTensorBinding | None = None
    mapped: bool = False
    address_reserved: bool = False

    @property
    def label(self) -> str:
        return f"rank{self.descriptor.owner_rank}/{self.descriptor.key}"


def _validate_endpoint_exchange(
    endpoints: Sequence[object],
    *,
    world_size: int,
    expected_schema: tuple[tuple[str, int], ...],
    expected_fingerprint: str,
    expected_lease_id: str,
) -> dict[int, PeerRankEndpoint]:
    if len(endpoints) != world_size:
        raise ValueError(f"endpoint exchange returned {len(endpoints)} ranks, expected " f"{world_size}")
    by_rank: dict[int, PeerRankEndpoint] = {}
    device_indices: set[int] = set()
    for value in endpoints:
        if not isinstance(value, PeerRankEndpoint):
            raise TypeError("endpoint exchange returned an unexpected object")
        if value.rank in by_rank:
            raise ValueError(f"duplicate peer endpoint rank {value.rank}")
        if not 0 <= value.rank < world_size:
            raise ValueError(f"peer endpoint rank {value.rank} is out of range")
        if value.device_index < 0 or value.bare_tgid <= 0:
            raise ValueError("peer endpoint has invalid device or bare TGID")
        if value.lease_id != expected_lease_id:
            raise ValueError(f"peer endpoint rank {value.rank} lease id does not match " "the startup generation")
        if value.device_index in device_indices:
            raise ValueError(f"duplicate peer endpoint device index {value.device_index}")
        if value.schema_version != PACKED_VMM_PEER_SCHEMA_VERSION:
            raise ValueError(
                f"peer endpoint rank {value.rank} uses schema version "
                f"{value.schema_version}, expected {PACKED_VMM_PEER_SCHEMA_VERSION}"
            )
        if value.allocation_schema != expected_schema:
            raise ValueError(
                f"peer endpoint rank {value.rank} allocation key/size schema " "does not match the local arena"
            )
        if value.metadata_fingerprint != expected_fingerprint:
            raise ValueError(
                f"peer endpoint rank {value.rank} packed-plan fingerprint " "does not match the local arena"
            )
        device_indices.add(value.device_index)
        by_rank[value.rank] = value
    if set(by_rank) != set(range(world_size)):
        raise ValueError("endpoint exchange did not cover every rank")
    return by_rank


def _flatten_handle_exchange(
    batches: Sequence[object],
    *,
    world_size: int,
    endpoints: dict[int, PeerRankEndpoint],
    expected_schema: tuple[tuple[str, int], ...],
) -> tuple[PeerHandleDescriptor, ...]:
    if len(batches) != world_size:
        raise ValueError(f"handle exchange returned {len(batches)} ranks, expected " f"{world_size}")
    descriptors: list[PeerHandleDescriptor] = []
    seen: set[tuple[int, str]] = set()
    for expected_rank, batch in enumerate(batches):
        if not isinstance(batch, tuple):
            raise TypeError("handle exchange batches must be tuples")
        batch_schema = tuple(
            (value.key, value.size_bytes) for value in batch if isinstance(value, PeerHandleDescriptor)
        )
        if len(batch_schema) != len(batch) or batch_schema != expected_schema:
            raise ValueError(f"rank {expected_rank} handle key/size schema does not match " "the endpoint exchange")
        for value in batch:
            if not isinstance(value, PeerHandleDescriptor):
                raise TypeError("handle exchange returned an unexpected object")
            if value.owner_rank != expected_rank:
                raise ValueError("handle exchange batch owner does not match its control " "group rank")
            if value.owner_device_index != endpoints[expected_rank].device_index:
                raise ValueError("handle descriptor device does not match its endpoint")
            identity = (value.owner_rank, value.key)
            if identity in seen:
                raise ValueError(f"duplicate peer handle descriptor {identity}")
            if not value.key:
                raise ValueError("peer handle key must be non-empty")
            if value.size_bytes <= 0 or value.size_bytes % CANN_VMM_GRANULARITY_BYTES:
                raise ValueError("peer handle size must be 2-MiB aligned")
            if not isinstance(value.shareable_handle, AclV2ShareableHandle):
                raise TypeError("peer handle descriptor has an invalid V2 handle")
            seen.add(identity)
            descriptors.append(value)
    return tuple(descriptors)


def _validate_stage_exchange(
    values: Sequence[object],
    *,
    stage: str,
    world_size: int,
) -> tuple[PeerStartupStageResult, ...]:
    if len(values) != world_size:
        raise ValueError(f"startup stage {stage!r} returned {len(values)} ranks, expected " f"{world_size}")
    reports: list[PeerStartupStageResult] = []
    for expected_rank, value in enumerate(values):
        if not isinstance(value, PeerStartupStageResult):
            raise TypeError(f"startup stage {stage!r} returned an unexpected object")
        if value.stage != stage or value.rank != expected_rank:
            raise ValueError(f"startup stage {stage!r} report identity does not match rank " f"{expected_rank}")
        reports.append(value)
    return tuple(reports)


def _arena_metadata_fingerprint(
    *,
    owner_arena: PackedArenaLease,
    allocation_schema: tuple[tuple[str, int], ...],
) -> str:
    plan = owner_arena.plan
    canonical_groups = tuple(
        (
            group.name,
            group.logical_blocks,
            tuple(
                (
                    component.name,
                    component.bucket,
                    component.page_size_bytes,
                    component.copies,
                    component.placement.value,
                    component.allocation_granularity_bytes,
                )
                for component in group.components
            ),
        )
        for group in plan.groups
    )
    canonical_scratch = tuple(
        (
            scratch.bucket,
            scratch.page_size_bytes,
            scratch.max_pages_per_rank,
            scratch.allocation_granularity_bytes,
        )
        for scratch in plan.scratch
    )
    canonical = repr(
        (
            PACKED_VMM_PEER_SCHEMA_VERSION,
            owner_arena.alignment_bytes,
            plan.global_block_capacity,
            plan.tp_size,
            canonical_groups,
            canonical_scratch,
            allocation_schema,
        )
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class PackedVmmPeerLease:
    """Collective lease over startup-imported packed C128 peer aliases."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device_index: int,
        backend: PeerVmmBackend,
        tensor_factory: PackedArenaTensorFactory,
        control: PeerControlGroup,
        fence: Callable[[], None],
        owner_arena: PackedArenaLease,
        owner_export_pin: PackedArenaExportPin,
        allocation_schema: tuple[tuple[str, int], ...],
        metadata_fingerprint: str,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device_index = device_index
        self._backend = backend
        self._tensor_factory = tensor_factory
        self._control = control
        self._fence = fence
        self._owner_arena = owner_arena
        self._owner_export_pin = owner_export_pin
        self._allocation_schema = allocation_schema
        self._metadata_fingerprint = metadata_fingerprint
        self._lease_id = ""
        self._state = PackedVmmPeerLeaseState.STARTING
        self._aliases: dict[tuple[int, str], _PeerAliasAllocation] = {}
        self._peer_access_lease: PeerAccessLease | None = None
        self._fenced = False
        self._imports_released = False
        self._owner_export_pin_released = False
        self._counter_values: dict[str, int] = {
            field_name: 0 for field_name in PackedVmmPeerCounters.__dataclass_fields__
        }

    @classmethod
    def open(
        cls,
        *,
        owner_arena: PackedArenaLease,
        shared_buckets: Sequence[str],
        backend: PeerVmmBackend,
        tensor_factory: PackedArenaTensorFactory,
        control: PeerControlGroup,
        fence: Callable[[], None],
    ) -> PackedVmmPeerLease:
        rank = control.rank
        world_size = control.world_size
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("control group rank/world size is invalid")
        if owner_arena.plan.tp_size != world_size:
            raise ValueError(
                f"owner arena TP size {owner_arena.plan.tp_size} does not " f"match control world size {world_size}"
            )
        if owner_arena.tp_rank != rank:
            raise ValueError(f"owner arena TP rank {owner_arena.tp_rank} does not match " f"control rank {rank}")
        owner_export_pin = owner_arena.acquire_export_pin()
        try:
            export_allocations = owner_export_pin.allocations()
        except BaseException:
            owner_export_pin.release()
            raise
        requested_buckets = tuple(shared_buckets)
        try:
            if not requested_buckets:
                raise ValueError("shared_buckets must not be empty")
            if len(requested_buckets) != len(set(requested_buckets)):
                raise ValueError("shared_buckets must be unique")
            if any(not bucket for bucket in requested_buckets):
                raise ValueError("shared bucket names must be non-empty")
        except BaseException:
            owner_export_pin.release()
            raise
        by_bucket = {allocation.bucket: allocation for allocation in export_allocations}
        missing_buckets = tuple(bucket for bucket in requested_buckets if bucket not in by_bucket)
        if missing_buckets:
            owner_export_pin.release()
            raise ValueError(f"owner arena does not contain shared buckets {missing_buckets!r}")
        selected_allocations = tuple(by_bucket[bucket] for bucket in requested_buckets)
        allocations = tuple(
            PeerOwnedAllocation(
                key=allocation.bucket,
                size_bytes=allocation.size_bytes,
                physical_handle=allocation.physical_handle,
            )
            for allocation in selected_allocations
        )
        allocations = tuple(sorted(allocations, key=lambda value: value.key))
        keys = [allocation.key for allocation in allocations]
        try:
            if not allocations:
                raise ValueError("owner arena has no exportable allocations")
            if len(keys) != len(set(keys)):
                raise ValueError("owned peer-allocation keys must be unique")
            for allocation in allocations:
                if not allocation.key:
                    raise ValueError("owned peer-allocation key must be non-empty")
                if allocation.size_bytes <= 0 or allocation.size_bytes % CANN_VMM_GRANULARITY_BYTES:
                    raise ValueError("owned peer-allocation size must be 2-MiB aligned")
                if not isinstance(allocation.physical_handle, AclPhysicalMemoryHandle):
                    raise TypeError("owned peer allocation requires AclPhysicalMemoryHandle")
                if allocation.physical_handle.device_index != owner_arena.device_index:
                    raise ValueError("owner allocation belongs to another device")
                if allocation.physical_handle.size_bytes != allocation.size_bytes:
                    raise ValueError("owner allocation size does not match its physical handle")
        except BaseException:
            owner_export_pin.release()
            raise

        device_index = owner_arena.device_index
        allocation_schema = tuple((allocation.key, allocation.size_bytes) for allocation in allocations)
        metadata_fingerprint = _arena_metadata_fingerprint(
            owner_arena=owner_arena,
            allocation_schema=allocation_schema,
        )
        lease = cls(
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            backend=backend,
            tensor_factory=tensor_factory,
            control=control,
            fence=fence,
            owner_arena=owner_arena,
            owner_export_pin=owner_export_pin,
            allocation_schema=allocation_schema,
            metadata_fingerprint=metadata_fingerprint,
        )
        try:
            lease._startup(allocations)
        except BaseException as error:
            if isinstance(error, _PackedVmmPeerControlError):
                cleanup_error = lease._cleanup_after_lost_control()
                raise PackedVmmPeerOpenError(
                    cause=error,
                    cleanup_error=cleanup_error,
                    lease=lease,
                ) from error
            try:
                lease._coordinated_startup_rollback(cause=error)
            except PackedVmmPeerOpenError:
                raise
            raise
        return lease

    @property
    def state(self) -> PackedVmmPeerLeaseState:
        return self._state

    @property
    def counters(self) -> PackedVmmPeerCounters:
        return PackedVmmPeerCounters(**self._counter_values)

    @property
    def aliases(self) -> tuple[PeerAlias, ...]:
        self._require_open()
        return tuple(
            PeerAlias(
                owner_rank=allocation.descriptor.owner_rank,
                key=allocation.descriptor.key,
                base_address=int(allocation.base_address or 0),
                size_bytes=allocation.descriptor.size_bytes,
            )
            for allocation in self._aliases.values()
        )

    def _increment(self, name: str) -> None:
        self._counter_values[name] += 1

    def _import_v2(
        self,
        *,
        descriptor: PeerHandleDescriptor,
    ) -> object:
        counter = "startup_import_calls" if self._state is PackedVmmPeerLeaseState.STARTING else "request_import_calls"
        self._increment(counter)
        return self._backend.import_v2(
            shareable_handle=descriptor.shareable_handle,
            size_bytes=descriptor.size_bytes,
            device_index=self.device_index,
        )

    def _map_alias(
        self,
        *,
        allocation: _PeerAliasAllocation,
    ) -> None:
        assert allocation.base_address is not None
        assert allocation.imported_handle is not None
        counter = "startup_map_calls" if self._state is PackedVmmPeerLeaseState.STARTING else "request_map_calls"
        self._increment(counter)
        self._backend.map_alias(
            base_address=allocation.base_address,
            size_bytes=allocation.descriptor.size_bytes,
            imported_handle=allocation.imported_handle,
            device_index=self.device_index,
        )

    def _require_open(self) -> None:
        if self._state is not PackedVmmPeerLeaseState.OPEN:
            raise PackedVmmPeerLeaseClosedError(f"packed VMM peer lease is {self._state.value}")

    def _exchange_startup_stage(
        self,
        *,
        stage: str,
        operation: Callable[[], object],
    ) -> tuple[PeerStartupStageResult, ...]:
        try:
            payload = operation()
            local = PeerStartupStageResult(
                stage=stage,
                rank=self.rank,
                ok=True,
                payload=payload,
            )
        except BaseException as error:
            local = PeerStartupStageResult(
                stage=stage,
                rank=self.rank,
                ok=False,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        self._increment("startup_control_exchange_calls")
        try:
            gathered = self._control.all_gather_object(local)
        except BaseException as error:
            raise _PackedVmmPeerControlError(stage=stage, cause=error) from error
        reports = _validate_stage_exchange(
            gathered,
            stage=stage,
            world_size=self.world_size,
        )
        failures = tuple(report for report in reports if not report.ok)
        if failures:
            raise PackedVmmPeerStartupError(stage=stage, failures=failures)
        return reports

    def _startup(
        self,
        allocations: tuple[PeerOwnedAllocation, ...],
    ) -> None:
        lease_id_reports = self._exchange_startup_stage(
            stage="lease_id",
            operation=lambda: secrets.token_hex(16) if self.rank == 0 else None,
        )
        rank_zero_lease_id = lease_id_reports[0].payload
        if not isinstance(rank_zero_lease_id, str) or not rank_zero_lease_id:
            raise ValueError("rank zero returned an invalid peer lease id")
        if any(report.payload is not None for report in lease_id_reports[1:]):
            raise ValueError("non-zero rank attempted to choose the peer lease id")
        self._lease_id = rank_zero_lease_id

        def make_endpoint() -> PeerRankEndpoint:
            return PeerRankEndpoint(
                rank=self.rank,
                device_index=self.device_index,
                bare_tgid=self._backend.bare_tgid(device_index=self.device_index),
                lease_id=self._lease_id,
                schema_version=PACKED_VMM_PEER_SCHEMA_VERSION,
                allocation_schema=self._allocation_schema,
                metadata_fingerprint=self._metadata_fingerprint,
            )

        endpoint_reports = self._exchange_startup_stage(
            stage="endpoint",
            operation=make_endpoint,
        )
        endpoints = _validate_endpoint_exchange(
            tuple(report.payload for report in endpoint_reports),
            world_size=self.world_size,
            expected_schema=self._allocation_schema,
            expected_fingerprint=self._metadata_fingerprint,
            expected_lease_id=self._lease_id,
        )
        peer_tgids = tuple(endpoint.bare_tgid for rank, endpoint in endpoints.items() if rank != self.rank)

        def enable_and_export() -> tuple[PeerHandleDescriptor, ...]:
            try:
                self._peer_access_lease = self._backend.acquire_peer_access(
                    device_index=self.device_index,
                    peer_device_indices=tuple(
                        endpoint.device_index for rank, endpoint in endpoints.items() if rank != self.rank
                    ),
                )
            except AclPeerAccessAcquireError as error:
                self._peer_access_lease = error.lease
                raise
            self._increment("startup_peer_access_acquire_calls")
            local_descriptors: list[PeerHandleDescriptor] = []
            for allocation in allocations:
                shareable_handle = self._backend.export_v2(
                    physical_handle=allocation.physical_handle,
                    device_index=self.device_index,
                )
                self._increment("startup_export_calls")
                if peer_tgids:
                    self._backend.authorize_v2(
                        shareable_handle=shareable_handle,
                        bare_tgids=peer_tgids,
                        device_index=self.device_index,
                    )
                    self._increment("startup_authorize_calls")
                local_descriptors.append(
                    PeerHandleDescriptor(
                        owner_rank=self.rank,
                        owner_device_index=self.device_index,
                        key=allocation.key,
                        size_bytes=allocation.size_bytes,
                        shareable_handle=shareable_handle,
                    )
                )
            return tuple(local_descriptors)

        handle_reports = self._exchange_startup_stage(
            stage="enable_export_authorize",
            operation=enable_and_export,
        )
        descriptors = _flatten_handle_exchange(
            tuple(report.payload for report in handle_reports),
            world_size=self.world_size,
            endpoints=endpoints,
            expected_schema=self._allocation_schema,
        )

        def import_and_bind() -> None:
            for descriptor in descriptors:
                if descriptor.owner_rank == self.rank:
                    continue
                self._open_alias(descriptor)

        self._exchange_startup_stage(
            stage="import_map_bind_ready",
            operation=import_and_bind,
        )
        self._state = PackedVmmPeerLeaseState.OPEN

    def _open_alias(self, descriptor: PeerHandleDescriptor) -> None:
        identity = (descriptor.owner_rank, descriptor.key)
        if identity in self._aliases:
            raise ValueError(f"peer alias {identity} is already open")
        allocation = _PeerAliasAllocation(descriptor=descriptor)
        self._aliases[identity] = allocation
        allocation.imported_handle = self._import_v2(descriptor=descriptor)
        allocation.base_address = self._backend.reserve_alias(
            size_bytes=descriptor.size_bytes,
            alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
            device_index=self.device_index,
        )
        allocation.address_reserved = True
        self._increment("startup_reserve_calls")
        try:
            self._map_alias(allocation=allocation)
        except AclVmmRollbackError:
            # ``map_alias`` only emits this composite error when aclrtMapMem
            # succeeded and the aclrtMemSetAccess rollback unmap failed.  Keep
            # that retained mapping visible so coordinated teardown can retry
            # the unmap before closing the imported physical handle.
            allocation.mapped = True
            raise
        allocation.mapped = True
        allocation.binding = self._tensor_factory.bind(
            base_address=allocation.base_address,
            size_bytes=descriptor.size_bytes,
            device_index=self.device_index,
        )
        self._increment("startup_bind_calls")

    def tensor(self, *, owner_rank: int, key: str) -> object:
        """Return a pre-bound alias without any import or map operation."""
        self._require_open()
        try:
            allocation = self._aliases[(owner_rank, key)]
        except KeyError as error:
            raise ValueError(f"unknown peer alias rank{owner_rank}/{key}") from error
        if allocation.binding is None:
            raise RuntimeError("peer alias has no tensor binding")
        self._increment("request_tensor_calls")
        # These counters remain explicit so production logs can prove the
        # request path did not regress into lazy import or lazy map behavior.
        return allocation.binding.tensor()

    def _cleanup_local_aliases(self) -> None:
        failures: list[PackedVmmPeerCleanupFailure] = []
        if not self._fenced and any(allocation.mapped for allocation in self._aliases.values()):
            try:
                self._fence()
                self._fenced = True
            except BaseException as error:
                self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
                raise PackedVmmPeerCleanupError(
                    (
                        PackedVmmPeerCleanupFailure(
                            operation="fence",
                            alias="*",
                            error=error,
                        ),
                    )
                ) from error

        for allocation in reversed(tuple(self._aliases.values())):
            if allocation.binding is not None:
                try:
                    allocation.binding.close()
                    allocation.binding = None
                except BaseException as error:
                    failures.append(
                        PackedVmmPeerCleanupFailure(
                            operation="drop_tensor_alias",
                            alias=allocation.label,
                            error=error,
                        )
                    )
                    continue

            if allocation.mapped:
                try:
                    assert allocation.base_address is not None
                    self._backend.unmap_alias(
                        base_address=allocation.base_address,
                        size_bytes=allocation.descriptor.size_bytes,
                        device_index=self.device_index,
                    )
                    allocation.mapped = False
                    self._increment("teardown_unmap_calls")
                except BaseException as error:
                    failures.append(
                        PackedVmmPeerCleanupFailure(
                            operation="unmap_alias",
                            alias=allocation.label,
                            error=error,
                        )
                    )
                    continue

            if allocation.imported_handle is not None:
                try:
                    self._backend.close_imported(
                        imported_handle=allocation.imported_handle,
                        device_index=self.device_index,
                    )
                    allocation.imported_handle = None
                    self._increment("teardown_close_import_calls")
                except BaseException as error:
                    failures.append(
                        PackedVmmPeerCleanupFailure(
                            operation="close_imported_handle",
                            alias=allocation.label,
                            error=error,
                        )
                    )
                    continue

            if allocation.address_reserved:
                try:
                    assert allocation.base_address is not None
                    self._backend.release_alias(
                        base_address=allocation.base_address,
                        size_bytes=allocation.descriptor.size_bytes,
                        device_index=self.device_index,
                    )
                    allocation.address_reserved = False
                    self._increment("teardown_release_alias_calls")
                except BaseException as error:
                    failures.append(
                        PackedVmmPeerCleanupFailure(
                            operation="release_alias_address",
                            alias=allocation.label,
                            error=error,
                        )
                    )

        if failures:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            raise PackedVmmPeerCleanupError(tuple(failures))

    def _release_peer_access(self) -> None:
        if self._peer_access_lease is None:
            return
        self._increment("teardown_peer_access_release_calls")
        self._peer_access_lease.release()
        self._peer_access_lease = None

    def _cleanup_local_resources(self) -> None:
        try:
            self._cleanup_local_aliases()
        except PackedVmmPeerCleanupError as error:
            raise error
        try:
            self._release_peer_access()
        except BaseException as error:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            raise PackedVmmPeerCleanupError(
                (
                    PackedVmmPeerCleanupFailure(
                        operation="release_peer_access",
                        alias="*",
                        error=error,
                    ),
                )
            ) from error

    def _release_export_pin_after_ack(self) -> None:
        if not self._owner_export_pin_released:
            self._owner_export_pin.release()
            self._owner_export_pin_released = True

    def commit_consumer_views(
        self,
        error: BaseException | None = None,
    ) -> None:
        """Collectively commit or reject post-map consumer construction.

        Every rank calls this exactly once after peer open.  A rank whose
        local cache reshape/materializer construction failed reports that
        failure here, ensuring peers do not publish while it enters teardown.
        """
        self._require_open()
        local = PeerStartupStageResult(
            stage="consumer_views",
            rank=self.rank,
            ok=error is None,
            error_type="" if error is None else type(error).__name__,
            error_message="" if error is None else str(error),
        )
        self._increment("startup_control_exchange_calls")
        try:
            gathered = self._control.all_gather_object(local)
            reports = _validate_stage_exchange(
                gathered,
                stage="consumer_views",
                world_size=self.world_size,
            )
        except BaseException as control_error:
            raise _PackedVmmPeerControlError(
                stage="consumer_views",
                cause=control_error,
            ) from control_error
        failures = tuple(report for report in reports if not report.ok)
        if failures:
            raise PackedVmmPeerStartupError(
                stage="consumer_views",
                failures=failures,
            )

    def _cleanup_after_lost_control(self) -> PackedVmmPeerCleanupError | None:
        try:
            self._cleanup_local_resources()
        except PackedVmmPeerCleanupError as cleanup_error:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            return cleanup_error
        self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
        return None

    def _coordinated_startup_rollback(self, *, cause: BaseException) -> None:
        local_cleanup_error: PackedVmmPeerCleanupError | None = None
        try:
            self._cleanup_local_resources()
            local_report = PeerStartupStageResult(
                stage="startup_rollback",
                rank=self.rank,
                ok=True,
            )
        except PackedVmmPeerCleanupError as error:
            local_cleanup_error = error
            local_report = PeerStartupStageResult(
                stage="startup_rollback",
                rank=self.rank,
                ok=False,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        self._increment("startup_control_exchange_calls")
        try:
            gathered = self._control.all_gather_object(local_report)
            reports = _validate_stage_exchange(
                gathered,
                stage="startup_rollback",
                world_size=self.world_size,
            )
        except BaseException as control_error:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            raise PackedVmmPeerOpenError(
                cause=_PackedVmmPeerControlError(
                    stage="startup_rollback",
                    cause=control_error,
                ),
                cleanup_error=local_cleanup_error,
                lease=self,
            ) from cause
        failed_reports = tuple(report for report in reports if not report.ok)
        if failed_reports:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            remote_failure = PackedVmmPeerCleanupError(
                tuple(
                    PackedVmmPeerCleanupFailure(
                        operation="startup_rollback",
                        alias=f"rank{report.rank}",
                        error=RuntimeError(f"{report.error_type}: {report.error_message}"),
                    )
                    for report in failed_reports
                )
            )
            raise PackedVmmPeerOpenError(
                cause=cause,
                cleanup_error=local_cleanup_error or remote_failure,
                lease=self,
            ) from cause

        self._imports_released = True
        try:
            self._release_export_pin_after_ack()
        except BaseException as owner_error:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            cleanup_error = PackedVmmPeerCleanupError(
                (
                    PackedVmmPeerCleanupFailure(
                        operation="release_owner_after_startup_rollback",
                        alias="*",
                        error=owner_error,
                    ),
                )
            )
            raise PackedVmmPeerOpenError(
                cause=cause,
                cleanup_error=cleanup_error,
                lease=self,
            ) from cause
        self._state = PackedVmmPeerLeaseState.CLOSED

    def close(self) -> None:
        if self._state is PackedVmmPeerLeaseState.CLOSED:
            return
        self._state = PackedVmmPeerLeaseState.CLOSING
        self._cleanup_local_resources()
        if not self._imports_released:
            try:
                self._control.barrier(stage="peer_imports_released")
                self._imports_released = True
            except BaseException as error:
                self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
                raise PackedVmmPeerCleanupError(
                    (
                        PackedVmmPeerCleanupFailure(
                            operation="peer_imports_released_barrier",
                            alias="*",
                            error=error,
                        ),
                    )
                ) from error
        try:
            self._release_export_pin_after_ack()
        except BaseException as error:
            self._state = PackedVmmPeerLeaseState.CLEANUP_FAILED
            raise PackedVmmPeerCleanupError(
                (
                    PackedVmmPeerCleanupFailure(
                        operation="release_owner_after_peer_ack",
                        alias="*",
                        error=error,
                    ),
                )
            ) from error
        self._state = PackedVmmPeerLeaseState.CLOSED

    def __enter__(self) -> PackedVmmPeerLease:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


def maybe_create_packed_vmm_peer_lease(
    *,
    enabled: bool,
    owner_arena: PackedArenaLease | None = None,
    shared_buckets: Sequence[str] | None = None,
    backend: PeerVmmBackend | None = None,
    tensor_factory: PackedArenaTensorFactory | None = None,
    control: PeerControlGroup | None = None,
    fence: Callable[[], None] | None = None,
) -> PackedVmmPeerLease | None:
    """Open peer aliases only after the explicit feature gate is enabled."""
    if not enabled:
        return None
    if owner_arena is None:
        raise ValueError("owner_arena is required when peer VMM is enabled")
    if shared_buckets is None:
        raise ValueError("shared_buckets is required when peer VMM is enabled")
    if backend is None:
        raise ValueError("backend is required when peer VMM is enabled")
    if tensor_factory is None:
        raise ValueError("tensor_factory is required when peer VMM is enabled")
    if control is None:
        raise ValueError("CPU control group is required when peer VMM is enabled")
    if fence is None:
        raise ValueError("stream fence is required when peer VMM is enabled")
    return PackedVmmPeerLease.open(
        owner_arena=owner_arena,
        shared_buckets=shared_buckets,
        backend=backend,
        tensor_factory=tensor_factory,
        control=control,
        fence=fence,
    )
