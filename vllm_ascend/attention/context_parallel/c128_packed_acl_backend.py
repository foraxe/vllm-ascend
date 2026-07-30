# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Lazy AscendCL backend for the packed C128 arena.

Importing this module does not load AscendCL or select an NPU.  Construct the
backend only inside the packed-arena feature gate.  The backend deliberately
wraps the public CANN 9.0 ``aclrtMem*`` API instead of allocator internals.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .c128_packed_arena import CANN_VMM_GRANULARITY_BYTES

ACL_SUCCESS = 0
ACL_HBM_MEM_NORMAL = 5
ACL_MEM_LOCATION_TYPE_DEVICE = 1
ACL_MEM_ALLOCATION_TYPE_PINNED = 0
ACL_MEM_HANDLE_TYPE_NONE = 0
ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM = 0
ACL_RT_MEM_ACCESS_FLAGS_READWRITE = 0x3

_REQUIRED_ACL_SYMBOLS = (
    "aclrtSetDevice",
    "aclrtMemGetAllocationGranularity",
    "aclrtReserveMemAddress",
    "aclrtReleaseMemAddress",
    "aclrtMallocPhysical",
    "aclrtFreePhysical",
    "aclrtMapMem",
    "aclrtUnmapMem",
    "aclrtMemSetAccess",
    "aclrtMemset",
)


class PackedArenaAdapterUnavailable(RuntimeError):
    """The installed runtime cannot provide the requested packed-arena seam."""


class AclRuntimeCallError(RuntimeError):
    """An AscendCL operation returned a non-success status."""

    def __init__(self, operation: str, status: int) -> None:
        self.operation = operation
        self.status = status
        super().__init__(f"{operation} failed with aclError={status}")


class AclVmmRollbackError(RuntimeError):
    """A VMM operation and its immediate failure rollback both failed."""

    def __init__(
        self,
        *,
        operation_error: BaseException,
        rollback_error: BaseException,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_error = rollback_error
        super().__init__(
            f"Ascend VMM operation failed and rollback was incomplete: {operation_error}; rollback: {rollback_error}"
        )


class _AclrtMemLocation(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_int),
    ]


class _AclrtPhysicalMemProp(ctypes.Structure):
    _fields_ = [
        ("handleType", ctypes.c_int),
        ("allocationType", ctypes.c_int),
        ("memAttr", ctypes.c_int),
        ("location", _AclrtMemLocation),
        ("reserve", ctypes.c_uint64),
    ]


class _AclrtMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_int),
        ("location", _AclrtMemLocation),
        ("rsv", ctypes.c_uint8 * 12),
    ]


@dataclass(frozen=True)
class AclPhysicalMemoryHandle:
    """A locally allocated ACL physical-memory handle and its contract."""

    value: int
    size_bytes: int
    device_index: int


def _physical_memory_properties(device_index: int) -> _AclrtPhysicalMemProp:
    return _AclrtPhysicalMemProp(
        handleType=ACL_MEM_HANDLE_TYPE_NONE,
        allocationType=ACL_MEM_ALLOCATION_TYPE_PINNED,
        memAttr=ACL_HBM_MEM_NORMAL,
        location=_AclrtMemLocation(
            id=device_index,
            type=ACL_MEM_LOCATION_TYPE_DEVICE,
        ),
        reserve=0,
    )


def _configure_acl_signatures(library: object) -> None:
    signatures: dict[str, tuple[list[object], object]] = {
        "aclrtSetDevice": ([ctypes.c_int32], ctypes.c_int),
        "aclrtMemGetAllocationGranularity": (
            [
                ctypes.POINTER(_AclrtPhysicalMemProp),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_size_t),
            ],
            ctypes.c_int,
        ),
        "aclrtReserveMemAddress": (
            [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint64,
            ],
            ctypes.c_int,
        ),
        "aclrtReleaseMemAddress": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMallocPhysical": (
            [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.POINTER(_AclrtPhysicalMemProp),
                ctypes.c_uint64,
            ],
            ctypes.c_int,
        ),
        "aclrtFreePhysical": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMapMem": (
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint64,
            ],
            ctypes.c_int,
        ),
        "aclrtUnmapMem": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMemSetAccess": (
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(_AclrtMemAccessDesc),
                ctypes.c_size_t,
            ],
            ctypes.c_int,
        ),
        "aclrtMemset": (
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int32,
                ctypes.c_size_t,
            ],
            ctypes.c_int,
        ),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype


def _load_acl_library(
    library_path: str,
    loader: Callable[..., object],
) -> object:
    try:
        library = loader(library_path)
    except OSError as error:
        raise PackedArenaAdapterUnavailable(f"cannot load AscendCL library {library_path!r}: {error}") from error
    missing = [name for name in _REQUIRED_ACL_SYMBOLS if not hasattr(library, name)]
    if missing:
        raise PackedArenaAdapterUnavailable("AscendCL runtime is missing packed-arena symbols: " + ", ".join(missing))
    _configure_acl_signatures(library)
    return library


class AscendAclPackedArenaBackend:
    """Concrete ``PackedArenaBackend`` over CANN 9.0 AscendCL VMM."""

    def __init__(
        self,
        *,
        library: object | None = None,
        library_path: str = "libascendcl.so",
        loader: Callable[..., object] = ctypes.CDLL,
    ) -> None:
        if library is None:
            self._library = _load_acl_library(library_path, loader)
        else:
            missing = [name for name in _REQUIRED_ACL_SYMBOLS if not hasattr(library, name)]
            if missing:
                raise PackedArenaAdapterUnavailable(
                    "AscendCL runtime is missing packed-arena symbols: " + ", ".join(missing)
                )
            _configure_acl_signatures(library)
            self._library = library
        self._reserved_addresses: dict[int, tuple[int, int]] = {}
        self._allocated_handles: dict[int, AclPhysicalMemoryHandle] = {}
        self._mapped_handles_by_address: dict[int, AclPhysicalMemoryHandle] = {}

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
    def _validate_size(size_bytes: int) -> None:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if size_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError(f"size_bytes must be a {CANN_VMM_GRANULARITY_BYTES}-byte multiple")

    @classmethod
    def _validate_range(
        cls,
        *,
        base_address: int,
        size_bytes: int,
    ) -> None:
        cls._validate_size(size_bytes)
        if base_address <= 0:
            raise ValueError("base_address must be positive")
        if base_address % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("base_address must be aligned to the measured 2-MiB CANN VMM granularity")

    def allocation_granularity(self, *, device_index: int) -> int:
        self._set_device(device_index)
        properties = _physical_memory_properties(device_index)
        granularity = ctypes.c_size_t()
        self._call(
            "aclrtMemGetAllocationGranularity",
            ctypes.byref(properties),
            ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
            ctypes.byref(granularity),
        )
        if granularity.value <= 0:
            raise PackedArenaAdapterUnavailable("aclrtMemGetAllocationGranularity returned zero")
        return int(granularity.value)

    def reserve_address(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        self._validate_size(size_bytes)
        if alignment_bytes != CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("packed C128 ACL backend requires 2-MiB address alignment")
        self._set_device(device_index)
        virtual_address = ctypes.c_void_p()
        # CANN 9.0 documents this argument as reserved and requires zero. The
        # returned address is checked against the measured granularity below.
        self._call(
            "aclrtReserveMemAddress",
            ctypes.byref(virtual_address),
            size_bytes,
            0,
            None,
            0,
        )
        base_address = int(virtual_address.value or 0)
        if base_address <= 0 or base_address % CANN_VMM_GRANULARITY_BYTES:
            invalid_address = ValueError(
                f"aclrtReserveMemAddress returned a null or non-2-MiB-aligned address: {base_address:#x}"
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
        if base_address in self._reserved_addresses:
            raise PackedArenaAdapterUnavailable(
                f"aclrtReserveMemAddress returned an address already tracked by this backend: {base_address:#x}"
            )
        self._reserved_addresses[base_address] = (
            size_bytes,
            device_index,
        )
        return base_address

    def allocate_physical(
        self,
        *,
        size_bytes: int,
        device_index: int,
    ) -> AclPhysicalMemoryHandle:
        self._validate_size(size_bytes)
        self._set_device(device_index)
        properties = _physical_memory_properties(device_index)
        handle = ctypes.c_void_p()
        self._call(
            "aclrtMallocPhysical",
            ctypes.byref(handle),
            size_bytes,
            ctypes.byref(properties),
            0,
        )
        if not handle.value:
            raise PackedArenaAdapterUnavailable("aclrtMallocPhysical succeeded but returned a null handle")
        result = AclPhysicalMemoryHandle(
            value=int(handle.value),
            size_bytes=size_bytes,
            device_index=device_index,
        )
        if result.value in self._allocated_handles:
            raise PackedArenaAdapterUnavailable(
                f"aclrtMallocPhysical returned a handle already tracked by this backend: {result.value:#x}"
            )
        self._allocated_handles[result.value] = result
        return result

    def map_physical(
        self,
        *,
        base_address: int,
        size_bytes: int,
        physical_handle: object,
        device_index: int,
    ) -> None:
        self._validate_range(
            base_address=base_address,
            size_bytes=size_bytes,
        )
        if not isinstance(physical_handle, AclPhysicalMemoryHandle):
            raise TypeError("physical_handle must be an AclPhysicalMemoryHandle")
        if physical_handle.value <= 0:
            raise ValueError("physical handle value must be positive")
        if physical_handle.size_bytes != size_bytes or physical_handle.device_index != device_index:
            raise ValueError("physical handle size/device does not match the mapping")
        if self._allocated_handles.get(physical_handle.value) != physical_handle:
            raise ValueError("physical handle is not owned by this backend")
        if self._reserved_addresses.get(base_address) != (
            size_bytes,
            device_index,
        ):
            raise ValueError("virtual address range is not reserved by this backend")
        if base_address in self._mapped_handles_by_address:
            raise ValueError("virtual address range is already mapped")
        if physical_handle in self._mapped_handles_by_address.values():
            raise ValueError("physical handle is already mapped")
        self._set_device(device_index)
        self._call(
            "aclrtMapMem",
            ctypes.c_void_p(base_address),
            size_bytes,
            0,
            ctypes.c_void_p(physical_handle.value),
            0,
        )
        self._mapped_handles_by_address[base_address] = physical_handle
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
                self._call(
                    "aclrtUnmapMem",
                    ctypes.c_void_p(base_address),
                )
                del self._mapped_handles_by_address[base_address]
            except BaseException as rollback_error:
                raise AclVmmRollbackError(
                    operation_error=operation_error,
                    rollback_error=rollback_error,
                ) from operation_error
            raise

    def zero_mapped(
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
        mapping = self._mapped_handles_by_address.get(base_address)
        if mapping is None:
            raise ValueError("virtual address range is not mapped")
        if mapping.size_bytes != size_bytes or mapping.device_index != device_index:
            raise ValueError("mapped range size/device does not match")
        self._set_device(device_index)
        self._call(
            "aclrtMemset",
            ctypes.c_void_p(base_address),
            size_bytes,
            0,
            size_bytes,
        )

    def unmap(
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
        mapping = self._mapped_handles_by_address.get(base_address)
        if mapping is None:
            raise ValueError("virtual address range is not mapped")
        if mapping.size_bytes != size_bytes or mapping.device_index != device_index:
            raise ValueError("mapped range size/device does not match")
        self._set_device(device_index)
        self._call(
            "aclrtUnmapMem",
            ctypes.c_void_p(base_address),
        )
        del self._mapped_handles_by_address[base_address]

    def free_physical(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> None:
        if not isinstance(physical_handle, AclPhysicalMemoryHandle):
            raise TypeError("physical_handle must be an AclPhysicalMemoryHandle")
        if physical_handle.value <= 0:
            raise ValueError("physical handle value must be positive")
        if physical_handle.device_index != device_index:
            raise ValueError("physical handle belongs to another device")
        if self._allocated_handles.get(physical_handle.value) != physical_handle:
            raise ValueError("physical handle is not owned by this backend")
        self._set_device(device_index)
        mapped_base = next(
            (
                base_address
                for base_address, mapped_handle in (self._mapped_handles_by_address.items())
                if mapped_handle == physical_handle
            ),
            None,
        )
        if mapped_base is not None:
            # The only normal route here is aclrtMemSetAccess failure followed
            # by a failed immediate unmap. PackedArenaLease does not mark the
            # mapping complete in that case, so retry the unmap here before
            # allowing physical memory to be released.
            self._call(
                "aclrtUnmapMem",
                ctypes.c_void_p(mapped_base),
            )
            del self._mapped_handles_by_address[mapped_base]
        self._call(
            "aclrtFreePhysical",
            ctypes.c_void_p(physical_handle.value),
        )
        del self._allocated_handles[physical_handle.value]

    def release_address(
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
        if self._reserved_addresses.get(base_address) != (
            size_bytes,
            device_index,
        ):
            raise ValueError("virtual address range is not reserved by this backend")
        if base_address in self._mapped_handles_by_address:
            raise RuntimeError("cannot release a virtual address range that is still mapped")
        self._set_device(device_index)
        self._call(
            "aclrtReleaseMemAddress",
            ctypes.c_void_p(base_address),
        )
        del self._reserved_addresses[base_address]
