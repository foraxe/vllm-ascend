# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only tests for the concrete packed C128 runtime adapters."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_acl_backend import (
    ACL_HBM_MEM_NORMAL,
    ACL_MEM_ALLOCATION_TYPE_PINNED,
    ACL_MEM_HANDLE_TYPE_NONE,
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
    AclPhysicalMemoryHandle,
    AclRuntimeCallError,
    AclVmmRollbackError,
    AscendAclPackedArenaBackend,
    PackedArenaAdapterUnavailable,
    _AclrtMemAccessDesc,
    _AclrtMemLocation,
    _AclrtPhysicalMemProp,
)
from vllm_ascend.attention.context_parallel.c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
    PackedArenaLease,
    PackedArenaOpenError,
    PackedArenaState,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
)
from vllm_ascend.attention.context_parallel.c128_packed_torch_npu import (
    TorchNpuPackedArenaTensorFactory,
    TorchNpuTensorBindingError,
)

pytestmark = pytest.mark.cpu_test

_BASE_ADDRESS = 0x1_0000_0000
_PHYSICAL_HANDLE = 0xA000_0000


def _value(value: object) -> object:
    return getattr(value, "value", value)


class _FakeAclFunction:
    def __init__(self, library: _FakeAclLibrary, name: str) -> None:
        self.library = library
        self.name = name
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *args: object) -> int:
        return self.library.invoke(self.name, args)


class _FakeAclLibrary:
    def __init__(
        self,
        *,
        statuses: dict[str, list[int]] | None = None,
        base_address: int = _BASE_ADDRESS,
        granularity: int = CANN_VMM_GRANULARITY_BYTES,
    ) -> None:
        self.statuses = {name: list(values) for name, values in (statuses or {}).items()}
        self.base_address = base_address
        self.granularity = granularity
        self.events: list[tuple[object, ...]] = []
        for name in (
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
        ):
            setattr(self, name, _FakeAclFunction(self, name))

    def _status(self, name: str) -> int:
        statuses = self.statuses.get(name, [])
        return statuses.pop(0) if statuses else 0

    def invoke(
        self,
        name: str,
        args: tuple[object, ...],
    ) -> int:
        status = self._status(name)
        if name == "aclrtSetDevice":
            self.events.append((name, int(_value(args[0]))))
        elif name == "aclrtMemGetAllocationGranularity":
            properties = ctypes.cast(
                args[0],
                ctypes.POINTER(_AclrtPhysicalMemProp),
            ).contents
            self.events.append(
                (
                    name,
                    properties.handleType,
                    properties.allocationType,
                    properties.memAttr,
                    properties.location.id,
                    properties.location.type,
                    int(_value(args[1])),
                )
            )
            if status == 0:
                ctypes.cast(
                    args[2],
                    ctypes.POINTER(ctypes.c_size_t),
                )[0] = self.granularity
        elif name == "aclrtReserveMemAddress":
            self.events.append(
                (
                    name,
                    int(_value(args[1])),
                    int(_value(args[2])),
                    _value(args[3]),
                    int(_value(args[4])),
                )
            )
            if status == 0:
                ctypes.cast(
                    args[0],
                    ctypes.POINTER(ctypes.c_void_p),
                )[0] = self.base_address
        elif name == "aclrtMallocPhysical":
            properties = ctypes.cast(
                args[2],
                ctypes.POINTER(_AclrtPhysicalMemProp),
            ).contents
            self.events.append(
                (
                    name,
                    int(_value(args[1])),
                    properties.location.id,
                    int(_value(args[3])),
                )
            )
            if status == 0:
                ctypes.cast(
                    args[0],
                    ctypes.POINTER(ctypes.c_void_p),
                )[0] = _PHYSICAL_HANDLE
        elif name == "aclrtMemSetAccess":
            access = ctypes.cast(
                args[2],
                ctypes.POINTER(_AclrtMemAccessDesc),
            ).contents
            self.events.append(
                (
                    name,
                    int(_value(args[0])),
                    int(_value(args[1])),
                    access.flags,
                    access.location.id,
                    access.location.type,
                    int(_value(args[3])),
                )
            )
        else:
            self.events.append((name, *(_value(argument) for argument in args)))
        return status


class _FakeDevice:
    def __init__(self, label: str) -> None:
        self.label = label

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDevice) and self.label == other.label


class _FakeTensor:
    def __init__(
        self,
        metadata: dict[str, object],
        *,
        pointer_delta: int = 0,
    ) -> None:
        self._metadata = metadata
        self._pointer_delta = pointer_delta
        self.dtype = metadata["dtype"]
        self.device = metadata["device"]

    def data_ptr(self) -> int:
        return int(self._metadata["data_ptr"]) + self._pointer_delta

    def numel(self) -> int:
        return int(self._metadata["size"][0])  # type: ignore[index]

    def element_size(self) -> int:
        return 1


class _FakeTorchNpuExtension:
    def __init__(
        self,
        *,
        pointer_delta: int = 0,
    ) -> None:
        self.pointer_delta = pointer_delta
        self.events: list[tuple[object, ...]] = []

    def _construct_storage_from_data_pointer(
        self,
        pointer: int,
        device: object,
        size_bytes: int,
    ) -> object:
        storage = object()
        self.events.append(("storage", pointer, device, size_bytes, storage))
        return storage

    def _construct_NPU_Tensor_From_Storage_And_Metadata(
        self,
        metadata: dict[str, object],
        storage: object,
    ) -> _FakeTensor:
        self.events.append(("tensor", metadata, storage))
        return _FakeTensor(
            metadata,
            pointer_delta=self.pointer_delta,
        )


def _fake_torch_modules(
    *,
    pointer_delta: int = 0,
) -> tuple[object, object, _FakeTorchNpuExtension]:
    uint8 = object()
    torch_module = SimpleNamespace(
        uint8=uint8,
        device=lambda label: _FakeDevice(label),
    )
    extension = _FakeTorchNpuExtension(pointer_delta=pointer_delta)
    torch_npu_module = SimpleNamespace(_C=extension)
    return torch_module, torch_npu_module, extension


def _single_bucket_plan() -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=3,
        tp_size=1,
        groups=(
            PackedPoolGroupSpec(
                name="cold",
                logical_blocks=2,
                components=(
                    PackedPoolComponentSpec(
                        name="c128",
                        bucket="wide",
                        page_size_bytes=128 * 1024,
                        copies=1,
                        placement=PackedPlacement.REPLICATED,
                        allocation_granularity_bytes=(CANN_VMM_GRANULARITY_BYTES),
                    ),
                ),
            ),
        ),
    )


def _reserve_and_allocate(
    backend: AscendAclPackedArenaBackend,
    *,
    device_index: int = 0,
) -> tuple[int, AclPhysicalMemoryHandle]:
    base_address = backend.reserve_address(
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=device_index,
    )
    handle = backend.allocate_physical(
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=device_index,
    )
    return base_address, handle


def test_ctypes_struct_layout_matches_cann_9_header() -> None:
    assert ctypes.sizeof(_AclrtMemLocation) == 8
    assert ctypes.sizeof(_AclrtPhysicalMemProp) == 32
    assert ctypes.sizeof(_AclrtMemAccessDesc) == 24


def test_backend_uses_canonical_two_mib_acl_lifecycle() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedArenaBackend(library=library)

    assert backend.allocation_granularity(device_index=3) == (CANN_VMM_GRANULARITY_BYTES)
    base_address = backend.reserve_address(
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=3,
    )
    handle = backend.allocate_physical(
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=3,
    )
    backend.map_physical(
        base_address=base_address,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        physical_handle=handle,
        device_index=3,
    )
    backend.zero_mapped(
        base_address=base_address,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=3,
    )
    backend.unmap(
        base_address=base_address,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=3,
    )
    backend.free_physical(
        physical_handle=handle,
        device_index=3,
    )
    backend.release_address(
        base_address=base_address,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=3,
    )

    assert isinstance(handle, AclPhysicalMemoryHandle)
    assert (
        "aclrtMemGetAllocationGranularity",
        ACL_MEM_HANDLE_TYPE_NONE,
        ACL_MEM_ALLOCATION_TYPE_PINNED,
        ACL_HBM_MEM_NORMAL,
        3,
        ACL_MEM_LOCATION_TYPE_DEVICE,
        0,
    ) in library.events
    assert (
        "aclrtReserveMemAddress",
        CANN_VMM_GRANULARITY_BYTES,
        0,
        None,
        0,
    ) in library.events
    assert (
        "aclrtMemSetAccess",
        _BASE_ADDRESS,
        CANN_VMM_GRANULARITY_BYTES,
        ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
        3,
        ACL_MEM_LOCATION_TYPE_DEVICE,
        1,
    ) in library.events
    assert (
        "aclrtMemset",
        _BASE_ADDRESS,
        CANN_VMM_GRANULARITY_BYTES,
        0,
        CANN_VMM_GRANULARITY_BYTES,
    ) in library.events


def test_backend_fails_closed_when_acl_symbol_is_absent() -> None:
    with pytest.raises(
        PackedArenaAdapterUnavailable,
        match="aclrtMemSetAccess",
    ):
        AscendAclPackedArenaBackend(library=SimpleNamespace())


def test_backend_rejects_non_two_mib_ranges_before_acl_calls() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedArenaBackend(library=library)

    with pytest.raises(ValueError, match="2097152-byte multiple"):
        backend.allocate_physical(size_bytes=1, device_index=0)
    assert library.events == []


def test_access_failure_unmaps_before_propagating() -> None:
    library = _FakeAclLibrary(
        statuses={"aclrtMemSetAccess": [507899]},
    )
    backend = AscendAclPackedArenaBackend(library=library)
    base_address, handle = _reserve_and_allocate(backend)
    library.events.clear()

    with pytest.raises(
        AclRuntimeCallError,
        match="aclrtMemSetAccess.*507899",
    ):
        backend.map_physical(
            base_address=base_address,
            size_bytes=CANN_VMM_GRANULARITY_BYTES,
            physical_handle=handle,
            device_index=0,
        )
    operations = [event[0] for event in library.events]
    assert operations[-3:] == [
        "aclrtMapMem",
        "aclrtMemSetAccess",
        "aclrtUnmapMem",
    ]


def test_access_and_unmap_failure_preserve_both_errors() -> None:
    library = _FakeAclLibrary(
        statuses={
            "aclrtMemSetAccess": [507899],
            "aclrtUnmapMem": [507018],
        },
    )
    backend = AscendAclPackedArenaBackend(library=library)
    base_address, handle = _reserve_and_allocate(backend)
    library.events.clear()

    with pytest.raises(AclVmmRollbackError) as captured:
        backend.map_physical(
            base_address=base_address,
            size_bytes=CANN_VMM_GRANULARITY_BYTES,
            physical_handle=handle,
            device_index=0,
        )
    assert isinstance(
        captured.value.operation_error,
        AclRuntimeCallError,
    )
    assert isinstance(
        captured.value.rollback_error,
        AclRuntimeCallError,
    )
    backend.free_physical(
        physical_handle=handle,
        device_index=0,
    )
    backend.release_address(
        base_address=base_address,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=0,
    )
    operations = [event[0] for event in library.events]
    assert operations[-5:] == [
        "aclrtSetDevice",
        "aclrtUnmapMem",
        "aclrtFreePhysical",
        "aclrtSetDevice",
        "aclrtReleaseMemAddress",
    ]


def test_lease_rolls_back_access_failure_without_freeing_a_mapping() -> None:
    library = _FakeAclLibrary(
        statuses={
            "aclrtMemSetAccess": [507899],
            "aclrtUnmapMem": [507018, 0],
        },
    )
    backend = AscendAclPackedArenaBackend(library=library)
    torch_module, torch_npu_module, _ = _fake_torch_modules()

    with pytest.raises(AclVmmRollbackError):
        PackedArenaLease.open(
            plan=_single_bucket_plan(),
            tp_rank=0,
            device_index=0,
            backend=backend,
            tensor_factory=TorchNpuPackedArenaTensorFactory(
                torch_module=torch_module,
                torch_npu_module=torch_npu_module,
            ),
            fence=lambda: None,
        )
    operations = [event[0] for event in library.events]
    assert operations[-5:] == [
        "aclrtSetDevice",
        "aclrtUnmapMem",
        "aclrtFreePhysical",
        "aclrtSetDevice",
        "aclrtReleaseMemAddress",
    ]


def test_incomplete_access_rollback_keeps_lease_retryable() -> None:
    library = _FakeAclLibrary(
        statuses={
            "aclrtMemSetAccess": [507899],
            "aclrtUnmapMem": [507018, 507018],
        },
    )
    backend = AscendAclPackedArenaBackend(library=library)
    torch_module, torch_npu_module, _ = _fake_torch_modules()

    with pytest.raises(PackedArenaOpenError) as captured:
        PackedArenaLease.open(
            plan=_single_bucket_plan(),
            tp_rank=0,
            device_index=0,
            backend=backend,
            tensor_factory=TorchNpuPackedArenaTensorFactory(
                torch_module=torch_module,
                torch_npu_module=torch_npu_module,
            ),
            fence=lambda: None,
        )
    lease = captured.value.lease
    assert lease.state is PackedArenaState.CLEANUP_FAILED
    assert "free_physical" in str(captured.value.cleanup_error)
    assert "release_address" in str(captured.value.cleanup_error)

    lease.close()
    assert lease.state is PackedArenaState.CLOSED


def test_unaligned_reserved_address_is_released() -> None:
    library = _FakeAclLibrary(base_address=_BASE_ADDRESS + 1)
    backend = AscendAclPackedArenaBackend(library=library)

    with pytest.raises(ValueError, match="non-2-MiB-aligned"):
        backend.reserve_address(
            size_bytes=CANN_VMM_GRANULARITY_BYTES,
            alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
            device_index=0,
        )
    assert library.events[-1] == (
        "aclrtReleaseMemAddress",
        _BASE_ADDRESS + 1,
    )


def test_torch_npu_factory_is_lazy_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def unavailable(name: str) -> object:
        imported.append(name)
        raise ImportError("not installed")

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.c128_packed_torch_npu.importlib.import_module",
        unavailable,
    )
    factory = TorchNpuPackedArenaTensorFactory()
    assert imported == []
    with pytest.raises(PackedArenaAdapterUnavailable, match="not installed"):
        factory.bind(
            base_address=_BASE_ADDRESS,
            size_bytes=CANN_VMM_GRANULARITY_BYTES,
            device_index=0,
        )
    assert imported == ["torch"]


def test_torch_npu_binding_is_non_owning_and_close_is_idempotent() -> None:
    torch_module, torch_npu_module, extension = _fake_torch_modules()
    factory = TorchNpuPackedArenaTensorFactory(
        torch_module=torch_module,
        torch_npu_module=torch_npu_module,
    )
    binding = factory.bind(
        base_address=_BASE_ADDRESS,
        size_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=5,
    )

    tensor = binding.tensor()
    assert tensor.data_ptr() == _BASE_ADDRESS  # type: ignore[attr-defined]
    metadata = extension.events[1][1]
    assert metadata["device"] == _FakeDevice("npu:5")
    assert metadata["size"] == (CANN_VMM_GRANULARITY_BYTES,)
    binding.close()
    binding.close()
    with pytest.raises(RuntimeError, match="closed"):
        binding.tensor()
    # The caller-held alias remains an ordinary non-owning object. Neither
    # close call invokes ACL free/unmap operations.
    assert tensor.data_ptr() == _BASE_ADDRESS  # type: ignore[attr-defined]


def test_tensor_validation_failure_drops_the_binding() -> None:
    torch_module, torch_npu_module, _ = _fake_torch_modules(
        pointer_delta=1,
    )
    factory = TorchNpuPackedArenaTensorFactory(
        torch_module=torch_module,
        torch_npu_module=torch_npu_module,
    )

    with pytest.raises(
        TorchNpuTensorBindingError,
        match="data_ptr mismatch",
    ):
        factory.bind(
            base_address=_BASE_ADDRESS,
            size_bytes=CANN_VMM_GRANULARITY_BYTES,
            device_index=0,
        )


def test_lease_orders_alias_fence_and_acl_teardown() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedArenaBackend(library=library)
    torch_module, torch_npu_module, _ = _fake_torch_modules()
    base_factory = TorchNpuPackedArenaTensorFactory(
        torch_module=torch_module,
        torch_npu_module=torch_npu_module,
    )
    lifetime_events: list[str] = []

    class RecordingBinding:
        def __init__(self, binding: object) -> None:
            self.binding = binding

        def tensor(self) -> object:
            return self.binding.tensor()  # type: ignore[attr-defined]

        def close(self) -> None:
            lifetime_events.append("close_alias")
            self.binding.close()  # type: ignore[attr-defined]

    class RecordingFactory:
        def bind(self, **kwargs: Any) -> RecordingBinding:
            return RecordingBinding(base_factory.bind(**kwargs))

    lease = PackedArenaLease.open(
        plan=_single_bucket_plan(),
        tp_rank=0,
        device_index=0,
        backend=backend,
        tensor_factory=RecordingFactory(),
        fence=lambda: lifetime_events.append("fence"),
    )
    library.events.clear()
    lease.close()

    acl_teardown = [event[0] for event in library.events if event[0] != "aclrtSetDevice"]
    assert lifetime_events == ["close_alias", "fence"]
    assert acl_teardown == [
        "aclrtUnmapMem",
        "aclrtFreePhysical",
        "aclrtReleaseMemAddress",
    ]
    assert lease.state is PackedArenaState.CLOSED
