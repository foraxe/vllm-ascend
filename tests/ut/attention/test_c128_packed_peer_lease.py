# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only tests for startup-only packed C128 VMM peer leases."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_acl_backend import (
    AclPhysicalMemoryHandle,
    _AclrtMemAccessDesc,
)
from vllm_ascend.attention.context_parallel.c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
    PackedArenaBusyError,
    PackedArenaExportAllocation,
    PackedArenaLease,
    PackedArenaState,
)
from vllm_ascend.attention.context_parallel.c128_packed_peer_lease import (
    ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
    ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
    ACL_V2_SHAREABLE_HANDLE_BYTES,
    PACKED_VMM_PEER_SCHEMA_VERSION,
    AclPeerAccessReleaseError,
    AclV2ShareableHandle,
    AscendAclPackedPeerBackend,
    PackedVmmPeerCleanupError,
    PackedVmmPeerLease,
    PackedVmmPeerLeaseState,
    PackedVmmPeerStartupError,
    PeerHandleDescriptor,
    PeerRankEndpoint,
    PeerStartupStageResult,
    TorchDistributedCpuControlGroup,
    _AclrtMemDefaultHandle,
    _arena_metadata_fingerprint,
    maybe_create_packed_vmm_peer_lease,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
)

pytestmark = pytest.mark.cpu_test

_LOCAL_RANK = 0
_REMOTE_RANK = 1
_LOCAL_DEVICE = 0
_REMOTE_DEVICE = 1
_LOCAL_TGID = 1001
_REMOTE_TGID = 1002
_SIZE = CANN_VMM_GRANULARITY_BYTES
_LOCAL_PHYSICAL_HANDLE = 0xA000_0000
_IMPORTED_PHYSICAL_HANDLE = 0xB000_0000
_ALIAS_ADDRESS = 0x1_0000_0000
_LOCAL_HANDLE_BYTES = bytes(range(ACL_V2_SHAREABLE_HANDLE_BYTES))
_REMOTE_HANDLE_BYTES = bytes(reversed(range(ACL_V2_SHAREABLE_HANDLE_BYTES)))


def _make_plan() -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=9,
        tp_size=2,
        groups=(
            PackedPoolGroupSpec(
                name="fake",
                logical_blocks=8,
                components=(
                    PackedPoolComponentSpec(
                        name="c128",
                        bucket="page_131072",
                        page_size_bytes=131072,
                        copies=1,
                        placement=PackedPlacement.C128_OWNER,
                        allocation_granularity_bytes=CANN_VMM_GRANULARITY_BYTES,
                    ),
                    PackedPoolComponentSpec(
                        name="narrow",
                        bucket="narrow",
                        page_size_bytes=4096,
                        copies=1,
                        placement=PackedPlacement.REPLICATED,
                        allocation_granularity_bytes=CANN_VMM_GRANULARITY_BYTES,
                    ),
                ),
            ),
        ),
    )


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
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.statuses: dict[str, list[int]] = {}
        for name in (
            "aclrtSetDevice",
            "aclrtDeviceGetBareTgid",
            "aclrtDeviceCanAccessPeer",
            "aclrtDeviceEnablePeerAccess",
            "aclrtDeviceDisablePeerAccess",
            "aclrtMemGetAllocationGranularity",
            "aclrtReserveMemAddress",
            "aclrtReleaseMemAddress",
            "aclrtMallocPhysical",
            "aclrtFreePhysical",
            "aclrtMapMem",
            "aclrtUnmapMem",
            "aclrtMemSetAccess",
            "aclrtMemset",
            "aclrtMemExportToShareableHandleV2",
            "aclrtMemSetPidToShareableHandleV2",
            "aclrtMemImportFromShareableHandleV2",
        ):
            setattr(self, name, _FakeAclFunction(self, name))

    def _status(self, name: str) -> int:
        values = self.statuses.get(name, [])
        return values.pop(0) if values else 0

    def invoke(
        self,
        name: str,
        args: tuple[object, ...],
    ) -> int:
        status = self._status(name)
        if name == "aclrtSetDevice":
            self.events.append((name, int(_value(args[0]))))
        elif name == "aclrtDeviceGetBareTgid":
            self.events.append((name,))
            if status == 0:
                ctypes.cast(
                    args[0],
                    ctypes.POINTER(ctypes.c_int32),
                )[0] = _LOCAL_TGID
        elif name == "aclrtDeviceCanAccessPeer":
            self.events.append(
                (
                    name,
                    int(_value(args[1])),
                    int(_value(args[2])),
                )
            )
            if status == 0:
                ctypes.cast(
                    args[0],
                    ctypes.POINTER(ctypes.c_int32),
                )[0] = 1
        elif name == "aclrtMemExportToShareableHandleV2":
            self.events.append(
                (
                    name,
                    int(_value(args[0])),
                    int(_value(args[1])),
                    int(_value(args[2])),
                )
            )
            if status == 0:
                output = ctypes.cast(
                    args[3],
                    ctypes.POINTER(_AclrtMemDefaultHandle),
                ).contents
                ctypes.memmove(
                    ctypes.addressof(output),
                    _LOCAL_HANDLE_BYTES,
                    ACL_V2_SHAREABLE_HANDLE_BYTES,
                )
        elif name == "aclrtMemSetPidToShareableHandleV2":
            pid_count = int(_value(args[3]))
            pids = tuple(
                ctypes.cast(
                    args[2],
                    ctypes.POINTER(ctypes.c_int32),
                )[index]
                for index in range(pid_count)
            )
            self.events.append((name, int(_value(args[1])), pids, pid_count))
        elif name == "aclrtMemImportFromShareableHandleV2":
            self.events.append((name, int(_value(args[1])), int(_value(args[2]))))
            if status == 0:
                ctypes.cast(
                    args[3],
                    ctypes.POINTER(ctypes.c_void_p),
                )[0] = _IMPORTED_PHYSICAL_HANDLE
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
                )[0] = _ALIAS_ADDRESS
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


def test_acl_backend_uses_exact_v2_abi_and_cleanup_order() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedPeerBackend(library=library)
    local_handle = AclPhysicalMemoryHandle(
        value=_LOCAL_PHYSICAL_HANDLE,
        size_bytes=_SIZE,
        device_index=_LOCAL_DEVICE,
    )

    assert backend.bare_tgid(device_index=_LOCAL_DEVICE) == _LOCAL_TGID
    peer_access = backend.acquire_peer_access(
        device_index=_LOCAL_DEVICE,
        peer_device_indices=(_REMOTE_DEVICE,),
    )
    shared = backend.export_v2(
        physical_handle=local_handle,
        device_index=_LOCAL_DEVICE,
    )
    assert shared.payload == _LOCAL_HANDLE_BYTES
    backend.authorize_v2(
        shareable_handle=shared,
        bare_tgids=(_REMOTE_TGID, _REMOTE_TGID),
        device_index=_LOCAL_DEVICE,
    )
    imported = backend.import_v2(
        shareable_handle=AclV2ShareableHandle(_REMOTE_HANDLE_BYTES),
        size_bytes=_SIZE,
        device_index=_LOCAL_DEVICE,
    )
    alias = backend.reserve_alias(
        size_bytes=_SIZE,
        alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
        device_index=_LOCAL_DEVICE,
    )
    backend.map_alias(
        base_address=alias,
        size_bytes=_SIZE,
        imported_handle=imported,
        device_index=_LOCAL_DEVICE,
    )
    backend.unmap_alias(
        base_address=alias,
        size_bytes=_SIZE,
        device_index=_LOCAL_DEVICE,
    )
    backend.close_imported(
        imported_handle=imported,
        device_index=_LOCAL_DEVICE,
    )
    backend.release_alias(
        base_address=alias,
        size_bytes=_SIZE,
        device_index=_LOCAL_DEVICE,
    )
    peer_access.release()

    significant = [event for event in library.events if event[0] != "aclrtSetDevice"]
    assert significant == [
        ("aclrtDeviceGetBareTgid",),
        ("aclrtDeviceCanAccessPeer", _LOCAL_DEVICE, _REMOTE_DEVICE),
        ("aclrtDeviceEnablePeerAccess", _REMOTE_DEVICE, 0),
        (
            "aclrtMemExportToShareableHandleV2",
            _LOCAL_PHYSICAL_HANDLE,
            ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
        ),
        (
            "aclrtMemSetPidToShareableHandleV2",
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            (_REMOTE_TGID,),
            1,
        ),
        (
            "aclrtMemImportFromShareableHandleV2",
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            0,
        ),
        ("aclrtReserveMemAddress", _SIZE, 0, None, 0),
        (
            "aclrtMapMem",
            _ALIAS_ADDRESS,
            _SIZE,
            0,
            _IMPORTED_PHYSICAL_HANDLE,
            0,
        ),
        (
            "aclrtMemSetAccess",
            _ALIAS_ADDRESS,
            _SIZE,
            3,
            _LOCAL_DEVICE,
            1,
            1,
        ),
        ("aclrtUnmapMem", _ALIAS_ADDRESS),
        ("aclrtFreePhysical", _IMPORTED_PHYSICAL_HANDLE),
        ("aclrtReleaseMemAddress", _ALIAS_ADDRESS),
        ("aclrtDeviceDisablePeerAccess", _REMOTE_DEVICE),
    ]


@dataclass(frozen=True)
class _Imported:
    value: int


class _FakePeerBackend:
    def __init__(
        self,
        trace: list[tuple[object, ...]],
        *,
        fail_map: bool = False,
        fail_unmap_once: bool = False,
    ) -> None:
        self.trace = trace
        self.fail_map = fail_map
        self.fail_unmap_once = fail_unmap_once

    def bare_tgid(self, *, device_index: int) -> int:
        self.trace.append(("bare_tgid", device_index))
        return _LOCAL_TGID

    def acquire_peer_access(
        self,
        *,
        device_index: int,
        peer_device_indices: tuple[int, ...],
    ) -> _FakePeerAccessLease:
        self.trace.append(("acquire_peer_access", device_index, peer_device_indices))
        return _FakePeerAccessLease(self.trace)

    def export_v2(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> AclV2ShareableHandle:
        self.trace.append(("export", physical_handle, device_index))
        return AclV2ShareableHandle(_LOCAL_HANDLE_BYTES)

    def authorize_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        bare_tgids: tuple[int, ...],
        device_index: int,
    ) -> None:
        self.trace.append(("authorize", bare_tgids, device_index, len(shareable_handle.payload)))

    def import_v2(
        self,
        *,
        shareable_handle: AclV2ShareableHandle,
        size_bytes: int,
        device_index: int,
    ) -> _Imported:
        self.trace.append(("import", size_bytes, device_index, len(shareable_handle.payload)))
        return _Imported(_IMPORTED_PHYSICAL_HANDLE)

    def reserve_alias(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        self.trace.append(("reserve", size_bytes, alignment_bytes, device_index))
        return _ALIAS_ADDRESS

    def map_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        imported_handle: object,
        device_index: int,
    ) -> None:
        self.trace.append(("map", base_address, size_bytes, imported_handle, device_index))
        if self.fail_map:
            raise RuntimeError("map failed")

    def unmap_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self.trace.append(("unmap", base_address, size_bytes, device_index))
        if self.fail_unmap_once:
            self.fail_unmap_once = False
            raise RuntimeError("unmap failed once")

    def close_imported(
        self,
        *,
        imported_handle: object,
        device_index: int,
    ) -> None:
        self.trace.append(("close_imported", imported_handle, device_index))

    def release_alias(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self.trace.append(("release_alias", base_address, size_bytes, device_index))


class _FakeBinding:
    def __init__(self, trace: list[tuple[object, ...]], tensor: object) -> None:
        self.trace = trace
        self._tensor: object | None = tensor

    def tensor(self) -> object:
        if self._tensor is None:
            raise RuntimeError("binding is closed")
        return self._tensor

    def close(self) -> None:
        self.trace.append(("drop_alias",))
        self._tensor = None


class _FakeTensorFactory:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.result = object()

    def bind(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> _FakeBinding:
        self.trace.append(("bind", base_address, size_bytes, device_index))
        return _FakeBinding(self.trace, self.result)


class _FakePeerAccessLease:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.released = False

    def release(self) -> None:
        self.trace.append(("release_peer_access",))
        self.released = True


class _FakeExportPin:
    def __init__(self, owner: _FakeOwnerArena) -> None:
        self.owner = owner
        self.released = False

    def allocations(self) -> tuple[PackedArenaExportAllocation, ...]:
        assert not self.released
        return self.owner.export_allocations

    def release(self) -> None:
        if self.released:
            return
        self.owner.trace.append(("release_export_pin",))
        self.owner.pin_active = False
        self.released = True


class _FakeOwnerArena:
    tp_rank = _LOCAL_RANK
    device_index = _LOCAL_DEVICE
    alignment_bytes = CANN_VMM_GRANULARITY_BYTES

    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.plan = _make_plan()
        self.pin_active = False
        self.closed = False
        self.local_handle = AclPhysicalMemoryHandle(
            value=_LOCAL_PHYSICAL_HANDLE,
            size_bytes=_SIZE,
            device_index=_LOCAL_DEVICE,
        )
        self.export_allocations = (
            PackedArenaExportAllocation(
                bucket="page_131072",
                size_bytes=_SIZE,
                physical_handle=self.local_handle,
            ),
            PackedArenaExportAllocation(
                bucket="narrow",
                size_bytes=_SIZE,
                physical_handle=AclPhysicalMemoryHandle(
                    value=_LOCAL_PHYSICAL_HANDLE + _SIZE,
                    size_bytes=_SIZE,
                    device_index=_LOCAL_DEVICE,
                ),
            ),
        )

    def acquire_export_pin(self) -> _FakeExportPin:
        assert not self.pin_active
        assert not self.closed
        self.pin_active = True
        return _FakeExportPin(self)

    def close(self) -> None:
        if self.pin_active:
            raise RuntimeError("owner closed while export pin is active")
        if not self.closed:
            self.trace.append(("release_owned",))
            self.closed = True


class _FakeControl:
    rank = _LOCAL_RANK
    world_size = 2

    def __init__(
        self,
        trace: list[tuple[object, ...]],
        *,
        remote_failure_stage: str | None = None,
        remote_schema: tuple[tuple[str, int], ...] | None = None,
        remote_lease_id: str | None = None,
        remote_fingerprint: str | None = None,
        fail_barrier_once: bool = False,
    ) -> None:
        self.trace = trace
        self.remote_failure_stage = remote_failure_stage
        self.remote_schema = remote_schema
        self.remote_lease_id = remote_lease_id
        self.remote_fingerprint = remote_fingerprint
        self.fail_barrier_once = fail_barrier_once

    def all_gather_object(self, value: object) -> tuple[object, ...]:
        assert isinstance(value, PeerStartupStageResult)
        self.trace.append(("gather_stage", value.stage, value.ok))
        if value.stage == self.remote_failure_stage:
            return (
                value,
                PeerStartupStageResult(
                    stage=value.stage,
                    rank=_REMOTE_RANK,
                    ok=False,
                    error_type="RemoteFailure",
                    error_message="injected remote failure",
                ),
            )
        if value.stage == "lease_id":
            remote_payload: object = None
        elif value.stage == "endpoint":
            assert isinstance(value.payload, PeerRankEndpoint)
            local_endpoint = value.payload
            remote_endpoint = PeerRankEndpoint(
                rank=_REMOTE_RANK,
                device_index=_REMOTE_DEVICE,
                bare_tgid=_REMOTE_TGID,
                lease_id=self.remote_lease_id or local_endpoint.lease_id,
                schema_version=PACKED_VMM_PEER_SCHEMA_VERSION,
                allocation_schema=(
                    self.remote_schema if self.remote_schema is not None else local_endpoint.allocation_schema
                ),
                metadata_fingerprint=(self.remote_fingerprint or local_endpoint.metadata_fingerprint),
            )
            remote_payload = remote_endpoint
        elif value.stage == "enable_export_authorize":
            assert isinstance(value.payload, tuple)
            remote_payload = tuple(
                PeerHandleDescriptor(
                    owner_rank=_REMOTE_RANK,
                    owner_device_index=_REMOTE_DEVICE,
                    key=descriptor.key,
                    size_bytes=descriptor.size_bytes,
                    shareable_handle=AclV2ShareableHandle(_REMOTE_HANDLE_BYTES),
                )
                for descriptor in value.payload
            )
        else:
            remote_payload = None
        return (
            value,
            PeerStartupStageResult(
                stage=value.stage,
                rank=_REMOTE_RANK,
                ok=True,
                payload=remote_payload,
            ),
        )

    def barrier(self, *, stage: str) -> None:
        self.trace.append(("barrier", stage))
        if self.fail_barrier_once:
            self.fail_barrier_once = False
            raise RuntimeError("barrier failed once")


def _open_fake_lease(
    trace: list[tuple[object, ...]],
    *,
    backend: _FakePeerBackend | None = None,
    control: _FakeControl | None = None,
) -> tuple[
    PackedVmmPeerLease,
    _FakePeerBackend,
    _FakeTensorFactory,
    _FakeOwnerArena,
]:
    selected_backend = backend or _FakePeerBackend(trace)
    tensor_factory = _FakeTensorFactory(trace)
    owner_arena = _FakeOwnerArena(trace)
    lease = PackedVmmPeerLease.open(
        owner_arena=owner_arena,  # type: ignore[arg-type]
        shared_buckets=("page_131072",),
        backend=selected_backend,
        tensor_factory=tensor_factory,
        control=control or _FakeControl(trace),
        fence=lambda: trace.append(("fence",)),
    )
    return lease, selected_backend, tensor_factory, owner_arena


def test_peer_lease_maps_once_then_request_path_only_returns_alias() -> None:
    trace: list[tuple[object, ...]] = []
    lease, _, tensor_factory, _ = _open_fake_lease(trace)

    assert lease.state is PackedVmmPeerLeaseState.OPEN
    assert len(lease.aliases) == 1
    # Validate fields without exposing the V2 handle.
    alias = lease.aliases[0]
    assert (
        alias.owner_rank,
        alias.key,
        alias.base_address,
        alias.size_bytes,
    ) == (_REMOTE_RANK, "page_131072", _ALIAS_ADDRESS, _SIZE)

    startup_trace = tuple(trace)
    assert lease.tensor(owner_rank=_REMOTE_RANK, key="page_131072") is tensor_factory.result
    assert lease.tensor(owner_rank=_REMOTE_RANK, key="page_131072") is tensor_factory.result
    assert tuple(trace) == startup_trace

    counters = lease.counters
    assert counters.startup_control_exchange_calls == 4
    assert counters.startup_peer_access_acquire_calls == 1
    assert counters.startup_export_calls == 1
    assert counters.startup_authorize_calls == 1
    assert counters.startup_import_calls == 1
    assert counters.startup_reserve_calls == 1
    assert counters.startup_map_calls == 1
    assert counters.startup_bind_calls == 1
    assert counters.request_tensor_calls == 2
    assert counters.request_import_calls == 0
    assert counters.request_map_calls == 0
    assert counters.request_control_collective_calls == 0
    assert counters.request_fence_calls == 0
    exported_handles = [event[1] for event in trace if event[0] == "export"]
    assert [handle.value for handle in exported_handles] == [_LOCAL_PHYSICAL_HANDLE]


def test_close_orders_fence_alias_drop_import_close_ack_then_owner_free() -> None:
    trace: list[tuple[object, ...]] = []
    lease, _, _, owner_arena = _open_fake_lease(trace)
    trace.clear()

    lease.close()

    assert trace == [
        ("fence",),
        ("drop_alias",),
        ("unmap", _ALIAS_ADDRESS, _SIZE, _LOCAL_DEVICE),
        ("close_imported", _Imported(_IMPORTED_PHYSICAL_HANDLE), _LOCAL_DEVICE),
        ("release_alias", _ALIAS_ADDRESS, _SIZE, _LOCAL_DEVICE),
        ("release_peer_access",),
        ("barrier", "peer_imports_released"),
        ("release_export_pin",),
        ("release_owned",),
    ]
    assert lease.state is PackedVmmPeerLeaseState.CLOSED
    counters = lease.counters
    assert counters.teardown_unmap_calls == 1
    assert counters.teardown_close_import_calls == 1
    assert counters.teardown_release_alias_calls == 1
    assert counters.teardown_peer_access_release_calls == 1
    assert owner_arena.closed


def test_cleanup_failure_never_acknowledges_or_releases_owner_and_is_retryable() -> None:
    trace: list[tuple[object, ...]] = []
    backend = _FakePeerBackend(trace, fail_unmap_once=True)
    lease, _, _, owner_arena = _open_fake_lease(trace, backend=backend)
    trace.clear()

    with pytest.raises(PackedVmmPeerCleanupError, match="unmap failed once"):
        lease.close()
    assert lease.state is PackedVmmPeerLeaseState.CLEANUP_FAILED
    assert ("barrier", "peer_imports_released") not in trace
    assert ("release_owned",) not in trace
    assert not owner_arena.closed

    trace.clear()
    lease.close()
    assert trace == [
        ("unmap", _ALIAS_ADDRESS, _SIZE, _LOCAL_DEVICE),
        ("close_imported", _Imported(_IMPORTED_PHYSICAL_HANDLE), _LOCAL_DEVICE),
        ("release_alias", _ALIAS_ADDRESS, _SIZE, _LOCAL_DEVICE),
        ("release_peer_access",),
        ("barrier", "peer_imports_released"),
        ("release_export_pin",),
        ("release_owned",),
    ]
    assert lease.state is PackedVmmPeerLeaseState.CLOSED


def test_failed_import_release_ack_keeps_owner_pinned_until_retry() -> None:
    trace: list[tuple[object, ...]] = []
    control = _FakeControl(trace, fail_barrier_once=True)
    lease, _, _, owner_arena = _open_fake_lease(trace, control=control)
    trace.clear()

    with pytest.raises(PackedVmmPeerCleanupError, match="barrier failed once"):
        lease.close()
    assert not owner_arena.closed
    assert owner_arena.pin_active
    assert ("release_export_pin",) not in trace

    lease.close()
    assert owner_arena.closed
    assert trace[-2:] == [("release_export_pin",), ("release_owned",)]


def test_startup_map_failure_is_globally_observed_then_fully_rolled_back() -> None:
    trace: list[tuple[object, ...]] = []
    backend = _FakePeerBackend(trace, fail_map=True)
    with pytest.raises(PackedVmmPeerStartupError, match="map failed"):
        _open_fake_lease(trace, backend=backend)

    assert ("close_imported", _Imported(_IMPORTED_PHYSICAL_HANDLE), _LOCAL_DEVICE) in trace
    assert ("release_alias", _ALIAS_ADDRESS, _SIZE, _LOCAL_DEVICE) in trace
    assert ("release_peer_access",) in trace
    assert ("release_export_pin",) in trace
    assert ("release_owned",) in trace
    assert ("gather_stage", "startup_rollback", True) in trace


def test_set_access_and_immediate_unmap_failure_retains_retryable_mapping() -> None:
    trace: list[tuple[object, ...]] = []
    library = _FakeAclLibrary()
    library.statuses["aclrtMemSetAccess"] = [7]
    library.statuses["aclrtUnmapMem"] = [8, 0]
    backend = AscendAclPackedPeerBackend(library=library)

    with pytest.raises(PackedVmmPeerStartupError, match="AclVmmRollbackError"):
        _open_fake_lease(
            trace,
            backend=backend,  # type: ignore[arg-type]
        )

    unmaps = [event for event in library.events if event[0] == "aclrtUnmapMem"]
    assert unmaps == [
        ("aclrtUnmapMem", _ALIAS_ADDRESS),
        ("aclrtUnmapMem", _ALIAS_ADDRESS),
    ]
    assert ("release_export_pin",) in trace
    assert ("release_owned",) in trace


def test_remote_startup_failure_rolls_back_local_alias_before_owner_close() -> None:
    trace: list[tuple[object, ...]] = []
    control = _FakeControl(
        trace,
        remote_failure_stage="import_map_bind_ready",
    )

    with pytest.raises(PackedVmmPeerStartupError, match="rank1:RemoteFailure"):
        _open_fake_lease(trace, control=control)

    assert trace.index(("drop_alias",)) < trace.index(("release_export_pin",))
    assert trace.index(("release_peer_access",)) < trace.index(("release_export_pin",))
    assert ("gather_stage", "startup_rollback", True) in trace
    assert ("release_owned",) in trace


def test_mismatched_rank_schema_aborts_before_export_or_import() -> None:
    trace: list[tuple[object, ...]] = []
    control = _FakeControl(
        trace,
        remote_schema=(("different", _SIZE),),
    )

    with pytest.raises(ValueError, match="allocation key/size schema"):
        _open_fake_lease(trace, control=control)

    assert not any(event[0] == "export" for event in trace)
    assert not any(event[0] == "import" for event in trace)
    assert ("gather_stage", "startup_rollback", True) in trace
    assert ("release_owned",) in trace


def test_identical_independent_plans_have_same_metadata_fingerprint() -> None:
    first = _FakeOwnerArena([])
    second = _FakeOwnerArena([])
    assert first.plan is not second.plan
    schema = (("page_131072", _SIZE),)

    assert _arena_metadata_fingerprint(
        owner_arena=first,  # type: ignore[arg-type]
        allocation_schema=schema,
    ) == _arena_metadata_fingerprint(
        owner_arena=second,  # type: ignore[arg-type]
        allocation_schema=schema,
    )


@pytest.mark.parametrize(
    ("control_kwargs", "match"),
    [
        ({"remote_lease_id": "stale-generation"}, "lease id"),
        ({"remote_fingerprint": "wrong-plan"}, "fingerprint"),
    ],
)
def test_mismatched_lease_or_plan_aborts_before_export_or_import(
    control_kwargs: dict[str, str],
    match: str,
) -> None:
    trace: list[tuple[object, ...]] = []
    control = _FakeControl(trace, **control_kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        _open_fake_lease(trace, control=control)

    assert not any(event[0] == "export" for event in trace)
    assert not any(event[0] == "import" for event in trace)
    assert ("release_owned",) in trace


def test_peer_access_refcounts_enable_once_and_disable_after_last_lease() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedPeerBackend(library=library)

    first = backend.acquire_peer_access(
        device_index=_LOCAL_DEVICE,
        peer_device_indices=(_REMOTE_DEVICE,),
    )
    second = backend.acquire_peer_access(
        device_index=_LOCAL_DEVICE,
        peer_device_indices=(_REMOTE_DEVICE,),
    )
    first.release()
    second.release()

    significant = [event for event in library.events if event[0] != "aclrtSetDevice"]
    assert significant == [
        ("aclrtDeviceCanAccessPeer", _LOCAL_DEVICE, _REMOTE_DEVICE),
        ("aclrtDeviceEnablePeerAccess", _REMOTE_DEVICE, 0),
        ("aclrtDeviceDisablePeerAccess", _REMOTE_DEVICE),
    ]


def test_peer_access_partial_disable_failure_is_retryable_per_pair() -> None:
    library = _FakeAclLibrary()
    backend = AscendAclPackedPeerBackend(library=library)
    lease = backend.acquire_peer_access(
        device_index=_LOCAL_DEVICE,
        peer_device_indices=(1, 2),
    )
    library.statuses["aclrtDeviceDisablePeerAccess"] = [0, 7, 0]

    with pytest.raises(AclPeerAccessReleaseError, match="0->1"):
        lease.release()
    assert not lease.released

    lease.release()
    assert lease.released
    disables = [event for event in library.events if event[0] == "aclrtDeviceDisablePeerAccess"]
    assert disables == [
        ("aclrtDeviceDisablePeerAccess", 2),
        ("aclrtDeviceDisablePeerAccess", 1),
        ("aclrtDeviceDisablePeerAccess", 1),
    ]


def test_feature_off_does_not_touch_runtime_dependencies() -> None:
    assert maybe_create_packed_vmm_peer_lease(enabled=False) is None


def test_export_pin_blocks_owner_close_before_any_cleanup_operation() -> None:
    trace: list[tuple[object, ...]] = []
    lease = PackedArenaLease(
        plan=object(),  # type: ignore[arg-type]
        tp_rank=0,
        device_index=0,
        alignment_bytes=CANN_VMM_GRANULARITY_BYTES,
        backend=object(),  # type: ignore[arg-type]
        tensor_factory=object(),  # type: ignore[arg-type]
        fence=lambda: trace.append(("owner_fence",)),
    )
    pin = lease.acquire_export_pin()

    with pytest.raises(PackedArenaBusyError, match="export pins"):
        lease.close()
    assert lease.state is PackedArenaState.OPEN
    assert trace == []

    pin.release()
    lease.close()
    assert lease.state is PackedArenaState.CLOSED
    assert trace == []


def test_shareable_handle_repr_is_redacted() -> None:
    handle = AclV2ShareableHandle(_LOCAL_HANDLE_BYTES)
    descriptor = PeerHandleDescriptor(
        owner_rank=0,
        owner_device_index=0,
        key="c128",
        size_bytes=_SIZE,
        shareable_handle=handle,
    )
    assert _LOCAL_HANDLE_BYTES.hex() not in repr(handle)
    assert _LOCAL_HANDLE_BYTES.hex() not in repr(descriptor)
    assert "redacted" in repr(handle)
    assert "redacted" in repr(descriptor)


class _FakeDist:
    def __init__(self, *, backend: str = "gloo") -> None:
        self.backend = backend
        self.events: list[tuple[object, ...]] = []

    def get_backend(self, group: object) -> str:
        self.events.append(("get_backend", group))
        return self.backend

    def get_rank(self, group: object) -> int:
        return 0

    def get_world_size(self, group: object) -> int:
        return 2

    def all_gather_object(
        self,
        output: list[object | None],
        value: object,
        *,
        group: object,
    ) -> None:
        self.events.append(("all_gather_object", value, group))
        output[:] = [value, "peer"]

    def monitored_barrier(
        self,
        *,
        group: object,
        timeout: object,
        wait_all_ranks: bool,
    ) -> None:
        self.events.append(("monitored_barrier", group, timeout, wait_all_ranks))


def test_torch_control_adapter_requires_gloo_and_uses_monitored_barrier() -> None:
    group = object()
    dist = _FakeDist()
    control = TorchDistributedCpuControlGroup(
        group=group,
        timeout_seconds=7,
        dist_module=dist,
    )
    assert control.rank == 0
    assert control.world_size == 2
    assert control.all_gather_object("local") == ("local", "peer")
    control.barrier(stage="imports_released")
    assert dist.events[-1][0] == "monitored_barrier"
    assert dist.events[-1][3] is True

    with pytest.raises(ValueError, match="Gloo"):
        bad_control = TorchDistributedCpuControlGroup(
            group=group,
            dist_module=_FakeDist(backend="hccl"),
        )
        assert bad_control.world_size == 2
