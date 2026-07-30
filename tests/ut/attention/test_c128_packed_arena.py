# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only tests for the packed C128 VMM arena lease."""

from dataclasses import replace

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
    PackedArenaCleanupError,
    PackedArenaClosedError,
    PackedArenaLease,
    PackedArenaState,
    maybe_create_packed_arena,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
    PackedPoolScratchSpec,
)

pytestmark = pytest.mark.cpu_test

WIDE_PAGE_BYTES = 128 * 1024
NARROW_PAGE_BYTES = 16_640


def _component(
    name: str,
    *,
    bucket: str,
    page_size_bytes: int,
    placement: PackedPlacement,
    copies: int = 1,
    granularity_bytes: int = CANN_VMM_GRANULARITY_BYTES,
) -> PackedPoolComponentSpec:
    return PackedPoolComponentSpec(
        name=name,
        bucket=bucket,
        page_size_bytes=page_size_bytes,
        copies=copies,
        placement=placement,
        allocation_granularity_bytes=granularity_bytes,
    )


def _plan(
    *,
    granularity_bytes: int = CANN_VMM_GRANULARITY_BYTES,
) -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=18,
        tp_size=2,
        groups=(
            PackedPoolGroupSpec(
                name="hot",
                logical_blocks=7,
                components=(
                    _component(
                        "replicated_wide",
                        bucket="wide",
                        page_size_bytes=WIDE_PAGE_BYTES,
                        placement=PackedPlacement.REPLICATED,
                        copies=2,
                        granularity_bytes=granularity_bytes,
                    ),
                ),
            ),
            PackedPoolGroupSpec(
                name="cold",
                logical_blocks=9,
                components=(
                    _component(
                        "owner_wide",
                        bucket="wide",
                        page_size_bytes=WIDE_PAGE_BYTES,
                        placement=PackedPlacement.C128_OWNER,
                        granularity_bytes=granularity_bytes,
                    ),
                    _component(
                        "owner_narrow",
                        bucket="narrow",
                        page_size_bytes=NARROW_PAGE_BYTES,
                        placement=PackedPlacement.C128_OWNER,
                        copies=2,
                        granularity_bytes=granularity_bytes,
                    ),
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=WIDE_PAGE_BYTES,
                max_pages_per_rank=3,
                allocation_granularity_bytes=granularity_bytes,
            ),
            PackedPoolScratchSpec(
                bucket="narrow",
                page_size_bytes=NARROW_PAGE_BYTES,
                max_pages_per_rank=5,
                allocation_granularity_bytes=granularity_bytes,
            ),
        ),
    )


class _FakeBackend:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_calls: dict[str, set[int]] | None = None,
    ) -> None:
        self.events = events
        self.fail_calls = fail_calls or {}
        self.call_counts: dict[str, int] = {}
        self.reserve_count = 0

    def _event(self, operation: str, *args: object) -> None:
        count = self.call_counts.get(operation, 0) + 1
        self.call_counts[operation] = count
        self.events.append((operation, *args))
        if count in self.fail_calls.get(operation, set()):
            raise RuntimeError(f"injected {operation} failure {count}")

    def allocation_granularity(self, *, device_index: int) -> int:
        self._event(
            "allocation_granularity",
            CANN_VMM_GRANULARITY_BYTES,
            device_index,
        )
        return CANN_VMM_GRANULARITY_BYTES

    def reserve_address(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        self.reserve_count += 1
        base_address = 0x1_0000_0000 * self.reserve_count
        self._event(
            "reserve",
            base_address,
            size_bytes,
            alignment_bytes,
            device_index,
        )
        return base_address

    def allocate_physical(
        self,
        *,
        size_bytes: int,
        device_index: int,
    ) -> object:
        handle = f"physical-{self.call_counts.get('allocate', 0) + 1}"
        self._event("allocate", handle, size_bytes, device_index)
        return handle

    def map_physical(
        self,
        *,
        base_address: int,
        size_bytes: int,
        physical_handle: object,
        device_index: int,
    ) -> None:
        self._event(
            "map",
            base_address,
            size_bytes,
            physical_handle,
            device_index,
        )

    def zero_mapped(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self._event("zero", base_address, size_bytes, device_index)

    def unmap(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self._event("unmap", base_address, size_bytes, device_index)

    def free_physical(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> None:
        self._event("free", physical_handle, device_index)

    def release_address(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self._event("release", base_address, size_bytes, device_index)


class _FakeBinding:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        base_address: int,
        token: object,
    ) -> None:
        self.events = events
        self.base_address = base_address
        self.token = token
        self.closed = False

    def tensor(self) -> object:
        if self.closed:
            raise RuntimeError("binding is closed")
        return self.token

    def close(self) -> None:
        self.events.append(("close_binding", self.base_address))
        self.closed = True


class _FakeTensorFactory:
    def __init__(
        self,
        events: list[tuple[object, ...]],
    ) -> None:
        self.events = events
        self.bindings: dict[int, _FakeBinding] = {}

    def bind(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> _FakeBinding:
        self.events.append(("bind", base_address, size_bytes, device_index))
        binding = _FakeBinding(
            self.events,
            base_address=base_address,
            token=object(),
        )
        self.bindings[base_address] = binding
        return binding


def _open(
    plan: PackedPoolPlan,
    *,
    rank: int = 1,
    backend: _FakeBackend | None = None,
    events: list[tuple[object, ...]] | None = None,
) -> tuple[
    PackedArenaLease,
    _FakeBackend,
    _FakeTensorFactory,
    list[tuple[object, ...]],
]:
    event_log = [] if events is None else events
    selected_backend = _FakeBackend(event_log) if backend is None else backend
    tensor_factory = _FakeTensorFactory(event_log)

    def fence() -> None:
        event_log.append(("fence",))

    lease = PackedArenaLease.open(
        plan=plan,
        tp_rank=rank,
        device_index=rank,
        backend=selected_backend,
        tensor_factory=tensor_factory,
        fence=fence,
    )
    return lease, selected_backend, tensor_factory, event_log


def test_disabled_factory_does_not_touch_any_runtime_dependency() -> None:
    def unexpected_plan() -> PackedPoolPlan:
        raise AssertionError("disabled path invoked the planner")

    assert (
        maybe_create_packed_arena(
            enabled=False,
            plan_factory=unexpected_plan,
        )
        is None
    )


def test_plan_must_use_measured_two_mib_vmm_granularity() -> None:
    events: list[tuple[object, ...]] = []
    backend = _FakeBackend(events)
    tensor_factory = _FakeTensorFactory(events)

    with pytest.raises(ValueError, match="expected 2097152"):
        PackedArenaLease.open(
            plan=_plan(granularity_bytes=1),
            tp_rank=0,
            device_index=0,
            backend=backend,
            tensor_factory=tensor_factory,
            fence=lambda: None,
        )
    assert events == []


def test_backend_runtime_granularity_must_match_plan_contract() -> None:
    events: list[tuple[object, ...]] = []

    class WrongGranularityBackend(_FakeBackend):
        def allocation_granularity(self, *, device_index: int) -> int:
            self._event("allocation_granularity", 64 * 1024, device_index)
            return 64 * 1024

    backend = WrongGranularityBackend(events)
    with pytest.raises(ValueError, match="backend reports 65536-byte"):
        PackedArenaLease.open(
            plan=_plan(),
            tp_rank=0,
            device_index=0,
            backend=backend,
            tensor_factory=_FakeTensorFactory(events),
            fence=lambda: None,
        )
    assert events == [("allocation_granularity", 64 * 1024, 0)]


def test_arena_sizes_bindings_offsets_and_scratch_match_plan() -> None:
    plan = _plan()
    lease, _, tensor_factory, events = _open(plan)

    assert lease.bucket_names == ("narrow", "wide")
    accounting_by_bucket = {accounting.bucket: accounting for accounting in plan.bucket_accounting}
    for bucket_name in lease.bucket_names:
        descriptor = lease.bucket(bucket_name)
        accounting = accounting_by_bucket[bucket_name]
        assert descriptor.size_bytes == (accounting.total_allocated_bytes_by_rank[1])
        assert descriptor.size_bytes % CANN_VMM_GRANULARITY_BYTES == 0
        assert descriptor.base_address % CANN_VMM_GRANULARITY_BYTES == 0
        assert lease.tensor(bucket_name) is (tensor_factory.bindings[descriptor.base_address].token)
        assert (
            "bind",
            descriptor.base_address,
            descriptor.size_bytes,
            1,
        ) in events

        scratch = lease.scratch_view(bucket_name)
        assert scratch is not None
        expected_scratch = next(spec for spec in plan.scratch if spec.bucket == bucket_name)
        assert scratch.byte_offset == (
            (descriptor.persistent_bytes + CANN_VMM_GRANULARITY_BYTES - 1)
            // CANN_VMM_GRANULARITY_BYTES
            * CANN_VMM_GRANULARITY_BYTES
        )
        assert scratch.size_bytes == (expected_scratch.max_pages_per_rank * expected_scratch.page_size_bytes)
        assert scratch.allocated_size_bytes == (
            (scratch.size_bytes + CANN_VMM_GRANULARITY_BYTES - 1)
            // CANN_VMM_GRANULARITY_BYTES
            * CANN_VMM_GRANULARITY_BYTES
        )
        assert scratch.data_ptr == (descriptor.base_address + scratch.byte_offset)

    replicated = plan.map_replicated(
        "hot",
        "replicated_wide",
        6,
        tp_rank=1,
        copy_index=1,
    )
    replicated_view = lease.resolve(replicated)
    assert replicated_view.byte_offset == replicated.physical_offset_bytes
    assert replicated_view.size_bytes == WIDE_PAGE_BYTES
    assert replicated_view.data_ptr == (lease.bucket("wide").base_address + replicated.physical_offset_bytes)
    sentinel = plan.sentinel_address(
        "hot",
        "replicated_wide",
        tp_rank=1,
        copy_index=1,
    )
    sentinel_view = lease.resolve(sentinel)
    assert sentinel_view.byte_offset == sentinel.segment_base_bytes
    assert sentinel_view.size_bytes == WIDE_PAGE_BYTES

    owner = next(
        address
        for block_id in range(1, 10)
        if (
            address := plan.map_c128(
                "cold",
                "owner_narrow",
                block_id,
                copy_index=1,
            )
        ).tp_rank
        == 1
    )
    owner_view = lease.resolve(owner)
    assert owner_view.data_ptr == (lease.bucket("narrow").base_address + owner.physical_offset_bytes)

    remote_owner = next(
        address
        for block_id in range(1, 10)
        if (
            address := plan.map_c128(
                "cold",
                "owner_wide",
                block_id,
            )
        ).tp_rank
        == 0
    )
    with pytest.raises(ValueError, match="not local rank 1"):
        lease.resolve(remote_owner)

    malformed = replace(
        replicated,
        physical_offset_bytes=replicated.segment_base_bytes + replicated.segment_allocated_bytes,
    )
    with pytest.raises(ValueError, match="outside bucket"):
        lease.resolve(malformed)


def test_close_fences_drops_bindings_then_releases_in_reverse_order() -> None:
    lease, _, _, events = _open(_plan())
    buckets = [lease.bucket(name) for name in lease.bucket_names]
    events.clear()

    lease.close()
    assert lease.state is PackedArenaState.CLOSED
    assert events == [
        ("close_binding", buckets[1].base_address),
        ("close_binding", buckets[0].base_address),
        ("fence",),
        (
            "unmap",
            buckets[1].base_address,
            buckets[1].size_bytes,
            1,
        ),
        ("free", "physical-2", 1),
        (
            "release",
            buckets[1].base_address,
            buckets[1].size_bytes,
            1,
        ),
        (
            "unmap",
            buckets[0].base_address,
            buckets[0].size_bytes,
            1,
        ),
        ("free", "physical-1", 1),
        (
            "release",
            buckets[0].base_address,
            buckets[0].size_bytes,
            1,
        ),
    ]

    lease.close()
    with pytest.raises(PackedArenaClosedError, match="closed"):
        lease.bucket("wide")


def test_close_marks_lease_non_open_before_callbacks_run() -> None:
    lease, _, _, _ = _open(_plan())
    callback_states: list[PackedArenaState] = []

    def inspect_state() -> None:
        callback_states.append(lease.state)
        with pytest.raises(PackedArenaClosedError, match="closing"):
            lease.tensor("wide")

    lease._fence = inspect_state
    lease.close()
    assert callback_states == [PackedArenaState.CLOSING]
    assert lease.state is PackedArenaState.CLOSED


def test_open_failure_rolls_back_partial_and_complete_buckets() -> None:
    events: list[tuple[object, ...]] = []
    backend = _FakeBackend(events, fail_calls={"map": {2}})
    plan = _plan()
    with pytest.raises(RuntimeError, match="injected map failure 2"):
        _open(plan, backend=backend, events=events)

    assert ("fence",) in events
    first_base = 0x1_0000_0000
    second_base = 0x2_0000_0000
    assert ("close_binding", first_base) in events
    assert ("free", "physical-2", 1) in events
    assert (
        "release",
        second_base,
        plan.bucket_accounting[1].total_allocated_bytes_by_rank[1],
        1,
    ) in events
    assert any(event[0] == "unmap" and event[1] == first_base for event in events)


def test_failed_unmap_keeps_backing_live_and_close_can_retry() -> None:
    events: list[tuple[object, ...]] = []
    backend = _FakeBackend(events, fail_calls={"unmap": {1}})
    lease, _, _, _ = _open(_plan(), backend=backend, events=events)
    buckets = [lease.bucket(name) for name in lease.bucket_names]
    events.clear()

    with pytest.raises(PackedArenaCleanupError, match="wide:unmap"):
        lease.close()
    assert lease.state is PackedArenaState.CLEANUP_FAILED
    assert (
        "free",
        "physical-2",
        1,
    ) not in events
    assert not any(event[0] == "release" and event[1] == buckets[1].base_address for event in events)
    assert ("free", "physical-1", 1) in events
    with pytest.raises(PackedArenaClosedError, match="cleanup_failed"):
        lease.tensor("wide")

    events.clear()
    lease.close()
    assert lease.state is PackedArenaState.CLOSED
    assert events == [
        (
            "unmap",
            buckets[1].base_address,
            buckets[1].size_bytes,
            1,
        ),
        ("free", "physical-2", 1),
        (
            "release",
            buckets[1].base_address,
            buckets[1].size_bytes,
            1,
        ),
    ]
