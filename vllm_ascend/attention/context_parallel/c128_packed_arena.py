# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Feature-gated lifetime contract for one rank's packed C128 VMM arenas.

The concrete CANN and torch_npu adapters intentionally live outside this
module.  Keeping those two seams behind protocols makes the allocation,
addressing, and teardown contract CPU-testable without importing torch.

Each page-size bucket is one contiguous local virtual address reservation with
locally owned physical backing.  The :class:`PackedPoolPlan` offsets select
segments and pages inside it.  Peer-import lifetime is a separate protocol and
is deliberately not implied by this lease.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .c128_packed_pool import (
    BucketPhysicalBytes,
    PackedBlockAddress,
    PackedPoolPlan,
)

CANN_VMM_GRANULARITY_BYTES = 2 * 1024 * 1024


class PackedArenaBackend(Protocol):
    """CANN VMM operations required by the packed-arena lease."""

    def allocation_granularity(self, *, device_index: int) -> int:
        """Query the device's minimum physical-allocation granularity."""

    def reserve_address(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        """Reserve an aligned local virtual address range."""

    def allocate_physical(
        self,
        *,
        size_bytes: int,
        device_index: int,
    ) -> object:
        """Allocate a locally owned physical handle for ``size_bytes``."""

    def map_physical(
        self,
        *,
        base_address: int,
        size_bytes: int,
        physical_handle: object,
        device_index: int,
    ) -> None:
        """Map ``physical_handle`` into the reserved local address range."""

    def zero_mapped(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        """Zero a newly mapped range, completing before this call returns."""

    def unmap(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        """Remove a physical mapping from a local virtual address range."""

    def free_physical(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> None:
        """Release a locally owned physical handle."""

    def release_address(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        """Release a local virtual address reservation."""


class PackedArenaTensorBinding(Protocol):
    """Owns the non-owning torch tensor/storage aliases for one VMM range.

    ``close`` must drop every alias created through this binding.  Callers must
    drop downstream tensor views before the arena lease is closed.
    """

    def tensor(self) -> object:
        """Return the root byte tensor spanning the complete arena."""

    def close(self) -> None:
        """Drop tensor/storage aliases without touching the VMM allocation."""


class PackedArenaTensorFactory(Protocol):
    """Construct a local-device tensor binding over mapped virtual memory."""

    def bind(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> PackedArenaTensorBinding:
        """Bind ``size_bytes`` at ``base_address`` to the local NPU device.

        The operation must be failure-atomic: if it raises, it must already
        have discarded every tensor/storage alias it constructed.
        """


class PackedArenaState(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLEANUP_FAILED = "cleanup_failed"
    CLOSED = "closed"


class PackedArenaClosedError(RuntimeError):
    """Raised when a closed or partially cleaned lease is dereferenced."""


@dataclass(frozen=True)
class PackedArenaCleanupFailure:
    operation: str
    bucket: str
    error: BaseException


class PackedArenaCleanupError(RuntimeError):
    """One or more teardown operations failed.

    The lease remains retryable when a driver resource could not be released.
    """

    def __init__(
        self,
        failures: tuple[PackedArenaCleanupFailure, ...],
    ) -> None:
        self.failures = failures
        detail = "; ".join(f"{failure.bucket}:{failure.operation}: {failure.error}" for failure in failures)
        super().__init__(f"packed-arena cleanup failed: {detail}")


class PackedArenaOpenError(RuntimeError):
    """Allocation failed and rollback also left resources for explicit retry."""

    def __init__(
        self,
        *,
        cause: BaseException,
        cleanup_error: PackedArenaCleanupError,
        lease: PackedArenaLease,
    ) -> None:
        self.cause = cause
        self.cleanup_error = cleanup_error
        self.lease = lease
        super().__init__(f"packed-arena creation failed and rollback was incomplete: {cause}; {cleanup_error}")


@dataclass(frozen=True)
class PackedArenaBucket:
    """One rank-local bucket arena and its plan-derived accounting."""

    bucket: str
    page_size_bytes: int
    base_address: int
    size_bytes: int
    persistent_bytes: int
    scratch_region_bytes: int


@dataclass(frozen=True)
class PackedArenaByteView:
    """A validated byte interval inside a bucket arena."""

    bucket: str
    byte_offset: int
    size_bytes: int
    data_ptr: int


@dataclass(frozen=True)
class PackedArenaScratchView:
    """Scratch payload and its tail allocation inside a bucket arena."""

    bucket: str
    byte_offset: int
    size_bytes: int
    allocated_size_bytes: int
    data_ptr: int


@dataclass
class _BucketAllocation:
    descriptor: PackedArenaBucket
    physical_handle: object | None = None
    binding: PackedArenaTensorBinding | None = None
    mapped: bool = False
    address_reserved: bool = True


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _validate_plan_for_vmm(
    plan: PackedPoolPlan,
    *,
    alignment_bytes: int,
) -> None:
    if alignment_bytes != CANN_VMM_GRANULARITY_BYTES:
        raise ValueError(f"packed C128 VMM arenas require the measured 2 MiB CANN granularity, got {alignment_bytes}")
    for group in plan.groups:
        for component in group.components:
            if component.allocation_granularity_bytes != alignment_bytes:
                raise ValueError(
                    f"component {group.name}/{component.name} uses "
                    f"{component.allocation_granularity_bytes}-byte allocation "
                    f"granularity, expected {alignment_bytes}"
                )
    for scratch in plan.scratch:
        if scratch.allocation_granularity_bytes != alignment_bytes:
            raise ValueError(
                f"scratch bucket {scratch.bucket} uses "
                f"{scratch.allocation_granularity_bytes}-byte allocation "
                f"granularity, expected {alignment_bytes}"
            )
    for bucket in plan.bucket_accounting:
        for rank, size_bytes in enumerate(bucket.total_allocated_bytes_by_rank):
            if size_bytes <= 0 or size_bytes % alignment_bytes:
                raise ValueError(
                    f"bucket {bucket.bucket} rank {rank} size {size_bytes} is "
                    f"not a positive {alignment_bytes}-byte multiple"
                )


class PackedArenaLease:
    """Explicit lease over all packed bucket arenas for one TP rank."""

    def __init__(
        self,
        *,
        plan: PackedPoolPlan,
        tp_rank: int,
        device_index: int,
        alignment_bytes: int,
        backend: PackedArenaBackend,
        tensor_factory: PackedArenaTensorFactory,
        fence: Callable[[], None],
    ) -> None:
        self.plan = plan
        self.tp_rank = tp_rank
        self.device_index = device_index
        self.alignment_bytes = alignment_bytes
        self._backend = backend
        self._tensor_factory = tensor_factory
        self._fence = fence
        self._allocations: dict[str, _BucketAllocation] = {}
        self._state = PackedArenaState.OPEN
        self._bindings_closed = False
        self._fenced = False

    @classmethod
    def open(
        cls,
        *,
        plan: PackedPoolPlan,
        tp_rank: int,
        device_index: int,
        backend: PackedArenaBackend,
        tensor_factory: PackedArenaTensorFactory,
        fence: Callable[[], None],
        alignment_bytes: int = CANN_VMM_GRANULARITY_BYTES,
    ) -> PackedArenaLease:
        """Allocate, map, zero, and bind every bucket for one rank."""
        if not 0 <= tp_rank < plan.tp_size:
            raise ValueError(f"tp_rank {tp_rank} is outside [0, {plan.tp_size})")
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        _validate_plan_for_vmm(plan, alignment_bytes=alignment_bytes)
        runtime_granularity = backend.allocation_granularity(device_index=device_index)
        if runtime_granularity != alignment_bytes:
            raise ValueError(f"backend reports {runtime_granularity}-byte VMM granularity, expected {alignment_bytes}")

        lease = cls(
            plan=plan,
            tp_rank=tp_rank,
            device_index=device_index,
            alignment_bytes=alignment_bytes,
            backend=backend,
            tensor_factory=tensor_factory,
            fence=fence,
        )
        try:
            for accounting in plan.bucket_accounting:
                lease._open_bucket(accounting)
        except BaseException as error:
            try:
                lease.close()
            except PackedArenaCleanupError as cleanup_error:
                raise PackedArenaOpenError(
                    cause=error,
                    cleanup_error=cleanup_error,
                    lease=lease,
                ) from error
            raise
        return lease

    @property
    def state(self) -> PackedArenaState:
        return self._state

    @property
    def bucket_names(self) -> tuple[str, ...]:
        return tuple(self._allocations)

    def _require_open(self) -> None:
        if self._state is not PackedArenaState.OPEN:
            raise PackedArenaClosedError(f"packed-arena lease is {self._state.value}")

    def _open_bucket(self, accounting: BucketPhysicalBytes) -> None:
        size_bytes = accounting.total_allocated_bytes_by_rank[self.tp_rank]
        base_address = self._backend.reserve_address(
            size_bytes=size_bytes,
            alignment_bytes=self.alignment_bytes,
            device_index=self.device_index,
        )
        descriptor = PackedArenaBucket(
            bucket=accounting.bucket,
            page_size_bytes=accounting.page_size_bytes,
            base_address=base_address,
            size_bytes=size_bytes,
            persistent_bytes=(accounting.persistent_allocated_bytes_by_rank[self.tp_rank]),
            scratch_region_bytes=(accounting.scratch_region_bytes_by_rank[self.tp_rank]),
        )
        allocation = _BucketAllocation(descriptor=descriptor)
        self._allocations[accounting.bucket] = allocation

        if base_address <= 0 or base_address % self.alignment_bytes:
            raise ValueError(
                f"backend returned unaligned base address {base_address:#x} for bucket {accounting.bucket}"
            )
        allocation.physical_handle = self._backend.allocate_physical(
            size_bytes=size_bytes,
            device_index=self.device_index,
        )
        self._backend.map_physical(
            base_address=base_address,
            size_bytes=size_bytes,
            physical_handle=allocation.physical_handle,
            device_index=self.device_index,
        )
        allocation.mapped = True
        self._backend.zero_mapped(
            base_address=base_address,
            size_bytes=size_bytes,
            device_index=self.device_index,
        )
        allocation.binding = self._tensor_factory.bind(
            base_address=base_address,
            size_bytes=size_bytes,
            device_index=self.device_index,
        )

    def bucket(self, bucket: str) -> PackedArenaBucket:
        self._require_open()
        try:
            return self._allocations[bucket].descriptor
        except KeyError as error:
            raise ValueError(f"unknown packed-arena bucket: {bucket}") from error

    def tensor(self, bucket: str) -> object:
        """Return the root tensor while retaining lease ownership."""
        self._require_open()
        try:
            binding = self._allocations[bucket].binding
        except KeyError as error:
            raise ValueError(f"unknown packed-arena bucket: {bucket}") from error
        if binding is None:
            raise RuntimeError(f"bucket {bucket} has no tensor binding")
        return binding.tensor()

    def resolve(
        self,
        address: PackedBlockAddress,
        *,
        size_bytes: int | None = None,
    ) -> PackedArenaByteView:
        """Resolve a plan address into this rank's mapped virtual range."""
        self._require_open()
        if address.tp_rank != self.tp_rank:
            raise ValueError(f"address belongs to TP rank {address.tp_rank}, not local rank {self.tp_rank}")
        bucket = self.bucket(address.bucket)
        resolved_size = bucket.page_size_bytes if size_bytes is None else size_bytes
        if resolved_size <= 0:
            raise ValueError("size_bytes must be positive")
        segment_stop = address.segment_base_bytes + address.segment_allocated_bytes
        view_stop = address.physical_offset_bytes + resolved_size
        if (
            address.segment_base_bytes < 0
            or address.physical_offset_bytes < address.segment_base_bytes
            or view_stop > segment_stop
            or segment_stop > bucket.size_bytes
        ):
            raise ValueError(
                f"address interval [{address.physical_offset_bytes}, "
                f"{view_stop}) is outside bucket {address.bucket} or its "
                "component segment"
            )
        return PackedArenaByteView(
            bucket=address.bucket,
            byte_offset=address.physical_offset_bytes,
            size_bytes=resolved_size,
            data_ptr=bucket.base_address + address.physical_offset_bytes,
        )

    def scratch_view(
        self,
        bucket: str,
    ) -> PackedArenaScratchView | None:
        """Return the bounded scratch payload, excluding alignment padding."""
        self._require_open()
        descriptor = self.bucket(bucket)
        scratch_spec = next(
            (item for item in self.plan.scratch if item.bucket == bucket),
            None,
        )
        if scratch_spec is None or scratch_spec.max_pages_per_rank == 0:
            return None
        payload_bytes = scratch_spec.max_pages_per_rank * scratch_spec.page_size_bytes
        allocated_bytes = _round_up(payload_bytes, self.alignment_bytes)
        scratch_base = descriptor.size_bytes - allocated_bytes
        if scratch_base + payload_bytes > descriptor.size_bytes:
            raise RuntimeError(f"scratch payload for bucket {bucket} exceeds its arena")
        if scratch_base < descriptor.persistent_bytes:
            raise RuntimeError(f"scratch allocation for bucket {bucket} overlaps persistent segments")
        return PackedArenaScratchView(
            bucket=bucket,
            byte_offset=scratch_base,
            size_bytes=payload_bytes,
            allocated_size_bytes=allocated_bytes,
            data_ptr=descriptor.base_address + scratch_base,
        )

    def close(self) -> None:
        """Drop owned aliases, fence queued work, then tear down VMM resources.

        Teardown order is ``unmap -> free physical -> release address``.  If
        unmap fails, that allocation keeps its physical handle and address so a
        later ``close`` call can retry without creating a use-after-free.
        """
        if self._state is PackedArenaState.CLOSED:
            return
        self._state = PackedArenaState.CLOSING

        failures: list[PackedArenaCleanupFailure] = []
        if not self._bindings_closed:
            for bucket, allocation in reversed(tuple(self._allocations.items())):
                if allocation.binding is None:
                    continue
                try:
                    allocation.binding.close()
                    allocation.binding = None
                except BaseException as error:
                    failures.append(
                        PackedArenaCleanupFailure(
                            operation="close_tensor_binding",
                            bucket=bucket,
                            error=error,
                        )
                    )
            if failures:
                self._state = PackedArenaState.CLEANUP_FAILED
                raise PackedArenaCleanupError(tuple(failures))
            self._bindings_closed = True

        if not self._fenced:
            try:
                if any(allocation.mapped for allocation in self._allocations.values()):
                    self._fence()
            except BaseException as error:
                self._state = PackedArenaState.CLEANUP_FAILED
                raise PackedArenaCleanupError(
                    (
                        PackedArenaCleanupFailure(
                            operation="fence",
                            bucket="*",
                            error=error,
                        ),
                    )
                ) from error
            self._fenced = True

        for bucket, allocation in reversed(tuple(self._allocations.items())):
            descriptor = allocation.descriptor
            if allocation.mapped:
                try:
                    self._backend.unmap(
                        base_address=descriptor.base_address,
                        size_bytes=descriptor.size_bytes,
                        device_index=self.device_index,
                    )
                    allocation.mapped = False
                except BaseException as error:
                    failures.append(
                        PackedArenaCleanupFailure(
                            operation="unmap",
                            bucket=bucket,
                            error=error,
                        )
                    )
                    continue

            if allocation.physical_handle is not None:
                try:
                    self._backend.free_physical(
                        physical_handle=allocation.physical_handle,
                        device_index=self.device_index,
                    )
                    allocation.physical_handle = None
                except BaseException as error:
                    failures.append(
                        PackedArenaCleanupFailure(
                            operation="free_physical",
                            bucket=bucket,
                            error=error,
                        )
                    )

            if allocation.address_reserved:
                try:
                    self._backend.release_address(
                        base_address=descriptor.base_address,
                        size_bytes=descriptor.size_bytes,
                        device_index=self.device_index,
                    )
                    allocation.address_reserved = False
                except BaseException as error:
                    failures.append(
                        PackedArenaCleanupFailure(
                            operation="release_address",
                            bucket=bucket,
                            error=error,
                        )
                    )

        if failures:
            self._state = PackedArenaState.CLEANUP_FAILED
            raise PackedArenaCleanupError(tuple(failures))
        self._state = PackedArenaState.CLOSED

    def __enter__(self) -> PackedArenaLease:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


def maybe_create_packed_arena(
    *,
    enabled: bool,
    plan_factory: Callable[[], PackedPoolPlan] | None = None,
    tp_rank: int = 0,
    device_index: int = 0,
    backend: PackedArenaBackend | None = None,
    tensor_factory: PackedArenaTensorFactory | None = None,
    fence: Callable[[], None] | None = None,
    alignment_bytes: int = CANN_VMM_GRANULARITY_BYTES,
) -> PackedArenaLease | None:
    """Create a lease only after the feature gate is enabled.

    The disabled path returns before invoking the planner or touching backend,
    tensor, or synchronization objects.  No production module imports this
    prototype yet, so the existing feature-off hot path is unchanged.
    """
    if not enabled:
        return None
    if plan_factory is None:
        raise ValueError("plan_factory is required when packed arena is enabled")
    if backend is None:
        raise ValueError("backend is required when packed arena is enabled")
    if tensor_factory is None:
        raise ValueError("tensor_factory is required when packed arena is enabled")
    if fence is None:
        raise ValueError("fence is required when packed arena is enabled")
    return PackedArenaLease.open(
        plan=plan_factory(),
        tp_rank=tp_rank,
        device_index=device_index,
        backend=backend,
        tensor_factory=tensor_factory,
        fence=fence,
        alignment_bytes=alignment_bytes,
    )
