# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Pure planning model for a packed DeepSeek-V4 C128 cache pool.

This module deliberately has no torch, allocator, scheduler, or worker
dependency.  It defines the address and byte-accounting contract that a later
feature-gated runtime implementation must preserve.

Block zero is a table-padding sentinel.  Data block IDs are one-based within a
cache group.  When packing is enabled, each group receives a disjoint global
range and each physical component maps that range either to a dense replicated
pool or to a compact C128 owner pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SENTINEL_BLOCK_ID = 0


class PackedPlacement(str, Enum):
    """Physical placement of a packed-pool component."""

    REPLICATED = "replicated"
    C128_OWNER = "c128_owner"


class SentinelBlockError(ValueError):
    """Raised when block-table padding is used as a physical cache address."""


@dataclass(frozen=True)
class PackedPoolComponentSpec:
    """One physical cache family driven by a group's logical block IDs.

    ``copies`` represents independent layer/cache copies with the same page
    shape.  Copies receive disjoint aligned byte segments.
    """

    name: str
    bucket: str
    page_size_bytes: int
    copies: int
    placement: PackedPlacement
    allocation_granularity_bytes: int = 1


@dataclass(frozen=True)
class PackedPoolGroupSpec:
    """A scheduler cache group and all physical components it addresses."""

    name: str
    logical_blocks: int
    components: tuple[PackedPoolComponentSpec, ...]


@dataclass(frozen=True)
class PackedPoolScratchSpec:
    """A reusable, per-rank scratch bound for one page-size bucket."""

    bucket: str
    page_size_bytes: int
    max_pages_per_rank: int
    allocation_granularity_bytes: int = 1


@dataclass(frozen=True)
class PackedLogicalRange:
    """Disjoint global data-block interval assigned to one cache group."""

    group_name: str
    start: int
    stop: int

    @property
    def logical_blocks(self) -> int:
        return self.stop - self.start

    def contains(self, global_block_id: int) -> bool:
        return self.start <= global_block_id < self.stop


@dataclass(frozen=True)
class LogicalBlockRef:
    """A group-local block recovered from a packed global block ID."""

    group_name: str
    group_block_id: int


@dataclass(frozen=True)
class PackedBlockAddress:
    """A physical packed-pool address."""

    bucket: str
    placement: PackedPlacement
    physical_slot: int
    copy_index: int
    global_block_id: int
    tp_rank: int
    segment_base_bytes: int
    physical_offset_bytes: int
    segment_allocated_bytes: int
    owner_rank: int | None = None


@dataclass(frozen=True)
class BucketPhysicalBytes:
    """Exact persistent and bounded-scratch accounting for one bucket."""

    bucket: str
    page_size_bytes: int
    replicated_pages_per_rank: int
    owner_pages_by_rank: tuple[int, ...]
    sentinel_pages_per_rank: int
    scratch_pages_per_rank: int
    persistent_allocated_bytes_by_rank: tuple[int, ...]
    scratch_region_bytes_by_rank: tuple[int, ...]
    total_allocated_bytes_by_rank: tuple[int, ...]

    @property
    def persistent_pages_by_rank(self) -> tuple[int, ...]:
        return tuple(
            self.replicated_pages_per_rank + owner_pages + self.sentinel_pages_per_rank
            for owner_pages in self.owner_pages_by_rank
        )

    @property
    def persistent_bytes_by_rank(self) -> tuple[int, ...]:
        return self.persistent_allocated_bytes_by_rank

    @property
    def payload_bytes_by_rank(self) -> tuple[int, ...]:
        return tuple(pages * self.page_size_bytes for pages in self.persistent_pages_by_rank)

    @property
    def total_bytes_by_rank(self) -> tuple[int, ...]:
        return self.total_allocated_bytes_by_rank

    @property
    def padding_bytes_by_rank(self) -> tuple[int, ...]:
        scratch_payload_bytes = self.scratch_pages_per_rank * self.page_size_bytes
        return tuple(
            total_bytes - payload_bytes - scratch_payload_bytes
            for total_bytes, payload_bytes in zip(
                self.total_allocated_bytes_by_rank,
                self.payload_bytes_by_rank,
            )
        )


@dataclass(frozen=True)
class _ComponentLayout:
    group_range: PackedLogicalRange
    spec: PackedPoolComponentSpec
    owner_counts: tuple[int, ...] | None
    segment_bases_by_copy: tuple[tuple[int, ...], ...]
    segment_allocated_bytes_by_copy: tuple[tuple[int, ...], ...]


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _first_owned_block(
    logical_range: PackedLogicalRange,
    owner_rank: int,
    tp_size: int,
) -> int:
    return logical_range.start + (owner_rank - logical_range.start) % tp_size


def _owned_block_count(
    logical_range: PackedLogicalRange,
    owner_rank: int,
    tp_size: int,
) -> int:
    first = _first_owned_block(logical_range, owner_rank, tp_size)
    if first >= logical_range.stop:
        return 0
    return (logical_range.stop - 1 - first) // tp_size + 1


class PackedPoolPlan:
    """Deterministic logical ranges, physical mappings, and byte accounting."""

    def __init__(
        self,
        *,
        global_block_capacity: int,
        tp_size: int,
        groups: tuple[PackedPoolGroupSpec, ...],
        scratch: tuple[PackedPoolScratchSpec, ...] = (),
    ) -> None:
        if global_block_capacity <= 1:
            raise ValueError("global_block_capacity must include sentinel block 0 and at least one data block")
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if not groups:
            raise ValueError("at least one packed-pool group is required")

        self.global_block_capacity = global_block_capacity
        self.tp_size = tp_size
        self.groups = groups
        self.scratch = scratch
        self.usable_data_capacity = global_block_capacity - 1

        self._ranges: dict[str, PackedLogicalRange] = {}
        self._component_layouts: dict[tuple[str, str], _ComponentLayout] = {}
        self._bucket_page_sizes: dict[str, int] = {}

        group_names: set[str] = set()
        cursor = 1
        for group in groups:
            self._validate_group(group, group_names)
            logical_range = PackedLogicalRange(
                group_name=group.name,
                start=cursor,
                stop=cursor + group.logical_blocks,
            )
            self._ranges[group.name] = logical_range
            cursor = logical_range.stop

        self.used_logical_blocks = cursor - 1
        if self.used_logical_blocks > self.usable_data_capacity:
            raise ValueError(
                "sum of group logical_blocks exceeds usable data capacity "
                "after reserving sentinel block 0: "
                f"{self.used_logical_blocks} > {self.usable_data_capacity}"
            )

        replicated_pages: dict[str, int] = {}
        owner_pages: dict[str, list[int]] = {}
        sentinel_pages: dict[str, int] = {}
        bucket_byte_cursors = {bucket: [0] * tp_size for bucket in self._bucket_page_sizes}
        for group in groups:
            logical_range = self._ranges[group.name]
            for component in group.components:
                key = (group.name, component.name)
                if component.placement is PackedPlacement.REPLICATED:
                    data_counts = (group.logical_blocks,) * tp_size
                    replicated_pages[component.bucket] = (
                        replicated_pages.get(component.bucket, 0) + group.logical_blocks * component.copies
                    )
                    owner_counts = None
                else:
                    data_counts = tuple(
                        _owned_block_count(
                            logical_range,
                            rank,
                            tp_size,
                        )
                        for rank in range(tp_size)
                    )
                    owner_counts = data_counts
                    current_owner_pages = owner_pages.setdefault(
                        component.bucket,
                        [0] * tp_size,
                    )
                    owner_pages[component.bucket] = [
                        current + count * component.copies
                        for current, count in zip(
                            current_owner_pages,
                            data_counts,
                        )
                    ]
                sentinel_pages[component.bucket] = sentinel_pages.get(component.bucket, 0) + component.copies
                allocation_counts = tuple(data_count + 1 for data_count in data_counts)

                copy_bases: list[tuple[int, ...]] = []
                copy_allocations: list[tuple[int, ...]] = []
                for _copy_index in range(component.copies):
                    bases: list[int] = []
                    allocations: list[int] = []
                    for rank, count in enumerate(allocation_counts):
                        cursor_bytes = bucket_byte_cursors[component.bucket][rank]
                        base_bytes = _round_up(
                            cursor_bytes,
                            component.allocation_granularity_bytes,
                        )
                        allocated_bytes = _round_up(
                            count * component.page_size_bytes,
                            component.allocation_granularity_bytes,
                        )
                        bucket_byte_cursors[component.bucket][rank] = base_bytes + allocated_bytes
                        bases.append(base_bytes)
                        allocations.append(allocated_bytes)
                    copy_bases.append(tuple(bases))
                    copy_allocations.append(tuple(allocations))

                self._component_layouts[key] = _ComponentLayout(
                    group_range=logical_range,
                    spec=component,
                    owner_counts=owner_counts,
                    segment_bases_by_copy=tuple(copy_bases),
                    segment_allocated_bytes_by_copy=tuple(copy_allocations),
                )

        scratch_by_bucket = self._validate_scratch(scratch)
        persistent_bytes_by_bucket = {bucket: tuple(cursors) for bucket, cursors in bucket_byte_cursors.items()}
        scratch_region_bytes_by_bucket: dict[str, tuple[int, ...]] = {}
        total_bytes_by_bucket: dict[str, tuple[int, ...]] = {}
        for bucket, cursors in bucket_byte_cursors.items():
            scratch_spec = scratch_by_bucket.get(bucket)
            totals: list[int] = []
            scratch_regions: list[int] = []
            for persistent_bytes in cursors:
                if scratch_spec is None or scratch_spec.max_pages_per_rank == 0:
                    totals.append(persistent_bytes)
                    scratch_regions.append(0)
                    continue
                scratch_base = _round_up(
                    persistent_bytes,
                    scratch_spec.allocation_granularity_bytes,
                )
                scratch_allocated_bytes = _round_up(
                    scratch_spec.max_pages_per_rank * scratch_spec.page_size_bytes,
                    scratch_spec.allocation_granularity_bytes,
                )
                total_bytes = scratch_base + scratch_allocated_bytes
                totals.append(total_bytes)
                scratch_regions.append(total_bytes - persistent_bytes)
            total_bytes_by_bucket[bucket] = tuple(totals)
            scratch_region_bytes_by_bucket[bucket] = tuple(scratch_regions)

        bucket_names = set(self._bucket_page_sizes)
        self._bucket_accounting = tuple(
            BucketPhysicalBytes(
                bucket=bucket,
                page_size_bytes=self._bucket_page_sizes[bucket],
                replicated_pages_per_rank=replicated_pages.get(bucket, 0),
                owner_pages_by_rank=tuple(owner_pages.get(bucket, [0] * tp_size)),
                sentinel_pages_per_rank=sentinel_pages.get(bucket, 0),
                scratch_pages_per_rank=(
                    scratch_by_bucket[bucket].max_pages_per_rank if bucket in scratch_by_bucket else 0
                ),
                persistent_allocated_bytes_by_rank=(persistent_bytes_by_bucket[bucket]),
                scratch_region_bytes_by_rank=(scratch_region_bytes_by_bucket[bucket]),
                total_allocated_bytes_by_rank=total_bytes_by_bucket[bucket],
            )
            for bucket in sorted(bucket_names)
        )

    def _validate_group(
        self,
        group: PackedPoolGroupSpec,
        group_names: set[str],
    ) -> None:
        if not group.name:
            raise ValueError("group name must not be empty")
        if group.name in group_names:
            raise ValueError(f"duplicate group name: {group.name}")
        group_names.add(group.name)
        if group.logical_blocks <= 0:
            raise ValueError(f"group {group.name} logical_blocks must be positive")
        if not group.components:
            raise ValueError(f"group {group.name} requires at least one physical component")

        component_names: set[str] = set()
        for component in group.components:
            if not component.name:
                raise ValueError("component name must not be empty")
            if component.name in component_names:
                raise ValueError(f"duplicate component name in group {group.name}: {component.name}")
            component_names.add(component.name)
            if not component.bucket:
                raise ValueError("bucket name must not be empty")
            if component.page_size_bytes <= 0:
                raise ValueError(f"component {component.name} page_size_bytes must be positive")
            if component.copies <= 0:
                raise ValueError(f"component {component.name} copies must be positive")
            if component.allocation_granularity_bytes <= 0:
                raise ValueError(f"component {component.name} allocation granularity must be positive")
            if not isinstance(component.placement, PackedPlacement):
                raise ValueError(f"component {component.name} has invalid placement {component.placement!r}")
            known_page_size = self._bucket_page_sizes.setdefault(
                component.bucket,
                component.page_size_bytes,
            )
            if known_page_size != component.page_size_bytes:
                raise ValueError(
                    f"bucket {component.bucket} mixes page sizes {known_page_size} and {component.page_size_bytes}"
                )

    def _validate_scratch(
        self,
        scratch: tuple[PackedPoolScratchSpec, ...],
    ) -> dict[str, PackedPoolScratchSpec]:
        scratch_by_bucket: dict[str, PackedPoolScratchSpec] = {}
        for spec in scratch:
            if spec.bucket not in self._bucket_page_sizes:
                raise ValueError(f"scratch references unknown bucket: {spec.bucket}")
            if spec.bucket in scratch_by_bucket:
                raise ValueError(f"duplicate scratch bound for bucket: {spec.bucket}")
            if spec.page_size_bytes != self._bucket_page_sizes[spec.bucket]:
                raise ValueError(f"scratch page size for bucket {spec.bucket} does not match its persistent page size")
            if not 0 <= spec.max_pages_per_rank <= self.global_block_capacity:
                raise ValueError(f"scratch for bucket {spec.bucket} must be bounded by global_block_capacity")
            if spec.allocation_granularity_bytes <= 0:
                raise ValueError(f"scratch for bucket {spec.bucket} allocation granularity must be positive")
            scratch_by_bucket[spec.bucket] = spec
        return scratch_by_bucket

    @property
    def unused_logical_blocks(self) -> int:
        return self.usable_data_capacity - self.used_logical_blocks

    @property
    def group_ranges(self) -> tuple[PackedLogicalRange, ...]:
        return tuple(self._ranges[group.name] for group in self.groups)

    @property
    def bucket_accounting(self) -> tuple[BucketPhysicalBytes, ...]:
        return self._bucket_accounting

    def group_range(self, group_name: str) -> PackedLogicalRange:
        try:
            return self._ranges[group_name]
        except KeyError as error:
            raise ValueError(f"unknown packed-pool group: {group_name}") from error

    def encode_group_block_id(
        self,
        group_name: str,
        group_block_id: int,
    ) -> int:
        """Translate a group-local ID, preserving block-zero padding.

        Block zero is preserved for the worker's null-page adapter.
        """
        logical_range = self.group_range(group_name)
        if group_block_id == SENTINEL_BLOCK_ID:
            return SENTINEL_BLOCK_ID
        if not 1 <= group_block_id <= logical_range.logical_blocks:
            raise ValueError(
                f"group block ID {group_block_id} is outside [1, {logical_range.logical_blocks}] for {group_name}"
            )
        return logical_range.start + group_block_id - 1

    def encode_group_block_ids(
        self,
        group_name: str,
        group_block_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(self.encode_group_block_id(group_name, block_id) for block_id in group_block_ids)

    def decode_global_block_id(
        self,
        global_block_id: int,
    ) -> LogicalBlockRef | None:
        """Recover a group-local ID from a packed global ID."""
        if global_block_id == SENTINEL_BLOCK_ID:
            return None
        if global_block_id < 0 or global_block_id > self.usable_data_capacity:
            raise ValueError(f"global block ID {global_block_id} is outside [0, {self.usable_data_capacity}]")
        for logical_range in self.group_ranges:
            if logical_range.contains(global_block_id):
                return LogicalBlockRef(
                    group_name=logical_range.group_name,
                    group_block_id=global_block_id - logical_range.start + 1,
                )
        raise ValueError(f"global block ID {global_block_id} is unassigned by this plan")

    def decode_for_group(
        self,
        group_name: str,
        encoded_block_id: int,
    ) -> int:
        """Inverse of :meth:`encode_group_block_id`."""
        logical_range = self.group_range(group_name)
        if encoded_block_id == SENTINEL_BLOCK_ID:
            return SENTINEL_BLOCK_ID
        if not logical_range.contains(encoded_block_id):
            raise ValueError(
                f"global block ID {encoded_block_id} is outside group "
                f"{group_name} range [{logical_range.start}, "
                f"{logical_range.stop})"
            )
        return encoded_block_id - logical_range.start + 1

    def _component_layout(
        self,
        group_name: str,
        component_name: str,
        expected_placement: PackedPlacement,
    ) -> _ComponentLayout:
        try:
            layout = self._component_layouts[(group_name, component_name)]
        except KeyError as error:
            raise ValueError(f"unknown component {group_name}/{component_name}") from error
        if layout.spec.placement is not expected_placement:
            raise ValueError(
                f"component {group_name}/{component_name} uses "
                f"{layout.spec.placement.value}, not {expected_placement.value}"
            )
        return layout

    @staticmethod
    def _validate_copy_index(
        layout: _ComponentLayout,
        copy_index: int,
    ) -> None:
        if not 0 <= copy_index < layout.spec.copies:
            raise ValueError(f"copy_index {copy_index} is outside [0, {layout.spec.copies}) for {layout.spec.name}")

    def _validate_tp_rank(self, tp_rank: int) -> None:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank {tp_rank} is outside [0, {self.tp_size})")

    def sentinel_address(
        self,
        group_name: str,
        component_name: str,
        *,
        tp_rank: int,
        copy_index: int = 0,
    ) -> PackedBlockAddress:
        """Return the reserved local dummy page for block-table padding."""
        try:
            layout = self._component_layouts[(group_name, component_name)]
        except KeyError as error:
            raise ValueError(f"unknown component {group_name}/{component_name}") from error
        self._validate_copy_index(layout, copy_index)
        self._validate_tp_rank(tp_rank)
        segment_base_bytes = layout.segment_bases_by_copy[copy_index][tp_rank]
        segment_allocated_bytes = layout.segment_allocated_bytes_by_copy[copy_index][tp_rank]
        return PackedBlockAddress(
            bucket=layout.spec.bucket,
            placement=layout.spec.placement,
            physical_slot=SENTINEL_BLOCK_ID,
            copy_index=copy_index,
            global_block_id=SENTINEL_BLOCK_ID,
            tp_rank=tp_rank,
            segment_base_bytes=segment_base_bytes,
            physical_offset_bytes=segment_base_bytes,
            segment_allocated_bytes=segment_allocated_bytes,
        )

    def map_replicated(
        self,
        group_name: str,
        component_name: str,
        group_block_id: int,
        *,
        tp_rank: int,
        copy_index: int = 0,
    ) -> PackedBlockAddress:
        """Map a non-C128 block to a dense slot present on every rank."""
        layout = self._component_layout(
            group_name,
            component_name,
            PackedPlacement.REPLICATED,
        )
        self._validate_copy_index(layout, copy_index)
        self._validate_tp_rank(tp_rank)
        global_block_id = self.encode_group_block_id(
            group_name,
            group_block_id,
        )
        if global_block_id == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("sentinel block 0 has no persistent physical slot")
        ordinal = global_block_id - layout.group_range.start
        physical_slot = ordinal + 1
        segment_base_bytes = layout.segment_bases_by_copy[copy_index][tp_rank]
        segment_allocated_bytes = layout.segment_allocated_bytes_by_copy[copy_index][tp_rank]
        return PackedBlockAddress(
            bucket=layout.spec.bucket,
            placement=layout.spec.placement,
            physical_slot=physical_slot,
            copy_index=copy_index,
            global_block_id=global_block_id,
            tp_rank=tp_rank,
            segment_base_bytes=segment_base_bytes,
            physical_offset_bytes=(segment_base_bytes + physical_slot * layout.spec.page_size_bytes),
            segment_allocated_bytes=segment_allocated_bytes,
        )

    def unmap_replicated(
        self,
        group_name: str,
        component_name: str,
        physical_slot: int,
        *,
        tp_rank: int,
        copy_index: int,
    ) -> tuple[LogicalBlockRef, int]:
        """Inverse of :meth:`map_replicated` for one component segment."""
        layout = self._component_layout(
            group_name,
            component_name,
            PackedPlacement.REPLICATED,
        )
        self._validate_copy_index(layout, copy_index)
        self._validate_tp_rank(tp_rank)
        if physical_slot == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("physical slot 0 is the reserved sentinel page")
        if not 1 <= physical_slot <= layout.group_range.logical_blocks:
            raise ValueError(f"physical slot {physical_slot} is outside component {group_name}/{component_name}")
        global_block_id = layout.group_range.start + physical_slot - 1
        decoded = self.decode_global_block_id(global_block_id)
        assert decoded is not None
        return decoded, copy_index

    def map_c128(
        self,
        group_name: str,
        component_name: str,
        group_block_id: int,
        *,
        copy_index: int = 0,
    ) -> PackedBlockAddress:
        """Map a C128 block to its TP owner and dense owner-local slot."""
        layout = self._component_layout(
            group_name,
            component_name,
            PackedPlacement.C128_OWNER,
        )
        self._validate_copy_index(layout, copy_index)
        global_block_id = self.encode_group_block_id(
            group_name,
            group_block_id,
        )
        if global_block_id == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("sentinel block 0 has no persistent physical slot")

        owner_rank = global_block_id % self.tp_size
        assert layout.owner_counts is not None
        first_owned = _first_owned_block(
            layout.group_range,
            owner_rank,
            self.tp_size,
        )
        owner_ordinal = (global_block_id - first_owned) // self.tp_size
        physical_slot = owner_ordinal + 1
        segment_base_bytes = layout.segment_bases_by_copy[copy_index][owner_rank]
        segment_allocated_bytes = layout.segment_allocated_bytes_by_copy[copy_index][owner_rank]
        return PackedBlockAddress(
            bucket=layout.spec.bucket,
            placement=layout.spec.placement,
            physical_slot=physical_slot,
            copy_index=copy_index,
            global_block_id=global_block_id,
            tp_rank=owner_rank,
            segment_base_bytes=segment_base_bytes,
            physical_offset_bytes=(segment_base_bytes + physical_slot * layout.spec.page_size_bytes),
            segment_allocated_bytes=segment_allocated_bytes,
            owner_rank=owner_rank,
        )

    def unmap_c128(
        self,
        group_name: str,
        component_name: str,
        owner_rank: int,
        physical_slot: int,
        *,
        copy_index: int,
    ) -> tuple[LogicalBlockRef, int]:
        """Inverse of :meth:`map_c128` for one owner's component segment."""
        layout = self._component_layout(
            group_name,
            component_name,
            PackedPlacement.C128_OWNER,
        )
        self._validate_copy_index(layout, copy_index)
        self._validate_tp_rank(owner_rank)
        assert layout.owner_counts is not None
        owner_count = layout.owner_counts[owner_rank]
        if physical_slot == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("physical slot 0 is the reserved sentinel page")
        if not 1 <= physical_slot <= owner_count:
            raise ValueError(
                f"physical slot {physical_slot} is outside owner {owner_rank} component {group_name}/{component_name}"
            )
        global_block_id = (
            _first_owned_block(
                layout.group_range,
                owner_rank,
                self.tp_size,
            )
            + (physical_slot - 1) * self.tp_size
        )
        decoded = self.decode_global_block_id(global_block_id)
        assert decoded is not None
        return decoded, copy_index

    def total_physical_bytes_by_rank(
        self,
        *,
        include_scratch: bool = True,
    ) -> tuple[int, ...]:
        """Sum exact candidate bytes across all buckets."""
        totals = [0] * self.tp_size
        for bucket in self.bucket_accounting:
            bucket_bytes = bucket.total_bytes_by_rank if include_scratch else bucket.persistent_bytes_by_rank
            totals = [total + current for total, current in zip(totals, bucket_bytes)]
        return tuple(totals)

    def quota_replicated_bytes_by_rank(self) -> tuple[int, ...]:
        """Return bytes if each declared packed quota were replicated.

        Scratch is excluded.  This is a same-quota synthetic comparator, not
        a reconstruction of a multi-group shared BlockPool; real baseline
        storage must be deduplicated from its raw allocations.
        """
        bytes_per_rank = 0
        for group in self.groups:
            for component in group.components:
                bytes_per_rank += (group.logical_blocks + 1) * component.copies * component.page_size_bytes
        return (bytes_per_rank,) * self.tp_size


def translate_group_block_ids(
    plan: PackedPoolPlan | None,
    group_name: str,
    group_block_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Feature gate for scheduler-to-worker block ID translation.

    Feature-off is represented by ``plan is None`` and is an unconditional
    identity.  No packed quotas, owner accounting, or range validation exists
    on that path.
    """
    if plan is None:
        return group_block_ids
    return plan.encode_group_block_ids(group_name, group_block_ids)
