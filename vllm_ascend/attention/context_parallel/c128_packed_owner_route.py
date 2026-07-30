# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Validated routing seam from a packed C128 plan to owner-local storage.

The scheduler/worker block-table contract uses group-global block IDs:

* block zero is a padding sentinel;
* each cache group owns one disjoint positive interval;
* a C128 data block is stored on ``global_block_id % tp_size``;
* owner-local physical slot zero is reserved for the component sentinel.

The serialized packed-pool plan is the authority for group ranges, component
copies, persistent segment offsets, and bounded scratch.  This module parses
that plan without importing torch or an allocator.  It can therefore certify
the routing contract before a VMM-backed tensor view exists.

``downstream_runtime_abi_ready`` is deliberately enforced separately.  A
planner-only plan may be parsed for reference tests, but it cannot be attached
to :class:`C128OwnerShardCache` until the producer marks the runtime ABI ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .c128_packed_pool import SENTINEL_BLOCK_ID, SentinelBlockError

PAD_BLOCK_ID = -1
SUPPORTED_SCHEMA_VERSION = 1
C128_OWNER_PLACEMENT = "c128_owner"


@dataclass(frozen=True)
class C128PackedSegment:
    """One rank-local persistent or scratch byte segment."""

    rank: int
    base_bytes: int
    allocated_bytes: int

    @property
    def stop_bytes(self) -> int:
        return self.base_bytes + self.allocated_bytes


@dataclass(frozen=True)
class C128PackedPageAddress:
    """Persistent address of one positive group-global C128 block."""

    global_block_id: int
    group_block_id: int
    owner_rank: int
    owner_local_slot: int
    segment_base_bytes: int
    physical_offset_bytes: int
    page_size_bytes: int


@dataclass(frozen=True)
class C128PackedScatterEntry:
    """One compressor output row routed to an owner-local tensor page."""

    source_row: int
    global_block_id: int
    in_page_offset: int
    owner_local_slot: int
    flat_tensor_slot: int
    physical_page_offset_bytes: int


@dataclass(frozen=True)
class C128PackedScratchEntry:
    """One selected persistent page staged into destination-rank scratch."""

    global_block_id: int
    owner_rank: int
    owner_local_slot: int
    persistent_offset_bytes: int
    scratch_slot: int
    scratch_offset_bytes: int


@dataclass(frozen=True)
class C128PackedMaterialization:
    """A bounded selected-page materialization plan for one consumer rank."""

    destination_rank: int
    entries: tuple[C128PackedScratchEntry, ...]
    max_scratch_pages: int

    @property
    def selected_global_block_ids(self) -> tuple[int, ...]:
        return tuple(entry.global_block_id for entry in self.entries)

    def remap_block_table(
        self,
        global_block_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Map selected data pages to dense scratch slots.

        ``-1`` and sentinel block zero both become attention padding ``-1``.
        A positive block absent from this materialization fails closed.
        """
        scratch_slot_by_block = {entry.global_block_id: entry.scratch_slot for entry in self.entries}
        remapped: list[int] = []
        for block_id in global_block_ids:
            if block_id in (PAD_BLOCK_ID, SENTINEL_BLOCK_ID):
                remapped.append(PAD_BLOCK_ID)
                continue
            try:
                remapped.append(scratch_slot_by_block[block_id])
            except KeyError as error:
                raise ValueError(
                    "block table references a positive C128 block absent " f"from bounded scratch: {block_id}"
                ) from error
        return tuple(remapped)


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_sequence(value: object, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return value


def _indexed_by_rank(
    raw_segments: object,
    *,
    tp_size: int,
    field: str,
) -> tuple[C128PackedSegment, ...]:
    segments: list[C128PackedSegment | None] = [None] * tp_size
    for raw_segment in _require_sequence(raw_segments, field):
        segment = _require_mapping(raw_segment, f"{field} entry")
        rank = _require_int(segment.get("rank"), f"{field}.rank")
        if not 0 <= rank < tp_size:
            raise ValueError(f"{field}.rank {rank} is outside [0, {tp_size})")
        if segments[rank] is not None:
            raise ValueError(f"{field} contains duplicate rank {rank}")
        base_bytes = _require_int(
            segment.get("segment_base_bytes"),
            f"{field}.segment_base_bytes",
        )
        allocated_bytes = _require_int(
            segment.get("segment_allocated_bytes"),
            f"{field}.segment_allocated_bytes",
        )
        if base_bytes < 0 or allocated_bytes <= 0:
            raise ValueError(f"{field} segment offsets must be nonnegative and sizes positive")
        segments[rank] = C128PackedSegment(
            rank=rank,
            base_bytes=base_bytes,
            allocated_bytes=allocated_bytes,
        )
    if any(segment is None for segment in segments):
        missing = [rank for rank, segment in enumerate(segments) if segment is None]
        raise ValueError(f"{field} is missing ranks {missing}")
    return tuple(segment for segment in segments if segment is not None)


def _validated_groups(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate the serialized plan's complete disjoint group partition."""
    raw_groups = _require_sequence(metadata.get("groups"), "groups")
    groups: list[Mapping[str, Any]] = []
    cursor = 1
    names: set[str] = set()
    for expected_index, raw_group in enumerate(raw_groups):
        group = _require_mapping(raw_group, "groups entry")
        group_index = _require_int(group.get("group_index"), "group_index")
        if group_index != expected_index:
            raise ValueError(
                "packed group indices must be unique, ordered, and dense: "
                f"expected {expected_index}, got {group_index}"
            )
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("packed group name must be a nonempty string")
        if group_name in names:
            raise ValueError(f"duplicate packed group name: {group_name}")
        names.add(group_name)
        logical_start = _require_int(
            group.get("logical_start"),
            f"{group_name}.logical_start",
        )
        logical_stop = _require_int(
            group.get("logical_stop"),
            f"{group_name}.logical_stop",
        )
        logical_blocks = _require_int(
            group.get("logical_blocks"),
            f"{group_name}.logical_blocks",
        )
        if logical_blocks <= 0 or logical_start != cursor or logical_stop - logical_start != logical_blocks:
            raise ValueError(f"packed group {group_name} has inconsistent logical range")
        cursor = logical_stop
        groups.append(group)

    if not groups:
        raise ValueError("packed C128 metadata requires at least one group")
    used_logical_blocks = _require_int(
        metadata.get("used_logical_blocks"),
        "used_logical_blocks",
    )
    if used_logical_blocks != cursor - 1:
        raise ValueError("used_logical_blocks does not match the serialized group ranges")
    global_block_capacity = _require_int(
        metadata.get("global_block_capacity"),
        "global_block_capacity",
    )
    usable_data_capacity = _require_int(
        metadata.get("usable_data_capacity"),
        "usable_data_capacity",
    )
    if (
        global_block_capacity <= 1
        or usable_data_capacity != global_block_capacity - 1
        or used_logical_blocks > usable_data_capacity
    ):
        raise ValueError("packed global block capacity metadata is inconsistent")
    return tuple(groups)


def _int_vector(
    value: object,
    *,
    size: int,
    field: str,
) -> tuple[int, ...]:
    raw_values = _require_sequence(value, field)
    if len(raw_values) != size:
        raise ValueError(f"{field} must contain exactly {size} ranks")
    values = tuple(_require_int(raw_value, f"{field}[{rank}]") for rank, raw_value in enumerate(raw_values))
    if any(value < 0 for value in values):
        raise ValueError(f"{field} values must be nonnegative")
    return values


def _validate_bucket_accounting(
    metadata: Mapping[str, Any],
    *,
    bucket_name: object,
    page_size_bytes: int,
    tp_size: int,
    scratch_segments: tuple[C128PackedSegment, ...],
) -> None:
    """Validate every persistent copy and the scratch tail in one bucket."""
    if not isinstance(bucket_name, str) or not bucket_name:
        raise ValueError("packed C128 bucket must be a nonempty string")
    matching_buckets = [
        _require_mapping(bucket, "buckets entry")
        for bucket in _require_sequence(metadata.get("buckets"), "buckets")
        if _require_mapping(bucket, "buckets entry").get("bucket") == bucket_name
    ]
    if len(matching_buckets) != 1:
        raise ValueError(f"expected one packed bucket {bucket_name!r}, " f"found {len(matching_buckets)}")
    bucket = matching_buckets[0]
    if bucket.get("page_size_bytes") != page_size_bytes:
        raise ValueError("packed C128 bucket page size mismatch")
    persistent_bytes = _int_vector(
        bucket.get("persistent_allocated_bytes_by_rank"),
        size=tp_size,
        field=f"bucket[{bucket_name}].persistent_allocated_bytes_by_rank",
    )
    scratch_region_bytes = _int_vector(
        bucket.get("scratch_region_bytes_by_rank"),
        size=tp_size,
        field=f"bucket[{bucket_name}].scratch_region_bytes_by_rank",
    )
    total_bytes = _int_vector(
        bucket.get("total_allocated_bytes_by_rank"),
        size=tp_size,
        field=f"bucket[{bucket_name}].total_allocated_bytes_by_rank",
    )

    intervals_by_rank: list[list[tuple[int, int, str]]] = [[] for _ in range(tp_size)]
    for group in _validated_groups(metadata):
        group_name = group["name"]
        for raw_component in _require_sequence(
            group.get("components"),
            f"{group_name}.components",
        ):
            component = _require_mapping(
                raw_component,
                f"{group_name}.components entry",
            )
            if component.get("bucket") != bucket_name:
                continue
            component_name = component.get("name")
            if not isinstance(component_name, str) or not component_name:
                raise ValueError("packed component name must be nonempty")
            if component.get("page_size_bytes") != page_size_bytes:
                raise ValueError(f"component {group_name}/{component_name} page size " "does not match its bucket")
            granularity = _require_int(
                component.get("allocation_granularity_bytes"),
                (f"{group_name}/{component_name}." "allocation_granularity_bytes"),
            )
            copies = _require_int(
                component.get("copies"),
                f"{group_name}/{component_name}.copies",
            )
            if granularity <= 0 or copies <= 0:
                raise ValueError(
                    f"component {group_name}/{component_name} has invalid " "allocation granularity or copy count"
                )
            raw_copies = _require_sequence(
                component.get("segments"),
                f"{group_name}/{component_name}.segments",
            )
            copy_indices = [
                _require_int(
                    _require_mapping(
                        raw_copy,
                        "component segment copy",
                    ).get("copy_index"),
                    "component segment copy_index",
                )
                for raw_copy in raw_copies
            ]
            if copy_indices != list(range(copies)):
                raise ValueError(
                    f"component {group_name}/{component_name} segment " "copies must be unique, ordered, and dense"
                )
            for copy_index, raw_copy in enumerate(raw_copies):
                copy = _require_mapping(
                    raw_copy,
                    "component segment copy",
                )
                raw_rank_segments = tuple(
                    _require_mapping(
                        entry,
                        "persistent rank segment",
                    )
                    for entry in _require_sequence(
                        copy.get("ranks"),
                        "persistent ranks",
                    )
                )
                segments = _indexed_by_rank(
                    raw_rank_segments,
                    tp_size=tp_size,
                    field=(f"{group_name}/{component_name}/copy_{copy_index} " "persistent segments"),
                )
                rank_metadata = {
                    _require_int(
                        entry.get("rank"),
                        "persistent rank",
                    ): entry
                    for entry in raw_rank_segments
                }
                for rank, segment in enumerate(segments):
                    if rank_metadata[rank].get("sentinel_offset_bytes") != segment.base_bytes:
                        raise ValueError(f"rank {rank} sentinel offset must equal its " "segment base")
                    if segment.base_bytes % granularity or segment.allocated_bytes % granularity:
                        raise ValueError(f"rank {rank} persistent segment is not aligned")
                    intervals_by_rank[rank].append(
                        (
                            segment.base_bytes,
                            segment.stop_bytes,
                            (f"{group_name}/{component_name}/" f"copy_{copy_index}"),
                        )
                    )

    for rank in range(tp_size):
        intervals = sorted(intervals_by_rank[rank])
        if not intervals:
            raise ValueError(f"bucket {bucket_name!r} has no rank {rank} persistent " "segments")
        previous_stop = 0
        for start, stop, label in intervals:
            if start < previous_stop:
                raise ValueError(f"rank {rank} packed persistent segments overlap at " f"{label}")
            previous_stop = stop
        if previous_stop != persistent_bytes[rank]:
            raise ValueError(f"rank {rank} packed persistent extent does not match " "bucket accounting")
        scratch_segment = scratch_segments[rank]
        if (
            scratch_region_bytes[rank] != total_bytes[rank] - persistent_bytes[rank]
            or scratch_segment.base_bytes < persistent_bytes[rank]
            or scratch_segment.stop_bytes != total_bytes[rank]
        ):
            raise ValueError(f"rank {rank} C128 scratch does not match bucket accounting")


@dataclass(frozen=True)
class C128PackedOwnerRoute:
    """One validated packed C128 group/component/copy routing contract."""

    schema_version: int
    runtime_abi_ready: bool
    tp_size: int
    group_index: int
    group_name: str
    component_name: str
    layer_name: str
    copy_index: int
    logical_start: int
    logical_stop: int
    page_size_bytes: int
    allocation_granularity_bytes: int
    persistent_segments: tuple[C128PackedSegment, ...]
    scratch_segments: tuple[C128PackedSegment, ...]
    max_scratch_pages: int

    @property
    def view_key(self) -> str:
        """Return the matching :class:`PackedArenaRuntime` component key."""
        return f"component/{self.group_name}/{self.component_name}/" f"{self.copy_index}"

    @classmethod
    def from_serialized_plan(
        cls,
        metadata: Mapping[str, Any],
        *,
        group_index: int,
        group_name: str,
        component_name: str,
        layer_name: str,
        copy_index: int,
        allow_planner_only: bool = False,
    ) -> C128PackedOwnerRoute:
        """Parse exactly one C128 component copy from serialized metadata.

        The explicit group index/name and component/layer/copy tuple prevent a
        caller from silently binding a layer to a different planner group.
        Runtime callers must keep ``allow_planner_only=False``.
        """
        metadata = _require_mapping(metadata, "packed metadata")
        if not isinstance(allow_planner_only, bool):
            raise ValueError("allow_planner_only must be a boolean")
        schema_version = _require_int(
            metadata.get("schema_version"),
            "schema_version",
        )
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                "unsupported packed C128 schema_version " f"{schema_version}; expected {SUPPORTED_SCHEMA_VERSION}"
            )
        if metadata.get("sentinel_block_id") != SENTINEL_BLOCK_ID:
            raise ValueError("packed C128 metadata must reserve sentinel block 0")
        runtime_abi_ready = metadata.get("downstream_runtime_abi_ready")
        if not isinstance(runtime_abi_ready, bool):
            raise ValueError("downstream_runtime_abi_ready must be a JSON boolean")
        planner_only = metadata.get("planner_only")
        if not isinstance(planner_only, bool):
            raise ValueError("planner_only must be a JSON boolean")
        if planner_only == runtime_abi_ready:
            raise ValueError("planner_only and downstream_runtime_abi_ready are inconsistent")
        if not runtime_abi_ready and not allow_planner_only:
            raise RuntimeError("packed C128 plan is planner-only; downstream runtime ABI " "is not ready")
        tp_size = _require_int(metadata.get("tp_size"), "tp_size")
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        group_index = _require_int(group_index, "group_index")
        copy_index = _require_int(copy_index, "copy_index")
        if group_index < 0:
            raise ValueError("group_index must be nonnegative")
        if copy_index < 0:
            raise ValueError("copy_index must be nonnegative")

        groups = _validated_groups(metadata)
        if group_index >= len(groups):
            raise ValueError(f"expected one packed group with group_index={group_index}, " "found 0")
        group = groups[group_index]
        if group.get("name") != group_name:
            raise ValueError(
                "packed group identity mismatch: "
                f"index {group_index} is {group.get('name')!r}, "
                f"not {group_name!r}"
            )
        logical_start = _require_int(
            group.get("logical_start"),
            f"{group_name}.logical_start",
        )
        logical_stop = _require_int(
            group.get("logical_stop"),
            f"{group_name}.logical_stop",
        )
        assert logical_start > SENTINEL_BLOCK_ID
        assert logical_stop > logical_start

        raw_components = _require_sequence(
            group.get("components"),
            f"{group_name}.components",
        )
        matching_components = [
            _require_mapping(component, f"{group_name}.components entry")
            for component in raw_components
            if _require_mapping(
                component,
                f"{group_name}.components entry",
            ).get("name")
            == component_name
        ]
        if len(matching_components) != 1:
            raise ValueError(
                f"expected one component {group_name}/{component_name}, " f"found {len(matching_components)}"
            )
        component = matching_components[0]
        if component.get("placement") != C128_OWNER_PLACEMENT:
            raise ValueError(f"component {group_name}/{component_name} is not c128_owner")
        copies = _require_int(
            component.get("copies"),
            f"{group_name}/{component_name}.copies",
        )
        if copies <= 0:
            raise ValueError(f"component {group_name}/{component_name} must have copies")
        layer_names = tuple(
            _require_sequence(
                component.get("layer_names"),
                f"{group_name}/{component_name}.layer_names",
            )
        )
        if (
            len(layer_names) != copies
            or any(not isinstance(current_layer, str) or not current_layer for current_layer in layer_names)
            or len(set(layer_names)) != len(layer_names)
        ):
            raise ValueError(f"component {group_name}/{component_name} has an invalid " "layer/copy mapping")
        group_layer_names = tuple(
            _require_sequence(
                group.get("layer_names"),
                f"{group_name}.layer_names",
            )
        )
        if any(component_layer not in group_layer_names for component_layer in layer_names):
            raise ValueError(f"component {group_name}/{component_name} references a " "layer outside its group")
        if not 0 <= copy_index < copies:
            raise ValueError(f"copy_index {copy_index} is outside [0, {copies}) for " f"{group_name}/{component_name}")
        if layer_names[copy_index] != layer_name:
            raise ValueError(
                "packed component layer/copy mismatch: "
                f"copy {copy_index} is {layer_names[copy_index]!r}, "
                f"not {layer_name!r}"
            )

        page_size_bytes = _require_int(
            component.get("page_size_bytes"),
            f"{group_name}/{component_name}.page_size_bytes",
        )
        allocation_granularity_bytes = _require_int(
            component.get("allocation_granularity_bytes"),
            (f"{group_name}/{component_name}." "allocation_granularity_bytes"),
        )
        if page_size_bytes <= 0 or allocation_granularity_bytes <= 0:
            raise ValueError("packed C128 page size and granularity must be positive")

        raw_copies = _require_sequence(
            component.get("segments"),
            f"{group_name}/{component_name}.segments",
        )
        serialized_copy_indices = [
            _require_int(
                _require_mapping(copy, "component segment copy").get("copy_index"),
                "component segment copy_index",
            )
            for copy in raw_copies
        ]
        if serialized_copy_indices != list(range(copies)):
            raise ValueError(
                f"component {group_name}/{component_name} segment copies " "must be unique, ordered, and dense"
            )
        matching_copies = [
            _require_mapping(copy, "component segment copy")
            for copy in raw_copies
            if _require_mapping(
                copy,
                "component segment copy",
            ).get("copy_index")
            == copy_index
        ]
        if len(matching_copies) != 1:
            raise ValueError(f"expected one segment set for copy_index={copy_index}, " f"found {len(matching_copies)}")
        persistent_segments = _indexed_by_rank(
            matching_copies[0].get("ranks"),
            tp_size=tp_size,
            field=(f"{group_name}/{component_name}/copy_{copy_index} " "persistent segments"),
        )

        bucket = component.get("bucket")
        matching_scratch = [
            _require_mapping(scratch, "scratch entry")
            for scratch in _require_sequence(
                metadata.get("scratch"),
                "scratch",
            )
            if _require_mapping(
                scratch,
                "scratch entry",
            ).get("bucket")
            == bucket
        ]
        if len(matching_scratch) != 1:
            raise ValueError(
                f"expected one bounded scratch segment for bucket {bucket!r}, " f"found {len(matching_scratch)}"
            )
        scratch = matching_scratch[0]
        if scratch.get("page_size_bytes") != page_size_bytes:
            raise ValueError("packed C128 scratch page size mismatch")
        if scratch.get("allocation_granularity_bytes") != allocation_granularity_bytes:
            raise ValueError("packed C128 scratch granularity mismatch")
        max_scratch_pages = _require_int(
            scratch.get("max_pages_per_rank"),
            f"scratch[{bucket}].max_pages_per_rank",
        )
        if max_scratch_pages <= 0:
            raise ValueError("packed C128 scratch must have a positive page bound")
        scratch_segments = _indexed_by_rank(
            scratch.get("segments"),
            tp_size=tp_size,
            field=f"scratch[{bucket}] segments",
        )
        _validate_bucket_accounting(
            metadata,
            bucket_name=bucket,
            page_size_bytes=page_size_bytes,
            tp_size=tp_size,
            scratch_segments=scratch_segments,
        )

        route = cls(
            schema_version=schema_version,
            runtime_abi_ready=runtime_abi_ready,
            tp_size=tp_size,
            group_index=group_index,
            group_name=group_name,
            component_name=component_name,
            layer_name=layer_name,
            copy_index=copy_index,
            logical_start=logical_start,
            logical_stop=logical_stop,
            page_size_bytes=page_size_bytes,
            allocation_granularity_bytes=allocation_granularity_bytes,
            persistent_segments=persistent_segments,
            scratch_segments=scratch_segments,
            max_scratch_pages=max_scratch_pages,
        )
        route._validate_segment_capacities()
        return route

    @property
    def logical_blocks(self) -> int:
        return self.logical_stop - self.logical_start

    def assert_runtime_ready(self) -> None:
        """Reject attaching planner-only metadata to runtime cache tensors."""
        if not self.runtime_abi_ready:
            raise RuntimeError(
                "packed C128 route cannot attach to runtime tensors until " "downstream_runtime_abi_ready is true"
            )

    def _validate_tp_rank(self, tp_rank: int) -> None:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank {tp_rank} is outside [0, {self.tp_size})")

    def first_owned_global_block(self, owner_rank: int) -> int:
        self._validate_tp_rank(owner_rank)
        return self.logical_start + (owner_rank - self.logical_start) % self.tp_size

    def owner_page_count(self, owner_rank: int) -> int:
        first_owned = self.first_owned_global_block(owner_rank)
        if first_owned >= self.logical_stop:
            return 0
        return (self.logical_stop - 1 - first_owned) // self.tp_size + 1

    def required_persistent_pages(self, owner_rank: int) -> int:
        """Include the component-local sentinel page at physical slot zero."""
        return self.owner_page_count(owner_rank) + 1

    def _validate_segment_capacities(self) -> None:
        for rank, segment in enumerate(self.persistent_segments):
            required_bytes = self.required_persistent_pages(rank) * self.page_size_bytes
            if segment.allocated_bytes < required_bytes:
                raise ValueError(
                    f"rank {rank} persistent segment is too small: " f"{segment.allocated_bytes} < {required_bytes}"
                )
            if segment.base_bytes % self.allocation_granularity_bytes:
                raise ValueError(f"rank {rank} persistent segment base is not aligned")
            if segment.allocated_bytes % self.allocation_granularity_bytes:
                raise ValueError(f"rank {rank} persistent segment size is not aligned")
        required_scratch_bytes = self.max_scratch_pages * self.page_size_bytes
        for rank, segment in enumerate(self.scratch_segments):
            if segment.allocated_bytes < required_scratch_bytes:
                raise ValueError(
                    f"rank {rank} scratch segment is too small: "
                    f"{segment.allocated_bytes} < {required_scratch_bytes}"
                )
            if segment.base_bytes % self.allocation_granularity_bytes:
                raise ValueError(f"rank {rank} scratch segment base is not aligned")
            if segment.allocated_bytes % self.allocation_granularity_bytes:
                raise ValueError(f"rank {rank} scratch segment size is not aligned")

    def validate_runtime_tensor_pages(
        self,
        *,
        tp_rank: int,
        persistent_pages: int,
        scratch_pages: int,
    ) -> None:
        """Validate concrete segment views before any cache write/read."""
        self.assert_runtime_ready()
        self._validate_tp_rank(tp_rank)
        persistent_pages = _require_int(
            persistent_pages,
            "persistent_pages",
        )
        scratch_pages = _require_int(scratch_pages, "scratch_pages")
        if persistent_pages <= 0 or scratch_pages <= 0:
            raise ValueError("runtime tensor page counts must be positive")
        persistent_segment = self.persistent_segments[tp_rank]
        scratch_segment = self.scratch_segments[tp_rank]
        if persistent_pages * self.page_size_bytes > persistent_segment.allocated_bytes:
            raise ValueError(
                f"persistent tensor exposes {persistent_pages} pages beyond "
                f"rank {tp_rank} segment capacity "
                f"{persistent_segment.allocated_bytes // self.page_size_bytes}"
            )
        if scratch_pages * self.page_size_bytes > scratch_segment.allocated_bytes:
            raise ValueError(
                f"scratch tensor exposes {scratch_pages} pages beyond rank "
                f"{tp_rank} segment capacity "
                f"{scratch_segment.allocated_bytes // self.page_size_bytes}"
            )
        required_persistent = self.required_persistent_pages(tp_rank)
        if persistent_pages < required_persistent:
            raise ValueError(
                f"persistent tensor exposes {persistent_pages} pages, "
                f"needs {required_persistent} for rank {tp_rank}"
            )
        if scratch_pages < self.max_scratch_pages:
            raise ValueError(
                f"scratch tensor exposes {scratch_pages} pages, " f"needs bounded capacity {self.max_scratch_pages}"
            )

    def map_global_block(
        self,
        global_block_id: int,
    ) -> C128PackedPageAddress:
        """Map one positive group-global ID to owner rank/local slot."""
        if global_block_id == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("sentinel block 0 is not a persistent C128 data page")
        if not self.logical_start <= global_block_id < self.logical_stop:
            raise ValueError(
                f"global block ID {global_block_id} is outside "
                f"{self.group_name} range "
                f"[{self.logical_start}, {self.logical_stop})"
            )
        owner_rank = global_block_id % self.tp_size
        first_owned = self.first_owned_global_block(owner_rank)
        owner_local_slot = (global_block_id - first_owned) // self.tp_size + 1
        segment = self.persistent_segments[owner_rank]
        physical_offset_bytes = segment.base_bytes + owner_local_slot * self.page_size_bytes
        if physical_offset_bytes + self.page_size_bytes > segment.stop_bytes:
            raise ValueError("packed C128 page address exceeds its persistent segment")
        return C128PackedPageAddress(
            global_block_id=global_block_id,
            group_block_id=global_block_id - self.logical_start + 1,
            owner_rank=owner_rank,
            owner_local_slot=owner_local_slot,
            segment_base_bytes=segment.base_bytes,
            physical_offset_bytes=physical_offset_bytes,
            page_size_bytes=self.page_size_bytes,
        )

    def sentinel_offset_bytes(self, tp_rank: int) -> int:
        """Return the local dummy-page address for table sentinel zero."""
        self._validate_tp_rank(tp_rank)
        return self.persistent_segments[tp_rank].base_bytes

    def validate_no_hybrid_expansion(
        self,
        *,
        physical_block_size: int,
        logical_block_size: int,
        blocks_per_physical_block: int,
    ) -> None:
        """C128 routes require one block-table ID per physical C128 page."""
        if (
            physical_block_size <= 0
            or logical_block_size <= 0
            or blocks_per_physical_block != 1
            or physical_block_size != logical_block_size
        ):
            raise ValueError("packed C128 owner routing does not support hybrid " "block expansion")

    def plan_owned_scatter(
        self,
        slot_mapping: Iterable[tuple[int, int]],
        *,
        tp_rank: int,
        tokens_per_page: int,
    ) -> tuple[C128PackedScatterEntry, ...]:
        """Route compressor rows into the local persistent segment.

        Padding ``(-1, -1)`` is ignored.  Sentinel zero and partially padded
        rows are rejected because neither may mutate the dummy page.
        """
        self._validate_tp_rank(tp_rank)
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        entries: list[C128PackedScatterEntry] = []
        seen_flat_slots: set[int] = set()
        for source_row, raw_slot in enumerate(slot_mapping):
            try:
                global_block_id, in_page_offset = raw_slot
            except (TypeError, ValueError) as error:
                raise ValueError("slot_mapping entries must be (global_block_id, " "in_page_offset)") from error
            if global_block_id == PAD_BLOCK_ID and in_page_offset == PAD_BLOCK_ID:
                continue
            if global_block_id == SENTINEL_BLOCK_ID:
                raise SentinelBlockError("scatter cannot write the packed sentinel page")
            if global_block_id < 0 or in_page_offset < 0:
                raise ValueError("partially padded C128 scatter slot is invalid")
            if in_page_offset >= tokens_per_page:
                raise ValueError(f"in-page offset {in_page_offset} is outside " f"[0, {tokens_per_page})")
            address = self.map_global_block(global_block_id)
            if address.owner_rank != tp_rank:
                continue
            flat_tensor_slot = address.owner_local_slot * tokens_per_page + in_page_offset
            if flat_tensor_slot in seen_flat_slots:
                raise ValueError("one packed C128 scatter cannot write a slot twice")
            seen_flat_slots.add(flat_tensor_slot)
            entries.append(
                C128PackedScatterEntry(
                    source_row=source_row,
                    global_block_id=global_block_id,
                    in_page_offset=in_page_offset,
                    owner_local_slot=address.owner_local_slot,
                    flat_tensor_slot=flat_tensor_slot,
                    physical_page_offset_bytes=(address.physical_offset_bytes),
                )
            )
        return tuple(entries)

    def plan_materialization(
        self,
        global_block_ids: Iterable[int],
        *,
        destination_rank: int,
    ) -> C128PackedMaterialization:
        """Assign sorted unique selected pages to bounded local scratch."""
        self._validate_tp_rank(destination_rank)
        selected: set[int] = set()
        for block_id in global_block_ids:
            if block_id in (PAD_BLOCK_ID, SENTINEL_BLOCK_ID):
                continue
            # Validate before adding so a foreign group cannot disappear via
            # de-duplication or padding handling.
            self.map_global_block(block_id)
            selected.add(block_id)
        selected_ids = tuple(sorted(selected))
        if len(selected_ids) > self.max_scratch_pages:
            raise ValueError(
                f"packed C128 materialization needs {len(selected_ids)} "
                f"scratch pages, bound is {self.max_scratch_pages}"
            )
        scratch = self.scratch_segments[destination_rank]
        entries: list[C128PackedScratchEntry] = []
        for scratch_slot, block_id in enumerate(selected_ids):
            address = self.map_global_block(block_id)
            scratch_offset_bytes = scratch.base_bytes + scratch_slot * self.page_size_bytes
            if scratch_offset_bytes + self.page_size_bytes > scratch.stop_bytes:
                raise ValueError("packed C128 scratch address exceeds its segment")
            entries.append(
                C128PackedScratchEntry(
                    global_block_id=block_id,
                    owner_rank=address.owner_rank,
                    owner_local_slot=address.owner_local_slot,
                    persistent_offset_bytes=(address.physical_offset_bytes),
                    scratch_slot=scratch_slot,
                    scratch_offset_bytes=scratch_offset_bytes,
                )
            )
        return C128PackedMaterialization(
            destination_rank=destination_rank,
            entries=tuple(entries),
            max_scratch_pages=self.max_scratch_pages,
        )


@dataclass(frozen=True)
class C128PackedOwnerRouteTable:
    """Immutable, explicit layer-to-route activation contract.

    Runtime setup builds this table from the worker-delivered serialized plan
    and passes the selected route to each C128 cache view. Keeping the table
    caller-owned avoids another data-pointer or process-global registry.
    """

    routes: tuple[C128PackedOwnerRoute, ...]

    @classmethod
    def from_serialized_plan(
        cls,
        metadata: Mapping[str, Any],
        *,
        expected_group_layer_names: Sequence[Sequence[str]] | None = None,
        allow_planner_only: bool = False,
    ) -> C128PackedOwnerRouteTable:
        """Build every C128 owner route after exact group/layer validation."""
        metadata = _require_mapping(metadata, "packed metadata")
        if not isinstance(allow_planner_only, bool):
            raise ValueError("allow_planner_only must be a boolean")
        groups = _validated_groups(metadata)
        if expected_group_layer_names is not None and len(expected_group_layer_names) != len(groups):
            raise ValueError(
                "packed owner route cache-group count mismatch: " f"{len(expected_group_layer_names)} != {len(groups)}"
            )

        routes: list[C128PackedOwnerRoute] = []
        seen_layer_names: set[str] = set()
        for group_index, group in enumerate(groups):
            group_name = group.get("name")
            assert isinstance(group_name, str)
            raw_group_layers = _require_sequence(
                group.get("layer_names"),
                f"{group_name}.layer_names",
            )
            group_layers = tuple(raw_group_layers)
            if any(not isinstance(layer_name, str) or not layer_name for layer_name in group_layers):
                raise ValueError(f"{group_name}.layer_names must contain nonempty strings")
            if expected_group_layer_names is not None:
                expected_layers = tuple(expected_group_layer_names[group_index])
                if group_layers != expected_layers:
                    raise ValueError(
                        f"packed owner route layer mapping mismatch for group "
                        f"{group_index}: {group_layers!r} != "
                        f"{expected_layers!r}"
                    )

            components = _require_sequence(
                group.get("components"),
                f"{group_name}.components",
            )
            for component_index, raw_component in enumerate(components):
                component = _require_mapping(
                    raw_component,
                    f"{group_name}.components[{component_index}]",
                )
                if component.get("placement") != C128_OWNER_PLACEMENT:
                    continue
                component_name = component.get("name")
                if not isinstance(component_name, str) or not component_name:
                    raise ValueError(f"{group_name}.components[{component_index}].name " "must be a nonempty string")
                layer_names = tuple(
                    _require_sequence(
                        component.get("layer_names"),
                        f"{group_name}/{component_name}.layer_names",
                    )
                )
                if not layer_names or any(
                    not isinstance(layer_name, str) or not layer_name for layer_name in layer_names
                ):
                    raise ValueError(
                        f"{group_name}/{component_name}.layer_names must " "contain at least one nonempty string"
                    )
                for copy_index, layer_name in enumerate(layer_names):
                    if layer_name in seen_layer_names:
                        raise ValueError(f"packed C128 layer {layer_name!r} is routed more " "than once")
                    route = C128PackedOwnerRoute.from_serialized_plan(
                        metadata,
                        group_index=group_index,
                        group_name=group_name,
                        component_name=component_name,
                        layer_name=layer_name,
                        copy_index=copy_index,
                        allow_planner_only=allow_planner_only,
                    )
                    routes.append(route)
                    seen_layer_names.add(layer_name)

        if not routes:
            raise ValueError("packed metadata contains no C128 owner routes")
        return cls(routes=tuple(routes))

    def for_layer(self, layer_name: str) -> C128PackedOwnerRoute:
        """Resolve one exact C128 layer without mutable lookup state."""
        matches = tuple(route for route in self.routes if route.layer_name == layer_name)
        if len(matches) != 1:
            raise ValueError(f"expected one packed C128 route for layer {layer_name!r}, " f"found {len(matches)}")
        return matches[0]
