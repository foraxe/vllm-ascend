# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Collective-free C128 peer-root materialization protocol.

This module is deliberately not connected to the production DSA path.  A
future peer-lease integration may provide startup-imported tensors through
``C128PeerRootProvider`` and invoke the plans defined here only after producer
visibility has been established.  The high-level execution path itself does
not import, map, synchronize, or communicate.

Historical planning consumes scheduler-owned Python integer metadata. The
CPU overlay plan is a preflight oracle; the production-shaped overlay consumes
the worker's retained device ``[rows, 2]`` global mapping directly. Execution
then uses only tensor search/index/copy operations on the caller's stream:

``peer owner pages -> bounded local scratch -> current-row overlay``.

The overlay is ordered after historical-page copies, so current compressor
rows replace stale values from the peer snapshot exactly as they do in a
replicated cache.  Its mapping must be scheduler-owned group-global metadata,
captured before ``PackedBlockTableTranslator.translate_slot_mapping_`` turns
remote writes into padding.  Cross-device producer visibility remains a
peer-lease lifetime/fencing precondition, not a responsibility of this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import torch

from .c128_packed_owner_route import (
    PAD_BLOCK_ID,
    C128PackedOwnerRoute,
    C128PackedScratchEntry,
)
from .c128_packed_pool import SENTINEL_BLOCK_ID, SentinelBlockError

_SIGNED_BLOCK_TABLE_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_GLOBAL_ID_DTYPES = {
    torch.int32,
    torch.int64,
}


class C128PeerRootProvider(Protocol):
    """Borrow startup-imported owner roots without owning their lifetime."""

    @property
    def tp_size(self) -> int:
        """Number of canonical C128 owners represented by the provider."""

    def tensor_for_owner(self, owner_rank: int) -> torch.Tensor:
        """Return one sentinel-prefixed owner-local persistent tensor."""


@dataclass(frozen=True)
class C128PeerTensorRoots:
    """CPU-testable peer-root provider backed by explicit tensors.

    The tensors are borrowed.  Their VMM mappings, exporter lifetimes, and
    teardown ordering remain owned by the future peer lease.
    """

    roots: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.roots:
            raise ValueError("C128 peer roots must not be empty")
        if any(not isinstance(root, torch.Tensor) for root in self.roots):
            raise TypeError("every C128 peer root must be a torch.Tensor")
        if len({id(root) for root in self.roots}) != len(self.roots):
            raise ValueError("one tensor object cannot represent two C128 owners")

    @property
    def tp_size(self) -> int:
        return len(self.roots)

    def tensor_for_owner(self, owner_rank: int) -> torch.Tensor:
        if not 0 <= owner_rank < self.tp_size:
            raise ValueError(f"owner_rank {owner_rank} is outside [0, {self.tp_size})")
        return self.roots[owner_rank]


@dataclass(frozen=True)
class C128PeerRouteIdentity:
    """Stable route identity carried from planning into execution."""

    tp_size: int
    group_index: int
    group_name: str
    component_name: str
    layer_name: str
    copy_index: int
    logical_start: int
    logical_stop: int

    @classmethod
    def from_route(
        cls,
        route: C128PackedOwnerRoute,
    ) -> C128PeerRouteIdentity:
        return cls(
            tp_size=route.tp_size,
            group_index=route.group_index,
            group_name=route.group_name,
            component_name=route.component_name,
            layer_name=route.layer_name,
            copy_index=route.copy_index,
            logical_start=route.logical_start,
            logical_stop=route.logical_stop,
        )


@dataclass(frozen=True)
class C128PeerRoutingGeometry:
    """Layer-independent routing geometry reusable by all component copies."""

    tp_size: int
    group_index: int
    group_name: str
    logical_start: int
    logical_stop: int
    page_size_bytes: int
    max_scratch_pages: int

    @classmethod
    def from_route(
        cls,
        route: C128PackedOwnerRoute,
    ) -> C128PeerRoutingGeometry:
        return cls(
            tp_size=route.tp_size,
            group_index=route.group_index,
            group_name=route.group_name,
            logical_start=route.logical_start,
            logical_stop=route.logical_stop,
            page_size_bytes=route.page_size_bytes,
            max_scratch_pages=route.max_scratch_pages,
        )


@dataclass(frozen=True)
class C128PeerOwnerBatch:
    """One owner's source slots and their dense scratch destinations."""

    owner_rank: int
    global_block_ids: tuple[int, ...]
    owner_local_slots: tuple[int, ...]
    scratch_slots: tuple[int, ...]


@dataclass(frozen=True)
class C128PeerMaterializationPlan:
    """Pure scheduler-side plan for one bounded historical-page view."""

    route_identity: C128PeerRouteIdentity
    destination_rank: int
    block_table_shape: tuple[int, int]
    scratch_local_block_table: tuple[tuple[int, ...], ...]
    page_bindings: tuple[C128PackedScratchEntry, ...]
    owner_batches: tuple[C128PeerOwnerBatch, ...]
    max_scratch_pages: int

    @property
    def selected_global_block_ids(self) -> tuple[int, ...]:
        return tuple(binding.global_block_id for binding in self.page_bindings)

    @property
    def staged_pages(self) -> int:
        return len(self.page_bindings)


@dataclass(frozen=True)
class C128CurrentRowOverlayEntry:
    """One current compressor row written after historical materialization."""

    source_row: int
    global_block_id: int
    in_page_offset: int
    scratch_slot: int
    flat_scratch_slot: int


@dataclass(frozen=True)
class C128GlobalCurrentRowMapping:
    """Pre-owner-translation row mapping retained by worker metadata.

    This named wrapper prevents an ordinary owner-local/PAD ``slot_mapping``
    from being passed to the overlay planner by accident. The worker seam can
    construct it directly from the CPU-side flat global slots immediately
    before ``PackedBlockTableTranslator.translate_slot_mapping_`` mutates the
    normal write mapping.
    """

    tokens_per_page: int
    slots: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        tokens_per_page = _require_int(self.tokens_per_page, "tokens_per_page")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        normalized_slots = []
        for source_row, raw_slot in enumerate(self.slots):
            try:
                raw_global_block_id, raw_in_page_offset = raw_slot
            except (TypeError, ValueError) as error:
                raise ValueError("global current-row entries must be " "(global_block_id, in_page_offset)") from error
            normalized_slots.append(
                (
                    _require_int(
                        raw_global_block_id,
                        f"global_current_rows[{source_row}].global_block_id",
                    ),
                    _require_int(
                        raw_in_page_offset,
                        f"global_current_rows[{source_row}].in_page_offset",
                    ),
                )
            )
        object.__setattr__(self, "tokens_per_page", tokens_per_page)
        object.__setattr__(self, "slots", tuple(normalized_slots))

    @classmethod
    def from_pre_translation_flat_slots(
        cls,
        flat_global_slots: Iterable[int],
        *,
        tokens_per_page: int,
    ) -> C128GlobalCurrentRowMapping:
        """Capture the worker's flat global slots before owner translation."""
        tokens_per_page = _require_int(tokens_per_page, "tokens_per_page")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        slots = []
        for source_row, raw_flat_slot in enumerate(flat_global_slots):
            flat_slot = _require_int(
                raw_flat_slot,
                f"flat_global_slots[{source_row}]",
            )
            if flat_slot == PAD_BLOCK_ID:
                slots.append((PAD_BLOCK_ID, PAD_BLOCK_ID))
            elif flat_slot < 0:
                raise ValueError(f"flat global slot {flat_slot} is below padding -1")
            else:
                slots.append(divmod(flat_slot, tokens_per_page))
        return cls(tokens_per_page=tokens_per_page, slots=tuple(slots))


@dataclass(frozen=True)
class C128CurrentRowOverlayPlan:
    """Pure plan for same-stream current-row replacement in scratch."""

    route_identity: C128PeerRouteIdentity
    destination_rank: int
    selected_global_block_ids: tuple[int, ...]
    tokens_per_page: int
    expected_source_rows: int
    entries: tuple[C128CurrentRowOverlayEntry, ...]


@dataclass(frozen=True)
class C128PeerCompiledOwnerBatch:
    """One owner's request indices uploaded once for repeated layer use."""

    owner_rank: int
    owner_local_slots: torch.Tensor
    scratch_slots: torch.Tensor


@dataclass(frozen=True)
class C128PeerCompiledMaterializationPlan:
    """Device metadata compiled once per C128 group/chunk."""

    route_geometry: C128PeerRoutingGeometry
    destination_rank: int
    selected_global_block_ids: tuple[int, ...]
    selected_global_block_ids_tensor: torch.Tensor
    owner_batches: tuple[C128PeerCompiledOwnerBatch, ...]
    scratch_local_block_table: torch.Tensor
    block_table_shape: tuple[int, int]
    staged_pages: int


@dataclass(frozen=True)
class C128PeerCompiledCurrentRowOverlay:
    """Device overlay indices compiled once per C128 group/chunk."""

    route_geometry: C128PeerRoutingGeometry
    destination_rank: int
    selected_global_block_ids: tuple[int, ...]
    tokens_per_page: int
    expected_source_rows: int
    entry_count: int
    source_rows: torch.Tensor
    flat_scratch_slots: torch.Tensor


@dataclass(frozen=True)
class C128PeerCompiledDeviceCurrentRowOverlay:
    """Certified worker device mapping plus reusable valid-row indices."""

    route_geometry: C128PeerRoutingGeometry
    destination_rank: int
    selected_global_block_ids: tuple[int, ...]
    tokens_per_page: int
    expected_source_rows: int
    entry_count: int
    mapping_dtype: torch.dtype
    source_rows: torch.Tensor
    flat_scratch_slots: torch.Tensor
    mapping_matches_certificate: torch.Tensor
    global_current_row_mapping: torch.Tensor


@dataclass(frozen=True)
class C128PeerMaterializedView:
    """Fixed-capacity scratch and its scratch-local attention table."""

    cache: torch.Tensor
    block_table: torch.Tensor


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(value)


def _rectangular_block_table(
    block_table: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(block_table, (str, bytes)) or not isinstance(
        block_table,
        Sequence,
    ):
        raise TypeError("block_table must be a sequence of integer rows")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(block_table):
        if isinstance(raw_row, (str, bytes)) or not isinstance(
            raw_row,
            Sequence,
        ):
            raise TypeError(f"block_table[{row_index}] must be a sequence")
        row = tuple(
            _require_int(value, f"block_table[{row_index}][{column_index}]")
            for column_index, value in enumerate(raw_row)
        )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("block_table rows must have equal lengths")
        rows.append(row)
    if width is None:
        return ()
    return tuple(rows)


def plan_c128_peer_materialization(
    route: C128PackedOwnerRoute,
    block_table: Sequence[Sequence[int]],
    *,
    destination_rank: int,
) -> C128PeerMaterializationPlan:
    """Map a packed-global table to owner roots and dense local scratch.

    Duplicate positive IDs share one scratch page.  Padding ``-1`` and the
    reserved sentinel zero both become scratch-table padding.  Other negative
    values, foreign groups, out-of-range IDs, and scratch overflow fail before
    any tensor operation is launched.
    """
    route.assert_runtime_ready()
    destination_rank = _require_int(destination_rank, "destination_rank")
    rows = _rectangular_block_table(block_table)
    width = len(rows[0]) if rows else 0
    flat_global_ids = tuple(value for row in rows for value in row)
    materialization = route.plan_materialization(
        flat_global_ids,
        destination_rank=destination_rank,
    )
    remapped_flat = materialization.remap_block_table(flat_global_ids)
    remapped_rows = (
        tuple(tuple(remapped_flat[offset : offset + width]) for offset in range(0, len(remapped_flat), width))
        if width
        else tuple(() for _ in rows)
    )

    owner_batches = []
    for owner_rank in range(route.tp_size):
        bindings = tuple(binding for binding in materialization.entries if binding.owner_rank == owner_rank)
        if not bindings:
            continue
        owner_batches.append(
            C128PeerOwnerBatch(
                owner_rank=owner_rank,
                global_block_ids=tuple(binding.global_block_id for binding in bindings),
                owner_local_slots=tuple(binding.owner_local_slot for binding in bindings),
                scratch_slots=tuple(binding.scratch_slot for binding in bindings),
            )
        )
    return C128PeerMaterializationPlan(
        route_identity=C128PeerRouteIdentity.from_route(route),
        destination_rank=destination_rank,
        block_table_shape=(len(rows), width),
        scratch_local_block_table=remapped_rows,
        page_bindings=materialization.entries,
        owner_batches=tuple(owner_batches),
        max_scratch_pages=materialization.max_scratch_pages,
    )


def plan_c128_current_row_overlay(
    route: C128PackedOwnerRoute,
    materialization: C128PeerMaterializationPlan,
    global_current_row_mapping: C128GlobalCurrentRowMapping,
) -> C128CurrentRowOverlayPlan:
    """Plan current rows that overwrite their staged historical page slots.

    ``global_current_row_mapping`` is an explicit, unmodified mapping from
    compressor-output row to ``(group-global block ID, in-page offset)``. It
    must be preserved by scheduler/worker metadata before the packed block
    translator rewrites the ordinary cache ``slot_mapping`` for owner-local
    persistent writes. Passing that rewritten mapping here would silently
    omit remote current rows, so this API intentionally does not accept it.
    """
    route_identity = C128PeerRouteIdentity.from_route(route)
    if materialization.route_identity != route_identity:
        raise ValueError("current-row overlay route does not match materialization")
    if not isinstance(global_current_row_mapping, C128GlobalCurrentRowMapping):
        raise TypeError("current-row overlay requires explicit pre-translation global " "worker metadata")
    tokens_per_page = global_current_row_mapping.tokens_per_page

    raw_slots = global_current_row_mapping.slots
    scratch_slot_by_block = {binding.global_block_id: binding.scratch_slot for binding in materialization.page_bindings}
    entries = []
    seen_flat_slots: set[int] = set()
    for source_row, raw_slot in enumerate(raw_slots):
        try:
            raw_global_block_id, raw_in_page_offset = raw_slot
        except (TypeError, ValueError) as error:
            raise ValueError(
                "global_current_row_mapping entries must be " "(global_block_id, in_page_offset)"
            ) from error
        global_block_id = _require_int(
            raw_global_block_id,
            f"global_current_row_mapping[{source_row}].global_block_id",
        )
        in_page_offset = _require_int(
            raw_in_page_offset,
            f"global_current_row_mapping[{source_row}].in_page_offset",
        )
        if global_block_id == PAD_BLOCK_ID and in_page_offset == PAD_BLOCK_ID:
            continue
        if global_block_id == SENTINEL_BLOCK_ID:
            raise SentinelBlockError("current-row overlay cannot write the packed sentinel page")
        if global_block_id < 0 or in_page_offset < 0:
            raise ValueError("partially padded current-row overlay is invalid")
        if in_page_offset >= tokens_per_page:
            raise ValueError(f"in-page offset {in_page_offset} is outside " f"[0, {tokens_per_page})")
        route.map_global_block(global_block_id)
        try:
            scratch_slot = scratch_slot_by_block[global_block_id]
        except KeyError as error:
            raise ValueError(
                "current-row overlay references a C128 page absent from " f"bounded scratch: {global_block_id}"
            ) from error
        flat_scratch_slot = scratch_slot * tokens_per_page + in_page_offset
        if flat_scratch_slot in seen_flat_slots:
            raise ValueError("one current-row overlay cannot write a slot twice")
        seen_flat_slots.add(flat_scratch_slot)
        entries.append(
            C128CurrentRowOverlayEntry(
                source_row=source_row,
                global_block_id=global_block_id,
                in_page_offset=in_page_offset,
                scratch_slot=scratch_slot,
                flat_scratch_slot=flat_scratch_slot,
            )
        )
    return C128CurrentRowOverlayPlan(
        route_identity=route_identity,
        destination_rank=materialization.destination_rank,
        selected_global_block_ids=(materialization.selected_global_block_ids),
        tokens_per_page=tokens_per_page,
        expected_source_rows=len(raw_slots),
        entries=tuple(entries),
    )


def compile_c128_peer_materialization(
    route: C128PackedOwnerRoute,
    plan: C128PeerMaterializationPlan,
    *,
    device: torch.device | str,
    block_table_dtype: torch.dtype = torch.int32,
    global_id_dtype: torch.dtype = torch.int32,
) -> C128PeerCompiledMaterializationPlan:
    """Upload immutable chunk indices once, outside the per-layer hot path."""
    route.assert_runtime_ready()
    if plan.route_identity != C128PeerRouteIdentity.from_route(route):
        raise ValueError("peer materialization plan route mismatch")
    if block_table_dtype not in _SIGNED_BLOCK_TABLE_DTYPES:
        raise TypeError("scratch-local block table requires a signed dtype")
    if global_id_dtype not in _GLOBAL_ID_DTYPES:
        raise TypeError("selected global IDs require int32 or int64")
    compiled_batches = tuple(
        C128PeerCompiledOwnerBatch(
            owner_rank=batch.owner_rank,
            owner_local_slots=torch.tensor(
                batch.owner_local_slots,
                dtype=torch.int64,
                device=device,
            ),
            scratch_slots=torch.tensor(
                batch.scratch_slots,
                dtype=torch.int64,
                device=device,
            ),
        )
        for batch in plan.owner_batches
    )
    if plan.block_table_shape[0] == 0:
        scratch_local_block_table = torch.empty(
            plan.block_table_shape,
            dtype=block_table_dtype,
            device=device,
        )
    else:
        scratch_local_block_table = torch.tensor(
            plan.scratch_local_block_table,
            dtype=block_table_dtype,
            device=device,
        )
    return C128PeerCompiledMaterializationPlan(
        route_geometry=C128PeerRoutingGeometry.from_route(route),
        destination_rank=plan.destination_rank,
        selected_global_block_ids=plan.selected_global_block_ids,
        selected_global_block_ids_tensor=torch.tensor(
            plan.selected_global_block_ids,
            dtype=global_id_dtype,
            device=device,
        ),
        owner_batches=compiled_batches,
        scratch_local_block_table=scratch_local_block_table,
        block_table_shape=plan.block_table_shape,
        staged_pages=plan.staged_pages,
    )


def compile_c128_current_row_overlay(
    route: C128PackedOwnerRoute,
    overlay: C128CurrentRowOverlayPlan,
    *,
    device: torch.device | str,
) -> C128PeerCompiledCurrentRowOverlay:
    """Upload current-row source/destination indices once per chunk."""
    route.assert_runtime_ready()
    if overlay.route_identity != C128PeerRouteIdentity.from_route(route):
        raise ValueError("current-row overlay route mismatch")
    return C128PeerCompiledCurrentRowOverlay(
        route_geometry=C128PeerRoutingGeometry.from_route(route),
        destination_rank=overlay.destination_rank,
        selected_global_block_ids=overlay.selected_global_block_ids,
        tokens_per_page=overlay.tokens_per_page,
        expected_source_rows=overlay.expected_source_rows,
        entry_count=len(overlay.entries),
        source_rows=torch.tensor(
            tuple(entry.source_row for entry in overlay.entries),
            dtype=torch.int64,
            device=device,
        ),
        flat_scratch_slots=torch.tensor(
            tuple(entry.flat_scratch_slot for entry in overlay.entries),
            dtype=torch.int64,
            device=device,
        ),
    )


def compile_c128_device_current_row_overlay(
    route: C128PackedOwnerRoute,
    materialization: C128PeerMaterializationPlan,
    compiled_materialization: C128PeerCompiledMaterializationPlan,
    validated_overlay: C128CurrentRowOverlayPlan,
    global_current_row_mapping: torch.Tensor,
) -> C128PeerCompiledDeviceCurrentRowOverlay:
    """Bind a retained device mapping to its CPU-validated certificate.

    The worker creates the tensor from the same pre-owner-translation CPU slots
    certified by ``validated_overlay``. This helper never reads it back. It
    resolves and checks fixed destinations on device once per C128 group/chunk;
    every compatible layer copy then reuses the resulting fixed indices.
    Certificate destinations are authoritative, so a mismatched or later
    mutated mapping cannot redirect a current row.
    """
    route.assert_runtime_ready()
    if materialization.route_identity != C128PeerRouteIdentity.from_route(route):
        raise ValueError("device current-row route mismatch")
    if (
        compiled_materialization.route_geometry != C128PeerRoutingGeometry.from_route(route)
        or compiled_materialization.destination_rank != materialization.destination_rank
        or compiled_materialization.selected_global_block_ids != materialization.selected_global_block_ids
    ):
        raise ValueError("compiled device current-row materialization mismatch")
    if validated_overlay.route_identity != materialization.route_identity:
        raise ValueError("device current-row certificate route mismatch")
    if (
        validated_overlay.destination_rank != materialization.destination_rank
        or validated_overlay.selected_global_block_ids != materialization.selected_global_block_ids
    ):
        raise ValueError("device current-row certificate plan mismatch")
    if not isinstance(global_current_row_mapping, torch.Tensor):
        raise TypeError("global_current_row_mapping must be a tensor")
    if global_current_row_mapping.dtype not in _GLOBAL_ID_DTYPES:
        raise TypeError("device global-row mapping requires int32 or int64")
    if global_current_row_mapping.ndim != 2 or tuple(global_current_row_mapping.shape) != (
        validated_overlay.expected_source_rows,
        2,
    ):
        raise ValueError("device global-current-row mapping must have certified [rows, 2] " "shape")
    if (
        global_current_row_mapping.device != compiled_materialization.selected_global_block_ids_tensor.device
        or global_current_row_mapping.dtype != compiled_materialization.selected_global_block_ids_tensor.dtype
    ):
        raise ValueError("device global-current-row mapping must match compiled selected " "ID device and dtype")
    source_rows = torch.tensor(
        tuple(entry.source_row for entry in validated_overlay.entries),
        dtype=torch.int64,
        device=global_current_row_mapping.device,
    )
    certified_global_ids = torch.tensor(
        tuple(entry.global_block_id for entry in validated_overlay.entries),
        dtype=global_current_row_mapping.dtype,
        device=global_current_row_mapping.device,
    )
    certified_offsets = torch.tensor(
        tuple(entry.in_page_offset for entry in validated_overlay.entries),
        dtype=global_current_row_mapping.dtype,
        device=global_current_row_mapping.device,
    )
    certified_flat_slots = torch.tensor(
        tuple(entry.flat_scratch_slot for entry in validated_overlay.entries),
        dtype=torch.int64,
        device=global_current_row_mapping.device,
    )
    selected_mapping = global_current_row_mapping.index_select(0, source_rows)
    mapping_global_ids = selected_mapping[:, 0].contiguous()
    mapping_offsets = selected_mapping[:, 1]
    candidate_scratch_slots = torch.searchsorted(
        compiled_materialization.selected_global_block_ids_tensor,
        mapping_global_ids,
    )
    candidate_flat_slots = candidate_scratch_slots * validated_overlay.tokens_per_page + mapping_offsets
    mapping_matches_certificate = (
        (mapping_global_ids == certified_global_ids)
        & (mapping_offsets == certified_offsets)
        & (candidate_flat_slots == certified_flat_slots)
    )
    flat_scratch_slots = torch.where(
        mapping_matches_certificate,
        candidate_flat_slots,
        certified_flat_slots,
    )
    return C128PeerCompiledDeviceCurrentRowOverlay(
        route_geometry=C128PeerRoutingGeometry.from_route(route),
        destination_rank=materialization.destination_rank,
        selected_global_block_ids=materialization.selected_global_block_ids,
        tokens_per_page=validated_overlay.tokens_per_page,
        expected_source_rows=validated_overlay.expected_source_rows,
        entry_count=len(validated_overlay.entries),
        mapping_dtype=global_current_row_mapping.dtype,
        source_rows=source_rows,
        flat_scratch_slots=flat_scratch_slots,
        mapping_matches_certificate=mapping_matches_certificate,
        global_current_row_mapping=global_current_row_mapping,
    )


class C128PeerMaterializer:
    """Execute a prevalidated plan from peer roots into local scratch.

    All peer tensors must already be imported on the scratch device and made
    producer-visible by their external lease.  This class borrows those roots
    and never maps, synchronizes, communicates, or owns cleanup.
    """

    def __init__(
        self,
        *,
        route: C128PackedOwnerRoute,
        destination_rank: int,
        peer_roots: C128PeerRootProvider,
        scratch: torch.Tensor,
    ) -> None:
        route.assert_runtime_ready()
        destination_rank = _require_int(destination_rank, "destination_rank")
        if peer_roots.tp_size != route.tp_size:
            raise ValueError(
                "C128 peer-root TP size does not match its packed route: " f"{peer_roots.tp_size} != {route.tp_size}"
            )
        if not 0 <= destination_rank < route.tp_size:
            raise ValueError(f"destination_rank {destination_rank} is outside " f"[0, {route.tp_size})")
        if not isinstance(scratch, torch.Tensor):
            raise TypeError("C128 peer scratch must be a torch.Tensor")
        if scratch.ndim < 2 or not scratch.is_contiguous():
            raise ValueError("C128 peer scratch must be contiguous and paged")
        if scratch.shape[0] == 0:
            raise ValueError("C128 peer scratch must expose at least one page")

        roots = tuple(peer_roots.tensor_for_owner(owner_rank) for owner_rank in range(route.tp_size))
        if len({id(root) for root in roots}) != len(roots):
            raise ValueError("one tensor object cannot represent two C128 owners")
        for owner_rank, root in enumerate(roots):
            if not isinstance(root, torch.Tensor):
                raise TypeError(f"C128 peer root {owner_rank} must be a torch.Tensor")
            if root.ndim < 2 or not root.is_contiguous():
                raise ValueError(f"C128 peer root {owner_rank} must be contiguous and paged")
            if root.shape[0] == 0:
                raise ValueError(f"C128 peer root {owner_rank} must expose a sentinel page")
            if root.dtype != scratch.dtype or root.device != scratch.device or root.shape[1:] != scratch.shape[1:]:
                raise ValueError(f"C128 peer root {owner_rank} does not match scratch " "dtype, device, and page shape")
            page_size_bytes = root[0].numel() * root.element_size()
            if page_size_bytes != route.page_size_bytes:
                raise ValueError(
                    f"C128 peer root {owner_rank} page size "
                    f"{page_size_bytes} does not match route "
                    f"{route.page_size_bytes}"
                )
            required_pages = route.required_persistent_pages(owner_rank)
            if root.shape[0] < required_pages:
                raise ValueError(
                    f"C128 peer root {owner_rank} exposes {root.shape[0]} " f"pages, needs {required_pages}"
                )
            persistent_segment = route.persistent_segments[owner_rank]
            if root.shape[0] * page_size_bytes > persistent_segment.allocated_bytes:
                raise ValueError(
                    f"C128 peer root {owner_rank} exposes {root.shape[0]} " "pages beyond its packed persistent segment"
                )
        scratch_page_bytes = scratch[0].numel() * scratch.element_size()
        if scratch_page_bytes != route.page_size_bytes:
            raise ValueError("C128 peer scratch page size does not match packed route")
        if scratch.shape[0] != route.max_scratch_pages:
            raise ValueError(
                f"C128 peer scratch exposes {scratch.shape[0]} pages, "
                f"requires exact fixed capacity {route.max_scratch_pages}"
            )
        route.validate_runtime_tensor_pages(
            tp_rank=destination_rank,
            persistent_pages=roots[destination_rank].shape[0],
            scratch_pages=scratch.shape[0],
        )

        self.route = route
        self.destination_rank = destination_rank
        self.peer_roots = peer_roots
        self._roots = roots
        self.scratch = scratch
        self._route_geometry = C128PeerRoutingGeometry.from_route(route)

    def _validate_materialization_plan(
        self,
        plan: C128PeerCompiledMaterializationPlan,
    ) -> None:
        if plan.route_geometry != self._route_geometry:
            raise ValueError("peer materialization routing geometry mismatch")
        if plan.destination_rank != self.destination_rank:
            raise ValueError("peer materialization destination-rank mismatch")
        if plan.staged_pages < 0 or plan.staged_pages > self.scratch.shape[0]:
            raise ValueError("peer materialization exceeds concrete scratch")
        if len(plan.selected_global_block_ids) != plan.staged_pages or plan.selected_global_block_ids != tuple(
            sorted(set(plan.selected_global_block_ids))
        ):
            raise ValueError("peer materialization selected-ID metadata is invalid")
        for global_block_id in plan.selected_global_block_ids:
            self.route.map_global_block(global_block_id)
        if (
            plan.selected_global_block_ids_tensor.ndim != 1
            or plan.selected_global_block_ids_tensor.shape[0] != plan.staged_pages
            or plan.selected_global_block_ids_tensor.device != self.scratch.device
            or plan.selected_global_block_ids_tensor.dtype not in _GLOBAL_ID_DTYPES
        ):
            raise ValueError("peer materialization selected-ID tensor is invalid")
        if (
            plan.scratch_local_block_table.ndim != 2
            or tuple(plan.scratch_local_block_table.shape) != plan.block_table_shape
            or plan.scratch_local_block_table.device != self.scratch.device
            or plan.scratch_local_block_table.dtype not in _SIGNED_BLOCK_TABLE_DTYPES
        ):
            raise ValueError("peer materialization block-table tensor is invalid")
        owner_ranks: set[int] = set()
        compiled_pages = 0
        for batch in plan.owner_batches:
            if batch.owner_rank in owner_ranks or not 0 <= batch.owner_rank < self.route.tp_size:
                raise ValueError("peer materialization owner batches are invalid")
            owner_ranks.add(batch.owner_rank)
            if (
                batch.owner_local_slots.ndim != 1
                or batch.scratch_slots.ndim != 1
                or batch.owner_local_slots.shape != batch.scratch_slots.shape
                or batch.owner_local_slots.dtype != torch.int64
                or batch.scratch_slots.dtype != torch.int64
                or batch.owner_local_slots.device != self.scratch.device
                or batch.scratch_slots.device != self.scratch.device
            ):
                raise ValueError("peer materialization compiled indices are invalid")
            compiled_pages += batch.owner_local_slots.shape[0]
        if compiled_pages != plan.staged_pages:
            raise ValueError("peer materialization compiled page count is invalid")

    def _validate_current_rows(
        self,
        plan: C128PeerCompiledMaterializationPlan,
        overlay: C128PeerCompiledCurrentRowOverlay,
        current_rows: torch.Tensor,
    ) -> None:
        if overlay.route_geometry != self._route_geometry:
            raise ValueError("current-row overlay routing geometry mismatch")
        if overlay.destination_rank != self.destination_rank:
            raise ValueError("current-row overlay destination-rank mismatch")
        if overlay.selected_global_block_ids != plan.selected_global_block_ids:
            raise ValueError("current-row overlay references another scratch plan")
        if overlay.tokens_per_page != self.scratch.shape[1]:
            raise ValueError("current-row overlay token-page size mismatch")
        if not isinstance(current_rows, torch.Tensor):
            raise TypeError("current_rows must be a torch.Tensor")
        if (
            current_rows.ndim != self.scratch.ndim - 1
            or current_rows.shape[0] != overlay.expected_source_rows
            or current_rows.shape[1:] != self.scratch.shape[2:]
            or current_rows.dtype != self.scratch.dtype
            or current_rows.device != self.scratch.device
        ):
            raise ValueError("current_rows do not match overlay row count, dtype, device, " "or scratch row shape")
        if (
            overlay.source_rows.device != current_rows.device
            or overlay.flat_scratch_slots.device != self.scratch.device
            or overlay.source_rows.dtype != torch.int64
            or overlay.flat_scratch_slots.dtype != torch.int64
            or overlay.source_rows.ndim != 1
            or overlay.flat_scratch_slots.ndim != 1
            or overlay.source_rows.shape[0] != overlay.entry_count
            or overlay.flat_scratch_slots.shape[0] != overlay.entry_count
        ):
            raise ValueError("current-row overlay compiled-device mismatch")

    def _overlay_current_rows(
        self,
        overlay: C128PeerCompiledCurrentRowOverlay,
        current_rows: torch.Tensor,
    ) -> None:
        if overlay.entry_count == 0:
            return
        selected_rows = current_rows.index_select(0, overlay.source_rows)
        flat_scratch = self.scratch.view(
            -1,
            *self.scratch.shape[2:],
        )
        selected_rows = selected_rows.view(
            overlay.entry_count,
            *flat_scratch.shape[1:],
        )
        flat_scratch.index_copy_(
            0,
            overlay.flat_scratch_slots,
            selected_rows,
        )

    def _validate_device_current_rows(
        self,
        plan: C128PeerCompiledMaterializationPlan,
        overlay: C128PeerCompiledDeviceCurrentRowOverlay,
        current_rows: torch.Tensor,
    ) -> None:
        """Overlay from certified device metadata without D2H or H2D.

        The compiled certificate's fixed source rows exclude padding. They
        come from the same CPU-validated block table used to build ``plan``;
        no device-dependent mask or dynamic result shape is introduced after
        the stateful compressor.
        """
        if overlay.route_geometry != self._route_geometry:
            raise ValueError("device current-row overlay geometry mismatch")
        if overlay.destination_rank != self.destination_rank:
            raise ValueError("device current-row destination-rank mismatch")
        if overlay.selected_global_block_ids != plan.selected_global_block_ids:
            raise ValueError("device current-row overlay references another plan")
        if overlay.tokens_per_page != self.scratch.shape[1]:
            raise ValueError("device current-row token-page size mismatch")
        if (
            overlay.global_current_row_mapping.ndim != 2
            or overlay.global_current_row_mapping.shape != (overlay.expected_source_rows, 2)
            or overlay.global_current_row_mapping.dtype != overlay.mapping_dtype
            or overlay.global_current_row_mapping.device != self.scratch.device
        ):
            raise ValueError(
                "device global-current-row mapping must match compiled "
                "row count, [rows, 2] shape, signed dtype, and device"
            )
        if not isinstance(current_rows, torch.Tensor):
            raise TypeError("current_rows must be a torch.Tensor")
        if (
            current_rows.ndim != self.scratch.ndim - 1
            or current_rows.shape[0] != overlay.expected_source_rows
            or current_rows.shape[1:] != self.scratch.shape[2:]
            or current_rows.dtype != self.scratch.dtype
            or current_rows.device != self.scratch.device
        ):
            raise ValueError(
                "current_rows do not match device overlay row count, dtype, " "device, or scratch row shape"
            )
        if (
            overlay.source_rows.device != self.scratch.device
            or overlay.flat_scratch_slots.device != self.scratch.device
            or overlay.mapping_matches_certificate.device != self.scratch.device
            or plan.selected_global_block_ids_tensor.device != self.scratch.device
            or plan.selected_global_block_ids_tensor.dtype != overlay.mapping_dtype
        ):
            raise ValueError("device current-row compiled-device mismatch")
        if (
            overlay.source_rows.dtype != torch.int64
            or overlay.flat_scratch_slots.dtype != torch.int64
            or overlay.mapping_matches_certificate.dtype != torch.bool
            or overlay.source_rows.ndim != 1
            or overlay.flat_scratch_slots.ndim != 1
            or overlay.mapping_matches_certificate.ndim != 1
            or overlay.entry_count < 0
            or overlay.entry_count > overlay.expected_source_rows
            or overlay.source_rows.shape[0] != overlay.entry_count
            or overlay.flat_scratch_slots.shape[0] != overlay.entry_count
            or overlay.mapping_matches_certificate.shape[0] != overlay.entry_count
        ):
            raise ValueError("device current-row compiled indices are invalid")

    def _overlay_device_current_rows(
        self,
        plan: C128PeerCompiledMaterializationPlan,
        overlay: C128PeerCompiledDeviceCurrentRowOverlay,
        current_rows: torch.Tensor,
    ) -> None:
        if plan.staged_pages == 0:
            return

        selected_rows = current_rows.index_select(0, overlay.source_rows)
        flat_scratch = self.scratch.view(
            -1,
            *self.scratch.shape[2:],
        )
        selected_rows = selected_rows.view(
            overlay.entry_count,
            *flat_scratch.shape[1:],
        )
        flat_scratch.index_copy_(0, overlay.flat_scratch_slots, selected_rows)

    def materialize(
        self,
        plan: C128PeerCompiledMaterializationPlan,
        *,
        overlay: (C128PeerCompiledCurrentRowOverlay | C128PeerCompiledDeviceCurrentRowOverlay | None) = None,
        current_rows: torch.Tensor | None = None,
    ) -> C128PeerMaterializedView:
        """Copy selected peer pages and overlay current rows on one stream."""
        self._validate_materialization_plan(plan)

        if (overlay is None) != (current_rows is None):
            raise ValueError("overlay and current_rows must be provided together")
        if overlay is not None and current_rows is not None:
            if isinstance(overlay, C128PeerCompiledDeviceCurrentRowOverlay):
                self._validate_device_current_rows(
                    plan,
                    overlay,
                    current_rows,
                )
            else:
                self._validate_current_rows(plan, overlay, current_rows)

        for owner_batch in plan.owner_batches:
            root = self._roots[owner_batch.owner_rank]
            selected_pages = root.index_select(
                0,
                owner_batch.owner_local_slots,
            )
            self.scratch.index_copy_(
                0,
                owner_batch.scratch_slots,
                selected_pages,
            )

        if overlay is not None and current_rows is not None:
            if isinstance(overlay, C128PeerCompiledDeviceCurrentRowOverlay):
                self._overlay_device_current_rows(
                    plan,
                    overlay,
                    current_rows,
                )
            else:
                self._overlay_current_rows(overlay, current_rows)

        return C128PeerMaterializedView(
            # Keep the attention cache shape fixed at the configured bound.
            # The remapped table is the authority for which prefix pages are
            # live, so stale tail pages are unreachable.
            cache=self.scratch,
            block_table=plan.scratch_local_block_table,
        )

    def materialize_prevalidated(
        self,
        plan: C128PeerCompiledMaterializationPlan,
        *,
        overlay: C128PeerCompiledDeviceCurrentRowOverlay,
        current_rows: torch.Tensor,
    ) -> C128PeerMaterializedView:
        """Execute startup/request-compile validated metadata on the hot path.

        The compressor may expose an ABI padding row.  Slice by the certified
        metadata row count without querying the asynchronous result's shape;
        every source index is already bounded by that certificate.
        """
        for owner_batch in plan.owner_batches:
            selected_pages = self._roots[owner_batch.owner_rank].index_select(
                0,
                owner_batch.owner_local_slots,
            )
            self.scratch.index_copy_(
                0,
                owner_batch.scratch_slots,
                selected_pages,
            )

        certified_rows = current_rows[: overlay.expected_source_rows]
        if overlay.entry_count:
            selected_rows = certified_rows.index_select(
                0,
                overlay.source_rows,
            )
            flat_scratch = self.scratch.view(
                -1,
                *self.scratch.shape[2:],
            )
            selected_rows = selected_rows.view(
                overlay.entry_count,
                *flat_scratch.shape[1:],
            )
            flat_scratch.index_copy_(
                0,
                overlay.flat_scratch_slots,
                selected_rows,
            )
        return C128PeerMaterializedView(
            cache=self.scratch,
            block_table=plan.scratch_local_block_table,
        )
