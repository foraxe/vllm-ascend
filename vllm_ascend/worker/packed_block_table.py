# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Worker-side block-table translation for a packed KV-cache component.

The packed-pool plan owns global group ranges and component-local physical
slots.  This adapter keeps those concerns out of ``BlockTable`` while making
the feature gate explicit: a table without an adapter retains the established
worker behavior without executing any packed translation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    SENTINEL_BLOCK_ID,
    PackedPlacement,
    PackedPoolPlan,
)


@dataclass(frozen=True)
class PackedBlockTableTranslator:
    """Translate one cache group's IDs for one packed physical component.

    The fixed-quota scheduler emits packed global IDs. Replicated components
    store component-local IDs in their worker block table so attention reads
    address the same dense pages used by cache writes. C128 owner components
    retain packed-global IDs for selected-page materialization, while their
    write slot mappings are translated to canonical owner-local slots and
    remote rows become ``PAD_SLOT_ID``.
    """

    plan: PackedPoolPlan
    group_name: str
    component_name: str
    placement: PackedPlacement
    tp_rank: int

    def __post_init__(self) -> None:
        logical_range = self.plan.group_range(self.group_name)
        if not 0 <= self.tp_rank < self.plan.tp_size:
            raise ValueError(f"tp_rank {self.tp_rank} is outside [0, {self.plan.tp_size})")
        # Validate the group/component placement eagerly. All packed groups
        # contain at least one data block by construction.
        if self.placement is PackedPlacement.C128_OWNER:
            self.plan.map_c128(
                self.group_name,
                self.component_name,
                1,
            )
        elif self.placement is PackedPlacement.REPLICATED:
            self.plan.map_replicated(
                self.group_name,
                self.component_name,
                1,
                tp_rank=self.tp_rank,
            )
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported packed placement: {self.placement}")
        if logical_range.start <= SENTINEL_BLOCK_ID:
            raise ValueError("packed data ranges must not contain sentinel block 0")

    @property
    def group_start(self) -> int:
        return self.plan.group_range(self.group_name).start

    @property
    def group_stop(self) -> int:
        return self.plan.group_range(self.group_name).stop

    def translate_group_block_ids(
        self,
        group_block_ids: np.ndarray,
    ) -> np.ndarray:
        """Validate scheduler-global IDs and encode the table's read domain.

        ``FixedQuotaBlockPool`` preserves one scheduler-global ID space and
        permanently assigns a disjoint range to each cache group. Replicated
        cache reads address a dense component-local tensor, so their table IDs
        become ``global - group_start + 1``. C128 owner reads first materialize
        selected packed-global pages, so their table IDs remain global. Keep
        sentinel zero unchanged and fail before mutating a block-table row if
        any non-sentinel ID belongs to another group.
        """
        block_ids = np.asarray(group_block_ids)
        if block_ids.ndim != 1:
            raise ValueError("scheduler block IDs must be one-dimensional")
        if block_ids.dtype.kind not in {"i", "u"}:
            raise TypeError("scheduler block IDs must have an integer dtype")
        translated = block_ids.copy()
        sentinel = block_ids == SENTINEL_BLOCK_ID
        data = ~sentinel
        if np.any(block_ids < SENTINEL_BLOCK_ID):
            invalid = int(block_ids[block_ids < SENTINEL_BLOCK_ID][0])
            raise ValueError(f"global block ID {invalid} is outside the packed range")
        in_group = (block_ids[data] >= self.group_start) & (block_ids[data] < self.group_stop)
        if not np.all(in_group):
            invalid = int(block_ids[data][~in_group][0])
            raise ValueError(
                f"global block ID {invalid} is outside group "
                f"{self.group_name} range [{self.group_start}, "
                f"{self.group_stop})"
            )
        if self.placement is PackedPlacement.REPLICATED:
            translated[data] -= self.group_start - 1
        translated[sentinel] = SENTINEL_BLOCK_ID
        return translated

    def translate_slot_mapping_(
        self,
        slot_mapping: torch.Tensor,
        *,
        logical_block_size: int,
        physical_block_size: int,
        blocks_per_phys_block: int,
    ) -> None:
        """Finish placement-specific translation of flat cache-write slots.

        The input is the normal ``BlockTable`` result after CP interleave.
        Replicated tables already produced component-local slots and remain
        unchanged. C128 owner tables still produced packed-global slots, which
        are translated to owner-local physical slots while preserving
        ``PAD_SLOT_ID`` and the sentinel page.
        """
        if logical_block_size <= 0 or physical_block_size <= 0:
            raise ValueError("block sizes must be positive")
        if blocks_per_phys_block <= 0:
            raise ValueError("blocks_per_phys_block must be positive")
        if logical_block_size * blocks_per_phys_block != physical_block_size:
            raise ValueError("logical_block_size * blocks_per_phys_block must equal " "physical_block_size")

        # Replicated table entries were already converted to component-local
        # physical IDs before optional hybrid expansion. The normal slot kernel
        # therefore produces the exact dense component-local write address.
        # Applying the global-range translation again would double-remap groups
        # whose packed range starts after one.
        if self.placement is PackedPlacement.REPLICATED:
            return

        valid = slot_mapping != PAD_SLOT_ID
        valid_slots = torch.where(
            valid,
            slot_mapping,
            torch.zeros_like(slot_mapping),
        )
        global_logical_blocks = torch.div(
            valid_slots,
            logical_block_size,
            rounding_mode="floor",
        )
        logical_offsets = torch.remainder(valid_slots, logical_block_size)
        global_physical_blocks = torch.div(
            global_logical_blocks,
            blocks_per_phys_block,
            rounding_mode="floor",
        )
        logical_subblocks = torch.remainder(
            global_logical_blocks,
            blocks_per_phys_block,
        )
        physical_offsets = logical_subblocks * logical_block_size + logical_offsets

        sentinel = global_physical_blocks == SENTINEL_BLOCK_ID
        in_group = (global_physical_blocks >= self.group_start) & (global_physical_blocks < self.group_stop)

        owners = torch.remainder(
            global_physical_blocks,
            self.plan.tp_size,
        )
        owned = in_group & (owners == self.tp_rank)
        first_owned = self.group_start + (self.tp_rank - self.group_start) % self.plan.tp_size
        owner_slots = (
            torch.div(
                global_physical_blocks - first_owned,
                self.plan.tp_size,
                rounding_mode="floor",
            )
            + 1
        )
        owned_slots = owner_slots * physical_block_size + physical_offsets
        translated = torch.where(
            owned,
            owned_slots,
            torch.full_like(slot_mapping, PAD_SLOT_ID),
        )

        # Sentinel page zero remains the component-local dummy page, including
        # a nonzero offset within that page.
        translated = torch.where(
            sentinel,
            physical_offsets,
            translated,
        )
        slot_mapping.copy_(
            torch.where(
                valid,
                translated,
                torch.full_like(slot_mapping, PAD_SLOT_ID),
            )
        )


def packed_block_table_translators_from_metadata(
    metadata: Mapping[str, object],
    *,
    tp_rank: int,
    kv_cache_group_layer_names: Sequence[Sequence[str]],
) -> tuple[PackedBlockTableTranslator, ...]:
    """Build one explicit worker translator per scheduler cache group.

    Serialized planner metadata is the only authority. The helper rebuilds and
    validates the complete runtime contract, matches every worker cache group
    by both index and ordered layer names, and rejects a group whose physical
    components require different placements. A single worker block table
    cannot safely produce two incompatible slot-mapping layouts.

    No process-global registry is used. Callers pass the returned tuple into
    :class:`NPUInputBatch`, keeping the activation dependency explicit.
    """
    # Import lazily to keep the feature-off worker import path independent from
    # the packed runtime lifecycle module.
    from vllm_ascend.worker.c128_packed_runtime import (
        packed_arena_contract_from_metadata,
    )

    contract = packed_arena_contract_from_metadata(metadata)
    plan = contract.plan
    if len(kv_cache_group_layer_names) != len(plan.groups):
        raise ValueError(
            "packed translator cache-group count mismatch: " f"{len(kv_cache_group_layer_names)} != {len(plan.groups)}"
        )

    raw_groups = metadata.get("groups")
    if not isinstance(raw_groups, (list, tuple)):
        raise ValueError("packed metadata groups must be an array")
    if len(raw_groups) != len(plan.groups):
        raise ValueError(
            "packed metadata group count does not match the validated plan: " f"{len(raw_groups)} != {len(plan.groups)}"
        )

    translators: list[PackedBlockTableTranslator] = []
    for group_index, (group_spec, expected_layers, raw_group) in enumerate(
        zip(
            plan.groups,
            kv_cache_group_layer_names,
            raw_groups,
        )
    ):
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"packed metadata groups[{group_index}] must be an object")
        if raw_group.get("group_index") != group_index:
            raise ValueError(
                f"packed metadata group index mismatch at {group_index}: " f"{raw_group.get('group_index')!r}"
            )
        if raw_group.get("name") != group_spec.name:
            raise ValueError(
                f"packed metadata group name mismatch at {group_index}: "
                f"{raw_group.get('name')!r} != {group_spec.name!r}"
            )
        raw_layer_names = raw_group.get("layer_names")
        if not isinstance(raw_layer_names, (list, tuple)) or any(
            not isinstance(layer_name, str) or not layer_name for layer_name in raw_layer_names
        ):
            raise ValueError(
                f"packed metadata groups[{group_index}].layer_names must be " "an array of nonempty strings"
            )
        expected_layer_names = tuple(expected_layers)
        if any(not isinstance(layer_name, str) or not layer_name for layer_name in expected_layer_names):
            raise ValueError(f"worker cache group {group_index} layer names must be " "nonempty strings")
        if tuple(raw_layer_names) != expected_layer_names:
            raise ValueError(
                f"packed metadata layer mapping mismatch for group "
                f"{group_index}: {tuple(raw_layer_names)!r} != "
                f"{expected_layer_names!r}"
            )

        placements = {component.placement for component in group_spec.components}
        if len(placements) != 1:
            placement_names = sorted(placement.value for placement in placements)
            raise ValueError(
                f"packed cache group {group_spec.name} mixes placements "
                f"{placement_names}; one block table cannot translate both"
            )
        placement = next(iter(placements))
        # Component choice only validates that the reconstructed plan contains
        # this placement. Slot numbering is group-wide and is identical for all
        # components with the same placement.
        component_name = group_spec.components[0].name
        translators.append(
            PackedBlockTableTranslator(
                plan=plan,
                group_name=group_spec.name,
                component_name=component_name,
                placement=placement,
                tp_rank=tp_rank,
            )
        )

    return tuple(translators)
