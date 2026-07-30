# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Worker-side block-table translation for a packed KV-cache component.

The packed-pool plan owns global group ranges and component-local physical
slots.  This adapter keeps those concerns out of ``BlockTable`` while making
the feature gate explicit: a table without an adapter retains the established
worker behavior without executing any packed translation.
"""

from __future__ import annotations

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

    Block-table entries retain packed global IDs so group ranges remain
    explicit.  Slot mappings are translated into the component segment's
    physical page numbers.  A C128 owner component exposes data slots only on
    the canonical owner rank and maps all remote rows to ``PAD_SLOT_ID``.
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
        """Encode group-local physical block IDs into the packed global range."""
        block_ids = np.asarray(group_block_ids)
        translated = block_ids.copy()
        sentinel = block_ids == SENTINEL_BLOCK_ID
        data = ~sentinel
        if np.any(block_ids < SENTINEL_BLOCK_ID):
            invalid = int(block_ids[block_ids < SENTINEL_BLOCK_ID][0])
            raise ValueError(f"group block ID {invalid} is outside the packed range")
        logical_blocks = self.group_stop - self.group_start
        if np.any(block_ids[data] > logical_blocks):
            invalid = int(block_ids[data][block_ids[data] > logical_blocks][0])
            raise ValueError(f"group block ID {invalid} is outside [1, {logical_blocks}] " f"for {self.group_name}")
        translated[data] += self.group_start - 1
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
        """Translate packed-global flat slots into component-local flat slots.

        The input is the normal ``BlockTable`` result after CP interleave.
        ``PAD_SLOT_ID`` is therefore preserved. Hybrid logical-page expansion
        is inverted before component placement, then its intra-physical-page
        offset is restored in the translated slot.
        """
        if logical_block_size <= 0 or physical_block_size <= 0:
            raise ValueError("block sizes must be positive")
        if blocks_per_phys_block <= 0:
            raise ValueError("blocks_per_phys_block must be positive")
        if logical_block_size * blocks_per_phys_block != physical_block_size:
            raise ValueError("logical_block_size * blocks_per_phys_block must equal " "physical_block_size")

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

        if self.placement is PackedPlacement.REPLICATED:
            physical_slots = global_physical_blocks - self.group_start + 1
            local_slots = physical_slots * physical_block_size + physical_offsets
            translated = torch.where(
                in_group,
                local_slots,
                torch.full_like(slot_mapping, PAD_SLOT_ID),
            )
        else:
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
