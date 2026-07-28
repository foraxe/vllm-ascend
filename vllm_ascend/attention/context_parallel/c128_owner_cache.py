# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""C128 canonical-owner storage with a staged local attention view.

The DeepSeek-V4 C128 compressor is stateful.  The first production gate keeps
the existing gathered-hidden/compressor execution intact, but changes the
placement contract after compression:

* a logical C128 cache page has exactly one TP owner, ``page % tp_size``;
* every rank writes only its owned pages to persistent storage;
* the sparse-attention kernel receives a temporary, local paged view containing
  exactly the pages referenced by its block table.

The temporary view is deliberately separate from the canonical cache.  It can
be backed by HCCL today and by an NPU VMM peer view later without changing page
ownership or compressor-state semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.logger import init_logger


logger = init_logger(__name__)


# A DSV4 layer's static-forward cache slot must remain a Tensor.  Inserting a
# Python wrapper there makes the first model execution leave the normal eager
# cache contract before DSACP gets a chance to consume it.  Keep ownership
# metadata out-of-band and resolve it from the persistent Tensor at the DSA
# seam instead.
_OWNER_CACHES_BY_DATA_PTR: dict[int, "C128OwnerShardCache"] = {}


def c128_owner(page_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    """Return the canonical TP owner of each logical cache page."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return torch.remainder(page_ids, world_size)


def c128_local_page(page_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    """Translate logical pages into their owner's compact local-page index."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return torch.div(page_ids, world_size, rounding_mode="floor")


def remap_c128_block_table(block_table: torch.Tensor, selected_pages: torch.Tensor) -> torch.Tensor:
    """Map global logical pages in ``block_table`` into a compact stage view.

    ``selected_pages`` must be sorted and unique.  Negative entries are padding
    and remain padding.  Raising on an unselected valid page avoids silently
    feeding sparse attention a wrong history page.
    """
    if selected_pages.ndim != 1:
        raise ValueError("selected_pages must be one-dimensional")
    if selected_pages.numel() and not bool(torch.all(selected_pages[1:] > selected_pages[:-1])):
        raise ValueError("selected_pages must be sorted and unique")

    result = torch.full_like(block_table, -1)
    valid = block_table >= 0
    if not bool(valid.any()):
        return result
    if selected_pages.numel() == 0:
        raise ValueError("block table references pages but the staged set is empty")

    indices = torch.searchsorted(selected_pages, block_table[valid])
    in_range = indices < selected_pages.numel()
    matched = torch.zeros_like(in_range, dtype=torch.bool)
    if bool(in_range.any()):
        matched[in_range] = selected_pages[indices[in_range]] == block_table[valid][in_range]
    if not bool(matched.all()):
        raise ValueError("block table references a page absent from the staged set")
    result[valid] = indices.to(result.dtype)
    return result


@dataclass
class C128OwnerShardCache:
    """One canonical C128 page shard and a reusable local execution view."""

    persistent_cache: torch.Tensor
    stage_cache: torch.Tensor
    tp_size: int

    @property
    def persistent_bytes(self) -> int:
        return self.persistent_cache.numel() * self.persistent_cache.element_size()

    @property
    def stage_capacity_pages(self) -> int:
        return self.stage_cache.shape[0]

    def scatter_owned(
        self,
        slot_mapping: torch.Tensor,
        compressed_kv: torch.Tensor | None,
        tp_rank: int,
    ) -> None:
        """Persist only this rank's owned compressed rows.

        The compressor result remains globally ordered and identical to the
        replicated baseline.  Filtering happens solely at the final cache
        placement, so state-cache semantics are unchanged.
        """
        if compressed_kv is None:
            return
        if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
            raise ValueError("C128 slot_mapping must have shape [rows, 2]")
        if slot_mapping.shape[0] != compressed_kv.shape[0]:
            raise ValueError("C128 slot_mapping and compressed_kv row counts differ")
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank={tp_rank} is outside TP size {self.tp_size}")

        page_ids = slot_mapping[:, 0]
        owner_mask = c128_owner(page_ids, self.tp_size) == tp_rank
        logger.info(
            "C128 owner scatter: rank=%d rows=%d owned_rows=%d",
            tp_rank,
            slot_mapping.shape[0],
            int(owner_mask.sum().item()),
        )
        if not bool(owner_mask.any()):
            return
        local_slot_mapping = slot_mapping[owner_mask].clone()
        local_slot_mapping[:, 0] = c128_local_page(local_slot_mapping[:, 0], self.tp_size)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(
            self.persistent_cache,
            local_slot_mapping,
            compressed_kv[owner_mask],
        )

    @staticmethod
    def _all_gather_fixed(tensor: torch.Tensor, group) -> list[torch.Tensor]:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(gathered, tensor, group=group)
        return gathered

    @classmethod
    def _all_gather_pages_union(cls, local_pages: torch.Tensor, group) -> torch.Tensor:
        """Return sorted page union required by any rank's local block table."""
        device = local_pages.device
        count = torch.tensor([local_pages.numel()], dtype=torch.int64, device=device)
        counts = torch.cat(cls._all_gather_fixed(count, group=group))
        max_count = int(counts.max().item())
        if max_count == 0:
            return local_pages
        padded = torch.full((max_count,), -1, dtype=local_pages.dtype, device=device)
        padded[: local_pages.numel()] = local_pages
        all_pages = torch.cat(cls._all_gather_fixed(padded, group=group))
        return torch.unique(all_pages[all_pages >= 0], sorted=True)

    def materialize_for_attention(
        self,
        block_table: torch.Tensor,
        *,
        tp_rank: int,
        group,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HCCL-stage the pages required by ``block_table`` into local memory.

        All ranks first form the union of pages required by any CP-local query
        shard.  Each owner then contributes only its pages from that union.
        The caller receives a compact stage cache and a block table whose page
        numbers address that cache.  This supports different local query
        windows without assuming that every rank has an identical table.
        """
        if self.tp_size != dist.get_world_size(group=group):
            raise RuntimeError("C128 owner-shard TP size does not match its HCCL group")
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank={tp_rank} is outside TP size {self.tp_size}")

        local_pages = torch.unique(block_table[block_table >= 0], sorted=True)
        union_pages = self._all_gather_pages_union(local_pages, group=group)
        logger.info(
            "C128 owner stage: rank=%d local_pages=%d union_pages=%d capacity=%d",
            tp_rank,
            local_pages.numel(),
            union_pages.numel(),
            self.stage_capacity_pages,
        )
        if union_pages.numel() > self.stage_capacity_pages:
            raise RuntimeError(
                f"C128 stage capacity {self.stage_capacity_pages} pages is smaller than "
                f"the required union of {union_pages.numel()} pages"
            )

        owned_union_pages = union_pages[c128_owner(union_pages, self.tp_size) == tp_rank]
        local_count = torch.tensor([owned_union_pages.numel()], dtype=torch.int64, device=block_table.device)
        owner_counts = torch.cat(self._all_gather_fixed(local_count, group=group))
        max_owned_count = int(owner_counts.max().item())

        page_shape = self.persistent_cache.shape[1:]
        padded_owned_pages = torch.zeros(
            (max_owned_count, *page_shape), dtype=self.persistent_cache.dtype, device=self.persistent_cache.device
        )
        if owned_union_pages.numel():
            padded_owned_pages[: owned_union_pages.numel()].copy_(
                self.persistent_cache[c128_local_page(owned_union_pages, self.tp_size)]
            )
        gathered_pages = self._all_gather_fixed(padded_owned_pages, group=group)
        logger.info(
            "C128 owner stage pages: rank=%d local_owned=%d max_owned=%d page_shape=%s",
            tp_rank,
            owned_union_pages.numel(),
            max_owned_count,
            tuple(page_shape),
        )

        # ``union_pages`` is sorted.  Per-owner filtering preserves that order,
        # so rank-major gathered chunks reconstruct the canonical logical view.
        self.stage_cache[: union_pages.numel()].zero_()
        for owner, owner_count in enumerate(owner_counts.tolist()):
            if owner_count == 0:
                continue
            owner_pages = union_pages[c128_owner(union_pages, self.tp_size) == owner]
            stage_slots = torch.searchsorted(union_pages, owner_pages)
            self.stage_cache[stage_slots] = gathered_pages[owner][:owner_count]

        remapped_block_table = remap_c128_block_table(block_table, union_pages)
        logger.info(
            "C128 owner stage complete: rank=%d staged_pages=%d",
            tp_rank,
            union_pages.numel(),
        )
        return self.stage_cache[: union_pages.numel()], remapped_block_table


def register_c128_owner_cache(cache: C128OwnerShardCache) -> torch.Tensor:
    """Register owner metadata while preserving the model's Tensor cache ABI."""
    persistent_cache = cache.persistent_cache
    _OWNER_CACHES_BY_DATA_PTR[persistent_cache.data_ptr()] = cache
    return persistent_cache


def get_c128_owner_cache(kv_cache: object) -> C128OwnerShardCache | None:
    """Return owner metadata for a static-forward cache tensor, if enabled."""
    if isinstance(kv_cache, C128OwnerShardCache):
        return kv_cache
    if isinstance(kv_cache, torch.Tensor):
        return _OWNER_CACHES_BY_DATA_PTR.get(kv_cache.data_ptr())
    return None
