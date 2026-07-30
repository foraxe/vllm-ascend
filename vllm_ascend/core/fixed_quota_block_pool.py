# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Fixed-quota scheduler block partitions for packed hybrid KV caches.

The scheduler continues to expose one logical block-ID space. Block zero is
reserved as the universal null/padding block, while every real block belongs
to exactly one KV-cache group for its entire lifetime. This stable ownership
lets workers deterministically translate a logical block ID to a packed
group-local physical offset without changing prefix-cache metadata.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig

FIXED_GROUP_BLOCK_QUOTAS_ATTR = "ascend_kv_cache_group_block_quotas"


@dataclass(frozen=True)
class FixedQuotaBlockRange:
    """Stable half-open logical block-ID range assigned to one cache group."""

    group_id: int
    start: int
    stop: int

    @property
    def quota(self) -> int:
        return self.stop - self.start

    def contains(self, block_id: int) -> bool:
        return self.start <= block_id < self.stop

    def to_group_local(self, block_id: int) -> int:
        if not self.contains(block_id):
            raise ValueError(f"Block {block_id} is outside group {self.group_id} range " f"[{self.start}, {self.stop})")
        return block_id - self.start


def get_fixed_group_block_quotas(
    kv_cache_config: KVCacheConfig,
) -> tuple[int, ...] | None:
    """Read the opt-in fixed-quota contract attached by cache planning.

    KVCacheConfig is copied from planning to the scheduler, so the planner can
    attach this vLLM-Ascend-only metadata after final pool sizing without
    changing the upstream dataclass. An absent attribute preserves the
    upstream shared-pool behavior exactly.
    """

    quotas = getattr(kv_cache_config, FIXED_GROUP_BLOCK_QUOTAS_ATTR, None)
    if quotas is None:
        return None
    if not isinstance(quotas, Sequence) or isinstance(quotas, (str, bytes)):
        raise TypeError(f"{FIXED_GROUP_BLOCK_QUOTAS_ATTR} must be a sequence of integers")
    if any(not isinstance(quota, int) or isinstance(quota, bool) for quota in quotas):
        raise TypeError(f"{FIXED_GROUP_BLOCK_QUOTAS_ATTR} must contain only integers")
    return tuple(quotas)


def make_fixed_group_block_ranges(
    group_block_quotas: Sequence[int],
) -> tuple[FixedQuotaBlockRange, ...]:
    """Build the scheduler/worker shared logical-ID range contract."""

    ranges: list[FixedQuotaBlockRange] = []
    next_block_id = 1
    for group_id, quota in enumerate(group_block_quotas):
        if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
            raise ValueError("Group block quotas must be non-negative integers")
        stop = next_block_id + quota
        ranges.append(FixedQuotaBlockRange(group_id, next_block_id, stop))
        next_block_id = stop
    return tuple(ranges)


class _PartitionedQueueGuard:
    """Read-only aggregate surface; allocation must use a group queue."""

    def __init__(self, pool: "FixedQuotaBlockPool") -> None:
        self._pool = pool

    @property
    def num_free_blocks(self) -> int:
        return self._pool.get_num_free_blocks()

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return [block for queue in self._pool._group_free_block_queues for block in queue.get_all_free_blocks()]

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Global free_block_queue.{name} is undefined for a fixed-quota " "pool; use a group block-pool view"
        )


class FixedQuotaBlockPool(BlockPool):
    """BlockPool whose allocatable IDs are permanently partitioned by group."""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        group_block_quotas: Sequence[int],
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        quotas = tuple(group_block_quotas)
        if not quotas:
            raise ValueError("Fixed-quota block pool requires at least one group")
        if any(not isinstance(quota, int) or isinstance(quota, bool) or quota < 0 for quota in quotas):
            raise ValueError("Group block quotas must be non-negative integers")
        expected_allocatable = num_gpu_blocks - 1
        if sum(quotas) != expected_allocatable:
            raise ValueError(
                "Group block quotas must cover every non-null logical block: "
                f"sum={sum(quotas)}, expected={expected_allocatable}"
            )

        super().__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        self.group_block_quotas = quotas
        self.group_block_ranges = make_fixed_group_block_ranges(quotas)
        self._block_id_to_group_id = [-1] * num_gpu_blocks
        for block_range in self.group_block_ranges:
            self._block_id_to_group_id[block_range.start : block_range.stop] = [
                block_range.group_id
            ] * block_range.quota
        self._group_free_block_queues = tuple(
            FreeKVCacheBlockQueue(self.blocks[block_range.start : block_range.stop])
            for block_range in self.group_block_ranges
        )
        # The queue built by BlockPool.__init__ is no longer valid after the
        # same block nodes have been relinked into group queues. Keep only a
        # read-only aggregate surface to make accidental global allocation fail
        # closed instead of corrupting queue links.
        self.free_block_queue = _PartitionedQueueGuard(self)  # type: ignore[assignment]
        self._group_views = tuple(GroupBlockPoolView(self, group_id) for group_id in range(len(quotas)))

    def get_group_block_range(self, group_id: int) -> FixedQuotaBlockRange:
        if group_id < 0:
            raise ValueError(f"Invalid KV-cache group ID {group_id}")
        try:
            return self.group_block_ranges[group_id]
        except IndexError as exc:
            raise ValueError(f"Invalid KV-cache group ID {group_id}") from exc

    def get_group_view(self, group_id: int) -> "GroupBlockPoolView":
        if group_id < 0:
            raise ValueError(f"Invalid KV-cache group ID {group_id}")
        try:
            return self._group_views[group_id]
        except IndexError as exc:
            raise ValueError(f"Invalid KV-cache group ID {group_id}") from exc

    def _validate_group_block(self, group_id: int, block: KVCacheBlock) -> None:
        if block.is_null:
            return
        block_range = self.get_group_block_range(group_id)
        if not block_range.contains(block.block_id):
            raise ValueError(
                f"Block {block.block_id} does not belong to KV-cache group "
                f"{group_id} range [{block_range.start}, {block_range.stop})"
            )

    def get_num_free_blocks_for_group(self, group_id: int) -> int:
        self.get_group_block_range(group_id)
        return self._group_free_block_queues[group_id].num_free_blocks

    def get_num_free_blocks(self) -> int:
        return sum(queue.num_free_blocks for queue in self._group_free_block_queues)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        raise RuntimeError("FixedQuotaBlockPool allocation requires a KV-cache group view")

    def get_new_blocks_for_group(self, group_id: int, num_blocks: int) -> list[KVCacheBlock]:
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool):
            raise TypeError("num_blocks must be an integer")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        num_free_blocks = self.get_num_free_blocks_for_group(group_id)
        if num_blocks > num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from KV-cache group " f"{group_id}; only {num_free_blocks} remain"
            )

        blocks = self._group_free_block_queues[group_id].popleft_n(num_blocks)
        for block in blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return blocks

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch blocks by their permanent owner group."""

        for block in blocks:
            if block.is_null:
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_accessed(block)
                continue
            group_id = self.get_group_id_for_block(block.block_id)
            self.touch_for_group(group_id, (block,))

    def touch_for_group(self, group_id: int, blocks: Sequence[KVCacheBlock]) -> None:
        self.get_group_block_range(group_id)
        for block in blocks:
            self._validate_group_block(group_id, block)
        queue = self._group_free_block_queues[group_id]
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    @staticmethod
    def _return_blocks_to_queue(
        queue: FreeKVCacheBlockQueue,
        blocks: list[KVCacheBlock],
        prepend: bool,
    ) -> None:
        if not prepend:
            queue.append_n(blocks)
            return
        if not blocks:
            return
        # vLLM-Ascend's retention patch adds prepend_n to the upstream queue.
        # Keep this pool independently importable for focused CPU tests too.
        prepend_n = getattr(queue, "prepend_n", None)
        if prepend_n is not None:
            prepend_n(blocks)
            return

        first_block = queue.fake_free_list_head.next_free_block
        assert first_block is not None
        previous_block = queue.fake_free_list_head
        for block in blocks:
            block.prev_free_block = previous_block
            previous_block.next_free_block = block
            previous_block = block
        previous_block.next_free_block = first_block
        first_block.prev_free_block = previous_block
        queue.num_free_blocks += len(blocks)

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        """Free blocks into the queue selected by their permanent owner."""

        blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in self.group_block_ranges]
        for block in ordered_blocks:
            if block.is_null:
                block.ref_cnt -= 1
                continue
            group_id = self.get_group_id_for_block(block.block_id)
            blocks_by_group[group_id].append(block)
        for group_id, blocks in enumerate(blocks_by_group):
            self.free_blocks_for_group(group_id, blocks, prepend=prepend)

    def free_blocks_for_group(
        self,
        group_id: int,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        self.get_group_block_range(group_id)
        blocks = list(ordered_blocks)
        for block in blocks:
            self._validate_group_block(group_id, block)
        for block in blocks:
            block.ref_cnt -= 1
        self._return_blocks_to_queue(
            self._group_free_block_queues[group_id],
            [block for block in blocks if block.ref_cnt == 0 and not block.is_null],
            prepend,
        )

    def get_group_id_for_block(self, block_id: int) -> int:
        if block_id == 0:
            raise ValueError("Null block zero has no owning KV-cache group")
        if block_id < 0 or block_id >= self.num_gpu_blocks:
            raise ValueError(f"Invalid logical block ID {block_id}")
        group_id = self._block_id_to_group_id[block_id]
        if group_id < 0:
            raise AssertionError(f"Logical block ID {block_id} has no owner")
        return group_id


class GroupBlockPoolView:
    """Group-bound BlockPool interface consumed by one cache manager."""

    def __init__(self, pool: FixedQuotaBlockPool, group_id: int) -> None:
        self._pool = pool
        self.group_id = group_id

    @property
    def null_block(self) -> KVCacheBlock:
        return self._pool.null_block

    @property
    def num_gpu_blocks(self) -> int:
        return self._pool.group_block_quotas[self.group_id]

    @property
    def free_block_queue(self) -> FreeKVCacheBlockQueue:
        return self._pool._group_free_block_queues[self.group_id]

    def get_num_free_blocks(self) -> int:
        return self._pool.get_num_free_blocks_for_group(self.group_id)

    def get_usage(self) -> float:
        quota = self.num_gpu_blocks
        if quota == 0:
            return 0.0
        return 1.0 - self.get_num_free_blocks() / quota

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._pool.get_new_blocks_for_group(self.group_id, num_blocks)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        self._pool.touch_for_group(self.group_id, blocks)

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        self._pool.free_blocks_for_group(
            self.group_id,
            ordered_blocks,
            prepend=prepend,
        )

    def cache_full_blocks(self, *args: Any, **kwargs: Any) -> None:
        kv_cache_group_id = kwargs.get("kv_cache_group_id")
        if kv_cache_group_id is None and len(args) >= 6:
            kv_cache_group_id = args[5]
        if kv_cache_group_id != self.group_id:
            raise ValueError(f"Group view {self.group_id} cannot cache blocks for group " f"{kv_cache_group_id}")
        self._pool.cache_full_blocks(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)
