# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)

from vllm_ascend.core.fixed_quota_block_pool import (
    FIXED_GROUP_BLOCK_QUOTAS_ATTR,
    FixedQuotaBlockPool,
    get_fixed_group_block_quotas,
    make_fixed_group_block_ranges,
)

pytestmark = pytest.mark.cpu_test


def _pool(
    num_blocks: int = 10,
    quotas: tuple[int, ...] = (3, 6),
    enable_caching: bool = False,
) -> FixedQuotaBlockPool:
    return FixedQuotaBlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=enable_caching,
        hash_block_size=16,
        group_block_quotas=quotas,
    )


def test_ranges_reserve_block_zero_and_translate_deterministically() -> None:
    pool = _pool()

    assert pool.null_block.block_id == 0
    assert pool.null_block.is_null
    assert [(block_range.group_id, block_range.start, block_range.stop) for block_range in pool.group_block_ranges] == [
        (0, 1, 4),
        (1, 4, 10),
    ]
    assert pool.get_group_block_range(0).to_group_local(3) == 2
    assert pool.get_group_block_range(1).to_group_local(9) == 5
    assert {
        block.block_id
        for group_id in range(2)
        for block in pool.get_group_view(group_id).free_block_queue.get_all_free_blocks()
    } == set(range(1, 10))
    assert make_fixed_group_block_ranges((3, 6)) == pool.group_block_ranges


def test_group_exhaustion_never_borrows_another_groups_blocks() -> None:
    pool = _pool()
    group0 = pool.get_group_view(0)
    group1 = pool.get_group_view(1)

    assert [block.block_id for block in group0.get_new_blocks(3)] == [1, 2, 3]
    assert group0.get_num_free_blocks() == 0
    assert group1.get_num_free_blocks() == 6
    assert pool.get_num_free_blocks() == 6
    with pytest.raises(ValueError, match="group 0"):
        group0.get_new_blocks(1)


def test_free_and_reallocate_preserves_group_range() -> None:
    pool = _pool()
    group1 = pool.get_group_view(1)
    blocks = group1.get_new_blocks(2)

    group1.free_blocks(reversed(blocks))
    assert group1.get_num_free_blocks() == 6
    reallocated = group1.get_new_blocks(6)

    assert all(pool.get_group_block_range(1).contains(block.block_id) for block in reallocated)
    assert {block.block_id for block in reallocated} == set(range(4, 10))


def test_prepend_recycles_uncached_block_before_unused_blocks() -> None:
    pool = _pool()
    group0 = pool.get_group_view(0)
    first = group0.get_new_blocks(1)[0]

    group0.free_blocks([first], prepend=True)

    assert group0.get_new_blocks(1) == [first]


def test_group_free_queue_supports_sink_style_reservation() -> None:
    pool = _pool()
    group1 = pool.get_group_view(1)

    sink_blocks = group1.free_block_queue.popleft_n(2)

    assert [block.block_id for block in sink_blocks] == [4, 5]
    assert group1.get_num_free_blocks() == 4
    assert pool.get_num_free_blocks() == 7


def test_cached_block_stays_in_partition_until_same_group_eviction() -> None:
    pool = _pool(num_blocks=3, quotas=(1, 1), enable_caching=True)
    group0 = pool.get_group_view(0)
    block = group0.get_new_blocks(1)[0]
    block_hash = BlockHash(b"prefix")
    block_hash_with_group = make_block_hash_with_group_id(block_hash, 0)
    block.block_hash = block_hash_with_group
    pool.cached_block_hash_to_block.insert(block_hash_with_group, block)

    group0.free_blocks([block])
    assert pool.get_cached_block(block_hash, [0]) == [block]
    group0.touch([block])
    assert group0.get_num_free_blocks() == 0
    group0.free_blocks([block])

    assert group0.get_new_blocks(1) == [block]
    assert block.block_hash is None
    assert pool.get_cached_block(block_hash, [0]) is None


def test_reset_prefix_cache_uses_aggregate_partition_free_count() -> None:
    pool = _pool(num_blocks=3, quotas=(1, 1), enable_caching=True)
    group0 = pool.get_group_view(0)
    block = group0.get_new_blocks(1)[0]
    block_hash = BlockHash(b"prefix")
    block_hash_with_group = make_block_hash_with_group_id(block_hash, 0)
    block.block_hash = block_hash_with_group
    pool.cached_block_hash_to_block.insert(block_hash_with_group, block)
    group0.free_blocks([block])

    assert pool.reset_prefix_cache()
    assert block.block_hash is None
    assert pool.get_cached_block(block_hash, [0]) is None


def test_group_view_rejects_cross_partition_blocks() -> None:
    pool = _pool()
    block = pool.get_group_view(0).get_new_blocks(1)[0]

    with pytest.raises(ValueError, match="does not belong"):
        pool.get_group_view(1).touch([block])


def test_fixed_quota_contract_is_opt_in_and_validates_metadata() -> None:
    config = SimpleNamespace()
    assert get_fixed_group_block_quotas(config) is None

    setattr(config, FIXED_GROUP_BLOCK_QUOTAS_ATTR, [2, 3])
    assert get_fixed_group_block_quotas(config) == (2, 3)

    setattr(config, FIXED_GROUP_BLOCK_QUOTAS_ATTR, [2, True])
    with pytest.raises(TypeError, match="only integers"):
        get_fixed_group_block_quotas(config)


def test_quota_sum_must_cover_every_non_null_block() -> None:
    with pytest.raises(ValueError, match="sum=8, expected=9"):
        _pool(quotas=(3, 5))
