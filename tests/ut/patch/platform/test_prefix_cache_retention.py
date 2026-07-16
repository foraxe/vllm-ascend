# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from vllm_ascend.patch.platform import patch_prefix_cache_retention as retention

pytestmark = pytest.mark.cpu_test


def _config(spec):
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _full_spec(block_size=16):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _sliding_spec(block_size=16, sliding_window=32):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


def test_retention_requires_sliding_window_group() -> None:
    with pytest.raises(ValueError, match="no sliding-window"):
        retention._validate_prefix_cache_retention_interval(64, 16, _config(_full_spec()))


@pytest.mark.parametrize("interval", [-16, 24])
def test_retention_requires_non_negative_scheduler_alignment(interval) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        retention._validate_prefix_cache_retention_interval(
            interval,
            16,
            _config(_sliding_spec()),
        )


def test_retention_reads_validated_vllm_environment_value() -> None:
    config = _config(_sliding_spec())
    with patch.object(retention.vllm_envs, retention.ENV_NAME, 64, create=True):
        assert retention.get_prefix_cache_retention_interval(config, 16) == 64


def test_segmented_sliding_window_mask_keeps_each_checkpoint_tail() -> None:
    mask = retention._sliding_window_reachable_block_mask(
        SlidingWindowManager,
        start_block=0,
        end_block=8,
        alignment_tokens=64,
        kv_cache_spec=_sliding_spec(),
        use_eagle=False,
        retention_interval=64,
        num_prompt_tokens=None,
    )

    assert [index for index, keep in enumerate(mask or []) if keep] == [2, 3, 6, 7]


def _cache_sparse_blocks(block_mask, *, enable_events):
    blocks = [SimpleNamespace(is_null=False, block_hash=None) for _ in block_mask]
    request = SimpleNamespace(
        block_hashes=[bytes([index + 1]) * 4 for index in range(len(block_mask))],
        all_token_ids=list(range(len(block_mask) * 4)),
        lora_request=None,
        mm_features=[],
        cache_salt=None,
        prompt_embeds=None,
    )
    block_pool = SimpleNamespace(
        hash_block_size=4,
        enable_kv_cache_events=enable_events,
        cached_block_hash_to_block=MagicMock(),
        kv_event_queue=[],
    )

    with patch.object(
        retention.block_pool_mod,
        "maybe_convert_block_hash",
        side_effect=lambda block_hash: block_hash,
    ):
        retention.BlockPool.cache_full_blocks(
            block_pool,
            request,
            blocks,
            num_cached_blocks=0,
            num_full_blocks=len(blocks),
            block_size=4,
            kv_cache_group_id=2,
            block_mask=block_mask,
        )
    return request, blocks, block_pool


def test_sparse_block_pool_only_caches_selected_blocks() -> None:
    _, blocks, block_pool = _cache_sparse_blocks(
        [False, True, False, True], enable_events=False
    )

    assert [block.block_hash is not None for block in blocks] == [
        False,
        True,
        False,
        True,
    ]
    assert block_pool.cached_block_hash_to_block.insert.call_count == 2


def test_sparse_block_pool_events_split_non_contiguous_runs() -> None:
    request, _, block_pool = _cache_sparse_blocks(
        [False, True, True, False, True], enable_events=True
    )

    assert len(block_pool.kv_event_queue) == 2
    first, second = block_pool.kv_event_queue
    assert first.block_hashes == request.block_hashes[1:3]
    assert first.parent_block_hash == request.block_hashes[0]
    assert first.token_ids == request.all_token_ids[4:12]
    assert len(first.extra_keys) == 2
    assert second.block_hashes == request.block_hashes[4:5]
    assert second.parent_block_hash == request.block_hashes[3]
    assert second.token_ids == request.all_token_ids[16:20]
    assert len(second.extra_keys) == 1


def test_sparse_block_pool_does_not_publish_empty_event() -> None:
    _, _, block_pool = _cache_sparse_blocks([False, False], enable_events=True)

    assert block_pool.kv_event_queue == []


def test_retention_falls_back_to_dense_when_window_fills_segment() -> None:
    mask = retention._sliding_window_reachable_block_mask(
        SlidingWindowManager,
        start_block=0,
        end_block=8,
        alignment_tokens=64,
        kv_cache_spec=_sliding_spec(sliding_window=64),
        use_eagle=False,
        retention_interval=64,
        num_prompt_tokens=None,
    )

    assert mask is None


def test_coordinator_cache_blocks_forwards_retention_alignment_and_eagle_per_group() -> None:
    managers = [MagicMock(kv_cache_group_id=0), MagicMock(kv_cache_group_id=1)]
    coordinator = SimpleNamespace(
        single_type_managers=managers,
        retention_interval=128,
        eagle_group_ids={1},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16, compress_ratio=4)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32, compress_ratio=1)),
            ]
        ),
    )
    request = MagicMock()

    retention._cache_blocks_with_retention(coordinator, request, 96)

    for group_id, manager in enumerate(managers):
        manager.cache_blocks.assert_called_once_with(
            request,
            96,
            retention_interval=128,
            alignment_tokens=64,
            use_eagle=group_id == 1,
        )


def test_free_queue_prepend_preserves_order_and_count() -> None:
    first = SimpleNamespace(prev_free_block=None)
    head = SimpleNamespace(next_free_block=first)
    queue = SimpleNamespace(fake_free_list_head=head, num_free_blocks=3)
    blocks = [
        SimpleNamespace(prev_free_block=None, next_free_block=None),
        SimpleNamespace(prev_free_block=None, next_free_block=None),
    ]

    FreeKVCacheBlockQueue.prepend_n(queue, blocks)

    assert head.next_free_block is blocks[0]
    assert blocks[0].next_free_block is blocks[1]
    assert blocks[1].next_free_block is first
    assert first.prev_free_block is blocks[1]
    assert queue.num_free_blocks == 5


def test_sliding_window_free_recycles_uncached_blocks_at_front() -> None:
    cached = SimpleNamespace(block_hash=b"cached")
    uncached = SimpleNamespace(block_hash=None)
    block_pool = MagicMock()
    manager = SimpleNamespace(
        req_to_blocks={"r1": [cached, uncached]},
        num_cached_block={"r1": 1},
        block_pool=block_pool,
    )

    SlidingWindowManager.free(manager, "r1")

    block_pool.free_blocks.assert_any_call([cached])
    block_pool.free_blocks.assert_any_call([uncached], prepend=True)
    assert "r1" not in manager.req_to_blocks
    assert "r1" not in manager.num_cached_block
