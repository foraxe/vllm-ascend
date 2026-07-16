# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (  # noqa: E402
    get_block_hashes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (  # noqa: E402
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)


def _hashes(num_blocks: int) -> list[bytes]:
    return [bytes([idx % 251]) * 32 for idx in range(num_blocks)]


class TestExternalCachedBlockPool(unittest.TestCase):
    def test_requires_hash_for_every_group(self):
        block_hash = b"h" * 32
        pool = ExternalCachedBlockPool({(0, block_hash)})

        self.assertIsNotNone(pool.get_cached_block(block_hash, [0]))
        self.assertIsNone(pool.get_cached_block(block_hash, [0, 1]))


class TestAscendStoreCoordinator(unittest.TestCase):
    def test_compressed_group_hits_on_effective_granularity(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(
            block_hashes,
            group_block_size=128 * 128,
            hash_block_size=128,
        )[0]
        coordinator = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=128))],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c128"],
        )

        _, hit_length = coordinator.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool({(0, bytes(grouped_hash))}),
        )

        self.assertEqual(hit_length, 128 * 128)

    def test_missing_required_group_returns_zero(self):
        block_hashes = _hashes(128)
        c1_exists = {(0, block_hash) for block_hash in block_hashes}
        coordinator = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=128)),
                KVCacheGroupSpec(["layer.1"], FullAttentionSpec(block_size=128)),
            ],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c128"],
        )

        _, hit_length = coordinator.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(c1_exists),
        )

        self.assertEqual(hit_length, 0)

    def test_store_mask_uses_manager_reachability(self):
        coordinator = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(
                    ["layer.0"],
                    SlidingWindowSpec(block_size=128, sliding_window=256),
                )
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )

        self.assertEqual(coordinator.store_mask(512), ([False, False, False, True],))

    def test_compressed_masks_stay_unmasked(self):
        coordinator = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(
                    ["layer.0"],
                    SlidingWindowSpec(block_size=128, sliding_window=512),
                )
            ],
            scheduler_block_size=2048,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c4"],
        )

        self.assertEqual(
            coordinator.store_mask(2048, num_prompt_tokens=2048),
            ([True] * 4,),
        )
        with patch.object(
            coordinator,
            "find_longest_cache_hit",
            return_value=(([False, False, False, True],), 2048),
        ):
            self.assertEqual(
                coordinator.load_mask(_hashes(16), 2048),
                ([True] * 4,),
            )

    def test_lookup_mask_uses_reachable_boundaries_only(self):
        coordinator = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(
                    ["layer.0"],
                    SlidingWindowSpec(block_size=128, sliding_window=256),
                ),
                KVCacheGroupSpec(["layer.1"], FullAttentionSpec(block_size=128)),
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c4"],
        )

        self.assertEqual(
            coordinator.lookup_mask(512),
            ([False, False, False, True], None),
        )

    def test_mtp_fallback_excludes_compressed_groups(self):
        coordinator = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["compressed"], FullAttentionSpec(block_size=128)),
                KVCacheGroupSpec(
                    ["sliding"],
                    SlidingWindowSpec(block_size=128, sliding_window=128),
                ),
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c4", "c1"],
            use_eagle=True,
        )

        self.assertEqual(coordinator.eagle_group_ids, {1})


if __name__ == "__main__":
    unittest.main()
