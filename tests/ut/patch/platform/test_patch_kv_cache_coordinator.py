# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.core.fixed_quota_block_pool import (
    FIXED_GROUP_BLOCK_QUOTAS_ATTR,
    FixedQuotaBlockPool,
    GroupBlockPoolView,
)
from vllm_ascend.patch.platform import patch_kv_cache_coordinator as coordinator_patch

pytestmark = pytest.mark.cpu_test


def _config(*specs):
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    )


def _call_factory(config):
    return coordinator_patch.get_kv_cache_coordinator(
        kv_cache_config=config,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=32,
        eagle_attn_layer_names=None,
        metrics_collector=None,
    )


def test_deepseek_v4_detection_supports_direct_and_nested_specs() -> None:
    direct = SimpleNamespace(model_version="deepseek_v4")
    nested = SimpleNamespace(
        kv_cache_specs={"layer": SimpleNamespace(model_version="deepseek_v4")}
    )
    unrelated = SimpleNamespace(model_version="llama")

    assert coordinator_patch._is_deepseek_v4_kv_cache_spec(direct)
    assert coordinator_patch._is_deepseek_v4_kv_cache_spec(nested)
    assert not coordinator_patch._is_deepseek_v4_kv_cache_spec(unrelated)
    assert coordinator_patch._is_deepseek_v4_kv_cache_config(_config(unrelated, nested))


def test_factory_builds_ascend_coordinator_for_dsv4() -> None:
    config = _config(SimpleNamespace(model_version="deepseek_v4"))
    with patch.object(
        coordinator_patch, "AscendHybridKVCacheCoordinator"
    ) as ascend_cls:
        result = _call_factory(config)

    assert result is ascend_cls.return_value
    assert ascend_cls.call_args.args[:3] == (config, 4096, True)
    assert ascend_cls.call_args.kwargs["max_num_batched_tokens"] == 1024


def test_factory_delegates_non_dsv4_config() -> None:
    original = MagicMock(return_value="upstream")
    config = _config(SimpleNamespace(model_version="llama"))
    with patch.object(coordinator_patch, "_orig_get_kv_cache_coordinator", original):
        result = _call_factory(config)

    assert result == "upstream"
    original.assert_called_once_with(
        kv_cache_config=config,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=32,
        eagle_attn_layer_names=None,
        metrics_collector=None,
    )


def test_factory_filters_unsupported_kwargs_for_non_dsv4_config() -> None:
    received = {}

    def original(
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        hash_block_size,
        metrics_collector=None,
    ):
        received.update(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        return "upstream"

    config = _config(SimpleNamespace(model_version="llama"))
    with patch.object(coordinator_patch, "_orig_get_kv_cache_coordinator", original):
        result = _call_factory(config)

    assert result == "upstream"
    assert received == {
        "kv_cache_config": config,
        "max_model_len": 4096,
        "max_num_batched_tokens": 1024,
        "use_eagle": True,
        "enable_caching": True,
        "enable_kv_cache_events": False,
        "dcp_world_size": 1,
        "pcp_world_size": 1,
        "hash_block_size": 32,
        "metrics_collector": None,
    }


def test_compressed_group_disables_eagle_adjustment_for_all_groups() -> None:
    compressed_spec = SimpleNamespace(block_size=128, compress_ratio=128)
    uncompressed_spec = SimpleNamespace(block_size=128, compress_ratio=1)
    config = SimpleNamespace(
        num_blocks=32,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=compressed_spec, is_eagle_group=False),
            SimpleNamespace(kv_cache_spec=uncompressed_spec, is_eagle_group=True),
        ],
    )
    with (
        patch.object(coordinator_patch, "BlockPool"),
        patch.object(
            coordinator_patch,
            "get_manager_for_kv_cache_spec",
            side_effect=[MagicMock(kv_cache_group_id=0), MagicMock(kv_cache_group_id=1)],
        ),
        patch.object(
            coordinator_patch.AscendHybridKVCacheCoordinator,
            "verify_and_split_kv_cache_groups",
        ),
    ):
        coordinator = coordinator_patch.AscendHybridKVCacheCoordinator(
            kv_cache_config=config,
            max_model_len=32768,
            use_eagle=True,
            enable_caching=False,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=128,
        )

    assert coordinator.eagle_group_ids == set()


def _two_group_coordinator_config(*, quotas=None):
    config = SimpleNamespace(
        num_blocks=10,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16, compress_ratio=1),
                is_eagle_group=False,
            ),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16, compress_ratio=2),
                is_eagle_group=False,
            ),
        ],
    )
    if quotas is not None:
        setattr(config, FIXED_GROUP_BLOCK_QUOTAS_ATTR, quotas)
    return config


def _construct_coordinator_with_mock_managers(config):
    managers = [MagicMock(kv_cache_group_id=0), MagicMock(kv_cache_group_id=1)]
    with (
        patch.object(
            coordinator_patch,
            "get_manager_for_kv_cache_spec",
            side_effect=managers,
        ) as manager_factory,
        patch.object(
            coordinator_patch.AscendHybridKVCacheCoordinator,
            "verify_and_split_kv_cache_groups",
        ),
    ):
        coordinator = coordinator_patch.AscendHybridKVCacheCoordinator(
            kv_cache_config=config,
            max_model_len=4096,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=16,
        )
    return coordinator, manager_factory


def test_feature_off_keeps_one_shared_upstream_block_pool() -> None:
    config = _two_group_coordinator_config()
    shared_pool = MagicMock()
    with patch.object(coordinator_patch, "BlockPool", return_value=shared_pool):
        coordinator, manager_factory = _construct_coordinator_with_mock_managers(config)

    assert coordinator.block_pool is shared_pool
    assert [call.kwargs["block_pool"] for call in manager_factory.call_args_list] == [shared_pool, shared_pool]


def test_feature_on_binds_each_manager_to_deterministic_group_view() -> None:
    config = _two_group_coordinator_config(quotas=(3, 6))
    coordinator, manager_factory = _construct_coordinator_with_mock_managers(config)

    assert isinstance(coordinator.block_pool, FixedQuotaBlockPool)
    manager_pools = [call.kwargs["block_pool"] for call in manager_factory.call_args_list]
    assert all(isinstance(pool, GroupBlockPoolView) for pool in manager_pools)
    assert [pool.group_id for pool in manager_pools] == [0, 1]
    assert [
        (pool.get_group_block_range(pool.group_id).start, pool.get_group_block_range(pool.group_id).stop)
        for pool in manager_pools
    ] == [(1, 4), (4, 10)]


def test_fixed_quota_admission_rejects_one_exhausted_partition() -> None:
    coordinator = object.__new__(coordinator_patch.AscendHybridKVCacheCoordinator)
    coordinator.block_pool = FixedQuotaBlockPool(
        num_gpu_blocks=5,
        enable_caching=False,
        hash_block_size=16,
        group_block_quotas=(1, 3),
    )
    coordinator.block_pool.get_group_view(0).get_new_blocks(1)
    managers = (MagicMock(), MagicMock())
    managers[0].get_num_blocks_to_allocate.return_value = 1
    managers[1].get_num_blocks_to_allocate.return_value = 0
    coordinator.single_type_managers = managers

    required = coordinator.get_num_blocks_to_allocate(
        request_id="request",
        num_tokens=16,
        new_computed_blocks=((), ()),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=16,
        apply_admission_cap=True,
    )

    assert required == coordinator.block_pool.get_num_free_blocks() + 1
    managers[0].get_num_blocks_to_allocate.assert_called_once()
    managers[1].get_num_blocks_to_allocate.assert_not_called()


def test_ascend_coordinator_cache_blocks_forwards_group_eagle_flags() -> None:
    coordinator = object.__new__(coordinator_patch.AscendHybridKVCacheCoordinator)
    coordinator.retention_interval = 16384
    coordinator.lcm_block_size = 16384
    coordinator.eagle_group_ids = {1}
    coordinator.single_type_managers = (
        MagicMock(kv_cache_group_id=0),
        MagicMock(kv_cache_group_id=1),
    )
    request = MagicMock()

    coordinator.cache_blocks(request, 32768)

    for group_id, manager in enumerate(coordinator.single_type_managers):
        manager.cache_blocks.assert_called_once_with(
            request,
            32768,
            retention_interval=16384,
            alignment_tokens=16384,
            use_eagle=group_id == 1,
        )


def _coordinator_for_hash_grouping(spec, manager_cls):
    coordinator = object.__new__(coordinator_patch.AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = _config(spec)
    coordinator.attention_groups = [(spec, [0], manager_cls)]
    coordinator.eagle_attn_group_indices = set()
    coordinator.block_pool = MagicMock()
    coordinator.lcm_block_size = 32
    coordinator.hash_block_size = 16
    coordinator.dcp_world_size = 2
    coordinator.pcp_world_size = 1
    return coordinator


def test_context_parallelism_groups_attention_hashes_to_effective_block_size() -> None:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    manager_cls = MagicMock()
    manager_cls.find_longest_cache_hit.return_value = ([MagicMock()],)
    coordinator = _coordinator_for_hash_grouping(spec, manager_cls)

    coordinator.find_longest_cache_hit([b"h0", b"h1"], 32)

    grouped_hashes = manager_cls.find_longest_cache_hit.call_args.kwargs["block_hashes"]
    assert len(grouped_hashes) == 1


def test_context_parallelism_does_not_rescale_mamba_hash_blocks() -> None:
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    manager_cls = MagicMock()
    manager_cls.find_longest_cache_hit.return_value = ([MagicMock(), MagicMock()],)
    coordinator = _coordinator_for_hash_grouping(spec, manager_cls)

    coordinator.find_longest_cache_hit([b"h0", b"h1"], 32)

    block_hashes = manager_cls.find_longest_cache_hit.call_args.kwargs["block_hashes"]
    assert len(block_hashes) == 2
