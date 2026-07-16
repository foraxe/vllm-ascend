# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

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


def test_factory_delegates_and_filters_kwargs_when_dsv4_patch_is_disabled() -> None:
    sentinel = object()
    received = {}

    def original(kv_cache_config, max_model_len, use_eagle):
        received.update(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
        )
        return sentinel

    config = _config(SimpleNamespace(model_version="deepseek_v4"))
    with (
        patch.object(coordinator_patch.envs, "VLLM_ASCEND_APPLY_DSV4_PATCH", False),
        patch.object(coordinator_patch, "_orig_get_kv_cache_coordinator", original),
    ):
        result = _call_factory(config)

    assert result is sentinel
    assert received == {
        "kv_cache_config": config,
        "max_model_len": 4096,
        "use_eagle": True,
    }


def test_factory_delegates_non_dsv4_config_even_when_patch_is_enabled() -> None:
    original = MagicMock(return_value="upstream")
    config = _config(SimpleNamespace(model_version="llama"))
    with (
        patch.object(coordinator_patch.envs, "VLLM_ASCEND_APPLY_DSV4_PATCH", True),
        patch.object(coordinator_patch, "_orig_get_kv_cache_coordinator", original),
    ):
        result = _call_factory(config)

    assert result == "upstream"
    original.assert_called_once()


def test_factory_builds_ascend_coordinator_only_for_enabled_dsv4() -> None:
    config = _config(SimpleNamespace(model_version="deepseek_v4"))
    with (
        patch.object(coordinator_patch.envs, "VLLM_ASCEND_APPLY_DSV4_PATCH", True),
        patch.object(coordinator_patch, "AscendHybridKVCacheCoordinator") as ascend_cls,
    ):
        result = _call_factory(config)

    assert result is ascend_cls.return_value
    assert ascend_cls.call_args.args[:3] == (config, 4096, True)
    assert ascend_cls.call_args.kwargs["max_num_batched_tokens"] == 1024


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
