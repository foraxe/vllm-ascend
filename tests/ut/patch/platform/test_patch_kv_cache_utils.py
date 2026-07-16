# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_utils

pytestmark = pytest.mark.cpu_test


def _vllm_config(block_size, dcp=1, pcp=1):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _kv_cache_config(*block_sizes):
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size))
            for block_size in block_sizes
        ]
    )


def test_single_group_scales_cache_block_size_by_context_parallelism() -> None:
    result = patch_kv_cache_utils._ascend_resolve_kv_cache_block_sizes(
        _kv_cache_config(32),
        _vllm_config(32, dcp=2, pcp=4),
    )

    assert result == (256, 256)


def test_multiple_groups_use_lcm_scaled_by_context_parallelism() -> None:
    result = patch_kv_cache_utils._ascend_resolve_kv_cache_block_sizes(
        _kv_cache_config(32, 64, 128),
        _vllm_config(32, dcp=2, pcp=2),
    )

    assert result == (512, 512)


def test_multiple_groups_without_context_parallelism_delegate_upstream() -> None:
    config = _kv_cache_config(32, 64)
    vllm_config = _vllm_config(32)
    with patch.object(
        patch_kv_cache_utils,
        "_orig_resolve_kv_cache_block_sizes",
        return_value=(64, 32),
    ) as original:
        result = patch_kv_cache_utils._ascend_resolve_kv_cache_block_sizes(
            config,
            vllm_config,
        )

    assert result == (64, 32)
    original.assert_called_once_with(config, vllm_config)
