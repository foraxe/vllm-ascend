# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.patch.worker import patch_deepseek_compressor as compressor_patch
from vllm_ascend.utils import AscendDeviceType

pytestmark = pytest.mark.cpu_test


def _config(block_size=64):
    cache_config = SimpleNamespace(block_size=block_size, cache_dtype="auto")
    return SimpleNamespace(cache_config=cache_config)


@pytest.mark.parametrize(
    ("state_dim", "compress_ratio", "expected_padding"),
    [(512, 4, 8320), (256, 128, 65536)],
)
def test_compressor_state_spec_uses_layout_padding(state_dim, compress_ratio, expected_padding) -> None:
    state_cache = object.__new__(compressor_patch.AscendCompressorStateCache)
    torch.nn.Module.__init__(state_cache)
    state_cache.state_dim = state_dim
    state_cache.dtype = torch.float32
    state_cache.compress_ratio = compress_ratio
    state_cache.sliding_window = compress_ratio
    state_cache.block_size = 4 if compress_ratio == 4 else 16

    with patch.object(
        compressor_patch,
        "get_dsv4_cache_sizes_for_config",
        return_value=[[64, 64, 4, 16], [8320, 65536]],
    ):
        spec = state_cache.get_kv_cache_spec(_config())

    assert spec.page_size_padded == expected_padding
    assert spec.block_size == state_cache.block_size


def test_indexer_spec_uses_a5_dtype_block_and_scale_layout() -> None:
    indexer = object.__new__(compressor_patch.AscendDeepseekV4IndexerCache)
    torch.nn.Module.__init__(indexer)
    indexer.head_dim = 128
    indexer.dtype = torch.bfloat16
    indexer.compress_ratio = 4
    indexer.cache_config = SimpleNamespace(cache_dtype="auto")
    vllm_config = _config(64)

    with (
        patch.object(compressor_patch, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch.object(
            compressor_patch,
            "get_dsv4_cache_sizes_for_config",
            return_value=[[64, 64, 4, 8], [8448, 40960]],
        ),
    ):
        spec = indexer.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 64
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.scale_dim == 1
    assert spec.scale_dtype == torch.float32
    assert vllm_config.cache_config.cache_dtype == "float8_e4m3fn"


def test_swa_cache_constructor_and_a5_spec_use_layout() -> None:
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
    with (
        patch.object(compressor_patch.DeepseekV4SWACache, "__init__", return_value=None),
        patch.object(
            compressor_patch,
            "get_dsv4_cache_sizes_for_config",
            return_value=[[32, 32, 2, 4], [4224, 20480]],
        ),
    ):
        swa_cache = compressor_patch.AscendDeepseekV4SWACache(
            head_dim=256,
            window_size=512,
            dtype=torch.bfloat16,
            prefix="swa",
            cache_config=cache_config,
        )

    assert swa_cache.block_size == 32
    swa_cache.head_dim = 256
    swa_cache.window_size = 512
    swa_cache.cache_config = cache_config
    vllm_config = SimpleNamespace(cache_config=cache_config)

    with patch.object(compressor_patch, "get_ascend_device_type", return_value=AscendDeviceType.A5):
        spec = swa_cache.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 32
    assert spec.head_size == 384
    assert spec.dtype == torch.float8_e4m3fn
    assert cache_config.cache_dtype == "float8_e4m3fn"
