# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.models.layer.attention import layer as dsv4_layer
from vllm_ascend.utils import AscendDeviceType

pytestmark = pytest.mark.cpu_test


def test_dsa_backend_supports_all_dsv4_group_block_sizes() -> None:
    assert AscendDSABackend.get_supported_kernel_block_sizes() == [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]


@pytest.mark.parametrize(
    ("device_type", "expected_128"),
    [
        (AscendDeviceType.A3, [[128, 128, 8, 32], [16640, 131072]]),
        (AscendDeviceType.A5, [[128, 128, 8, 16], [16896, 81920]]),
    ],
)
def test_dsv4_layout_table_is_device_specific(device_type, expected_128) -> None:
    with patch.object(dsv4_layer, "get_ascend_device_type", return_value=device_type):
        layouts = dsv4_layer.get_dsv4_block_sizes()

    assert layouts[128] == expected_128
    assert set(layouts) == {32, 64, 128}


@pytest.mark.parametrize(
    ("block_size", "expected"),
    [
        (32, [[32, 32, 2, 8], [4160, 32768]]),
        (64, [[64, 64, 4, 16], [8320, 65536]]),
        (128, [[128, 128, 8, 32], [16640, 131072]]),
        (16, [[32, 32, 2, 8], [4160, 32768]]),
    ],
)
def test_dsv4_cache_sizes_select_supported_layout_or_default(block_size, expected) -> None:
    layouts = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }
    with patch.object(dsv4_layer, "DSV4_BLOCK_SIZES", layouts):
        assert dsv4_layer.get_dsv4_cache_sizes(block_size) == expected


def test_dsv4_cache_sizes_keep_saved_user_size_after_runtime_rewrite() -> None:
    cache_config = SimpleNamespace(
        block_size=2,
        _ascend_dsv4_user_block_size=64,
    )
    layouts = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }

    with patch.object(dsv4_layer, "DSV4_BLOCK_SIZES", layouts):
        assert dsv4_layer.get_dsv4_cache_sizes_for_config(cache_config) == layouts[64]


def test_dsa_kv_cache_spec_uses_configured_block_size_on_a3() -> None:
    attention = object.__new__(dsv4_layer.DSAAttention)
    attention.compress_ratio = 4
    attention.kv_cache_dtype = "auto"
    attention.head_size = 256
    cache_config = SimpleNamespace(block_size=64, cache_dtype="auto")
    vllm_config = SimpleNamespace(cache_config=cache_config, model_config=SimpleNamespace())

    with (
        patch.object(dsv4_layer, "get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch.object(dsv4_layer, "kv_cache_dtype_str_to_dtype", return_value=torch.bfloat16),
        patch.object(
            dsv4_layer,
            "get_dsv4_cache_sizes_for_config",
            return_value=[[64, 64, 4, 16], [8320, 65536]],
        ),
    ):
        spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 64
    assert spec.head_size == 256
    assert spec.dtype == torch.bfloat16
    assert spec.compress_ratio == 4


def test_dsa_kv_cache_spec_uses_a5_dtype_and_padded_head() -> None:
    attention = object.__new__(dsv4_layer.DSAAttention)
    attention.compress_ratio = 128
    attention.kv_cache_dtype = "auto"
    attention.head_size = 256
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
    vllm_config = SimpleNamespace(cache_config=cache_config, model_config=SimpleNamespace())

    with (
        patch.object(dsv4_layer, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch.object(dsv4_layer, "kv_cache_dtype_str_to_dtype", return_value=torch.bfloat16),
        patch.object(
            dsv4_layer,
            "get_dsv4_cache_sizes_for_config",
            return_value=[[32, 32, 2, 4], [4224, 20480]],
        ),
    ):
        spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 32
    assert spec.head_size == 384
    assert spec.dtype == torch.float8_e4m3fn
    assert cache_config.cache_dtype == "float8_e4m3fn"
