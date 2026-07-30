# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import json
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
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size)) for block_size in block_sizes
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


class _FakeMLASpec:
    def __init__(
        self,
        *,
        block_size: int,
        compress_ratio: int,
        page_size_bytes: int,
    ) -> None:
        self.block_size = block_size
        self.compress_ratio = compress_ratio
        self.page_size_bytes = page_size_bytes


class _FakeSlidingWindowMLASpec:
    def __init__(
        self,
        *,
        block_size: int,
        sliding_window: int,
        page_size_bytes: int,
    ) -> None:
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.page_size_bytes = page_size_bytes


class _FakeUniformTypeKVCacheSpecs:
    def __init__(self, kv_cache_specs: dict[str, object]) -> None:
        self.kv_cache_specs = kv_cache_specs


def _fake_group(prefix: str, count: int, spec_factory):
    layer_names = [f"{prefix}.{index}" for index in range(count)]
    return SimpleNamespace(
        layer_names=layer_names,
        kv_cache_spec=_FakeUniformTypeKVCacheSpecs(
            {name: spec_factory(index) for index, name in enumerate(layer_names)}
        ),
    )


def _flash_groups():
    wide_page = 128 * 1024
    narrow_page = 16_640
    return [
        _fake_group(
            "c4",
            42,
            lambda index: _FakeMLASpec(
                block_size=128,
                compress_ratio=4,
                page_size_bytes=narrow_page if index < 21 else wide_page,
            ),
        ),
        _fake_group(
            "c128",
            20,
            lambda _index: _FakeMLASpec(
                block_size=128,
                compress_ratio=128,
                page_size_bytes=wide_page,
            ),
        ),
        _fake_group(
            "dense_swa_a",
            22,
            lambda _index: _FakeSlidingWindowMLASpec(
                block_size=128,
                sliding_window=4096,
                page_size_bytes=wide_page,
            ),
        ),
        _fake_group(
            "dense_swa_b",
            21,
            lambda _index: _FakeSlidingWindowMLASpec(
                block_size=128,
                sliding_window=4096,
                page_size_bytes=wide_page,
            ),
        ),
        _fake_group(
            "c4_state",
            42,
            lambda index: _FakeSlidingWindowMLASpec(
                block_size=8,
                sliding_window=8,
                page_size_bytes=narrow_page if index < 21 else wide_page,
            ),
        ),
        _fake_group(
            "c128_state",
            20,
            lambda _index: _FakeSlidingWindowMLASpec(
                block_size=32,
                sliding_window=128,
                page_size_bytes=wide_page,
            ),
        ),
    ]


def _packed_vllm_config(
    *,
    enabled: bool = True,
    activation: bool = False,
):
    additional_config = {
        patch_kv_cache_utils.ENABLE_C128_PACKED_POOL_PLANNER: enabled,
    }
    if activation:
        additional_config.update(
            {
                patch_kv_cache_utils.ENABLE_C128_PACKED_POOL_ACTIVATION: True,
                patch_kv_cache_utils.ENABLE_C128_PACKED_VMM_ARENA: True,
            }
        )
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=True,
            data_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            model="/models/DeepSeek-V4-Flash-w8a8",
            max_model_len=8_201,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=5_120,
            max_num_scheduled_tokens=None,
            max_num_seqs=1,
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            long_prefill_token_threshold=0,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )


def _patch_packed_spec_types():
    return (
        patch.object(
            patch_kv_cache_utils,
            "MLAAttentionSpec",
            _FakeMLASpec,
        ),
        patch.object(
            patch_kv_cache_utils,
            "SlidingWindowMLASpec",
            _FakeSlidingWindowMLASpec,
        ),
        patch.object(
            patch_kv_cache_utils,
            "UniformTypeKVCacheSpecs",
            _FakeUniformTypeKVCacheSpecs,
        ),
    )


def test_fixed_flash_quotas_match_current_manager_chunk_semantics() -> None:
    config = _packed_vllm_config()
    expected_quotas = [17, 1, 65, 65, 642, 165]

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with mla_patch, swa_patch, uniform_patch:
        quota_and_shape = [
            patch_kv_cache_utils._packed_group_quota(
                group,
                max_num_batched_tokens=(config.scheduler_config.max_num_batched_tokens),
            )
            for group in _flash_groups()
        ]

    actual_quotas = [quota for quota, _shape in quota_and_shape]
    assert actual_quotas == expected_quotas
    # The one generated output token is sampled after prefill; it does not
    # consume a new KV slot. C4 still uses the manager's floor-before-ceil rule:
    # ceil(floor(8200 / 4) / 128) == 17.
    assert [
        (
            shape["peak_live_blocks"],
            shape["admission_blocks"],
            shape["partition_blocks"],
        )
        for _quota, shape in quota_and_shape
    ] == [
        (17, 17, 17),
        (1, 1, 1),
        (57, 65, 65),
        (57, 65, 65),
        (640, 642, 642),
        (160, 165, 165),
    ]
    assert sum(actual_quotas) == 955


def test_packed_planner_feature_off_returns_exact_original_objects() -> None:
    config = SimpleNamespace(
        additional_config={patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY: {"caller_owned": True}}
    )
    original_configs = [SimpleNamespace(marker=object())]
    with patch.object(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_configs",
        return_value=original_configs,
    ) as original:
        result = patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [123],
        )

    assert result is original_configs
    assert config.additional_config == {patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY: {"caller_owned": True}}
    assert not hasattr(
        original_configs[0],
        patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
    )
    original.assert_called_once()


def test_packed_planner_serializes_exact_ranges_after_final_block_clamp() -> None:
    config = _packed_vllm_config()
    cache_config = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[SimpleNamespace(size=123, shared_by=["unchanged"])],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[cache_config],
        ),
    ):
        result = patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [999_999],
        )

    assert result == [cache_config]
    assert cache_config.kv_cache_tensors[0].size == 123
    metadata = getattr(
        cache_config,
        patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
    )
    assert config.additional_config[patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY] is metadata
    assert metadata["schema_version"] == 1
    assert metadata["profile"] == "dsv4_flash_prefill_8200_tokens_1out"
    assert metadata["planner_only"] is True
    assert metadata["downstream_runtime_abi_ready"] is False
    assert not hasattr(
        cache_config,
        patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
    )
    assert metadata["global_block_capacity"] == 4_190
    assert metadata["used_logical_blocks"] == 955
    assert metadata["unused_logical_blocks"] == 3_234
    assert [
        (
            group["logical_blocks"],
            group["logical_start"],
            group["logical_stop"],
        )
        for group in metadata["groups"]
    ] == [
        (17, 1, 18),
        (1, 18, 19),
        (65, 19, 84),
        (65, 84, 149),
        (642, 149, 791),
        (165, 791, 956),
    ]
    assert {component["placement"] for component in metadata["groups"][1]["components"]} == {"c128_owner"}
    assert metadata["groups"][0]["components"][0]["layer_names"] == [f"c4.{index}" for index in range(21)]
    assert metadata["groups"][0]["components"][1]["layer_names"] == [f"c4.{index}" for index in range(21, 42)]
    assert metadata["groups"][1]["components"][0]["layer_names"] == [f"c128.{index}" for index in range(20)]
    assert metadata["groups"][2]["components"][0]["layer_names"] == [f"dense_swa_a.{index}" for index in range(22)]
    assert metadata["groups"][3]["components"][0]["layer_names"] == [f"dense_swa_b.{index}" for index in range(21)]
    assert metadata["groups"][4]["components"][0]["layer_names"] == [f"c4_state.{index}" for index in range(21)]
    assert metadata["groups"][4]["components"][1]["layer_names"] == [f"c4_state.{index}" for index in range(21, 42)]
    for expected_group, serialized_group in zip(
        _flash_groups(),
        metadata["groups"],
    ):
        component_layers = [
            layer_name for component in serialized_group["components"] for layer_name in component["layer_names"]
        ]
        assert sorted(component_layers) == sorted(expected_group.layer_names)
        assert len(component_layers) == len(set(component_layers))
    # The full-B raw-tensor tuple paired C128 attention with its compressor
    # state in the legacy allocator. The packed manifest must move that state
    # family too; an attention-only owner sidecar cannot reclaim the tuple.
    assert metadata["groups"][5]["components"][0]["layer_names"] == [f"c128_state.{index}" for index in range(20)]
    assert metadata["buckets"]
    assert len(metadata["scratch"]) == 1
    assert metadata["scratch"][0]["max_pages_per_rank"] == 65
    assert metadata["scratch"][0]["allocation_granularity_bytes"] == 2 * 1024 * 1024
    assert len(metadata["scratch"][0]["segments"]) == 8
    for segment in metadata["scratch"][0]["segments"]:
        assert segment["segment_base_bytes"] % (2 * 1024 * 1024) == 0
        assert segment["segment_allocated_bytes"] == 10 * 1024 * 1024
    assert len(metadata["total_physical_bytes_by_rank"]) == 8
    # The C128 group quota is one page. A sentinel plus that page rounds to
    # the same 2-MiB segment whether it is owner-sharded or replicated.
    # Any bytes-vs-B0 delta for this fixed plan is quota-envelope reduction,
    # not owner placement.
    expected_fixed_bytes = [3_166_699_520] * 8
    assert metadata["total_physical_bytes_by_rank"] == expected_fixed_bytes
    assert metadata["aligned_quota_replicated_bytes_by_rank"] == expected_fixed_bytes
    assert json.loads(json.dumps(metadata, sort_keys=True)) == metadata


def test_packed_activation_publishes_same_b_quota_transaction() -> None:
    config = _packed_vllm_config(activation=True)
    worker_configs = [
        SimpleNamespace(
            num_blocks=4_190,
            kv_cache_groups=_flash_groups(),
            kv_cache_tensors=[],
        )
        for _ in range(2)
    ]

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=worker_configs,
        ),
    ):
        result = patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}, {"layer": object()}],
            [999_999, 999_999],
        )

    assert result is worker_configs
    metadata = getattr(
        worker_configs[0],
        patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
    )
    assert (
        getattr(
            worker_configs[1],
            patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
        )
        == metadata
    )
    assert config.additional_config[patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY] is metadata
    required_quotas = [17, 1, 65, 65, 642, 165]
    assigned_quotas = [17, 3_235, 65, 65, 642, 165]
    assert metadata["schema_version"] == 1
    assert metadata["planner_only"] is False
    assert metadata["downstream_runtime_abi_ready"] is True
    assert metadata["expert_parallel_size"] == 8
    assert metadata["global_block_capacity"] == 4_190
    assert metadata["required_group_block_quotas"] == required_quotas
    assert metadata["assigned_group_block_quotas"] == assigned_quotas
    assert [group["identity"] for group in metadata["groups"]] == [
        "c4_attention",
        "c128_attention",
        "dense_swa_a",
        "dense_swa_b",
        "c4_state",
        "c128_state",
    ]
    assert [group["required_logical_blocks"] for group in metadata["groups"]] == required_quotas
    assert [group["assigned_logical_blocks"] for group in metadata["groups"]] == assigned_quotas
    assert [group["scheduler_shape"]["partition_blocks"] for group in metadata["groups"]] == assigned_quotas
    assert [group["scheduler_shape"]["workload_required_blocks"] for group in metadata["groups"]] == required_quotas
    assert [
        (
            group["logical_start"],
            group["logical_stop"],
        )
        for group in metadata["groups"]
    ] == [
        (1, 18),
        (18, 3_253),
        (3_253, 3_318),
        (3_318, 3_383),
        (3_383, 4_025),
        (4_025, 4_190),
    ]
    assert metadata["used_logical_blocks"] == 4_189
    assert metadata["unused_logical_blocks"] == 0
    assert sum(component["copies"] for group in metadata["groups"] for component in group["components"]) == 167
    assert len(metadata["scratch"]) == 1
    assert metadata["total_physical_bytes_by_rank"] == [4_215_275_520] * 8
    assert metadata["scheduler_group_identities"] == [
        {
            "group_index": group_index,
            "group_name": f"group_{group_index}",
            "identity": identity,
            "required_blocks": required,
            "assigned_blocks": assigned,
        }
        for group_index, (identity, required, assigned) in enumerate(
            zip(
                [
                    "c4_attention",
                    "c128_attention",
                    "dense_swa_a",
                    "dense_swa_b",
                    "c4_state",
                    "c128_state",
                ],
                required_quotas,
                assigned_quotas,
            )
        )
    ]
    for worker_config in worker_configs:
        assert getattr(
            worker_config,
            patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
        ) == tuple(assigned_quotas)
    assert json.loads(json.dumps(metadata, sort_keys=True)) == metadata


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            patch_kv_cache_utils.ENABLE_C128_PACKED_POOL_PLANNER,
            False,
            "activation=true requires.*planner=true",
        ),
        (
            patch_kv_cache_utils.ENABLE_C128_PACKED_VMM_ARENA,
            False,
            "activation=true requires.*arena=true",
        ),
    ],
)
def test_packed_activation_requires_coherent_feature_gates(
    field: str,
    value: bool,
    message: str,
) -> None:
    config = _packed_vllm_config(activation=True)
    config.additional_config[field] = value
    cache_config = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[cache_config],
        ),
        pytest.raises(ValueError, match=message),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [999_999],
        )

    assert not hasattr(
        cache_config,
        patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
    )
    assert not hasattr(
        cache_config,
        patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
    )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("blocks", "requires final global_block_capacity=4190"),
        ("ep", "requires enable_expert_parallel=true"),
        ("group_order", "workload block vector drift"),
        ("component", "dense_swa_a component manifest drift"),
    ],
)
def test_packed_activation_rejects_profile_or_manifest_drift(
    drift: str,
    message: str,
) -> None:
    config = _packed_vllm_config(activation=True)
    groups = _flash_groups()
    cache_config = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=groups,
        kv_cache_tensors=[],
    )
    if drift == "blocks":
        cache_config.num_blocks = 4_189
    elif drift == "ep":
        config.parallel_config.enable_expert_parallel = False
    elif drift == "group_order":
        groups[0], groups[1] = groups[1], groups[0]
    else:
        swa_spec = next(iter(groups[2].kv_cache_spec.kv_cache_specs.values()))
        swa_spec.page_size_bytes = 16_640

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[cache_config],
        ),
        pytest.raises(ValueError, match=message),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [999_999],
        )

    assert not hasattr(
        cache_config,
        patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
    )
    assert not hasattr(
        cache_config,
        patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
    )


def test_packed_activation_worker_drift_publishes_nothing() -> None:
    config = _packed_vllm_config(activation=True)
    first = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )
    second_groups = _flash_groups()
    second_groups[-1].layer_names[0] = "different.worker.layer"
    second_groups[-1].kv_cache_spec.kv_cache_specs["different.worker.layer"] = second_groups[
        -1
    ].kv_cache_spec.kv_cache_specs.pop("c128_state.0")
    second = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=second_groups,
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[first, second],
        ),
        pytest.raises(ValueError, match="identical worker group schemas"),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}, {"layer": object()}],
            [999_999, 999_999],
        )

    for cache_config in (first, second):
        assert not hasattr(
            cache_config,
            patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
        )
        assert not hasattr(
            cache_config,
            patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
        )
    assert patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY not in config.additional_config


def test_packed_activation_publication_rolls_back_partial_attributes() -> None:
    class RejectQuotaConfig(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR:
                raise RuntimeError("reject quota publication")
            super().__setattr__(name, value)

    config = _packed_vllm_config(activation=True)
    first = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )
    second = RejectQuotaConfig(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[first, second],
        ),
        pytest.raises(RuntimeError, match="reject quota publication"),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}, {"layer": object()}],
            [999_999, 999_999],
        )

    for cache_config in (first, second):
        assert not hasattr(
            cache_config,
            patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY,
        )
        assert not hasattr(
            cache_config,
            patch_kv_cache_utils.FIXED_GROUP_BLOCK_QUOTAS_ATTR,
        )
    assert patch_kv_cache_utils.C128_PACKED_POOL_METADATA_KEY not in config.additional_config


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (
            ("parallel_config", "prefill_context_parallel_size"),
            2,
            "cache managers scale block_size",
        ),
        (
            ("scheduler_config", "max_num_seqs"),
            2,
            "requires max_num_seqs=1",
        ),
        (
            ("scheduler_config", "max_num_scheduled_tokens"),
            4_096,
            "effective max_num_scheduled_tokens=5120",
        ),
        (
            ("scheduler_config", "max_num_partial_prefills"),
            2,
            "requires max_num_partial_prefills=1",
        ),
        (
            ("model_config", "max_model_len"),
            32_768,
            "requires max_model_len=8201",
        ),
    ],
)
def test_packed_planner_rejects_unpinned_runtime_shape(
    field_path: tuple[str, str],
    value: int,
    message: str,
) -> None:
    config = _packed_vllm_config()
    setattr(getattr(config, field_path[0]), field_path[1], value)
    cache_config = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[cache_config],
        ),
        pytest.raises(ValueError, match=message),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [999_999],
        )


def test_packed_planner_rejects_worker_group_schema_drift() -> None:
    config = _packed_vllm_config()
    first = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )
    second_groups = _flash_groups()
    second_groups[-1].layer_names[0] = "different.worker.layer"
    second_groups[-1].kv_cache_spec.kv_cache_specs["different.worker.layer"] = second_groups[
        -1
    ].kv_cache_spec.kv_cache_specs.pop("c128_state.0")
    second = SimpleNamespace(
        num_blocks=4_190,
        kv_cache_groups=second_groups,
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[first, second],
        ),
        pytest.raises(ValueError, match="identical worker group schemas"),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}, {"layer": object()}],
            [999_999, 999_999],
        )


def test_packed_planner_fails_closed_when_group_quotas_exceed_pool() -> None:
    config = _packed_vllm_config()
    cache_config = SimpleNamespace(
        num_blocks=900,
        kv_cache_groups=_flash_groups(),
        kv_cache_tensors=[],
    )

    mla_patch, swa_patch, uniform_patch = _patch_packed_spec_types()
    with (
        mla_patch,
        swa_patch,
        uniform_patch,
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[cache_config],
        ),
        pytest.raises(ValueError, match="exceeds usable data capacity"),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [{"layer": object()}],
            [999_999],
        )


def test_packed_planner_rejects_non_boolean_feature_gate() -> None:
    config = SimpleNamespace(additional_config={patch_kv_cache_utils.ENABLE_C128_PACKED_POOL_PLANNER: "1"})
    with (
        patch.object(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_configs",
            return_value=[],
        ),
        pytest.raises(ValueError, match="must be a JSON boolean"),
    ):
        patch_kv_cache_utils._ascend_get_kv_cache_configs(
            config,
            [],
            [],
        )
