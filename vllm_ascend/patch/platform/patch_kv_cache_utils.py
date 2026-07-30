# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from typing import Any

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_configs = vllm.v1.core.kv_cache_utils.get_kv_cache_configs

ENABLE_C128_PACKED_POOL_PLANNER = "enable_c128_packed_pool_planner"
C128_PACKED_POOL_METADATA_KEY = "c128_packed_pool_metadata"
C128_PACKED_POOL_SCHEMA_VERSION = 1
C128_PACKED_POOL_PROFILE = "dsv4_flash_prefill_8200_tokens_1out"
C128_PACKED_POOL_PROMPT_TOKENS = 8_200
C128_PACKED_POOL_OUTPUT_TOKENS = 1
C128_PACKED_POOL_MAX_MODEL_LEN = C128_PACKED_POOL_PROMPT_TOKENS + C128_PACKED_POOL_OUTPUT_TOKENS
C128_PACKED_POOL_MAX_CONCURRENT_REQUESTS = 1
C128_PACKED_POOL_VMM_GRANULARITY_BYTES = 2 * 1024 * 1024


def _is_c128_packed_pool_planner_enabled(vllm_config: VllmConfig) -> bool:
    additional_config = vllm_config.additional_config
    if not additional_config:
        return False
    enabled = additional_config.get(ENABLE_C128_PACKED_POOL_PLANNER, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{ENABLE_C128_PACKED_POOL_PLANNER} must be a JSON boolean, " f"got {enabled!r}")
    return enabled


def _validate_c128_packed_pool_profile(vllm_config: VllmConfig) -> int:
    """Validate the only workload for which the first packed plan is sound."""
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    if tp_size != 8:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires tensor_parallel_size=8, " f"got {tp_size}")
    pp_size = parallel_config.pipeline_parallel_size
    if pp_size != 1:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires pipeline_parallel_size=1, " f"got {pp_size}")
    dcp_size = parallel_config.decode_context_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    if dcp_size != 1 or pcp_size != 1:
        raise ValueError(
            f"{C128_PACKED_POOL_PROFILE} requires "
            "decode_context_parallel_size=1 and "
            "prefill_context_parallel_size=1 because cache managers scale "
            f"block_size by dcp*pcp, got dcp={dcp_size}, pcp={pcp_size}"
        )

    model_config = vllm_config.model_config
    model_name = str(getattr(model_config, "model", ""))
    if model_name and "flash" not in model_name.lower():
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} supports DeepSeek-V4-Flash only, " f"got model={model_name!r}")
    max_model_len = model_config.max_model_len
    if max_model_len != C128_PACKED_POOL_MAX_MODEL_LEN:
        raise ValueError(
            f"{C128_PACKED_POOL_PROFILE} requires max_model_len="
            f"{C128_PACKED_POOL_MAX_MODEL_LEN}, got {max_model_len}"
        )

    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    scheduler_config = vllm_config.scheduler_config
    if max_num_batched_tokens != 5_120:
        raise ValueError(
            f"{C128_PACKED_POOL_PROFILE} requires " f"max_num_batched_tokens=5120, got {max_num_batched_tokens}"
        )
    max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens or max_num_batched_tokens
    if max_num_scheduled_tokens != 5_120:
        raise ValueError(
            f"{C128_PACKED_POOL_PROFILE} requires effective "
            "max_num_scheduled_tokens=5120, got "
            f"{max_num_scheduled_tokens}"
        )
    max_num_seqs = scheduler_config.max_num_seqs
    if max_num_seqs != C128_PACKED_POOL_MAX_CONCURRENT_REQUESTS:
        raise ValueError(
            f"{C128_PACKED_POOL_PROFILE} requires max_num_seqs="
            f"{C128_PACKED_POOL_MAX_CONCURRENT_REQUESTS}, got {max_num_seqs}"
        )
    if not scheduler_config.enable_chunked_prefill:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires chunked prefill enabled")
    if scheduler_config.max_num_partial_prefills != 1:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires max_num_partial_prefills=1")
    if scheduler_config.long_prefill_token_threshold != 0:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires " "long_prefill_token_threshold=0")
    if vllm_config.speculative_config is not None:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires MTP/speculative decode disabled")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(f"{C128_PACKED_POOL_PROFILE} requires prefix caching disabled")
    return max_num_batched_tokens


def _sliding_window_peak_blocks(
    *,
    prompt_tokens: int,
    max_num_batched_tokens: int,
    block_size: int,
    sliding_window: int,
) -> int:
    """Mirror the current SlidingWindowManager's chunk-by-chunk live peak."""
    computed_tokens = 0
    block_table_len = 0
    peak_live_blocks = 0
    while computed_tokens < prompt_tokens:
        scheduled_tokens = min(
            max_num_batched_tokens,
            prompt_tokens - computed_tokens,
        )
        skipped_tokens = max(0, computed_tokens - sliding_window + 1)
        skipped_blocks = min(skipped_tokens // block_size, block_table_len)
        live_blocks = block_table_len - skipped_blocks

        required_table_len = cdiv(
            computed_tokens + scheduled_tokens,
            block_size,
        )
        appended_blocks = max(0, required_table_len - block_table_len)
        live_blocks += appended_blocks
        peak_live_blocks = max(peak_live_blocks, live_blocks)

        block_table_len = required_table_len
        computed_tokens += scheduled_tokens
    return peak_live_blocks


def _packed_group_quota(
    group: KVCacheGroupSpec,
    *,
    max_num_batched_tokens: int,
) -> tuple[int, dict[str, int | str]]:
    """Return max(live peak, admission reservation) and scheduler shape."""
    uniform_spec = group.kv_cache_spec
    if not isinstance(uniform_spec, UniformTypeKVCacheSpecs):
        raise ValueError("packed C128 planner requires UniformTypeKVCacheSpecs")
    specs = tuple(uniform_spec.kv_cache_specs.values())
    if not specs:
        raise ValueError("packed C128 planner received an empty cache group")

    if all(isinstance(spec, MLAAttentionSpec) for spec in specs):
        compress_ratios = {spec.compress_ratio for spec in specs}
        block_sizes = {spec.block_size for spec in specs}
        if len(compress_ratios) != 1 or len(block_sizes) != 1:
            raise ValueError("full MLA packed group must have one block size and compression ratio")
        compress_ratio = next(iter(compress_ratios))
        block_size = next(iter(block_sizes))
        compressed_tokens = C128_PACKED_POOL_PROMPT_TOKENS // compress_ratio
        logical_blocks = cdiv(compressed_tokens, block_size)
        return logical_blocks, {
            "kind": "compressed_mla",
            "block_size": block_size,
            "compress_ratio": compress_ratio,
            "peak_live_blocks": logical_blocks,
            "admission_blocks": logical_blocks,
            "partition_blocks": logical_blocks,
        }

    if all(isinstance(spec, SlidingWindowMLASpec) for spec in specs):
        block_sizes = {spec.block_size for spec in specs}
        sliding_windows = {spec.sliding_window for spec in specs}
        if len(block_sizes) != 1 or len(sliding_windows) != 1:
            raise ValueError("SWA MLA packed group must have one block size and sliding window")
        block_size = next(iter(block_sizes))
        sliding_window = next(iter(sliding_windows))
        peak_live_blocks = _sliding_window_peak_blocks(
            prompt_tokens=C128_PACKED_POOL_PROMPT_TOKENS,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            sliding_window=sliding_window,
        )
        admission_cap = (
            cdiv(
                min(
                    sliding_window - 1 + max_num_batched_tokens,
                    C128_PACKED_POOL_MAX_MODEL_LEN,
                ),
                block_size,
            )
            + 1
        )
        admission_blocks = min(
            cdiv(C128_PACKED_POOL_PROMPT_TOKENS, block_size),
            admission_cap,
        )
        logical_blocks = max(peak_live_blocks, admission_blocks)
        return logical_blocks, {
            "kind": "sliding_window_mla",
            "block_size": block_size,
            "sliding_window": sliding_window,
            "peak_live_blocks": peak_live_blocks,
            "admission_blocks": admission_blocks,
            "partition_blocks": logical_blocks,
        }

    spec_types = sorted({type(spec).__name__ for spec in specs})
    raise ValueError(
        "packed C128 planner supports only homogeneous compressed MLA or " f"SWA MLA groups, got {spec_types}"
    )


def _packed_group_components(
    group: KVCacheGroupSpec,
    *,
    group_name: str,
) -> tuple[tuple[Any, ...], dict[str, list[str]]]:
    from vllm_ascend.attention.context_parallel.c128_packed_pool import (
        PackedPlacement,
        PackedPoolComponentSpec,
    )

    uniform_spec = group.kv_cache_spec
    assert isinstance(uniform_spec, UniformTypeKVCacheSpecs)
    component_layers: dict[tuple[int, PackedPlacement], list[str]] = defaultdict(list)
    for layer_name in group.layer_names:
        layer_spec = uniform_spec.kv_cache_specs[layer_name]
        placement = (
            PackedPlacement.C128_OWNER
            if isinstance(layer_spec, MLAAttentionSpec) and layer_spec.compress_ratio == 128
            else PackedPlacement.REPLICATED
        )
        component_layers[(layer_spec.page_size_bytes, placement)].append(layer_name)

    components = []
    layer_names_by_component: dict[str, list[str]] = {}
    for component_index, ((page_size, placement), layer_names) in enumerate(
        sorted(
            component_layers.items(),
            key=lambda item: (item[0][0], item[0][1].value),
        )
    ):
        component_name = f"{group_name}_component_{component_index}"
        components.append(
            PackedPoolComponentSpec(
                name=component_name,
                bucket=f"page_{page_size}",
                page_size_bytes=page_size,
                copies=len(layer_names),
                placement=placement,
                allocation_granularity_bytes=(C128_PACKED_POOL_VMM_GRANULARITY_BYTES),
            )
        )
        layer_names_by_component[component_name] = layer_names
    return tuple(components), layer_names_by_component


def _serialize_c128_packed_pool_plan(
    *,
    plan: Any,
    group_profiles: list[dict[str, Any]],
    max_num_batched_tokens: int,
) -> dict[str, Any]:
    groups = []
    range_by_name = {logical_range.group_name: logical_range for logical_range in plan.group_ranges}
    for group_spec, group_profile in zip(plan.groups, group_profiles):
        logical_range = range_by_name[group_spec.name]
        serialized_group_profile = dict(group_profile)
        layer_names_by_component = serialized_group_profile.pop("_layer_names_by_component")
        components = []
        for component in group_spec.components:
            copies = []
            for copy_index in range(component.copies):
                ranks = []
                for rank in range(plan.tp_size):
                    sentinel = plan.sentinel_address(
                        group_spec.name,
                        component.name,
                        tp_rank=rank,
                        copy_index=copy_index,
                    )
                    ranks.append(
                        {
                            "rank": rank,
                            "segment_base_bytes": sentinel.segment_base_bytes,
                            "segment_allocated_bytes": (sentinel.segment_allocated_bytes),
                            "sentinel_offset_bytes": (sentinel.physical_offset_bytes),
                        }
                    )
                copies.append({"copy_index": copy_index, "ranks": ranks})
            components.append(
                {
                    "name": component.name,
                    "bucket": component.bucket,
                    "page_size_bytes": component.page_size_bytes,
                    "copies": component.copies,
                    "placement": component.placement.value,
                    "allocation_granularity_bytes": (component.allocation_granularity_bytes),
                    "layer_names": layer_names_by_component[component.name],
                    "segments": copies,
                }
            )
        groups.append(
            {
                **serialized_group_profile,
                "name": group_spec.name,
                "logical_blocks": group_spec.logical_blocks,
                "logical_start": logical_range.start,
                "logical_stop": logical_range.stop,
                "components": components,
            }
        )

    buckets = [
        {
            "bucket": bucket.bucket,
            "page_size_bytes": bucket.page_size_bytes,
            "replicated_pages_per_rank": bucket.replicated_pages_per_rank,
            "owner_pages_by_rank": list(bucket.owner_pages_by_rank),
            "sentinel_pages_per_rank": bucket.sentinel_pages_per_rank,
            "scratch_pages_per_rank": bucket.scratch_pages_per_rank,
            "persistent_allocated_bytes_by_rank": list(bucket.persistent_allocated_bytes_by_rank),
            "scratch_region_bytes_by_rank": list(bucket.scratch_region_bytes_by_rank),
            "total_allocated_bytes_by_rank": list(bucket.total_allocated_bytes_by_rank),
        }
        for bucket in plan.bucket_accounting
    ]
    scratch = []
    for bucket in plan.bucket_accounting:
        if bucket.scratch_pages_per_rank == 0:
            continue
        scratch_allocated_bytes = round_up(
            bucket.scratch_pages_per_rank * bucket.page_size_bytes,
            C128_PACKED_POOL_VMM_GRANULARITY_BYTES,
        )
        scratch.append(
            {
                "bucket": bucket.bucket,
                "page_size_bytes": bucket.page_size_bytes,
                "max_pages_per_rank": bucket.scratch_pages_per_rank,
                "allocation_granularity_bytes": (C128_PACKED_POOL_VMM_GRANULARITY_BYTES),
                "segments": [
                    {
                        "rank": rank,
                        "segment_base_bytes": (total_bytes - scratch_allocated_bytes),
                        "segment_allocated_bytes": scratch_allocated_bytes,
                    }
                    for rank, total_bytes in enumerate(bucket.total_allocated_bytes_by_rank)
                ],
            }
        )

    return {
        "schema_version": C128_PACKED_POOL_SCHEMA_VERSION,
        "profile": C128_PACKED_POOL_PROFILE,
        "planner_only": True,
        "downstream_runtime_abi_ready": False,
        "prompt_tokens": C128_PACKED_POOL_PROMPT_TOKENS,
        "output_tokens": C128_PACKED_POOL_OUTPUT_TOKENS,
        "kv_slot_tokens": C128_PACKED_POOL_PROMPT_TOKENS,
        "max_model_len": C128_PACKED_POOL_MAX_MODEL_LEN,
        "max_concurrent_requests": C128_PACKED_POOL_MAX_CONCURRENT_REQUESTS,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_scheduled_tokens": max_num_batched_tokens,
        "max_num_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "tp_size": plan.tp_size,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "sentinel_block_id": 0,
        "global_block_capacity": plan.global_block_capacity,
        "usable_data_capacity": plan.usable_data_capacity,
        "used_logical_blocks": plan.used_logical_blocks,
        "unused_logical_blocks": plan.unused_logical_blocks,
        "groups": groups,
        "buckets": buckets,
        "scratch": scratch,
        "total_physical_bytes_by_rank": list(plan.total_physical_bytes_by_rank()),
        "quota_replicated_bytes_by_rank": list(plan.quota_replicated_bytes_by_rank()),
    }


def _build_c128_packed_pool_metadata(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> dict[str, Any]:
    from vllm_ascend.attention.context_parallel.c128_packed_pool import (
        PackedPlacement,
        PackedPoolGroupSpec,
        PackedPoolPlan,
        PackedPoolScratchSpec,
    )

    max_num_batched_tokens = _validate_c128_packed_pool_profile(vllm_config)
    packed_groups = []
    group_profiles: list[dict[str, Any]] = []
    c128_scratch_bucket: tuple[str, int] | None = None
    for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
        group_name = f"group_{group_index}"
        logical_blocks, scheduler_shape = _packed_group_quota(
            group,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        components, layer_names_by_component = _packed_group_components(
            group,
            group_name=group_name,
        )
        packed_groups.append(
            PackedPoolGroupSpec(
                name=group_name,
                logical_blocks=logical_blocks,
                components=components,
            )
        )
        uniform_spec = group.kv_cache_spec
        assert isinstance(uniform_spec, UniformTypeKVCacheSpecs)
        group_profiles.append(
            {
                "group_index": group_index,
                "layer_names": list(group.layer_names),
                "scheduler_shape": scheduler_shape,
                "_layer_names_by_component": layer_names_by_component,
            }
        )
        for component in components:
            if component.placement is PackedPlacement.C128_OWNER:
                candidate = (component.bucket, component.page_size_bytes)
                if c128_scratch_bucket is not None and c128_scratch_bucket != candidate:
                    raise ValueError("packed C128 planner found multiple C128 page buckets")
                c128_scratch_bucket = candidate

    if c128_scratch_bucket is None:
        raise ValueError("packed C128 planner requires a compress_ratio=128 cache group")
    scratch_bucket, scratch_page_size = c128_scratch_bucket
    selected_c128_rows = cdiv(C128_PACKED_POOL_PROMPT_TOKENS, 128)
    plan = PackedPoolPlan(
        global_block_capacity=kv_cache_config.num_blocks,
        tp_size=vllm_config.parallel_config.tensor_parallel_size,
        groups=tuple(packed_groups),
        scratch=(
            PackedPoolScratchSpec(
                bucket=scratch_bucket,
                page_size_bytes=scratch_page_size,
                max_pages_per_rank=selected_c128_rows,
                allocation_granularity_bytes=(C128_PACKED_POOL_VMM_GRANULARITY_BYTES),
            ),
        ),
    )
    return _serialize_c128_packed_pool_plan(
        plan=plan,
        group_profiles=group_profiles,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _ascend_get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Attach a final-block-count packed plan to worker and scheduler configs."""
    kv_cache_configs = _orig_get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        available_memory,
    )
    if not _is_c128_packed_pool_planner_enabled(vllm_config):
        return kv_cache_configs
    if not kv_cache_configs:
        raise ValueError("packed C128 planner requires at least one KV cache config")

    worker_metadata = [
        _build_c128_packed_pool_metadata(
            vllm_config,
            kv_cache_config,
        )
        for kv_cache_config in kv_cache_configs
    ]
    metadata = worker_metadata[0]
    if any(current != metadata for current in worker_metadata[1:]):
        raise ValueError(
            "packed C128 planner requires identical worker group schemas, " "ranges, and final block counts"
        )
    for kv_cache_config, current in zip(kv_cache_configs, worker_metadata):
        setattr(kv_cache_config, C128_PACKED_POOL_METADATA_KEY, current)

    # This mutation is EngineCore-local because worker processes have already
    # spawned. The worker ABI is the metadata attribute on the KVCacheConfig
    # passed by initialize_from_config's RPC.
    additional_config = vllm_config.additional_config
    assert additional_config is not None
    additional_config[C128_PACKED_POOL_METADATA_KEY] = metadata
    return kv_cache_configs


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        return scheduler_block_size, scheduler_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    # C128 attention and compressor-state layers can land in the same
    # page-size bucket.  The normal allocator intentionally aliases a raw
    # tensor across such layers.  Canonical C128 ownership needs an independent
    # raw allocation while preserving the state cache, so split only this
    # feature-gated family before the worker performs its owner-shard reshape.
    additional_config = vllm_config.additional_config or {}
    enable_c128_owner_shard = bool(additional_config.get("enable_c128_owner_shard", False))
    c128_attention_layers = {
        layer_name
        for group in kv_cache_groups
        for layer_name, layer_spec in group.kv_cache_spec.kv_cache_specs.items()
        if isinstance(layer_spec, MLAAttentionSpec) and layer_spec.compress_ratio == 128
    }

    def _append_tensor(page_size: int, names: list[str]) -> None:
        if names:
            kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=names))

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            if enable_c128_owner_shard:
                c128_layers = [name for name in shared_by if name in c128_attention_layers]
                non_c128_layers = [name for name in shared_by if name not in c128_layers]
                _append_tensor(ps, non_c128_layers)
                _append_tensor(ps, c128_layers)
            else:
                _append_tensor(ps, shared_by)
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = _ascend_get_kv_cache_configs

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.get_kv_cache_configs = _ascend_get_kv_cache_configs
