# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Fail-closed worker lifecycle for the packed C128 arena prototype.

This module owns no CANN or torch_npu adapter.  It validates the serialized
fixed-Flash plan, rebuilds the pure :class:`PackedPoolPlan`, opens the existing
``PackedArenaLease`` with injected adapters, and keeps that lease alive until
all registered model-runner views have been quiesced and released.

The production model runner must not call this module until the planner marks
the downstream runtime ABI ready.  Keeping the feature-off return before any
metadata or adapter access makes the current worker path inert.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm_ascend.attention.context_parallel.c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
    PackedArenaBackend,
    PackedArenaLease,
    PackedArenaTensorFactory,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    BucketPhysicalBytes,
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
    PackedPoolScratchSpec,
)

C128_PACKED_POOL_METADATA_KEY = "c128_packed_pool_metadata"
C128_PACKED_POOL_SCHEMA_VERSION = 1
C128_PACKED_POOL_PROFILE = "dsv4_flash_prefill_8200_tokens_1out"
C128_PACKED_POOL_PROMPT_TOKENS = 8_200
C128_PACKED_POOL_OUTPUT_TOKENS = 1
C128_PACKED_POOL_MAX_MODEL_LEN = 8_201
C128_PACKED_POOL_MAX_NUM_BATCHED_TOKENS = 5_120
C128_PACKED_POOL_TP_SIZE = 8
C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS = (17, 1, 42, 42, 642, 165)
C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS = (17, 3_281, 42, 42, 642, 165)
C128_PACKED_POOL_GROUP_IDENTITIES = (
    "c4_attention",
    "c128_attention",
    "dense_swa_a",
    "dense_swa_b",
    "c4_state",
    "c128_state",
)
C128_PACKED_POOL_COMPONENT_SIGNATURES = (
    (
        ("page_16640", 16_640, 21, PackedPlacement.REPLICATED),
        ("page_131072", 131_072, 21, PackedPlacement.REPLICATED),
    ),
    (("page_131072", 131_072, 20, PackedPlacement.C128_OWNER),),
    (("page_131072", 131_072, 22, PackedPlacement.REPLICATED),),
    (("page_131072", 131_072, 21, PackedPlacement.REPLICATED),),
    (
        ("page_16640", 16_640, 21, PackedPlacement.REPLICATED),
        ("page_131072", 131_072, 21, PackedPlacement.REPLICATED),
    ),
    (("page_131072", 131_072, 20, PackedPlacement.REPLICATED),),
)
C128_PACKED_POOL_GLOBAL_BLOCK_CAPACITY = 4_190
C128_PACKED_POOL_PERSISTENT_VIEW_COUNT = 167
C128_PACKED_POOL_SCRATCH_VIEW_COUNT = 1
C128_PACKED_POOL_SCRATCH_PAGES_PER_RANK = 65
C128_PACKED_POOL_NARROW_PERSISTENT_BYTES_PER_RANK = 308_281_344
C128_PACKED_POOL_WIDE_PERSISTENT_BYTES_PER_RANK = 3_716_153_344
C128_PACKED_POOL_SCRATCH_BYTES_PER_RANK = 10_485_760
C128_PACKED_POOL_TOTAL_BYTES_PER_RANK = 4_034_920_448


class PackedArenaMetadataError(ValueError):
    """Serialized packed-plan metadata is absent, malformed, or inconsistent."""


class PackedArenaRuntimeState(str, Enum):
    OPEN = "open"
    SEALED = "sealed"
    PUBLISHED = "published"
    CLOSING = "closing"
    VIEWS_RELEASED = "views_released"
    CLEANUP_FAILED = "cleanup_failed"
    CLOSED = "closed"


class PackedArenaRuntimeClosedError(RuntimeError):
    """A caller tried to borrow or register a view after teardown began."""


class PackedArenaRuntimeCleanupError(RuntimeError):
    """Quiescence, view release, or lease teardown failed."""

    def __init__(self, operation: str, cause: BaseException) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(f"packed-arena runtime {operation} failed: {cause}")


@dataclass(frozen=True)
class PackedArenaBucketAccounting:
    bucket: str
    page_size_bytes: int
    persistent_allocated_bytes: int
    scratch_region_bytes: int
    total_allocated_bytes: int


@dataclass(frozen=True)
class PackedArenaRankAccounting:
    """Exact plan-derived allocation bytes for one worker rank."""

    tp_rank: int
    buckets: tuple[PackedArenaBucketAccounting, ...]
    persistent_allocated_bytes: int
    scratch_region_bytes: int
    total_allocated_bytes: int

    def as_metadata(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "persistent_allocated_bytes": self.persistent_allocated_bytes,
            "scratch_region_bytes": self.scratch_region_bytes,
            "total_allocated_bytes": self.total_allocated_bytes,
            "buckets": [
                {
                    "bucket": bucket.bucket,
                    "page_size_bytes": bucket.page_size_bytes,
                    "persistent_allocated_bytes": (bucket.persistent_allocated_bytes),
                    "scratch_region_bytes": bucket.scratch_region_bytes,
                    "total_allocated_bytes": bucket.total_allocated_bytes,
                }
                for bucket in self.buckets
            ],
        }


@dataclass(frozen=True)
class PackedArenaExpectedView:
    """One required component-copy or scratch view in the runtime ABI."""

    key: str
    bucket: str


@dataclass(frozen=True)
class PackedArenaRuntimeContract:
    """Validated worker ABI reconstructed from JSON-safe planner metadata."""

    plan: PackedPoolPlan
    rank_accounting: tuple[PackedArenaRankAccounting, ...]
    expected_views: tuple[PackedArenaExpectedView, ...]
    metadata_fingerprint: str

    def accounting_for_rank(self, tp_rank: int) -> PackedArenaRankAccounting:
        if not 0 <= tp_rank < len(self.rank_accounting):
            raise ValueError(f"tp_rank {tp_rank} is outside [0, " f"{len(self.rank_accounting)})")
        return self.rank_accounting[tp_rank]


def validate_c128_packed_startup_contract(
    contract: PackedArenaRuntimeContract,
    metadata: Mapping[str, object],
) -> PackedArenaRuntimeContract:
    """Require the one fixed same-capacity worker activation contract.

    The generic metadata parser is also used by synthetic/unit plans.  The
    production startup path is deliberately narrower: no VMM or Torch-NPU
    adapter may be opened unless the scheduler quotas, complete component
    manifest, scratch window, and exact rank allocation all match the
    reviewed DSV4-Flash TP8 profile.
    """
    plan = contract.plan
    if plan.global_block_capacity != C128_PACKED_POOL_GLOBAL_BLOCK_CAPACITY:
        raise PackedArenaMetadataError(
            "global_block_capacity: production packed runtime requires "
            f"{C128_PACKED_POOL_GLOBAL_BLOCK_CAPACITY}, got "
            f"{plan.global_block_capacity}"
        )
    if metadata.get("expert_parallel_size") != C128_PACKED_POOL_TP_SIZE:
        raise PackedArenaMetadataError("expert_parallel_size: production packed runtime requires EP8")
    required_quotas = metadata.get("required_group_block_quotas")
    if (
        not isinstance(required_quotas, (list, tuple))
        or tuple(required_quotas) != C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS
    ):
        raise PackedArenaMetadataError(
            "required_group_block_quotas: production packed runtime requires "
            f"{C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS!r}"
        )
    serialized_assigned_quotas = metadata.get("assigned_group_block_quotas")
    if (
        not isinstance(serialized_assigned_quotas, (list, tuple))
        or tuple(serialized_assigned_quotas) != C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS
    ):
        raise PackedArenaMetadataError(
            "assigned_group_block_quotas: production packed runtime requires "
            f"{C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS!r}"
        )

    assigned_quotas = tuple(group.logical_blocks for group in plan.groups)
    if assigned_quotas != C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS:
        raise PackedArenaMetadataError(
            "groups.logical_blocks: production packed runtime requires "
            f"{C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS!r}, got "
            f"{assigned_quotas!r}"
        )
    serialized_groups = metadata.get("groups")
    if not isinstance(serialized_groups, (list, tuple)):
        raise PackedArenaMetadataError("groups: production packed runtime requires six groups")
    if len(serialized_groups) != len(C128_PACKED_POOL_GROUP_IDENTITIES):
        raise PackedArenaMetadataError("groups: production packed runtime requires exactly six groups")
    expected_scheduler_identities = []
    for group_index, (
        raw_group,
        identity,
        required_blocks,
        assigned_blocks,
    ) in enumerate(
        zip(
            serialized_groups,
            C128_PACKED_POOL_GROUP_IDENTITIES,
            C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS,
            C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
        )
    ):
        if not isinstance(raw_group, Mapping):
            raise PackedArenaMetadataError(f"groups[{group_index}]: expected object")
        expected_group = {
            "group_index": group_index,
            "name": f"group_{group_index}",
            "identity": identity,
            "required_logical_blocks": required_blocks,
            "assigned_logical_blocks": assigned_blocks,
        }
        for field, expected_value in expected_group.items():
            if raw_group.get(field) != expected_value:
                raise PackedArenaMetadataError(
                    f"groups[{group_index}].{field}: expected " f"{expected_value!r}, got {raw_group.get(field)!r}"
                )
        expected_scheduler_identities.append(
            {
                "group_index": group_index,
                "group_name": f"group_{group_index}",
                "identity": identity,
                "required_blocks": required_blocks,
                "assigned_blocks": assigned_blocks,
            }
        )
    if metadata.get("scheduler_group_identities") != expected_scheduler_identities:
        raise PackedArenaMetadataError(
            "scheduler_group_identities: production packed runtime group " "identity/order contract changed"
        )

    owner_component_groups = {
        group_index
        for group_index, group in enumerate(plan.groups)
        for component in group.components
        if component.placement is PackedPlacement.C128_OWNER
    }
    if owner_component_groups != {1}:
        raise PackedArenaMetadataError(
            "groups.components.placement: only group_1 C128 attention may " "use c128_owner placement"
        )
    component_signatures = tuple(
        tuple(
            (
                component.bucket,
                component.page_size_bytes,
                component.copies,
                component.placement,
            )
            for component in group.components
        )
        for group in plan.groups
    )
    if component_signatures != C128_PACKED_POOL_COMPONENT_SIGNATURES:
        raise PackedArenaMetadataError("groups.components: production packed runtime component " "signature changed")

    persistent_views = tuple(view for view in contract.expected_views if view.key.startswith("component/"))
    scratch_views = tuple(view for view in contract.expected_views if view.key.startswith("scratch/"))
    if len(persistent_views) != C128_PACKED_POOL_PERSISTENT_VIEW_COUNT:
        raise PackedArenaMetadataError(
            "expected_views: production packed runtime requires exactly "
            f"{C128_PACKED_POOL_PERSISTENT_VIEW_COUNT} persistent component "
            f"views, got {len(persistent_views)}"
        )
    if len(scratch_views) != C128_PACKED_POOL_SCRATCH_VIEW_COUNT:
        raise PackedArenaMetadataError(
            "expected_views: production packed runtime requires exactly "
            f"{C128_PACKED_POOL_SCRATCH_VIEW_COUNT} scratch view, got "
            f"{len(scratch_views)}"
        )
    if len(persistent_views) + len(scratch_views) != len(contract.expected_views):
        raise PackedArenaMetadataError("expected_views: contains an unsupported view kind")

    active_scratch = tuple(scratch for scratch in plan.scratch if scratch.max_pages_per_rank)
    if len(active_scratch) != C128_PACKED_POOL_SCRATCH_VIEW_COUNT:
        raise PackedArenaMetadataError(
            "scratch: production packed runtime requires exactly one active " "scratch region"
        )
    scratch = active_scratch[0]
    if scratch.max_pages_per_rank != C128_PACKED_POOL_SCRATCH_PAGES_PER_RANK:
        raise PackedArenaMetadataError(
            "scratch.max_pages_per_rank: production packed runtime requires "
            f"{C128_PACKED_POOL_SCRATCH_PAGES_PER_RANK}, got "
            f"{scratch.max_pages_per_rank}"
        )
    if scratch.bucket != "page_131072":
        raise PackedArenaMetadataError("scratch.bucket: production packed runtime requires " "'page_131072'")
    if scratch_views[0].key != f"scratch/{scratch.bucket}":
        raise PackedArenaMetadataError(
            "expected_views: scratch view does not match the active scratch " f"bucket {scratch.bucket!r}"
        )

    if len(contract.rank_accounting) != C128_PACKED_POOL_TP_SIZE:
        raise PackedArenaMetadataError(
            "rank_accounting: production packed runtime requires exactly " f"{C128_PACKED_POOL_TP_SIZE} ranks"
        )
    rank_bytes = tuple(accounting.total_allocated_bytes for accounting in contract.rank_accounting)
    expected_rank_bytes = (C128_PACKED_POOL_TOTAL_BYTES_PER_RANK,) * C128_PACKED_POOL_TP_SIZE
    if rank_bytes != expected_rank_bytes:
        raise PackedArenaMetadataError(
            "rank_accounting.total_allocated_bytes: production packed "
            f"runtime requires {expected_rank_bytes!r}, got {rank_bytes!r}"
        )
    for accounting in contract.rank_accounting:
        buckets = {bucket.bucket: bucket for bucket in accounting.buckets}
        if set(buckets) != {"page_16640", "page_131072"}:
            raise PackedArenaMetadataError(
                "rank_accounting.buckets: production packed runtime requires " "exactly page_16640 and page_131072"
            )
        narrow = buckets["page_16640"]
        wide = buckets["page_131072"]
        if (
            narrow.persistent_allocated_bytes != C128_PACKED_POOL_NARROW_PERSISTENT_BYTES_PER_RANK
            or narrow.scratch_region_bytes != 0
            or narrow.total_allocated_bytes != C128_PACKED_POOL_NARROW_PERSISTENT_BYTES_PER_RANK
        ):
            raise PackedArenaMetadataError(
                "rank_accounting.buckets[narrow]: allocation does not match " "the production packed runtime"
            )
        if (
            wide.persistent_allocated_bytes != C128_PACKED_POOL_WIDE_PERSISTENT_BYTES_PER_RANK
            or wide.scratch_region_bytes != C128_PACKED_POOL_SCRATCH_BYTES_PER_RANK
            or wide.total_allocated_bytes
            != (C128_PACKED_POOL_WIDE_PERSISTENT_BYTES_PER_RANK + C128_PACKED_POOL_SCRATCH_BYTES_PER_RANK)
        ):
            raise PackedArenaMetadataError(
                "rank_accounting.buckets[wide]: allocation does not match " "the production packed runtime"
            )
    return contract


@dataclass(frozen=True)
class _PackedArenaViewOwner:
    """Runtime-owned pin and failure-atomic view cleanup callback."""

    token: int
    bucket: str
    release: Callable[[], None]


def _metadata_error(path: str, detail: str) -> PackedArenaMetadataError:
    return PackedArenaMetadataError(f"{path}: {detail}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _metadata_error(path, f"expected object, got {type(value).__name__}")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise _metadata_error(path, f"expected array, got {type(value).__name__}")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise _metadata_error(path, f"expected integer, got {value!r}")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise _metadata_error(path, f"expected boolean, got {value!r}")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _metadata_error(path, f"expected string, got {value!r}")
    return value


def _integers(value: object, path: str) -> tuple[int, ...]:
    return tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path)))


def _strings(value: object, path: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path)))


def _require_equal(actual: object, expected: object, path: str) -> None:
    integer_type_mismatch = type(expected) is int and type(actual) is not int
    if integer_type_mismatch or actual != expected:
        raise _metadata_error(path, f"expected {expected!r}, got {actual!r}")


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _metadata_fingerprint(metadata: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise _metadata_error(
            "metadata",
            f"must be JSON serializable: {error}",
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _validate_fixed_profile(metadata: Mapping[str, object]) -> None:
    expected = {
        "schema_version": C128_PACKED_POOL_SCHEMA_VERSION,
        "profile": C128_PACKED_POOL_PROFILE,
        "prompt_tokens": C128_PACKED_POOL_PROMPT_TOKENS,
        "output_tokens": C128_PACKED_POOL_OUTPUT_TOKENS,
        "kv_slot_tokens": C128_PACKED_POOL_PROMPT_TOKENS,
        "max_model_len": C128_PACKED_POOL_MAX_MODEL_LEN,
        "max_concurrent_requests": 1,
        "max_num_batched_tokens": C128_PACKED_POOL_MAX_NUM_BATCHED_TOKENS,
        "max_num_scheduled_tokens": C128_PACKED_POOL_MAX_NUM_BATCHED_TOKENS,
        "max_num_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "tp_size": C128_PACKED_POOL_TP_SIZE,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "sentinel_block_id": 0,
    }
    for key, expected_value in expected.items():
        _require_equal(metadata.get(key), expected_value, key)


def _build_groups(
    serialized_groups: Sequence[object],
) -> tuple[PackedPoolGroupSpec, ...]:
    groups: list[PackedPoolGroupSpec] = []
    for group_index, raw_group in enumerate(serialized_groups):
        path = f"groups[{group_index}]"
        group = _mapping(raw_group, path)
        _require_equal(group.get("group_index"), group_index, f"{path}.group_index")
        name = _string(group.get("name"), f"{path}.name")
        _require_equal(name, f"group_{group_index}", f"{path}.name")
        logical_blocks = _integer(
            group.get("logical_blocks"),
            f"{path}.logical_blocks",
        )

        scheduler_shape = _mapping(
            group.get("scheduler_shape"),
            f"{path}.scheduler_shape",
        )
        _require_equal(
            scheduler_shape.get("partition_blocks"),
            logical_blocks,
            f"{path}.scheduler_shape.partition_blocks",
        )

        group_layer_names = _strings(
            group.get("layer_names"),
            f"{path}.layer_names",
        )
        if len(set(group_layer_names)) != len(group_layer_names):
            raise _metadata_error(f"{path}.layer_names", "contains duplicates")

        components: list[PackedPoolComponentSpec] = []
        component_layer_names: list[str] = []
        raw_components = _sequence(
            group.get("components"),
            f"{path}.components",
        )
        if not raw_components:
            raise _metadata_error(f"{path}.components", "must not be empty")
        for component_index, raw_component in enumerate(raw_components):
            component_path = f"{path}.components[{component_index}]"
            component = _mapping(raw_component, component_path)
            name_value = _string(component.get("name"), f"{component_path}.name")
            bucket = _string(component.get("bucket"), f"{component_path}.bucket")
            page_size_bytes = _integer(
                component.get("page_size_bytes"),
                f"{component_path}.page_size_bytes",
            )
            copies = _integer(
                component.get("copies"),
                f"{component_path}.copies",
            )
            placement_value = _string(
                component.get("placement"),
                f"{component_path}.placement",
            )
            try:
                placement = PackedPlacement(placement_value)
            except ValueError as error:
                raise _metadata_error(
                    f"{component_path}.placement",
                    f"unsupported placement {placement_value!r}",
                ) from error
            granularity = _integer(
                component.get("allocation_granularity_bytes"),
                f"{component_path}.allocation_granularity_bytes",
            )
            _require_equal(
                granularity,
                CANN_VMM_GRANULARITY_BYTES,
                f"{component_path}.allocation_granularity_bytes",
            )
            layer_names = _strings(
                component.get("layer_names"),
                f"{component_path}.layer_names",
            )
            _require_equal(
                len(layer_names),
                copies,
                f"{component_path}.layer_names",
            )
            component_layer_names.extend(layer_names)
            components.append(
                PackedPoolComponentSpec(
                    name=name_value,
                    bucket=bucket,
                    page_size_bytes=page_size_bytes,
                    copies=copies,
                    placement=placement,
                    allocation_granularity_bytes=granularity,
                )
            )

        if len(set(component_layer_names)) != len(component_layer_names) or set(component_layer_names) != set(
            group_layer_names
        ):
            raise _metadata_error(
                f"{path}.components",
                "component layer_names must partition group layer_names",
            )
        groups.append(
            PackedPoolGroupSpec(
                name=name,
                logical_blocks=logical_blocks,
                components=tuple(components),
            )
        )
    if not groups:
        raise _metadata_error("groups", "must not be empty")
    return tuple(groups)


def _build_scratch(
    serialized_scratch: Sequence[object],
) -> tuple[PackedPoolScratchSpec, ...]:
    scratch: list[PackedPoolScratchSpec] = []
    for index, raw_scratch in enumerate(serialized_scratch):
        path = f"scratch[{index}]"
        item = _mapping(raw_scratch, path)
        granularity = _integer(
            item.get("allocation_granularity_bytes"),
            f"{path}.allocation_granularity_bytes",
        )
        _require_equal(
            granularity,
            CANN_VMM_GRANULARITY_BYTES,
            f"{path}.allocation_granularity_bytes",
        )
        scratch.append(
            PackedPoolScratchSpec(
                bucket=_string(item.get("bucket"), f"{path}.bucket"),
                page_size_bytes=_integer(
                    item.get("page_size_bytes"),
                    f"{path}.page_size_bytes",
                ),
                max_pages_per_rank=_integer(
                    item.get("max_pages_per_rank"),
                    f"{path}.max_pages_per_rank",
                ),
                allocation_granularity_bytes=granularity,
            )
        )
    return tuple(scratch)


def _validate_ranges_and_segments(
    plan: PackedPoolPlan,
    serialized_groups: Sequence[object],
) -> None:
    for group_index, (group_spec, raw_group) in enumerate(zip(plan.groups, serialized_groups)):
        path = f"groups[{group_index}]"
        group = _mapping(raw_group, path)
        logical_range = plan.group_range(group_spec.name)
        _require_equal(
            group.get("logical_start"),
            logical_range.start,
            f"{path}.logical_start",
        )
        _require_equal(
            group.get("logical_stop"),
            logical_range.stop,
            f"{path}.logical_stop",
        )
        raw_components = _sequence(
            group.get("components"),
            f"{path}.components",
        )
        for component_index, (component_spec, raw_component) in enumerate(zip(group_spec.components, raw_components)):
            component_path = f"{path}.components[{component_index}]"
            component = _mapping(raw_component, component_path)
            raw_copies = _sequence(
                component.get("segments"),
                f"{component_path}.segments",
            )
            _require_equal(
                len(raw_copies),
                component_spec.copies,
                f"{component_path}.segments",
            )
            for copy_index, raw_copy in enumerate(raw_copies):
                copy_path = f"{component_path}.segments[{copy_index}]"
                copy = _mapping(raw_copy, copy_path)
                _require_equal(
                    copy.get("copy_index"),
                    copy_index,
                    f"{copy_path}.copy_index",
                )
                raw_ranks = _sequence(copy.get("ranks"), f"{copy_path}.ranks")
                _require_equal(
                    len(raw_ranks),
                    plan.tp_size,
                    f"{copy_path}.ranks",
                )
                for rank, raw_rank in enumerate(raw_ranks):
                    rank_path = f"{copy_path}.ranks[{rank}]"
                    rank_segment = _mapping(raw_rank, rank_path)
                    sentinel = plan.sentinel_address(
                        group_spec.name,
                        component_spec.name,
                        tp_rank=rank,
                        copy_index=copy_index,
                    )
                    expected = {
                        "rank": rank,
                        "segment_base_bytes": sentinel.segment_base_bytes,
                        "segment_allocated_bytes": (sentinel.segment_allocated_bytes),
                        "sentinel_offset_bytes": (sentinel.physical_offset_bytes),
                    }
                    for key, expected_value in expected.items():
                        _require_equal(
                            rank_segment.get(key),
                            expected_value,
                            f"{rank_path}.{key}",
                        )


def _validate_bucket(
    accounting: BucketPhysicalBytes,
    raw_bucket: object,
    index: int,
) -> None:
    path = f"buckets[{index}]"
    bucket = _mapping(raw_bucket, path)
    expected_scalars = {
        "bucket": accounting.bucket,
        "page_size_bytes": accounting.page_size_bytes,
        "replicated_pages_per_rank": accounting.replicated_pages_per_rank,
        "sentinel_pages_per_rank": accounting.sentinel_pages_per_rank,
        "scratch_pages_per_rank": accounting.scratch_pages_per_rank,
    }
    for key, expected_value in expected_scalars.items():
        _require_equal(bucket.get(key), expected_value, f"{path}.{key}")
    expected_vectors = {
        "owner_pages_by_rank": accounting.owner_pages_by_rank,
        "persistent_allocated_bytes_by_rank": (accounting.persistent_allocated_bytes_by_rank),
        "scratch_region_bytes_by_rank": accounting.scratch_region_bytes_by_rank,
        "total_allocated_bytes_by_rank": accounting.total_allocated_bytes_by_rank,
    }
    for key, expected_value in expected_vectors.items():
        _require_equal(
            _integers(bucket.get(key), f"{path}.{key}"),
            expected_value,
            f"{path}.{key}",
        )


def _validate_scratch_segments(
    plan: PackedPoolPlan,
    serialized_scratch: Sequence[object],
) -> None:
    accounting_by_bucket = {accounting.bucket: accounting for accounting in plan.bucket_accounting}
    for index, (scratch_spec, raw_scratch) in enumerate(zip(plan.scratch, serialized_scratch)):
        path = f"scratch[{index}]"
        item = _mapping(raw_scratch, path)
        accounting = accounting_by_bucket[scratch_spec.bucket]
        scratch_allocated_bytes = _round_up(
            scratch_spec.max_pages_per_rank * scratch_spec.page_size_bytes,
            CANN_VMM_GRANULARITY_BYTES,
        )
        segments = _sequence(item.get("segments"), f"{path}.segments")
        _require_equal(len(segments), plan.tp_size, f"{path}.segments")
        for rank, raw_segment in enumerate(segments):
            segment_path = f"{path}.segments[{rank}]"
            segment = _mapping(raw_segment, segment_path)
            expected = {
                "rank": rank,
                "segment_base_bytes": (accounting.total_allocated_bytes_by_rank[rank] - scratch_allocated_bytes),
                "segment_allocated_bytes": scratch_allocated_bytes,
            }
            for key, expected_value in expected.items():
                _require_equal(
                    segment.get(key),
                    expected_value,
                    f"{segment_path}.{key}",
                )


def _rank_accounting(
    plan: PackedPoolPlan,
) -> tuple[PackedArenaRankAccounting, ...]:
    ranks = []
    for rank in range(plan.tp_size):
        buckets = tuple(
            PackedArenaBucketAccounting(
                bucket=accounting.bucket,
                page_size_bytes=accounting.page_size_bytes,
                persistent_allocated_bytes=(accounting.persistent_allocated_bytes_by_rank[rank]),
                scratch_region_bytes=(accounting.scratch_region_bytes_by_rank[rank]),
                total_allocated_bytes=(accounting.total_allocated_bytes_by_rank[rank]),
            )
            for accounting in plan.bucket_accounting
        )
        ranks.append(
            PackedArenaRankAccounting(
                tp_rank=rank,
                buckets=buckets,
                persistent_allocated_bytes=sum(bucket.persistent_allocated_bytes for bucket in buckets),
                scratch_region_bytes=sum(bucket.scratch_region_bytes for bucket in buckets),
                total_allocated_bytes=sum(bucket.total_allocated_bytes for bucket in buckets),
            )
        )
    return tuple(ranks)


def _expected_views(
    plan: PackedPoolPlan,
) -> tuple[PackedArenaExpectedView, ...]:
    views = []
    for group in plan.groups:
        for component in group.components:
            for copy_index in range(component.copies):
                views.append(
                    PackedArenaExpectedView(
                        key=(f"component/{group.name}/" f"{component.name}/{copy_index}"),
                        bucket=component.bucket,
                    )
                )
    for scratch in plan.scratch:
        if scratch.max_pages_per_rank:
            views.append(
                PackedArenaExpectedView(
                    key=f"scratch/{scratch.bucket}",
                    bucket=scratch.bucket,
                )
            )
    return tuple(views)


def packed_arena_contract_from_metadata(
    metadata: Mapping[str, object],
    *,
    require_runtime_ready: bool = True,
) -> PackedArenaRuntimeContract:
    """Rebuild and independently verify one serialized fixed-Flash plan."""
    metadata = _mapping(metadata, "metadata")
    _validate_fixed_profile(metadata)
    planner_only = _boolean(metadata.get("planner_only"), "planner_only")
    runtime_ready = _boolean(
        metadata.get("downstream_runtime_abi_ready"),
        "downstream_runtime_abi_ready",
    )
    if require_runtime_ready and (planner_only or not runtime_ready):
        raise _metadata_error(
            "downstream_runtime_abi_ready",
            "packed runtime requires planner_only=false and " "downstream_runtime_abi_ready=true",
        )

    serialized_groups = _sequence(metadata.get("groups"), "groups")
    serialized_scratch = _sequence(metadata.get("scratch"), "scratch")
    plan = PackedPoolPlan(
        global_block_capacity=_integer(
            metadata.get("global_block_capacity"),
            "global_block_capacity",
        ),
        tp_size=_integer(metadata.get("tp_size"), "tp_size"),
        groups=_build_groups(serialized_groups),
        scratch=_build_scratch(serialized_scratch),
    )

    expected_plan_scalars = {
        "usable_data_capacity": plan.usable_data_capacity,
        "used_logical_blocks": plan.used_logical_blocks,
        "unused_logical_blocks": plan.unused_logical_blocks,
    }
    for key, expected_value in expected_plan_scalars.items():
        _require_equal(metadata.get(key), expected_value, key)
    _validate_ranges_and_segments(plan, serialized_groups)

    serialized_buckets = _sequence(metadata.get("buckets"), "buckets")
    _require_equal(
        len(serialized_buckets),
        len(plan.bucket_accounting),
        "buckets",
    )
    for index, (accounting, raw_bucket) in enumerate(zip(plan.bucket_accounting, serialized_buckets)):
        _validate_bucket(accounting, raw_bucket, index)
    _require_equal(
        len(serialized_scratch),
        len(plan.scratch),
        "scratch",
    )
    _validate_scratch_segments(plan, serialized_scratch)

    _require_equal(
        _integers(
            metadata.get("total_physical_bytes_by_rank"),
            "total_physical_bytes_by_rank",
        ),
        plan.total_physical_bytes_by_rank(),
        "total_physical_bytes_by_rank",
    )
    _require_equal(
        _integers(
            metadata.get("quota_replicated_bytes_by_rank"),
            "quota_replicated_bytes_by_rank",
        ),
        plan.quota_replicated_bytes_by_rank(),
        "quota_replicated_bytes_by_rank",
    )
    _require_equal(
        _integers(
            metadata.get("aligned_quota_replicated_bytes_by_rank"),
            "aligned_quota_replicated_bytes_by_rank",
        ),
        plan.aligned_quota_replicated_bytes_by_rank(),
        "aligned_quota_replicated_bytes_by_rank",
    )
    return PackedArenaRuntimeContract(
        plan=plan,
        rank_accounting=_rank_accounting(plan),
        expected_views=_expected_views(plan),
        metadata_fingerprint=_metadata_fingerprint(metadata),
    )


class PackedArenaRuntime:
    """Own one rank's lease until all model-runner aliases are gone."""

    def __init__(
        self,
        *,
        contract: PackedArenaRuntimeContract,
        lease: PackedArenaLease,
        tp_rank: int,
        quiesce: Callable[[], None],
    ) -> None:
        if lease.plan is not contract.plan:
            raise ValueError("lease must be opened from the validated contract plan")
        if lease.tp_rank != tp_rank:
            raise ValueError(f"lease rank {lease.tp_rank} does not match runtime rank {tp_rank}")
        self.contract = contract
        self._lease: PackedArenaLease | None = lease
        self._peer_lease: Any | None = None
        self.tp_rank = tp_rank
        self.device_index = lease.device_index
        self._quiesce = quiesce
        self._view_owners: list[_PackedArenaViewOwner] = []
        self._expected_view_buckets = {view.key: view.bucket for view in contract.expected_views}
        self._installed_view_keys: set[str] = set()
        self._next_view_token = 0
        self._installing_view = False
        self._quiesced = False
        self._state = PackedArenaRuntimeState.OPEN

    @classmethod
    def open_from_metadata(
        cls,
        *,
        metadata: Mapping[str, object],
        tp_rank: int,
        device_index: int,
        backend: PackedArenaBackend,
        tensor_factory: PackedArenaTensorFactory,
        arena_fence: Callable[[], None],
        quiesce: Callable[[], None],
    ) -> PackedArenaRuntime:
        contract = packed_arena_contract_from_metadata(metadata)
        return cls.open_from_contract(
            contract=contract,
            tp_rank=tp_rank,
            device_index=device_index,
            backend=backend,
            tensor_factory=tensor_factory,
            arena_fence=arena_fence,
            quiesce=quiesce,
        )

    @classmethod
    def open_from_contract(
        cls,
        *,
        contract: PackedArenaRuntimeContract,
        tp_rank: int,
        device_index: int,
        backend: PackedArenaBackend,
        tensor_factory: PackedArenaTensorFactory,
        arena_fence: Callable[[], None],
        quiesce: Callable[[], None],
    ) -> PackedArenaRuntime:
        """Open a lease from an already validated immutable contract."""
        lease = PackedArenaLease.open(
            plan=contract.plan,
            tp_rank=tp_rank,
            device_index=device_index,
            backend=backend,
            tensor_factory=tensor_factory,
            fence=arena_fence,
        )
        return cls(
            contract=contract,
            lease=lease,
            tp_rank=tp_rank,
            quiesce=quiesce,
        )

    @property
    def state(self) -> PackedArenaRuntimeState:
        return self._state

    @property
    def accounting(self) -> PackedArenaRankAccounting:
        return self.contract.accounting_for_rank(self.tp_rank)

    def _require_open(self) -> PackedArenaLease:
        if self._state is not PackedArenaRuntimeState.OPEN or self._lease is None:
            raise PackedArenaRuntimeClosedError(f"packed-arena runtime is {self._state.value}")
        return self._lease

    def open_peer_lease(
        self,
        opener: Callable[[PackedArenaLease], Any],
    ) -> Any:
        """Attach one startup-only peer lease without transferring ownership.

        ``PackedArenaRuntime`` remains the sole owner of the local arena. A
        partially opened peer lease carried by a coordinated-open exception
        is retained so :meth:`close` can retry its cleanup before the local
        allocation is released.
        """
        lease = self._require_open()
        if self._peer_lease is not None:
            raise RuntimeError("packed-arena peer lease is already attached")
        try:
            peer_lease = opener(lease)
        except BaseException as error:
            retained = getattr(error, "lease", None)
            if retained is not None:
                if not callable(getattr(retained, "close", None)):
                    raise TypeError(
                        "retained packed peer lease must be closeable"
                    ) from error
                self._peer_lease = retained
            raise
        if peer_lease is None or not callable(
            getattr(peer_lease, "close", None)
        ):
            raise TypeError(
                "packed-arena peer opener must return a closeable lease"
            )
        self._peer_lease = peer_lease
        return peer_lease

    @property
    def peer_lease(self) -> Any | None:
        return self._peer_lease

    def install_tensor_views(
        self,
        view_key: str,
        install: Callable[[object], Callable[[], None]],
    ) -> None:
        """Install derived views without exposing a caller-owned lease pin.

        ``install`` receives the bucket root and must return a retry-idempotent
        releaser.  The releaser must report success only after it has discarded
        every derived tensor/storage alias.  Like ``PackedArenaTensorFactory``,
        an installer that raises must be failure-atomic.

        The runtime owns the pin and removes it only after the releaser
        succeeds.  Consumers cannot independently unpin a still-live view.
        """
        lease = self._require_open()
        if self._installing_view:
            raise RuntimeError("packed-arena view installation is not reentrant")
        if view_key in self._installed_view_keys:
            raise ValueError(f"packed-arena view {view_key!r} is already installed")
        try:
            bucket = self._expected_view_buckets[view_key]
        except KeyError as error:
            raise ValueError(f"unknown packed-arena view key: {view_key}") from error
        tensor = lease.tensor(bucket)
        token = self._next_view_token
        self._next_view_token += 1
        incomplete_error: BaseException | None = None

        def reject_incomplete_install() -> None:
            raise RuntimeError(
                "packed-arena view installation did not complete " f"failure-atomically: {incomplete_error}"
            )

        self._view_owners.append(
            _PackedArenaViewOwner(
                token=token,
                bucket=bucket,
                release=reject_incomplete_install,
            )
        )
        self._installing_view = True
        try:
            release = install(tensor)
            if not callable(release):
                raise TypeError("packed-arena view installer must return a cleanup callable")
        except BaseException as error:
            incomplete_error = error
            self._state = PackedArenaRuntimeState.CLEANUP_FAILED
            raise
        finally:
            self._installing_view = False
        self._view_owners[-1] = _PackedArenaViewOwner(
            token=token,
            bucket=bucket,
            release=release,
        )
        self._installed_view_keys.add(view_key)

    def install_teardown_dependency(
        self,
        dependency_key: str,
        release: Callable[[], None],
    ) -> None:
        """Register a non-arena dependency released before peer unmapping."""
        self._require_open()
        if not dependency_key:
            raise ValueError("packed-arena dependency key must be non-empty")
        if not callable(release):
            raise TypeError("packed-arena dependency releaser must be callable")
        token = self._next_view_token
        self._next_view_token += 1
        self._view_owners.append(
            _PackedArenaViewOwner(
                token=token,
                bucket=f"dependency:{dependency_key}",
                release=release,
            )
        )

    def seal_views(self) -> None:
        """Prove every plan-required component-copy and scratch view exists."""
        self._require_open()
        missing = set(self._expected_view_buckets) - self._installed_view_keys
        if missing:
            raise RuntimeError("packed-arena runtime cannot be sealed; missing views: " f"{sorted(missing)}")
        self._state = PackedArenaRuntimeState.SEALED

    def publish(self) -> None:
        """Transfer a sealed runtime to its model-runner owner.

        Publication is a distinct lifecycle transition so a runtime cannot be
        exposed while required aliases are still missing.  The caller must
        publish immediately before storing the runtime in its long-lived owner.
        """
        if self._state is not PackedArenaRuntimeState.SEALED:
            raise RuntimeError("packed-arena runtime must be sealed before publication; " f"state={self._state.value}")
        if self._peer_lease is not None and getattr(
            self._peer_lease,
            "consumer_views_committed",
            False,
        ) is not True:
            raise RuntimeError(
                "packed-arena peer consumer views must commit before "
                "publication"
            )
        self._state = PackedArenaRuntimeState.PUBLISHED

    def release_tensor_views(self) -> None:
        """Quiesce work and drop every derived tensor view.

        This split teardown stage lets the production worker release imported
        peer mappings after no materializer/local view can reach them, while
        retaining the local owner allocation until every rank acknowledges
        that its imports are gone.
        """
        if self._state in {
            PackedArenaRuntimeState.VIEWS_RELEASED,
            PackedArenaRuntimeState.CLOSED,
        }:
            return
        self._state = PackedArenaRuntimeState.CLOSING

        if not self._quiesced:
            try:
                self._quiesce()
            except BaseException as error:
                self._state = PackedArenaRuntimeState.CLEANUP_FAILED
                raise PackedArenaRuntimeCleanupError(
                    "quiesce",
                    error,
                ) from error
            self._quiesced = True

        while self._view_owners:
            owner = self._view_owners[-1]
            try:
                owner.release()
            except BaseException as error:
                self._state = PackedArenaRuntimeState.CLEANUP_FAILED
                raise PackedArenaRuntimeCleanupError(
                    f"release_view[{owner.token}:{owner.bucket}]",
                    error,
                ) from error
            self._view_owners.pop()
        self._state = PackedArenaRuntimeState.VIEWS_RELEASED

    def close(self) -> None:
        """Drop views, then close the local owning lease exactly once.

        OPEN and SEALED runtimes may be closed to roll back construction.
        PUBLISHED runtimes follow the same ordered teardown after their
        model-runner owner has stopped making the aliases reachable.
        """
        if self._state is PackedArenaRuntimeState.CLOSED:
            return
        self.release_tensor_views()

        peer_lease = self._peer_lease
        if peer_lease is not None:
            try:
                peer_lease.close()
            except BaseException as error:
                self._state = PackedArenaRuntimeState.CLEANUP_FAILED
                raise PackedArenaRuntimeCleanupError(
                    "close_peer_lease",
                    error,
                ) from error
            self._peer_lease = None

        lease = self._lease
        if lease is None:
            self._state = PackedArenaRuntimeState.CLOSED
            return
        try:
            lease.close()
        except BaseException as error:
            self._state = PackedArenaRuntimeState.CLEANUP_FAILED
            raise PackedArenaRuntimeCleanupError(
                "close_lease",
                error,
            ) from error
        self._lease = None
        self._state = PackedArenaRuntimeState.CLOSED


def maybe_open_packed_arena_runtime(
    *,
    enabled: bool,
    metadata: Mapping[str, object] | None = None,
    tp_rank: int = 0,
    device_index: int = 0,
    backend: PackedArenaBackend | None = None,
    tensor_factory: PackedArenaTensorFactory | None = None,
    arena_fence: Callable[[], None] | None = None,
    quiesce: Callable[[], None] | None = None,
) -> PackedArenaRuntime | None:
    """Open only after the explicit worker feature gate is enabled."""
    if not enabled:
        return None
    if metadata is None:
        raise PackedArenaMetadataError(f"{C128_PACKED_POOL_METADATA_KEY} is required when packed runtime " "is enabled")
    if backend is None:
        raise ValueError("backend is required when packed runtime is enabled")
    if tensor_factory is None:
        raise ValueError("tensor_factory is required when packed runtime is enabled")
    if arena_fence is None:
        raise ValueError("arena_fence is required when packed runtime is enabled")
    if quiesce is None:
        raise ValueError("quiesce is required when packed runtime is enabled")
    return PackedArenaRuntime.open_from_metadata(
        metadata=metadata,
        tp_rank=tp_rank,
        device_index=device_index,
        backend=backend,
        tensor_factory=tensor_factory,
        arena_fence=arena_fence,
        quiesce=quiesce,
    )
