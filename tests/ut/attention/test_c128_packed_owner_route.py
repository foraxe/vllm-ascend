# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only contract tests for packed C128 owner routing."""

from copy import deepcopy

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_owner_route import (
    C128PackedOwnerRoute,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    SentinelBlockError,
)

pytestmark = pytest.mark.cpu_test

TP_SIZE = 4
PAGE_SIZE_BYTES = 128
GRANULARITY_BYTES = 1_024


def _rank_segments(base_bytes: int) -> list[dict[str, int]]:
    return [
        {
            "rank": rank,
            "segment_base_bytes": base_bytes,
            "segment_allocated_bytes": GRANULARITY_BYTES,
            "sentinel_offset_bytes": base_bytes,
        }
        for rank in range(TP_SIZE)
    ]


def _metadata(
    *,
    runtime_ready: bool = False,
) -> dict:
    return {
        "schema_version": 1,
        "planner_only": not runtime_ready,
        "downstream_runtime_abi_ready": runtime_ready,
        "tp_size": TP_SIZE,
        "sentinel_block_id": 0,
        "global_block_capacity": 14,
        "usable_data_capacity": 13,
        "used_logical_blocks": 10,
        "groups": [
            {
                "group_index": 0,
                "name": "group_0",
                "logical_blocks": 3,
                "logical_start": 1,
                "logical_stop": 4,
                "layer_names": ["c4.0"],
                "components": [
                    {
                        "name": "group_0_component_0",
                        "bucket": "page_64",
                        "page_size_bytes": 64,
                        "copies": 1,
                        "placement": "replicated",
                        "allocation_granularity_bytes": (GRANULARITY_BYTES),
                        "layer_names": ["c4.0"],
                        "segments": [
                            {
                                "copy_index": 0,
                                "ranks": _rank_segments(0),
                            }
                        ],
                    }
                ],
            },
            {
                "group_index": 1,
                "name": "group_1",
                "logical_blocks": 7,
                "logical_start": 4,
                "logical_stop": 11,
                "layer_names": ["c128.0", "c128.1"],
                "components": [
                    {
                        "name": "group_1_component_0",
                        "bucket": "page_128",
                        "page_size_bytes": PAGE_SIZE_BYTES,
                        "copies": 2,
                        "placement": "c128_owner",
                        "allocation_granularity_bytes": (GRANULARITY_BYTES),
                        "layer_names": ["c128.0", "c128.1"],
                        "segments": [
                            {
                                "copy_index": 0,
                                "ranks": _rank_segments(0),
                            },
                            {
                                "copy_index": 1,
                                "ranks": _rank_segments(GRANULARITY_BYTES),
                            },
                        ],
                    }
                ],
            },
        ],
        "buckets": [
            {
                "bucket": "page_128",
                "page_size_bytes": PAGE_SIZE_BYTES,
                "persistent_allocated_bytes_by_rank": [2 * GRANULARITY_BYTES] * TP_SIZE,
                "scratch_region_bytes_by_rank": [GRANULARITY_BYTES] * TP_SIZE,
                "total_allocated_bytes_by_rank": [3 * GRANULARITY_BYTES] * TP_SIZE,
            }
        ],
        "scratch": [
            {
                "bucket": "page_128",
                "page_size_bytes": PAGE_SIZE_BYTES,
                "max_pages_per_rank": 7,
                "allocation_granularity_bytes": GRANULARITY_BYTES,
                "segments": [
                    {
                        "rank": rank,
                        "segment_base_bytes": 2 * GRANULARITY_BYTES,
                        "segment_allocated_bytes": GRANULARITY_BYTES,
                    }
                    for rank in range(TP_SIZE)
                ],
            }
        ],
    }


def _route(
    *,
    runtime_ready: bool = False,
    allow_planner_only: bool = True,
    copy_index: int = 0,
    layer_name: str = "c128.0",
) -> C128PackedOwnerRoute:
    return C128PackedOwnerRoute.from_serialized_plan(
        _metadata(runtime_ready=runtime_ready),
        group_index=1,
        group_name="group_1",
        component_name="group_1_component_0",
        layer_name=layer_name,
        copy_index=copy_index,
        allow_planner_only=allow_planner_only,
    )


def test_planner_only_metadata_cannot_attach_to_runtime_by_default() -> None:
    with pytest.raises(RuntimeError, match="planner-only"):
        _route(allow_planner_only=False)

    reference_route = _route()
    with pytest.raises(RuntimeError, match="runtime tensors"):
        reference_route.assert_runtime_ready()


def test_every_global_block_maps_to_exact_owner_and_dense_local_slot() -> None:
    route = _route()
    addresses = [route.map_global_block(block_id) for block_id in range(route.logical_start, route.logical_stop)]

    assert [address.owner_rank for address in addresses] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
    ]
    assert [address.owner_local_slot for address in addresses] == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    assert [route.owner_page_count(rank) for rank in range(TP_SIZE)] == [
        2,
        2,
        2,
        1,
    ]
    for address in addresses:
        assert address.group_block_id == address.global_block_id - 3
        assert address.physical_offset_bytes == address.segment_base_bytes + address.owner_local_slot * PAGE_SIZE_BYTES


def test_copy_index_binds_exact_layer_and_disjoint_segment() -> None:
    first = _route()
    second = _route(copy_index=1, layer_name="c128.1")
    assert first.map_global_block(4).physical_offset_bytes == PAGE_SIZE_BYTES
    assert second.map_global_block(4).physical_offset_bytes == GRANULARITY_BYTES + PAGE_SIZE_BYTES

    with pytest.raises(ValueError, match="layer/copy mismatch"):
        _route(copy_index=1, layer_name="c128.0")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.__setitem__("schema_version", 2),
            "schema_version",
        ),
        (
            lambda metadata: metadata.__setitem__("sentinel_block_id", -1),
            "sentinel block 0",
        ),
        (
            lambda metadata: metadata["groups"][1].__setitem__(
                "logical_stop",
                12,
            ),
            "inconsistent logical range",
        ),
        (
            lambda metadata: metadata["groups"][1]["components"][0].__setitem__(
                "placement",
                "replicated",
            ),
            "not c128_owner",
        ),
        (
            lambda metadata: metadata["groups"][1]["components"][0]["segments"][0].__setitem__(
                "ranks",
                _rank_segments(1),
            ),
            "segment is not aligned",
        ),
        (
            lambda metadata: metadata["scratch"][0].__setitem__(
                "max_pages_per_rank",
                9,
            ),
            "scratch segment is too small",
        ),
        (
            lambda metadata: metadata["scratch"][0]["segments"][0].__setitem__(
                "segment_base_bytes",
                GRANULARITY_BYTES,
            ),
            "scratch does not match bucket accounting",
        ),
        (
            lambda metadata: metadata["groups"][1]["components"][0]["segments"][1].__setitem__(
                "ranks",
                _rank_segments(0),
            ),
            "persistent segments overlap",
        ),
    ],
)
def test_serialized_contract_corruption_fails_closed(
    mutation,
    message: str,
) -> None:
    metadata = _metadata()
    mutation(metadata)
    with pytest.raises((ValueError, RuntimeError), match=message):
        C128PackedOwnerRoute.from_serialized_plan(
            metadata,
            group_index=1,
            group_name="group_1",
            component_name="group_1_component_0",
            layer_name="c128.0",
            copy_index=0,
            allow_planner_only=True,
        )


def test_group_index_name_and_component_must_match_same_plan_entry() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="group identity mismatch"):
        C128PackedOwnerRoute.from_serialized_plan(
            metadata,
            group_index=1,
            group_name="group_0",
            component_name="group_1_component_0",
            layer_name="c128.0",
            copy_index=0,
            allow_planner_only=True,
        )
    with pytest.raises(ValueError, match="expected one component"):
        C128PackedOwnerRoute.from_serialized_plan(
            metadata,
            group_index=1,
            group_name="group_1",
            component_name="missing",
            layer_name="c128.0",
            copy_index=0,
            allow_planner_only=True,
        )


def test_sentinel_foreign_group_and_hybrid_routes_fail_closed() -> None:
    route = _route()
    with pytest.raises(SentinelBlockError, match="sentinel"):
        route.map_global_block(0)
    with pytest.raises(ValueError, match=r"group_1 range \[4, 11\)"):
        route.map_global_block(3)
    with pytest.raises(ValueError, match="hybrid block expansion"):
        route.validate_no_hybrid_expansion(
            physical_block_size=128,
            logical_block_size=64,
            blocks_per_physical_block=2,
        )
    route.validate_no_hybrid_expansion(
        physical_block_size=128,
        logical_block_size=128,
        blocks_per_physical_block=1,
    )
    assert route.sentinel_offset_bytes(tp_rank=2) == 0


def test_owned_scatter_uses_reserved_slot_one_and_skips_remote_pages() -> None:
    route = _route()
    entries = route.plan_owned_scatter(
        [
            (-1, -1),
            (4, 0),
            (5, 1),
            (8, 3),
            (9, 4),
        ],
        tp_rank=0,
        tokens_per_page=8,
    )
    assert [
        (
            entry.source_row,
            entry.global_block_id,
            entry.owner_local_slot,
            entry.flat_tensor_slot,
        )
        for entry in entries
    ] == [
        (1, 4, 1, 8),
        (3, 8, 2, 19),
    ]

    with pytest.raises(SentinelBlockError, match="cannot write"):
        route.plan_owned_scatter(
            [(0, 0)],
            tp_rank=0,
            tokens_per_page=8,
        )
    with pytest.raises(ValueError, match="write a slot twice"):
        route.plan_owned_scatter(
            [(4, 0), (4, 0)],
            tp_rank=0,
            tokens_per_page=8,
        )


def test_materialization_is_sorted_deduplicated_and_bounded() -> None:
    route = _route()
    materialization = route.plan_materialization(
        [0, 10, -1, 4, 7, 4],
        destination_rank=3,
    )
    assert materialization.selected_global_block_ids == (4, 7, 10)
    assert [
        (
            entry.owner_rank,
            entry.owner_local_slot,
            entry.scratch_slot,
        )
        for entry in materialization.entries
    ] == [
        (0, 1, 0),
        (3, 1, 1),
        (2, 2, 2),
    ]
    assert [entry.scratch_offset_bytes for entry in materialization.entries] == [
        2 * GRANULARITY_BYTES,
        2 * GRANULARITY_BYTES + PAGE_SIZE_BYTES,
        2 * GRANULARITY_BYTES + 2 * PAGE_SIZE_BYTES,
    ]
    assert materialization.remap_block_table([10, 0, 4, -1, 7]) == (2, -1, 0, -1, 1)

    with pytest.raises(ValueError, match="absent from bounded scratch"):
        materialization.remap_block_table([5])

    constrained_metadata = _metadata()
    constrained_metadata["scratch"][0]["max_pages_per_rank"] = 3
    constrained_route = C128PackedOwnerRoute.from_serialized_plan(
        constrained_metadata,
        group_index=1,
        group_name="group_1",
        component_name="group_1_component_0",
        layer_name="c128.0",
        copy_index=0,
        allow_planner_only=True,
    )
    with pytest.raises(ValueError, match="bound is 3"):
        constrained_route.plan_materialization(
            range(4, 8),
            destination_rank=0,
        )


def test_runtime_tensor_views_must_cover_persistent_and_scratch_segments() -> None:
    route = _route(
        runtime_ready=True,
        allow_planner_only=False,
    )
    route.validate_runtime_tensor_pages(
        tp_rank=3,
        persistent_pages=2,
        scratch_pages=7,
    )
    route.validate_runtime_tensor_pages(
        tp_rank=0,
        persistent_pages=8,
        scratch_pages=8,
    )
    with pytest.raises(ValueError, match="needs 3"):
        route.validate_runtime_tensor_pages(
            tp_rank=0,
            persistent_pages=2,
            scratch_pages=7,
        )
    with pytest.raises(ValueError, match="bounded capacity 7"):
        route.validate_runtime_tensor_pages(
            tp_rank=0,
            persistent_pages=3,
            scratch_pages=6,
        )
    with pytest.raises(ValueError, match="persistent tensor.*beyond"):
        route.validate_runtime_tensor_pages(
            tp_rank=0,
            persistent_pages=9,
            scratch_pages=8,
        )
    with pytest.raises(ValueError, match="scratch tensor.*beyond"):
        route.validate_runtime_tensor_pages(
            tp_rank=0,
            persistent_pages=8,
            scratch_pages=9,
        )


def test_metadata_fixture_is_not_mutated_during_parse() -> None:
    metadata = _metadata()
    original = deepcopy(metadata)
    C128PackedOwnerRoute.from_serialized_plan(
        metadata,
        group_index=1,
        group_name="group_1",
        component_name="group_1_component_0",
        layer_name="c128.0",
        copy_index=0,
        allow_planner_only=True,
    )
    assert metadata == original
