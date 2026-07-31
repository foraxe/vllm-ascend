# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Fixed-profile table-domain gates for the six DSV4-Flash cache groups."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable
from vllm_ascend.worker.c128_packed_runtime import (
    C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
    C128_PACKED_POOL_COMPONENT_SIGNATURES,
    C128_PACKED_POOL_GLOBAL_BLOCK_CAPACITY,
    C128_PACKED_POOL_GROUP_IDENTITIES,
)
from vllm_ascend.worker.packed_block_table import (
    PackedBlockTableTranslator,
    packed_block_table_translators_from_metadata,
)

pytestmark = pytest.mark.cpu_test

_GROUP_IDENTITIES = (
    "c4_attention",
    "c128_attention",
    "dense_swa_a",
    "dense_swa_b",
    "c4_state",
    "c128_state",
)
_GROUP_NAMES = tuple(f"group_{index}" for index in range(len(_GROUP_IDENTITIES)))
_SAME_B_QUOTAS = (17, 3_281, 42, 42, 642, 165)
_GLOBAL_BLOCK_CAPACITY = 4_190


def _component(
    name: str,
    *,
    placement: PackedPlacement,
    copies: int,
    page_size_bytes: int = 128 * 1024,
) -> PackedPoolComponentSpec:
    return PackedPoolComponentSpec(
        name=name,
        bucket=f"page_{page_size_bytes}",
        page_size_bytes=page_size_bytes,
        copies=copies,
        placement=placement,
    )


def _group(
    name: str,
    logical_blocks: int,
    *components: PackedPoolComponentSpec,
) -> PackedPoolGroupSpec:
    return PackedPoolGroupSpec(
        name=name,
        logical_blocks=logical_blocks,
        components=components,
    )


def _same_b_plan() -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=_GLOBAL_BLOCK_CAPACITY,
        tp_size=8,
        groups=(
            _group(
                "group_0",
                _SAME_B_QUOTAS[0],
                _component(
                    "group_0_component_0",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    page_size_bytes=16_640,
                ),
                _component(
                    "group_0_component_1",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                ),
            ),
            _group(
                "group_1",
                _SAME_B_QUOTAS[1],
                _component(
                    "group_1_component_0",
                    placement=PackedPlacement.C128_OWNER,
                    copies=20,
                ),
            ),
            _group(
                "group_2",
                _SAME_B_QUOTAS[2],
                _component(
                    "group_2_component_0",
                    placement=PackedPlacement.REPLICATED,
                    copies=22,
                ),
            ),
            _group(
                "group_3",
                _SAME_B_QUOTAS[3],
                _component(
                    "group_3_component_0",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                ),
            ),
            _group(
                "group_4",
                _SAME_B_QUOTAS[4],
                _component(
                    "group_4_component_0",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    page_size_bytes=16_640,
                ),
                _component(
                    "group_4_component_1",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                ),
            ),
            _group(
                "group_5",
                _SAME_B_QUOTAS[5],
                _component(
                    "group_5_component_0",
                    placement=PackedPlacement.REPLICATED,
                    copies=20,
                ),
            ),
        ),
    )


def _translators(
    plan: PackedPoolPlan,
) -> tuple[PackedBlockTableTranslator, ...]:
    layer_names = tuple(
        tuple(
            f"{identity}.{layer_index}"
            for layer_index in range(sum(component.copies for component in group.components))
        )
        for identity, group in zip(_GROUP_IDENTITIES, plan.groups)
    )
    metadata = {
        "groups": [
            {
                "group_index": group_index,
                "name": group.name,
                "layer_names": list(group_layer_names),
            }
            for group_index, (group, group_layer_names) in enumerate(zip(plan.groups, layer_names))
        ]
    }
    with patch(
        "vllm_ascend.worker.c128_packed_runtime." "packed_arena_contract_from_metadata",
        return_value=SimpleNamespace(plan=plan),
    ):
        return packed_block_table_translators_from_metadata(
            metadata,
            tp_rank=0,
            kv_cache_group_layer_names=layer_names,
        )


def _scheduler_rows(
    plan: PackedPoolPlan,
) -> tuple[list[int], ...]:
    return tuple(
        [
            0,
            plan.group_range(group.name).start,
            plan.group_range(group.name).stop - 1,
        ]
        for group in plan.groups
    )


def _table(
    *,
    translators: tuple[PackedBlockTableTranslator, ...] | None,
) -> MultiGroupBlockTable:
    return MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=8_200,
        max_num_batched_tokens=5_120,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[128, 128, 128, 128, 8, 32],
        max_num_blocks=[4] * len(_GROUP_NAMES),
        kernel_sizes=[[128], [128], [128], [128], [8], [32]],
        packed_translators=translators,
    )


def test_fixed_flash_group_order_places_only_c128_attention_on_owner() -> None:
    plan = _same_b_plan()

    assert C128_PACKED_POOL_GROUP_IDENTITIES == _GROUP_IDENTITIES
    assert C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS == _SAME_B_QUOTAS
    assert C128_PACKED_POOL_GLOBAL_BLOCK_CAPACITY == _GLOBAL_BLOCK_CAPACITY
    assert tuple(group.name for group in plan.groups) == _GROUP_NAMES
    assert tuple(group.logical_blocks for group in plan.groups) == _SAME_B_QUOTAS
    assert plan.used_logical_blocks == _GLOBAL_BLOCK_CAPACITY - 1
    assert tuple({component.placement for component in group.components} for group in plan.groups) == (
        {PackedPlacement.REPLICATED},
        {PackedPlacement.C128_OWNER},
        {PackedPlacement.REPLICATED},
        {PackedPlacement.REPLICATED},
        {PackedPlacement.REPLICATED},
        {PackedPlacement.REPLICATED},
    )
    assert (
        tuple(
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
        == C128_PACKED_POOL_COMPONENT_SIGNATURES
    )


def test_six_group_worker_tables_keep_only_c128_attention_global() -> None:
    plan = _same_b_plan()
    table = _table(translators=_translators(plan))
    scheduler_rows = _scheduler_rows(plan)

    table.add_row(scheduler_rows, row_idx=0)

    expected_execution_rows = (
        [0, 1, 17],
        [0, 18, 3_298],
        [0, 1, 42],
        [0, 1, 42],
        [0, 1, 642],
        [0, 1, 165],
    )
    for group_name, block_table, expected in zip(
        _GROUP_NAMES,
        table.block_tables,
        expected_execution_rows,
    ):
        np.testing.assert_array_equal(
            block_table.block_table.np[0, :3],
            np.asarray(expected, dtype=np.int32),
            err_msg=f"{group_name} execution table uses the wrong ID domain",
        )

    c128_global_ids = table.block_tables[1].block_table.np[0, 1:3]
    owner_addresses = tuple(
        plan.map_c128(
            "group_1",
            "group_1_component_0",
            plan.decode_for_group("group_1", int(global_id)),
        )
        for global_id in c128_global_ids
    )
    assert tuple(address.global_block_id for address in owner_addresses) == (
        18,
        3_298,
    )
    assert all(address.owner_rank == address.global_block_id % plan.tp_size for address in owner_addresses)

    for group_index in (0, 2, 3, 4, 5):
        group_name = _GROUP_IDENTITIES[group_index]
        packed_group_name = _GROUP_NAMES[group_index]
        logical_range = plan.group_range(packed_group_name)
        block_table = table.block_tables[group_index]
        block_table.add_row(
            [logical_range.start, logical_range.start + 1],
            row_idx=0,
        )
        block_table.compute_slot_mapping_draft(
            req_indices=np.zeros(2, dtype=np.int32),
            positions=np.asarray(
                [0, block_table.block_size],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            block_table.slot_mapping.np[:2],
            np.asarray(
                [block_table.block_size, 2 * block_table.block_size],
                dtype=np.int32,
            ),
            err_msg=f"{group_name} write slots are not component-local",
        )


def test_prefix_tail_scheduler_updates_preserve_execution_domains() -> None:
    """The 5120-token prefix mapping stays stable when the tail is appended."""
    plan = _same_b_plan()
    table = _table(translators=_translators(plan))
    prefix_scheduler_rows = tuple(
        [
            plan.group_range(group.name).start,
            plan.group_range(group.name).start + 1,
        ]
        for group in plan.groups
    )
    tail_scheduler_rows = tuple([plan.group_range(group.name).start + 2] for group in plan.groups)

    table.add_row(prefix_scheduler_rows, row_idx=0)
    prefix_execution_rows = tuple(block_table.block_table.np[0, :2].copy() for block_table in table.block_tables)
    table.append_row(tail_scheduler_rows, row_idx=0)

    for group_index, (
        identity,
        packed_group,
        block_table,
        prefix_execution,
    ) in enumerate(
        zip(
            _GROUP_IDENTITIES,
            plan.groups,
            table.block_tables,
            prefix_execution_rows,
        )
    ):
        np.testing.assert_array_equal(
            block_table.block_table.np[0, :2],
            prefix_execution,
            err_msg=f"{identity} prefix mapping changed after tail append",
        )
        expected = (
            np.asarray(
                [
                    plan.group_range(packed_group.name).start,
                    plan.group_range(packed_group.name).start + 1,
                    plan.group_range(packed_group.name).start + 2,
                ],
                dtype=np.int32,
            )
            if group_index == 1
            else np.asarray([1, 2, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            block_table.block_table.np[0, :3],
            expected,
            err_msg=f"{identity} tail update uses the wrong ID domain",
        )
        assert block_table.num_blocks_per_row[0] == 3

        if group_index == 1:
            continue
        block_table.compute_slot_mapping_draft(
            req_indices=np.zeros(3, dtype=np.int32),
            positions=np.asarray(
                [
                    0,
                    block_table.block_size,
                    2 * block_table.block_size,
                ],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            block_table.slot_mapping.np[:3],
            np.asarray(
                [
                    block_table.block_size,
                    2 * block_table.block_size,
                    3 * block_table.block_size,
                ],
                dtype=np.int32,
            ),
            err_msg=f"{identity} tail write slots are not component-local",
        )


def test_feature_off_preserves_all_six_scheduler_global_rows() -> None:
    plan = _same_b_plan()
    table = _table(translators=None)
    scheduler_rows = _scheduler_rows(plan)

    table.add_row(scheduler_rows, row_idx=0)

    for group_name, block_table, scheduler_row in zip(
        _GROUP_NAMES,
        table.block_tables,
        scheduler_rows,
    ):
        np.testing.assert_array_equal(
            block_table.block_table.np[0, :3],
            np.asarray(scheduler_row, dtype=np.int32),
            err_msg=f"{group_name} changed with packed translation disabled",
        )
