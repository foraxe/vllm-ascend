# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
)
from vllm_ascend.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm_ascend.worker.packed_block_table import PackedBlockTableTranslator

pytestmark = pytest.mark.cpu_test


def _component(
    name: str,
    placement: PackedPlacement,
) -> PackedPoolComponentSpec:
    return PackedPoolComponentSpec(
        name=name,
        bucket="wide",
        page_size_bytes=128,
        copies=1,
        placement=placement,
    )


def _group(
    name: str,
    logical_blocks: int,
    component: PackedPoolComponentSpec,
) -> PackedPoolGroupSpec:
    return PackedPoolGroupSpec(
        name=name,
        logical_blocks=logical_blocks,
        components=(component,),
    )


def _plan() -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=14,
        tp_size=4,
        groups=(
            _group(
                "c4",
                3,
                _component("c4_replicated", PackedPlacement.REPLICATED),
            ),
            _group(
                "c128",
                7,
                _component("c128_owner", PackedPlacement.C128_OWNER),
            ),
            _group(
                "swa",
                2,
                _component("swa_replicated", PackedPlacement.REPLICATED),
            ),
        ),
    )


def _translator(
    plan: PackedPoolPlan,
    group_name: str,
    component_name: str,
    placement: PackedPlacement,
    *,
    tp_rank: int = 0,
) -> PackedBlockTableTranslator:
    return PackedBlockTableTranslator(
        plan=plan,
        group_name=group_name,
        component_name=component_name,
        placement=placement,
        tp_rank=tp_rank,
    )


def _block_table(
    *,
    block_size: int,
    kernel_size: int,
    translator: PackedBlockTableTranslator | None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    interleave: int = 1,
) -> BlockTable:
    dcp_group = MagicMock(world_size=cp_world_size, rank_in_group=cp_rank)
    pcp_group = MagicMock(world_size=1, rank_in_group=0)
    with (
        patch(
            "vllm_ascend.worker.block_table.get_dcp_group",
            return_value=dcp_group,
        ),
        patch(
            "vllm_ascend.worker.block_table.get_pcp_group",
            return_value=pcp_group,
        ),
    ):
        return BlockTable(
            block_size=block_size,
            max_num_reqs=2,
            max_num_blocks_per_req=16,
            max_num_batched_tokens=64,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[kernel_size],
            cp_kv_cache_interleave_size=interleave,
            packed_translator=translator,
        )


def test_multigroup_translation_uses_exact_disjoint_group_ranges() -> None:
    plan = _plan()
    translators = [
        _translator(
            plan,
            "c4",
            "c4_replicated",
            PackedPlacement.REPLICATED,
        ),
        _translator(
            plan,
            "c128",
            "c128_owner",
            PackedPlacement.C128_OWNER,
            tp_rank=1,
        ),
        _translator(
            plan,
            "swa",
            "swa_replicated",
            PackedPlacement.REPLICATED,
        ),
    ]
    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[128, 128, 128],
        max_num_blocks=[8, 8, 8],
        kernel_sizes=[[128], [128], [128]],
        packed_translators=translators,
    )

    table.add_row(
        (
            [0, 1, 3],
            [0, 1, 4, 7],
            [0, 1, 2],
        ),
        row_idx=0,
    )

    np.testing.assert_array_equal(
        table[0].block_table.np[0, :3],
        np.array([0, 1, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        table[1].block_table.np[0, :4],
        np.array([0, 4, 7, 10], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        table[2].block_table.np[0, :3],
        np.array([0, 11, 12], dtype=np.int32),
    )

    table.append_row(
        (
            [2],
            [2],
            [2],
        ),
        row_idx=0,
    )
    assert table[0].block_table.np[0, 3] == 2
    assert table[1].block_table.np[0, 4] == 5
    assert table[2].block_table.np[0, 3] == 12


def test_feature_off_append_retains_existing_hybrid_expansion() -> None:
    table = _block_table(
        block_size=128,
        kernel_size=64,
        translator=None,
    )
    table.add_row([0, 1], row_idx=0)

    # This is the pre-feature behavior, including expansion of block zero.
    np.testing.assert_array_equal(
        table.block_table.np[0, :4],
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_feature_on_hybrid_expansion_preserves_sentinel_zero() -> None:
    plan = PackedPoolPlan(
        global_block_capacity=7,
        tp_size=2,
        groups=(
            _group(
                "prefix",
                2,
                _component("prefix_replicated", PackedPlacement.REPLICATED),
            ),
            _group(
                "flash",
                4,
                _component("replicated", PackedPlacement.REPLICATED),
            ),
        ),
    )
    table = _block_table(
        block_size=128,
        kernel_size=64,
        translator=_translator(
            plan,
            "flash",
            "replicated",
            PackedPlacement.REPLICATED,
        ),
    )
    table.add_row([0, 1], row_idx=0)

    np.testing.assert_array_equal(
        table.block_table.np[0, :4],
        np.array([0, 1, 6, 7], dtype=np.int32),
    )
    assert table.num_blocks_per_row[0] == 4


def test_packed_c128_rejects_hybrid_block_expansion() -> None:
    plan = _plan()
    with pytest.raises(
        ValueError,
        match="requires matching physical and kernel block sizes",
    ):
        _block_table(
            block_size=128,
            kernel_size=64,
            translator=_translator(
                plan,
                "c128",
                "c128_owner",
                PackedPlacement.C128_OWNER,
                tp_rank=1,
            ),
        )


def test_hybrid_slot_translation_restores_component_physical_offsets() -> None:
    plan = PackedPoolPlan(
        global_block_capacity=5,
        tp_size=2,
        groups=(
            _group(
                "flash",
                4,
                _component("replicated", PackedPlacement.REPLICATED),
            ),
        ),
    )
    table = _block_table(
        block_size=128,
        kernel_size=64,
        translator=_translator(
            plan,
            "flash",
            "replicated",
            PackedPlacement.REPLICATED,
        ),
    )
    table.add_row([0, 1], row_idx=0)
    table.compute_slot_mapping_draft(
        req_indices=np.zeros(4, dtype=np.int32),
        positions=np.array([0, 64, 128, 192], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        table.slot_mapping.np[:4],
        np.array([0, 64, 128, 192], dtype=np.int32),
    )


def test_c128_slots_are_owner_local_and_reserve_physical_slot_zero() -> None:
    plan = _plan()
    table = _block_table(
        block_size=128,
        kernel_size=128,
        translator=_translator(
            plan,
            "c128",
            "c128_owner",
            PackedPlacement.C128_OWNER,
            tp_rank=1,
        ),
    )
    table.add_row([0, 1, 2, 3, 4, 5, 6, 7], row_idx=0)
    table.compute_slot_mapping_draft(
        req_indices=np.zeros(8, dtype=np.int32),
        positions=np.arange(0, 8 * 128, 128, dtype=np.int32),
    )

    # C128 range [4, 11): rank 1 owns global blocks 5 and 9, which
    # become dense physical slots 1 and 2. Global page zero remains the
    # reserved component-local dummy page; other owners are filtered.
    np.testing.assert_array_equal(
        table.slot_mapping.np[:8],
        np.array([0, -1, 128, -1, -1, -1, 256, -1], dtype=np.int32),
    )


@pytest.mark.parametrize("tp_rank", range(4))
def test_c128_slot_translation_matches_plan_oracle(tp_rank: int) -> None:
    plan = _plan()
    translator = _translator(
        plan,
        "c128",
        "c128_owner",
        PackedPlacement.C128_OWNER,
        tp_rank=tp_rank,
    )
    logical_size = 128
    offset = 17
    encoded = plan.encode_group_block_ids(
        "c128",
        tuple(range(1, 8)),
    )
    slots = torch.tensor(
        [offset, *(block_id * logical_size + offset for block_id in encoded)],
        dtype=torch.int32,
    )

    translator.translate_slot_mapping_(
        slots,
        logical_block_size=logical_size,
        physical_block_size=logical_size,
        blocks_per_phys_block=1,
    )

    expected = [offset]
    for group_block_id in range(1, 8):
        address = plan.map_c128(
            "c128",
            "c128_owner",
            group_block_id,
        )
        expected.append(address.physical_slot * logical_size + offset if address.owner_rank == tp_rank else -1)
    assert slots.tolist() == expected


def test_c128_owner_translation_preserves_cp_interleave_padding() -> None:
    plan = PackedPoolPlan(
        global_block_capacity=7,
        tp_size=2,
        groups=(
            _group(
                "c4",
                2,
                _component("c4_replicated", PackedPlacement.REPLICATED),
            ),
            _group(
                "c128",
                4,
                _component("c128_owner", PackedPlacement.C128_OWNER),
            ),
        ),
    )
    table = _block_table(
        block_size=4,
        kernel_size=4,
        translator=_translator(
            plan,
            "c128",
            "c128_owner",
            PackedPlacement.C128_OWNER,
            tp_rank=1,
        ),
        cp_world_size=2,
        cp_rank=1,
        interleave=1,
    )
    table.add_row([1], row_idx=0)
    table.compute_slot_mapping_draft(
        req_indices=np.zeros(8, dtype=np.int32),
        positions=np.arange(8, dtype=np.int32),
    )

    # Packed global block 3 belongs to owner rank 1 and becomes owner slot 1.
    # The CP rank still receives only odd positions; its -1 mask is unchanged.
    np.testing.assert_array_equal(
        table.slot_mapping.np[:8],
        np.array([-1, 4, -1, 5, -1, 6, -1, 7], dtype=np.int32),
    )


def test_multigroup_rejects_partial_translator_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="packed_translators length",
    ):
        MultiGroupBlockTable(
            max_num_reqs=2,
            max_model_len=1024,
            max_num_batched_tokens=64,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[128, 128],
            max_num_blocks=[8, 8],
            kernel_sizes=[[128], [128]],
            packed_translators=[None],
        )


def test_multigroup_validation_is_atomic_across_groups() -> None:
    plan = _plan()
    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[128, 128],
        max_num_blocks=[8, 8],
        kernel_sizes=[[128], [128]],
        packed_translators=[
            _translator(
                plan,
                "c4",
                "c4_replicated",
                PackedPlacement.REPLICATED,
            ),
            _translator(
                plan,
                "c128",
                "c128_owner",
                PackedPlacement.C128_OWNER,
                tp_rank=1,
            ),
        ],
    )
    table.add_row(([1], [1]), row_idx=0)
    before = [block_table.block_table.np[0].copy() for block_table in table.block_tables]
    before_counts = [int(block_table.num_blocks_per_row[0]) for block_table in table.block_tables]

    with pytest.raises(ValueError, match="outside"):
        table.append_row(([2], [8]), row_idx=0)

    for block_table, expected, expected_count in zip(
        table.block_tables,
        before,
        before_counts,
    ):
        np.testing.assert_array_equal(block_table.block_table.np[0], expected)
        assert block_table.num_blocks_per_row[0] == expected_count


def test_multigroup_rejects_block_id_group_count_mismatch() -> None:
    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[128, 128],
        max_num_blocks=[8, 8],
        kernel_sizes=[[128], [128]],
    )
    with pytest.raises(ValueError, match="block_ids group count"):
        table.add_row(([1],), row_idx=0)
