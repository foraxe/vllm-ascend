# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU reference gates for the DSA-CP owner-compute design.

These tests intentionally model data ownership and cache placement only. They
do not claim that a Python loop is a distributed/NPU implementation. A later
HCCL or VMM transport must match this oracle before it can replace the current
hidden-state AllGather path.
"""

import pytest
import torch

from vllm_ascend.attention.context_parallel.c128_owner_cache import (
    C128LocalCompressorPlan,
    C128OwnerShardCache,
    c128_local_page,
    c128_owner,
    get_c128_owner_cache,
    make_c128_local_compressor_plan,
    register_c128_owner_cache,
    remap_c128_block_table,
    slice_c128_local_compressor_output,
    slice_c128_local_compressor_rope,
)

pytestmark = pytest.mark.cpu_test


def _place_kv(cache: torch.Tensor, slots: torch.Tensor, kv: torch.Tensor) -> None:
    if slots.ndim != 1 or kv.ndim != 2 or slots.numel() != kv.shape[0]:
        raise ValueError("slots must be 1-D and match the KV token dimension")
    if slots.unique().numel() != slots.numel():
        raise ValueError("one direct-placement operation cannot write a slot twice")
    cache[:, slots] = kv.unsqueeze(0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_owner_wkv_direct_placement_matches_allgather_baseline(world_size: int) -> None:
    """One owner WKV per sequence shard gives the same replicated KV cache.

    This is the first DSA-CP semantic gate: compute WKV only at the source
    token owner, then place that produced KV at every required local cache.
    It replaces an AllGather of hidden states followed by repeated WKV.
    """
    torch.manual_seed(7)
    tokens_per_owner, hidden_dim, kv_dim = 3, 5, 4
    total_tokens = world_size * tokens_per_owner
    hidden = torch.randn(total_tokens, hidden_dim)
    wkv = torch.randn(hidden_dim, kv_dim)
    slots = torch.randperm(total_tokens, generator=torch.Generator().manual_seed(11))

    baseline = torch.zeros(world_size, total_tokens, kv_dim)
    _place_kv(baseline, slots, hidden @ wkv)

    owner_direct = torch.zeros_like(baseline)
    for owner in range(world_size):
        start = owner * tokens_per_owner
        end = start + tokens_per_owner
        owner_kv = hidden[start:end] @ wkv
        _place_kv(owner_direct, slots[start:end], owner_kv)

    torch.testing.assert_close(owner_direct, baseline)


def test_direct_placement_rejects_colliding_slots() -> None:
    cache = torch.zeros(2, 8, 3)
    with pytest.raises(ValueError, match="write a slot twice"):
        _place_kv(cache, torch.tensor([2, 2]), torch.ones(2, 3))


def _materialize_owner_rows(
    owner_shards: list[torch.Tensor], selected_rows: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Read logical selected rows from rank-major owner-page shards.

    This models the prefill-only consumer contract.  The returned tensor is a
    bounded local workspace for dequantize/attention, not another persistent
    cache replica.  Transport is deliberately abstract: HCCL staging and a
    VMM peer view must both preserve this mapping.
    """
    world_size = len(owner_shards)
    rows = []
    for row in selected_rows.tolist():
        logical_page, in_page = divmod(row, page_size)
        owner = logical_page % world_size
        owner_page = logical_page // world_size
        rows.append(owner_shards[owner][owner_page, in_page])
    return torch.stack(rows)


def _materialize_owner_quantized_rows(
    owner_quantized_shards: list[torch.Tensor],
    owner_scale_shards: list[torch.Tensor],
    selected_rows: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Dequantize selected owner rows into the local attention workspace."""
    quantized = _materialize_owner_rows(owner_quantized_shards, selected_rows, page_size)
    scales = _materialize_owner_rows(owner_scale_shards, selected_rows, page_size)
    return quantized.to(torch.float32) * scales.to(torch.float32)


def test_owner_sharded_c128_materialization_matches_replicated_history() -> None:
    """Frozen C128 rows need one owner copy and a local selected-row workspace.

    This gate starts *after* compression has emitted its history rows.  It
    proves the #49741-style consumer contract independently from the ordered
    compressor-state production problem covered by the next test.
    """
    torch.manual_seed(19)
    world_size, page_size, pages_per_owner, kv_dim = 4, 2, 3, 5
    total_pages = world_size * pages_per_owner
    replicated_history = torch.randn(total_pages, page_size, kv_dim)

    owner_shards = [
        replicated_history[owner::world_size].clone() for owner in range(world_size)
    ]
    selected_rows = torch.tensor([0, 3, 4, 7, 10, 17, 22])

    materialized = _materialize_owner_rows(owner_shards, selected_rows, page_size)
    expected = replicated_history.flatten(0, 1)[selected_rows]
    torch.testing.assert_close(materialized, expected)


def test_c128_owner_page_map_and_compact_stage_block_table() -> None:
    """The production owner map preserves every logical page exactly once."""
    pages = torch.arange(12, dtype=torch.int64)
    owners = c128_owner(pages, world_size=4)
    local_pages = c128_local_page(pages, world_size=4)
    torch.testing.assert_close(owners, torch.tensor([0, 1, 2, 3] * 3))
    torch.testing.assert_close(local_pages, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))

    # A rank needs a compact local stage view for only its referenced pages;
    # padding must remain -1 so the kernel does not consume a stale stage row.
    selected_pages = torch.tensor([1, 4, 7, 10], dtype=torch.int32)
    block_table = torch.tensor([[10, 4, -1, 1], [7, -1, -1, -1]], dtype=torch.int32)
    remapped = remap_c128_block_table(block_table, selected_pages)
    torch.testing.assert_close(remapped, torch.tensor([[3, 1, -1, 0], [2, -1, -1, -1]], dtype=torch.int32))


def test_c128_stage_block_table_rejects_missing_page() -> None:
    with pytest.raises(ValueError, match="absent from the staged set"):
        remap_c128_block_table(torch.tensor([[3]], dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32))


def test_owner_cache_registry_preserves_tensor_static_forward_abi() -> None:
    """Owner metadata must not replace the Tensor model layers receive."""
    persistent = torch.zeros(3, 2, 4)
    stage = torch.empty(6, 2, 4)
    owner_cache = C128OwnerShardCache(persistent, stage, tp_size=2)

    layer_cache = [register_c128_owner_cache(owner_cache)]
    static_forward_cache = [layer_cache]

    assert static_forward_cache[0][0] is persistent
    assert isinstance(static_forward_cache[0][0], torch.Tensor)
    assert get_c128_owner_cache(static_forward_cache[0][0]) is owner_cache
    assert get_c128_owner_cache(torch.empty_like(persistent)) is None


def test_owner_scatter_plan_preserves_rows_and_compact_addresses() -> None:
    """Pre-compress planning selects the same owner rows as final placement."""
    # Two owners, three pages per owner, and two tokens per page.
    owner_cache = C128OwnerShardCache(
        persistent_cache=torch.empty(3, 2, 1, 4),
        stage_cache=torch.empty(6, 2, 1, 4),
        tp_size=2,
    )
    slot_mapping = torch.tensor(
        [
            [0, 1],  # rank 0, ignored by rank 1
            [1, 0],  # rank 1 -> compact page 0, flat row 0
            [4, 1],  # rank 0, ignored by rank 1
            [5, 1],  # rank 1 -> compact page 2, flat row 5
            [-1, -1],  # padding
        ],
        dtype=torch.int64,
    )

    owner_rows, flat_slots, expected_rows = owner_cache.prepare_owned_scatter(slot_mapping, tp_rank=1)

    torch.testing.assert_close(owner_rows, torch.tensor([1, 3]))
    torch.testing.assert_close(flat_slots, torch.tensor([[0], [5]]))
    assert expected_rows == slot_mapping.shape[0]


@pytest.mark.parametrize("world_size", [2, 4, 16])
def test_owner_sharded_quantized_c128_dequantizes_only_selected_rows(world_size: int) -> None:
    """The #49741-style prefill consumer reads quantized owner rows locally.

    Persistent storage contains one quantized C128 page and one scale page per
    logical page. Only selected rows are dequantized into a temporary local
    workspace; neither the quantized page nor the BF16 result is replicated.
    """
    torch.manual_seed(23)
    page_size, pages_per_owner, kv_dim = 2, 3, 5
    total_pages = world_size * pages_per_owner
    quantized_history = torch.randint(
        -127, 128, (total_pages, page_size, kv_dim), dtype=torch.int8
    )
    scales_history = torch.rand(total_pages, page_size, 1, dtype=torch.float32) + 0.01
    owner_quantized_shards = [
        quantized_history[owner::world_size].clone() for owner in range(world_size)
    ]
    owner_scale_shards = [
        scales_history[owner::world_size].clone() for owner in range(world_size)
    ]
    selected_rows = torch.tensor([0, 3, 4, 7, total_pages * page_size - 1])

    materialized = _materialize_owner_quantized_rows(
        owner_quantized_shards, owner_scale_shards, selected_rows, page_size
    )
    expected = (
        quantized_history.flatten(0, 1)[selected_rows].to(torch.float32)
        * scales_history.flatten(0, 1)[selected_rows]
    )
    torch.testing.assert_close(materialized, expected)

    owner_persistent_bytes = sum(
        quantized.numel() * quantized.element_size()
        + scale.numel() * scale.element_size()
        for quantized, scale in zip(owner_quantized_shards, owner_scale_shards)
    )
    replicated_persistent_bytes = world_size * (
        quantized_history.numel() * quantized_history.element_size()
        + scales_history.numel() * scales_history.element_size()
    )
    assert owner_persistent_bytes * world_size == replicated_persistent_bytes


def test_c128_local_compressor_plan_accepts_aligned_tp8_chunk() -> None:
    """A 5120-token TP8 chunk splits into eight independent 5-row C128 slices."""
    positions = torch.arange(5120, dtype=torch.int64)
    plans = [
        make_c128_local_compressor_plan(
            positions,
            local_start=rank * 640,
            local_end=(rank + 1) * 640,
            tp_size=8,
        )
        for rank in range(8)
    ]
    assert all(plan is not None for plan in plans)
    assert [(plan.slot_start, plan.slot_end) for plan in plans if plan] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 35),
        (35, 40),
    ]


def test_c128_local_compressor_plan_rejects_unaligned_tp8_tail() -> None:
    """The 3080-token TP8 tail has 385-token rank shards and must fall back."""
    positions = torch.arange(5120, 8200, dtype=torch.int64)
    assert make_c128_local_compressor_plan(
        positions,
        local_start=0,
        local_end=385,
        tp_size=8,
        sequence_start_pos=5120,
    ) is None


def test_c128_local_compressor_rope_retains_per_shard_padding_row() -> None:
    """CANN requires five C128 rows plus one RoPE padding row per TP shard."""
    rope = torch.arange(41, dtype=torch.float32).unsqueeze(1)
    plan = C128LocalCompressorPlan(slot_start=35, slot_end=40)
    local_rope = slice_c128_local_compressor_rope(rope, plan)
    torch.testing.assert_close(
        local_rope.squeeze(1), torch.tensor([35, 36, 37, 38, 39, 40], dtype=rope.dtype)
    )


def test_c128_local_compressor_output_drops_final_padding_row() -> None:
    """Only the five mapped data rows may enter the static TP exchange."""
    plan = C128LocalCompressorPlan(slot_start=35, slot_end=40)
    compressed_kv = torch.arange(6 * 3, dtype=torch.float32).view(6, 3)
    local_rows = slice_c128_local_compressor_output(compressed_kv, plan)
    torch.testing.assert_close(local_rows, compressed_kv[:5])


def test_c128_static_collective_layout_restores_rank_major_slot_order() -> None:
    """The fixed all-to-all buffer reconstructs the pre-existing slot ABI.

    Each source rank broadcasts its local C128 rows into equal destination
    chunks.  ``all_to_all_single`` then presents every receiver with source
    rank-major chunks, exactly matching the global compressor-slot ordering
    consumed by the existing owner-scatter plan.
    """
    world_size, rows, kv_dim = 8, 5, 3
    local_rows = [
        torch.arange(rank * rows * kv_dim, (rank + 1) * rows * kv_dim).view(rows, kv_dim)
        for rank in range(world_size)
    ]
    sends = []
    for compressed_kv in local_rows:
        send = torch.empty(world_size * rows, kv_dim, dtype=compressed_kv.dtype)
        send.view(world_size, rows, kv_dim).copy_(compressed_kv)
        sends.append(send)

    expected = torch.cat(local_rows)
    for receiver in range(world_size):
        received = torch.cat(
            [send.view(world_size, rows, kv_dim)[receiver] for send in sends]
        )
        torch.testing.assert_close(received, expected)
