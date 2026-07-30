# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU gates for the packed C128 DSA scatter/materialization seam."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.attention.context_parallel.c128_owner_cache as owner_cache_module
from vllm_ascend.attention.context_parallel.c128_owner_cache import (
    C128OwnerShardCache,
)
from vllm_ascend.attention.context_parallel.c128_packed_owner_route import (
    C128PackedOwnerRoute,
    C128PackedSegment,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    _compressor_state_execution_block_table,
    _materialize_c128_owner_cache,
    _prepare_c128_owner_scatter,
    _swa_execution_block_table,
)

pytestmark = pytest.mark.cpu_test


def _packed_route(
    *,
    tp_size: int,
    logical_start: int,
    logical_stop: int,
    page_size_bytes: int,
    persistent_pages: int,
    scratch_pages: int,
) -> C128PackedOwnerRoute:
    persistent_bytes = persistent_pages * page_size_bytes
    scratch_bytes = scratch_pages * page_size_bytes
    return C128PackedOwnerRoute(
        schema_version=1,
        runtime_abi_ready=True,
        tp_size=tp_size,
        group_index=1,
        group_name="c128",
        component_name="compress_kv",
        layer_name="model.layers.0.self_attn",
        copy_index=0,
        logical_start=logical_start,
        logical_stop=logical_stop,
        page_size_bytes=page_size_bytes,
        allocation_granularity_bytes=page_size_bytes,
        persistent_segments=tuple(
            C128PackedSegment(
                rank=rank,
                base_bytes=0,
                allocated_bytes=persistent_bytes,
            )
            for rank in range(tp_size)
        ),
        scratch_segments=tuple(
            C128PackedSegment(
                rank=rank,
                base_bytes=persistent_bytes,
                allocated_bytes=scratch_bytes,
            )
            for rank in range(tp_size)
        ),
        max_scratch_pages=scratch_pages,
    )


def _packed_cache(
    *,
    tp_size: int,
    logical_start: int,
    logical_stop: int,
    persistent_pages: int,
    scratch_pages: int,
    debug: bool = False,
) -> C128OwnerShardCache:
    persistent = torch.zeros(persistent_pages, 2, 1, 3)
    scratch = torch.zeros(scratch_pages, 2, 1, 3)
    page_size_bytes = persistent[0].numel() * persistent.element_size()
    return C128OwnerShardCache(
        persistent_cache=persistent,
        stage_cache=scratch,
        tp_size=tp_size,
        debug=debug,
        packed_route=_packed_route(
            tp_size=tp_size,
            logical_start=logical_start,
            logical_stop=logical_stop,
            page_size_bytes=page_size_bytes,
            persistent_pages=persistent_pages,
            scratch_pages=scratch_pages,
        ),
    )


def test_packed_scatter_consumes_owner_local_slots_without_second_translation() -> None:
    """Worker slots already include the sentinel-prefixed owner-local page."""
    owner_cache = _packed_cache(
        tp_size=2,
        logical_start=4,
        logical_stop=10,
        persistent_pages=4,
        scratch_pages=3,
    )
    # PAD_SLOT_ID=-1 becomes (-1, page_size - 1) in DSA's 2-D mapping.
    slot_mapping = torch.tensor(
        [
            [-1, 1],
            [1, 0],
            [1, 1],
            [-1, 1],
            [3, 0],
        ],
        dtype=torch.int32,
    )

    owner_rows, flat_slots, expected_rows = _prepare_c128_owner_scatter(
        owner_cache,
        slot_mapping,
        tp_rank=1,
    )

    torch.testing.assert_close(owner_rows, torch.tensor([1, 2, 4]))
    torch.testing.assert_close(flat_slots, torch.tensor([[2], [3], [6]]))
    assert expected_rows == 5


def test_packed_scatter_masks_sentinel_and_padding_without_device_sync() -> None:
    """CPU translation rejects bad IDs; the device seam never writes page 0."""
    owner_cache = _packed_cache(
        tp_size=2,
        logical_start=4,
        logical_stop=10,
        persistent_pages=4,
        scratch_pages=3,
    )
    slot_mapping = torch.tensor(
        [[0, 0], [-1, 1], [2, 0]],
        dtype=torch.int32,
    )

    owner_rows, flat_slots, expected_rows = owner_cache.prepare_owned_scatter(slot_mapping, tp_rank=1)

    torch.testing.assert_close(owner_rows, torch.tensor([2]))
    torch.testing.assert_close(flat_slots, torch.tensor([[4]]))
    assert expected_rows == 3


def test_packed_route_conversion_has_no_device_value_sync() -> None:
    """Packed route/scatter conversion cannot branch on NPU tensor values."""
    for helper in (
        C128OwnerShardCache._prepare_packed_owned_scatter,
        C128OwnerShardCache._packed_owner_index_block_table,
    ):
        source = inspect.getsource(helper)
        for synchronizing_expression in ("bool(", ".any(", ".item("):
            assert synchronizing_expression not in source


def test_replicated_consumer_calls_receive_worker_execution_tables() -> None:
    """Compressor state and SWA consume the component-local worker table."""
    component_local_table = torch.tensor(
        [[0, 1, 2]],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        req_metadata=SimpleNamespace(
            block_table=component_local_table,
        )
    )

    assert _compressor_state_execution_block_table(metadata) is component_local_table
    assert _swa_execution_block_table(metadata) is component_local_table


def test_packed_materialization_converts_global_ids_only_at_cache_boundary() -> None:
    """The private owner index recovers route owner rank and local page."""
    owner_cache = _packed_cache(
        tp_size=4,
        logical_start=7,
        logical_stop=15,
        persistent_pages=3,
        scratch_pages=4,
    )
    packed_global = torch.tensor(
        [[7, 8, 9, 10, 11, 0, -1]],
        dtype=torch.int32,
    )

    owner_index = owner_cache._packed_owner_index_block_table(
        packed_global,
        tp_rank=2,
    )

    # global 7..10 are the first page of owners 3,0,1,2. Global 11 is
    # owner 3's second page. Sentinel zero and padding both become -1.
    torch.testing.assert_close(
        owner_index,
        torch.tensor([[7, 4, 5, 6, 11, -1, -1]], dtype=torch.int32),
    )


def test_packed_materialization_masks_non_group_ids_at_device_boundary() -> None:
    """Foreign IDs cannot become an owner-cache or scratch address."""
    owner_cache = _packed_cache(
        tp_size=4,
        logical_start=7,
        logical_stop=15,
        persistent_pages=3,
        scratch_pages=4,
    )
    block_table = torch.tensor([[6, 15, -2]], dtype=torch.int32)

    owner_index = owner_cache._packed_owner_index_block_table(
        block_table,
        tp_rank=0,
    )

    torch.testing.assert_close(
        owner_index,
        torch.full_like(block_table, -1),
    )


def test_packed_materialization_rejects_unsigned_block_table() -> None:
    """Scratch padding requires a signed block-table dtype."""
    owner_cache = _packed_cache(
        tp_size=1,
        logical_start=4,
        logical_stop=7,
        persistent_pages=4,
        scratch_pages=3,
    )

    with pytest.raises(TypeError, match="signed integer dtype"):
        owner_cache._packed_owner_index_block_table(
            torch.tensor([[4]], dtype=torch.uint8),
            tp_rank=0,
        )


@pytest.mark.parametrize("selective", [False, True])
def test_packed_materialization_reads_sentinel_prefixed_owner_pages(
    monkeypatch: pytest.MonkeyPatch,
    selective: bool,
) -> None:
    """Packed global pages become a dense, zero-based attention scratch."""
    owner_cache = _packed_cache(
        tp_size=1,
        logical_start=4,
        logical_stop=7,
        persistent_pages=4,
        scratch_pages=3,
    )
    owner_cache.persistent_cache[1].fill_(11)
    owner_cache.persistent_cache[2].fill_(22)
    owner_cache.persistent_cache[3].fill_(33)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "get_world_size",
        lambda group: 1,
    )

    def _all_gather(
        outputs: list[torch.Tensor],
        tensor: torch.Tensor,
        *,
        group,
    ) -> None:
        outputs[0].copy_(tensor)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "all_gather",
        _all_gather,
    )

    def _all_to_all_single(
        recv: torch.Tensor,
        send: torch.Tensor,
        **kwargs,
    ) -> None:
        recv.copy_(send)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "all_to_all_single",
        _all_to_all_single,
    )

    packed_global = torch.tensor([[4, 6, 0, -1]], dtype=torch.int32)
    staged, scratch_local = _materialize_c128_owner_cache(
        owner_cache,
        packed_global,
        selective=selective,
        tp_rank=0,
        group=object(),
    )

    torch.testing.assert_close(staged[0], torch.full_like(staged[0], 11))
    torch.testing.assert_close(staged[1], torch.full_like(staged[1], 33))
    torch.testing.assert_close(
        scratch_local,
        torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
    )
    # The DSA helper does not mutate the packed-global producer metadata.
    torch.testing.assert_close(
        packed_global,
        torch.tensor([[4, 6, 0, -1]], dtype=torch.int32),
    )


def test_packed_full_materialization_routes_two_owner_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TP2 union reads each packed page from its canonical owner slot."""
    owner_cache = _packed_cache(
        tp_size=2,
        logical_start=4,
        logical_stop=8,
        persistent_pages=3,
        scratch_pages=4,
    )
    # This test executes rank 0. Its owner-local slots contain global blocks
    # 4 and 6; the fake rank-1 payload below contains global blocks 5 and 7.
    owner_cache.persistent_cache[1].fill_(40)
    owner_cache.persistent_cache[2].fill_(60)
    remote_owner_pages = torch.empty_like(owner_cache.persistent_cache[1:])
    remote_owner_pages[0].fill_(50)
    remote_owner_pages[1].fill_(70)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "get_world_size",
        lambda group: 2,
    )
    collective_call = 0

    def _all_gather(
        outputs: list[torch.Tensor],
        tensor: torch.Tensor,
        *,
        group,
    ) -> None:
        nonlocal collective_call
        if collective_call == 0:
            outputs[0].fill_(4)
            outputs[1].fill_(4)
        elif collective_call == 1:
            pages = torch.tensor([2, 3, 4, 5], dtype=tensor.dtype)
            outputs[0].copy_(pages)
            outputs[1].copy_(pages)
        elif collective_call == 2:
            outputs[0].fill_(2)
            outputs[1].fill_(2)
        elif collective_call == 3:
            outputs[0].copy_(tensor)
            outputs[1].copy_(remote_owner_pages)
        else:  # pragma: no cover - guards collective contract drift
            raise AssertionError("unexpected packed materialization collective")
        collective_call += 1

    monkeypatch.setattr(
        owner_cache_module.dist,
        "all_gather",
        _all_gather,
    )

    staged, scratch_local = owner_cache.materialize_for_attention(
        torch.tensor([[4, 5, 6, 7]], dtype=torch.int32),
        tp_rank=0,
        group=object(),
    )

    assert collective_call == 4
    torch.testing.assert_close(
        staged[:, 0, 0, 0],
        torch.tensor([40, 50, 60, 70], dtype=staged.dtype),
    )
    torch.testing.assert_close(
        scratch_local,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
    )


def test_dsa_materialization_helper_forwards_packed_global_table_unchanged() -> None:
    """DSA delegates the ID-domain boundary to its registered owner cache."""
    owner_cache = MagicMock(spec=C128OwnerShardCache)
    block_table = torch.tensor([[7, 9, 0]], dtype=torch.int32)
    staged = torch.empty(2, 2, 1, 3)
    remapped = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    owner_cache.materialize_selected_for_attention.return_value = (
        staged,
        remapped,
    )

    result = _materialize_c128_owner_cache(
        owner_cache,
        block_table,
        selective=True,
        tp_rank=2,
        group="hccl",
    )

    assert result[0] is staged
    assert result[1] is remapped
    owner_cache.materialize_selected_for_attention.assert_called_once_with(
        block_table,
        tp_rank=2,
        group="hccl",
    )
    assert owner_cache.materialize_selected_for_attention.call_args.args[0] is block_table
