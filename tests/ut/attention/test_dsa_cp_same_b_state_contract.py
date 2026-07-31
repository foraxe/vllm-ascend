# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU contract oracle for replicated DSA compressor-state continuation."""

from types import SimpleNamespace

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
    _swa_execution_block_table,
)

pytestmark = pytest.mark.cpu_test

_PREFIX_TOKENS = 5_120
_TAIL_TOKENS = 3_080
_TOTAL_TOKENS = _PREFIX_TOKENS + _TAIL_TOKENS
_STATE_BLOCK_SIZE = 32
_STATE_COMPONENT_PAGES = 165


def _metadata(block_table: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        req_metadata=SimpleNamespace(
            block_table=block_table,
        )
    )


def _advance_replicated_state(
    token_values: torch.Tensor,
    *,
    start_pos: int,
    state_cache: torch.Tensor,
    state_execution_table: torch.Tensor,
) -> torch.Tensor:
    """Deterministic recurrence over one compact replicated component view.

    The recurrence is an interface oracle rather than an implementation of
    the CANN compressor. It makes state continuity observable: the first token
    of chunk two must read the last state written by chunk one through the
    same component-local table and tensor.
    """
    table = state_execution_table[0]
    outputs = torch.empty_like(token_values)
    for chunk_offset, token_value in enumerate(token_values):
        position = start_pos + chunk_offset
        if position == 0:
            previous = state_cache.new_zeros(())
        else:
            previous_position = position - 1
            previous_page = table[(previous_position // _STATE_BLOCK_SIZE) % table.numel()]
            previous = state_cache[
                previous_page,
                previous_position % _STATE_BLOCK_SIZE,
            ]
        page = table[(position // _STATE_BLOCK_SIZE) % table.numel()]
        value = previous + token_value
        state_cache[page, position % _STATE_BLOCK_SIZE] = value
        outputs[chunk_offset] = value
    return outputs


def _packed_owner_cache() -> C128OwnerShardCache:
    persistent = torch.zeros(5, 2, 1, 1)
    scratch = torch.zeros(4, 2, 1, 1)
    page_size_bytes = persistent[0].numel() * persistent.element_size()
    return C128OwnerShardCache(
        persistent_cache=persistent,
        stage_cache=scratch,
        tp_size=1,
        packed_route=C128PackedOwnerRoute(
            schema_version=1,
            runtime_abi_ready=True,
            tp_size=1,
            group_index=1,
            group_name="group_1",
            component_name="group_1_component_0",
            layer_name="model.layers.0.self_attn",
            copy_index=0,
            logical_start=18,
            logical_stop=22,
            page_size_bytes=page_size_bytes,
            allocation_granularity_bytes=page_size_bytes,
            persistent_segments=(
                C128PackedSegment(
                    rank=0,
                    base_bytes=0,
                    allocated_bytes=persistent.numel() * persistent.element_size(),
                ),
            ),
            scratch_segments=(
                C128PackedSegment(
                    rank=0,
                    base_bytes=persistent.numel() * persistent.element_size(),
                    allocated_bytes=scratch.numel() * scratch.element_size(),
                ),
            ),
            max_scratch_pages=4,
        ),
    )


def test_replicated_consumer_tables_are_forwarded_by_identity() -> None:
    c4_state_table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    c128_state_table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    swa_a_table = torch.tensor([[1, 2]], dtype=torch.int32)
    swa_b_table = torch.tensor([[1, 2]], dtype=torch.int32)

    assert _compressor_state_execution_block_table(_metadata(c4_state_table)) is c4_state_table
    assert _compressor_state_execution_block_table(_metadata(c128_state_table)) is c128_state_table
    assert _swa_execution_block_table(_metadata(swa_a_table)) is swa_a_table
    assert _swa_execution_block_table(_metadata(swa_b_table)) is swa_b_table


@pytest.mark.parametrize("selective", [False, True])
def test_c128_state_interface_continues_across_5120_plus_3080_on_replicated_view(
    monkeypatch: pytest.MonkeyPatch,
    selective: bool,
) -> None:
    """Only the C128 attention table crosses the owner materialization seam."""
    state_execution_table = torch.arange(
        1,
        _STATE_COMPONENT_PAGES + 1,
        dtype=torch.int32,
    ).view(1, -1)
    state_metadata = _metadata(state_execution_table)
    state_cache = torch.zeros(
        _STATE_COMPONENT_PAGES + 1,
        _STATE_BLOCK_SIZE,
        dtype=torch.float64,
    )
    token_values = torch.arange(
        1,
        _TOTAL_TOKENS + 1,
        dtype=torch.float64,
    )

    c128_attention_global_table = torch.tensor(
        [[18, 19, 20, 21]],
        dtype=torch.int32,
    )
    owner_cache = _packed_owner_cache()
    canonical_attention_rows = torch.tensor(
        [180.0, 190.0, 200.0, 210.0],
    )
    for page, value in enumerate(canonical_attention_rows, start=1):
        owner_cache.persistent_cache[page].fill_(value)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "get_world_size",
        lambda group: 1,
    )

    def _all_gather(
        outputs: list[torch.Tensor],
        tensor: torch.Tensor,
        *,
        group: object,
    ) -> None:
        outputs[0].copy_(tensor)

    monkeypatch.setattr(
        owner_cache_module.dist,
        "all_gather",
        _all_gather,
    )
    monkeypatch.setattr(
        owner_cache_module.dist,
        "all_to_all_single",
        lambda recv, send, **kwargs: recv.copy_(send),
    )

    materialized_tables: list[torch.Tensor] = []
    method_name = "materialize_selected_for_attention" if selective else "materialize_for_attention"
    real_materialize = getattr(owner_cache, method_name)

    def _record_materialization(
        block_table: torch.Tensor,
        *,
        tp_rank: int,
        group: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        materialized_tables.append(block_table)
        return real_materialize(
            block_table,
            tp_rank=tp_rank,
            group=group,
        )

    monkeypatch.setattr(
        owner_cache,
        method_name,
        _record_materialization,
    )

    prefix_table = _compressor_state_execution_block_table(state_metadata)
    prefix = _advance_replicated_state(
        token_values[:_PREFIX_TOKENS],
        start_pos=0,
        state_cache=state_cache,
        state_execution_table=prefix_table,
    )
    first_attention_view, first_attention_table = _materialize_c128_owner_cache(
        owner_cache,
        c128_attention_global_table,
        peer_materialization_plan=None,
        peer_current_row_overlay=None,
        current_rows=None,
        selective=selective,
        tp_rank=0,
        group="hccl",
    )
    torch.testing.assert_close(
        first_attention_view[:, 0, 0, 0],
        canonical_attention_rows,
    )
    torch.testing.assert_close(
        first_attention_table,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
    )
    stable_prefix_mapping = state_execution_table.clone()
    owner_cache.stage_cache.fill_(-999.0)

    tail_table = _compressor_state_execution_block_table(state_metadata)
    tail = _advance_replicated_state(
        token_values[_PREFIX_TOKENS:],
        start_pos=_PREFIX_TOKENS,
        state_cache=state_cache,
        state_execution_table=tail_table,
    )
    second_attention_view, second_attention_table = _materialize_c128_owner_cache(
        owner_cache,
        c128_attention_global_table,
        peer_materialization_plan=None,
        peer_current_row_overlay=None,
        current_rows=None,
        selective=selective,
        tp_rank=0,
        group="hccl",
    )

    expected = torch.cumsum(token_values, dim=0)
    torch.testing.assert_close(
        torch.cat((prefix, tail)),
        expected,
        rtol=0,
        atol=0,
    )
    assert prefix_table is state_execution_table
    assert tail_table is state_execution_table
    torch.testing.assert_close(state_execution_table, stable_prefix_mapping)
    assert tail[0] == prefix[-1] + token_values[_PREFIX_TOKENS]
    assert state_execution_table.min().item() == 1
    assert state_execution_table.max().item() == _STATE_COMPONENT_PAGES
    torch.testing.assert_close(
        second_attention_view[:, 0, 0, 0],
        canonical_attention_rows,
    )
    torch.testing.assert_close(
        second_attention_table,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
    )
    assert not torch.any(second_attention_view == -999.0)
    assert materialized_tables == [
        c128_attention_global_table,
        c128_attention_global_table,
    ]
    assert all(table is not state_execution_table for table in materialized_tables)
