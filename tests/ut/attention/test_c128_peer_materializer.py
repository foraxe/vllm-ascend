# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU contracts for collective-free C128 peer-root materialization."""

import inspect
from dataclasses import replace

import pytest
import torch

from vllm_ascend.attention.context_parallel.c128_packed_owner_route import (
    C128PackedOwnerRoute,
    C128PackedSegment,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    SentinelBlockError,
)
from vllm_ascend.attention.context_parallel.c128_peer_materializer import (
    C128GlobalCurrentRowMapping,
    C128PeerMaterializer,
    C128PeerTensorRoots,
    compile_c128_current_row_overlay,
    compile_c128_device_current_row_overlay,
    compile_c128_peer_materialization,
    plan_c128_current_row_overlay,
    plan_c128_peer_materialization,
)

pytestmark = pytest.mark.cpu_test

_TOKENS_PER_PAGE = 8
_ROW_WIDTH = 2


def _route(
    *,
    tp_size: int = 4,
    logical_start: int = 18,
    logical_stop: int = 34,
    scratch_pages: int = 16,
) -> C128PackedOwnerRoute:
    page_size_bytes = _TOKENS_PER_PAGE * _ROW_WIDTH * torch.empty((), dtype=torch.float32).element_size()
    prototype = C128PackedOwnerRoute(
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
                allocated_bytes=(logical_stop - logical_start + 1) * page_size_bytes,
            )
            for rank in range(tp_size)
        ),
        scratch_segments=tuple(
            C128PackedSegment(
                rank=rank,
                base_bytes=(logical_stop - logical_start + 1) * page_size_bytes,
                allocated_bytes=scratch_pages * page_size_bytes,
            )
            for rank in range(tp_size)
        ),
        max_scratch_pages=scratch_pages,
    )
    persistent_segments = tuple(
        C128PackedSegment(
            rank=rank,
            base_bytes=0,
            allocated_bytes=(prototype.required_persistent_pages(rank) * page_size_bytes),
        )
        for rank in range(tp_size)
    )
    return C128PackedOwnerRoute(
        schema_version=prototype.schema_version,
        runtime_abi_ready=True,
        tp_size=tp_size,
        group_index=prototype.group_index,
        group_name=prototype.group_name,
        component_name=prototype.component_name,
        layer_name=prototype.layer_name,
        copy_index=prototype.copy_index,
        logical_start=logical_start,
        logical_stop=logical_stop,
        page_size_bytes=page_size_bytes,
        allocation_granularity_bytes=page_size_bytes,
        persistent_segments=persistent_segments,
        scratch_segments=tuple(
            C128PackedSegment(
                rank=rank,
                base_bytes=persistent_segments[rank].allocated_bytes,
                allocated_bytes=scratch_pages * page_size_bytes,
            )
            for rank in range(tp_size)
        ),
        max_scratch_pages=scratch_pages,
    )


def _roots(
    route: C128PackedOwnerRoute,
    *,
    value_offset: float = 0.0,
) -> C128PeerTensorRoots:
    roots = tuple(
        torch.full(
            (
                route.required_persistent_pages(owner_rank),
                _TOKENS_PER_PAGE,
                _ROW_WIDTH,
            ),
            -1_000.0 - owner_rank,
        )
        for owner_rank in range(route.tp_size)
    )
    for global_block_id in range(route.logical_start, route.logical_stop):
        address = route.map_global_block(global_block_id)
        roots[address.owner_rank][address.owner_local_slot].fill_(float(global_block_id) + value_offset)
    return C128PeerTensorRoots(roots)


def _scratch(route: C128PackedOwnerRoute) -> torch.Tensor:
    return torch.full(
        (route.max_scratch_pages, _TOKENS_PER_PAGE, _ROW_WIDTH),
        -9_999.0,
    )


def _materializer(
    route: C128PackedOwnerRoute,
    *,
    destination_rank: int = 0,
    value_offset: float = 0.0,
) -> tuple[C128PeerTensorRoots, torch.Tensor, C128PeerMaterializer]:
    roots = _roots(route, value_offset=value_offset)
    scratch = _scratch(route)
    return (
        roots,
        scratch,
        C128PeerMaterializer(
            route=route,
            destination_rank=destination_rank,
            peer_roots=roots,
            scratch=scratch,
        ),
    )


def _global_rows(
    slots: tuple[tuple[int, int], ...],
    *,
    tokens_per_page: int = _TOKENS_PER_PAGE,
) -> C128GlobalCurrentRowMapping:
    return C128GlobalCurrentRowMapping(
        tokens_per_page=tokens_per_page,
        slots=slots,
    )


def test_exhaustive_global_owner_slot_peer_mapping_and_dense_table() -> None:
    route = _route()
    roots, scratch, materializer = _materializer(route, destination_rank=2)
    global_ids = tuple(range(route.logical_start, route.logical_stop))
    table = (
        global_ids[:8] + (0, -1),
        global_ids[8:] + (global_ids[0], global_ids[-1]),
    )

    plan = plan_c128_peer_materialization(
        route,
        table,
        destination_rank=2,
    )

    assert plan.selected_global_block_ids == global_ids
    bindings = {entry.global_block_id: entry for entry in plan.page_bindings}
    for global_block_id in global_ids:
        address = route.map_global_block(global_block_id)
        binding = bindings[global_block_id]
        assert binding.owner_rank == address.owner_rank
        assert binding.owner_local_slot == address.owner_local_slot
        assert torch.equal(
            roots.tensor_for_owner(binding.owner_rank)[binding.owner_local_slot],
            torch.full(
                (_TOKENS_PER_PAGE, _ROW_WIDTH),
                float(global_block_id),
            ),
        )

    compiled = compile_c128_peer_materialization(
        route,
        plan,
        device=scratch.device,
    )
    view = materializer.materialize(compiled)
    assert view.cache is scratch
    assert tuple(view.cache.shape) == tuple(scratch.shape)
    expected_live_pages = torch.stack(
        [
            torch.full(
                (_TOKENS_PER_PAGE, _ROW_WIDTH),
                float(global_block_id),
            )
            for global_block_id in global_ids
        ]
    )
    torch.testing.assert_close(view.cache[: len(global_ids)], expected_live_pages)
    torch.testing.assert_close(
        view.block_table,
        torch.tensor(
            (
                tuple(range(8)) + (-1, -1),
                tuple(range(8, 16)) + (0, 15),
            ),
            dtype=torch.int32,
        ),
    )


@pytest.mark.parametrize(
    ("block_table", "error", "message"),
    [
        (((18, -2),), ValueError, "outside"),
        (((17,),), ValueError, "outside"),
        (((34,),), ValueError, "outside"),
        (((18, True),), ValueError, "must be an integer"),
        (((18,), (19, 20)), ValueError, "equal lengths"),
    ],
)
def test_materialization_planning_fails_closed(
    block_table: tuple[tuple[object, ...], ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        plan_c128_peer_materialization(
            _route(),
            block_table,
            destination_rank=0,
        )


def test_materialization_rejects_scratch_overflow() -> None:
    route = _route(scratch_pages=2)
    with pytest.raises(ValueError, match="needs 3 scratch pages"):
        plan_c128_peer_materialization(
            route,
            ((18, 19, 20),),
            destination_rank=0,
        )


def test_peer_roots_and_runtime_views_fail_closed() -> None:
    route = _route()
    roots = _roots(route)
    with pytest.raises(ValueError, match="cannot represent two"):
        C128PeerTensorRoots((roots.roots[0], roots.roots[0]))
    with pytest.raises(ValueError, match="TP size"):
        C128PeerMaterializer(
            route=route,
            destination_rank=0,
            peer_roots=C128PeerTensorRoots(roots.roots[:-1]),
            scratch=_scratch(route),
        )
    short_roots = list(roots.roots)
    short_roots[0] = short_roots[0][:-1]
    with pytest.raises(ValueError, match="needs"):
        C128PeerMaterializer(
            route=route,
            destination_rank=0,
            peer_roots=C128PeerTensorRoots(tuple(short_roots)),
            scratch=_scratch(route),
        )


def test_current_rows_overlay_remote_and_local_pages_after_history_copy() -> None:
    route = _route()
    _, scratch, materializer = _materializer(route, destination_rank=0)
    plan = plan_c128_peer_materialization(
        route,
        ((18, 19, 22),),
        destination_rank=0,
    )
    # This is an explicit scheduler-global mapping. Global page 19 is owned
    # by rank 3 and would be PAD in rank 0's translated write slot mapping.
    global_current_row_mapping = C128GlobalCurrentRowMapping.from_pre_translation_flat_slots(
        (18 * _TOKENS_PER_PAGE + 2, 19 * _TOKENS_PER_PAGE + 5, -1, 22 * _TOKENS_PER_PAGE + 7),
        tokens_per_page=_TOKENS_PER_PAGE,
    )
    overlay = plan_c128_current_row_overlay(
        route,
        plan,
        global_current_row_mapping,
    )
    current_rows = torch.tensor([[1802.0, 1802.5], [1905.0, 1905.5], [-1.0, -1.0], [2207.0, 2207.5]])
    compiled_plan = compile_c128_peer_materialization(
        route,
        plan,
        device=scratch.device,
    )
    compiled_overlay = compile_c128_current_row_overlay(
        route,
        overlay,
        device=scratch.device,
    )
    view = materializer.materialize(
        compiled_plan,
        overlay=compiled_overlay,
        current_rows=current_rows,
    )

    expected = torch.stack([torch.full((_TOKENS_PER_PAGE, _ROW_WIDTH), float(block_id)) for block_id in (18, 19, 22)])
    expected[0, 2] = current_rows[0]
    expected[1, 5] = current_rows[1]
    expected[2, 7] = current_rows[3]
    torch.testing.assert_close(view.cache[:3], expected)
    assert scratch[1, 5, 0] == 1905.0

    _, device_scratch, device_materializer = _materializer(route)
    device_mapping = torch.tensor(
        ((20, 0), (19, 5), (-1, -1), (22, 7)),
        dtype=torch.int32,
    )
    compiled_device_overlay = compile_c128_device_current_row_overlay(
        route,
        plan,
        compiled_plan,
        overlay,
        device_mapping,
    )
    # A mismatched value at compile time and mutation after compile have no
    # authority over the certificate's fixed scratch destinations.
    device_mapping.fill_(-1)
    device_view = device_materializer.materialize(
        compiled_plan,
        overlay=compiled_device_overlay,
        current_rows=current_rows,
    )
    torch.testing.assert_close(device_view.cache[:3], expected)
    assert device_view.cache is device_scratch


def test_compiled_indices_are_reused_across_layer_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_0 = _route()
    route_1 = replace(
        route_0,
        layer_name="model.layers.1.self_attn",
        copy_index=1,
    )
    plan = plan_c128_peer_materialization(
        route_0,
        ((18, 19, 22),),
        destination_rank=0,
    )
    overlay = plan_c128_current_row_overlay(
        route_0,
        plan,
        _global_rows(((18, 0), (19, 1), (22, 2))),
    )
    compiled_plan = compile_c128_peer_materialization(
        route_0,
        plan,
        device="cpu",
    )
    device_global_mapping = torch.tensor(
        ((18, 0), (19, 1), (22, 2)),
        dtype=torch.int32,
    )
    compiled_device_overlay = compile_c128_device_current_row_overlay(
        route_0,
        plan,
        compiled_plan,
        overlay,
        device_global_mapping,
    )
    compiled_tensor_ids = (
        id(compiled_plan.scratch_local_block_table),
        id(compiled_plan.selected_global_block_ids_tensor),
        *(id(batch.owner_local_slots) for batch in compiled_plan.owner_batches),
        *(id(batch.scratch_slots) for batch in compiled_plan.owner_batches),
        id(compiled_device_overlay.source_rows),
        id(compiled_device_overlay.global_current_row_mapping),
    )
    _, scratch_0, materializer_0 = _materializer(route_0)
    _, scratch_1, materializer_1 = _materializer(
        route_1,
        value_offset=1_000.0,
    )
    current_rows = torch.arange(6, dtype=torch.float32).view(3, 2)

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("per-layer device metadata upload was attempted")

    monkeypatch.setattr(torch, "tensor", _forbidden)
    materializer_0.materialize(
        compiled_plan,
        overlay=compiled_device_overlay,
        current_rows=current_rows,
    )
    materializer_1.materialize(
        compiled_plan,
        overlay=compiled_device_overlay,
        current_rows=current_rows,
    )
    monkeypatch.undo()

    assert compiled_tensor_ids == (
        id(compiled_plan.scratch_local_block_table),
        id(compiled_plan.selected_global_block_ids_tensor),
        *(id(batch.owner_local_slots) for batch in compiled_plan.owner_batches),
        *(id(batch.scratch_slots) for batch in compiled_plan.owner_batches),
        id(compiled_device_overlay.source_rows),
        id(compiled_device_overlay.global_current_row_mapping),
    )
    assert compiled_device_overlay.global_current_row_mapping is device_global_mapping
    # Current rows overwrite three positions equally; every other historical
    # row proves that the two layer caches came from distinct peer roots.
    torch.testing.assert_close(scratch_1[0, 1:], scratch_0[0, 1:] + 1_000.0)


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        (((0, 0),), SentinelBlockError, "sentinel"),
        (((-1, 0),), ValueError, "partially padded"),
        (((18, -1),), ValueError, "partially padded"),
        (((18, 8),), ValueError, "outside"),
        (((20, 0),), ValueError, "absent"),
        (((18, 0), (18, 0)), ValueError, "twice"),
    ],
)
def test_current_row_overlay_mapping_fails_closed(
    mapping: tuple[tuple[int, int], ...],
    error: type[Exception],
    message: str,
) -> None:
    route = _route()
    plan = plan_c128_peer_materialization(
        route,
        ((18, 19),),
        destination_rank=0,
    )
    with pytest.raises(error, match=message):
        plan_c128_current_row_overlay(
            route,
            plan,
            _global_rows(mapping),
        )


def test_overlay_rejects_ordinary_translated_slot_mapping() -> None:
    route = _route()
    plan = plan_c128_peer_materialization(
        route,
        ((18, 19),),
        destination_rank=0,
    )
    with pytest.raises(TypeError, match="explicit pre-translation global"):
        plan_c128_current_row_overlay(
            route,
            plan,
            ((18, 0), (19, 0)),  # type: ignore[arg-type]
        )


def test_global_current_rows_capture_flat_worker_slots_before_translation() -> None:
    mapping = C128GlobalCurrentRowMapping.from_pre_translation_flat_slots(
        (18 * 128 + 7, -1, 19 * 128 + 3),
        tokens_per_page=128,
    )
    assert mapping.slots == ((18, 7), (-1, -1), (19, 3))
    with pytest.raises(ValueError, match="below padding"):
        C128GlobalCurrentRowMapping.from_pre_translation_flat_slots(
            (-2,),
            tokens_per_page=128,
        )


def test_invalid_compiled_inputs_do_not_mutate_scratch() -> None:
    route = _route()
    _, scratch, materializer = _materializer(route)
    plan = plan_c128_peer_materialization(route, ((18, 19),), destination_rank=0)
    compiled_plan = compile_c128_peer_materialization(route, plan, device="cpu")
    before = scratch.clone()

    with pytest.raises(ValueError, match="provided together"):
        materializer.materialize(
            compiled_plan,
            current_rows=torch.zeros(1, _ROW_WIDTH),
        )
    torch.testing.assert_close(scratch, before)

    with pytest.raises(ValueError, match="exceeds concrete scratch"):
        materializer.materialize(replace(compiled_plan, staged_pages=-1))
    torch.testing.assert_close(scratch, before)


def test_remote_root_cannot_exceed_its_packed_segment() -> None:
    route = _route()
    roots = list(_roots(route).roots)
    roots[1] = torch.cat((roots[1], roots[1][:1]), dim=0)
    with pytest.raises(ValueError, match="beyond its packed persistent segment"):
        C128PeerMaterializer(
            route=route,
            destination_rank=0,
            peer_roots=C128PeerTensorRoots(tuple(roots)),
            scratch=_scratch(route),
        )


def test_materializer_snapshots_provider_roots_at_construction() -> None:
    route = _route()
    stable_roots = _roots(route).roots

    class StartupOnlyProvider:
        tp_size = route.tp_size

        def __init__(self) -> None:
            self.calls = 0

        def tensor_for_owner(self, owner_rank: int) -> torch.Tensor:
            self.calls += 1
            if self.calls > self.tp_size:
                raise AssertionError("peer roots were reacquired in hot path")
            return stable_roots[owner_rank]

    provider = StartupOnlyProvider()
    scratch = _scratch(route)
    materializer = C128PeerMaterializer(
        route=route,
        destination_rank=0,
        peer_roots=provider,
        scratch=scratch,
    )
    plan = plan_c128_peer_materialization(route, ((18, 19),), destination_rank=0)
    compiled = compile_c128_peer_materialization(route, plan, device="cpu")
    materializer.materialize(compiled)
    assert provider.calls == route.tp_size


def test_5120_plus_3080_current_rows_match_replicated_continuation_oracle() -> None:
    """Two chunk overlays preserve old rows and replace new rows in order."""
    prefix_rows = 5_120 // 128
    # The 3080-token tail produces 24 complete persistent C128 rows; the
    # remaining eight tokens stay in compressor state for the continuation.
    tail_rows = 3_080 // 128
    total_rows = prefix_rows + tail_rows
    # Use one 128-row cache page for this interface oracle. The real C128
    # tensor has the same row-addressing contract and wider rows.
    assert total_rows == 64
    page = torch.zeros(128, 1)
    page_size_bytes = page[0].numel() * page.element_size() * page.shape[0]
    route = C128PackedOwnerRoute(
        schema_version=1,
        runtime_abi_ready=True,
        tp_size=1,
        group_index=1,
        group_name="c128",
        component_name="compress_kv",
        layer_name="model.layers.0.self_attn",
        copy_index=0,
        logical_start=18,
        logical_stop=19,
        page_size_bytes=page_size_bytes,
        allocation_granularity_bytes=page_size_bytes,
        persistent_segments=(C128PackedSegment(0, 0, 2 * page_size_bytes),),
        scratch_segments=(C128PackedSegment(0, 2 * page_size_bytes, page_size_bytes),),
        max_scratch_pages=1,
    )
    peer_page = torch.zeros(2, 128, 1)
    roots = C128PeerTensorRoots((peer_page,))
    scratch = torch.full((1, 128, 1), -1.0)
    materializer = C128PeerMaterializer(
        route=route,
        destination_rank=0,
        peer_roots=roots,
        scratch=scratch,
    )
    plan = plan_c128_peer_materialization(route, ((18,),), destination_rank=0)
    compiled_plan = compile_c128_peer_materialization(
        route,
        plan,
        device=scratch.device,
    )
    replicated = torch.zeros(1, 128, 1)

    prefix_mapping = tuple((18, row) for row in range(prefix_rows))
    prefix_values = torch.arange(1, prefix_rows + 1, dtype=torch.float32).view(-1, 1)
    prefix_overlay = plan_c128_current_row_overlay(
        route,
        plan,
        _global_rows(prefix_mapping, tokens_per_page=128),
    )
    compiled_prefix_overlay = compile_c128_device_current_row_overlay(
        route,
        plan,
        compiled_plan,
        prefix_overlay,
        torch.tensor(prefix_mapping, dtype=torch.int32),
    )
    materializer.materialize(
        compiled_plan,
        overlay=compiled_prefix_overlay,
        current_rows=prefix_values,
    )
    replicated[0, :prefix_rows] = prefix_values
    torch.testing.assert_close(scratch, replicated)

    # Owner persistence becomes the next historical peer snapshot.
    peer_page[1].copy_(scratch[0])
    scratch.fill_(-1.0)
    tail_mapping = tuple((18, prefix_rows + row) for row in range(tail_rows))
    tail_values = torch.arange(
        prefix_rows + 1,
        total_rows + 1,
        dtype=torch.float32,
    ).view(-1, 1)
    tail_overlay = plan_c128_current_row_overlay(
        route,
        plan,
        _global_rows(tail_mapping, tokens_per_page=128),
    )
    compiled_tail_overlay = compile_c128_device_current_row_overlay(
        route,
        plan,
        compiled_plan,
        tail_overlay,
        torch.tensor(tail_mapping, dtype=torch.int32),
    )
    materializer.materialize(
        compiled_plan,
        overlay=compiled_tail_overlay,
        current_rows=tail_values,
    )
    replicated[0, prefix_rows:total_rows] = tail_values
    torch.testing.assert_close(scratch, replicated)


def test_request_execution_uses_no_collective_sync_or_device_value_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    _, scratch, materializer = _materializer(route)
    plan = plan_c128_peer_materialization(
        route,
        ((18, 19, 22),),
        destination_rank=0,
    )
    overlay = plan_c128_current_row_overlay(
        route,
        plan,
        _global_rows(((18, 0), (19, 1), (22, 2))),
    )
    current_rows = torch.arange(6, dtype=torch.float32).view(3, 2)
    compiled_plan = compile_c128_peer_materialization(
        route,
        plan,
        device=scratch.device,
    )
    device_global_mapping = torch.tensor(
        ((18, 0), (19, 1), (22, 2)),
        dtype=torch.int32,
        device=scratch.device,
    )
    compiled_device_overlay = compile_c128_device_current_row_overlay(
        route,
        plan,
        compiled_plan,
        overlay,
        device_global_mapping,
    )

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("request-time collective or host sync was called")

    monkeypatch.setattr(torch.distributed, "all_gather", _forbidden)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", _forbidden)
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        monkeypatch.setattr(torch.npu, "synchronize", _forbidden)
    monkeypatch.setattr(torch.Tensor, "item", _forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbidden)
    monkeypatch.setattr(torch, "tensor", _forbidden)
    monkeypatch.setattr(torch, "arange", _forbidden)

    view = materializer.materialize(
        compiled_plan,
        overlay=compiled_device_overlay,
        current_rows=current_rows,
    )
    assert view.cache is scratch


def test_execution_source_contains_no_forbidden_request_path_calls() -> None:
    for implementation in (
        C128PeerMaterializer.materialize,
        C128PeerMaterializer._overlay_current_rows,
        C128PeerMaterializer._overlay_device_current_rows,
    ):
        source = inspect.getsource(implementation)
        for forbidden in (
            "dist.all_gather(",
            "all_to_all_single(",
            "torch.npu.synchronize(",
            ".item(",
            ".tolist(",
            "torch.tensor(",
            "torch.empty(",
            "torch.arange(",
            "masked_select(",
        ):
            assert forbidden not in source
