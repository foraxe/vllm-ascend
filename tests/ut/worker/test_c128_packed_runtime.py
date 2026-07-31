# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only tests for the packed-arena worker lifecycle contract."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_arena import (
    CANN_VMM_GRANULARITY_BYTES,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
    PackedPoolScratchSpec,
)
from vllm_ascend.worker.c128_packed_runtime import (
    C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
    C128_PACKED_POOL_COMPONENT_SIGNATURES,
    C128_PACKED_POOL_GROUP_IDENTITIES,
    C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS,
    C128_PACKED_POOL_TOTAL_BYTES_PER_RANK,
    PackedArenaMetadataError,
    PackedArenaRuntime,
    PackedArenaRuntimeCleanupError,
    PackedArenaRuntimeClosedError,
    PackedArenaRuntimeState,
    maybe_open_packed_arena_runtime,
    packed_arena_contract_from_metadata,
    validate_c128_packed_startup_contract,
)

pytestmark = pytest.mark.cpu_test

PAGE_BYTES = 128 * 1024
REPLICATED_VIEW_KEY = "component/group_0/group_0_component_0/0"
OWNER_VIEW_KEY_0 = "component/group_1/group_1_component_0/0"
OWNER_VIEW_KEY_1 = "component/group_1/group_1_component_0/1"
SCRATCH_VIEW_KEY = "scratch/wide"
EXPECTED_VIEW_KEYS = (
    REPLICATED_VIEW_KEY,
    OWNER_VIEW_KEY_0,
    OWNER_VIEW_KEY_1,
    SCRATCH_VIEW_KEY,
)


def _production_startup_contract():
    groups = tuple(
        SimpleNamespace(
            logical_blocks=logical_blocks,
            components=tuple(
                SimpleNamespace(
                    bucket=bucket,
                    page_size_bytes=page_size_bytes,
                    copies=copies,
                    placement=placement,
                )
                for (
                    bucket,
                    page_size_bytes,
                    copies,
                    placement,
                ) in component_signatures
            ),
        )
        for logical_blocks, component_signatures in zip(
            C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
            C128_PACKED_POOL_COMPONENT_SIGNATURES,
        )
    )
    plan = SimpleNamespace(
        global_block_capacity=4_190,
        groups=groups,
        scratch=(
            SimpleNamespace(
                bucket="page_131072",
                max_pages_per_rank=65,
            ),
        ),
    )
    expected_views = tuple(
        SimpleNamespace(
            key=f"component/group_{group_index}/component_{component_index}/{copy_index}",
            bucket=bucket,
        )
        for group_index, component_signatures in enumerate(C128_PACKED_POOL_COMPONENT_SIGNATURES)
        for component_index, (
            bucket,
            _page_size_bytes,
            copies,
            _placement,
        ) in enumerate(component_signatures)
        for copy_index in range(copies)
    ) + (
        SimpleNamespace(
            key="scratch/page_131072",
            bucket="page_131072",
        ),
    )
    narrow = SimpleNamespace(
        bucket="page_16640",
        persistent_allocated_bytes=308_281_344,
        scratch_region_bytes=0,
        total_allocated_bytes=308_281_344,
    )
    wide = SimpleNamespace(
        bucket="page_131072",
        persistent_allocated_bytes=3_716_153_344,
        scratch_region_bytes=10_485_760,
        total_allocated_bytes=3_726_639_104,
    )
    rank_accounting = tuple(
        SimpleNamespace(
            total_allocated_bytes=C128_PACKED_POOL_TOTAL_BYTES_PER_RANK,
            buckets=(narrow, wide),
        )
        for _ in range(8)
    )
    metadata_groups = [
        {
            "group_index": index,
            "name": f"group_{index}",
            "identity": identity,
            "required_logical_blocks": required,
            "assigned_logical_blocks": assigned,
        }
        for index, (identity, required, assigned) in enumerate(
            zip(
                C128_PACKED_POOL_GROUP_IDENTITIES,
                C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS,
                C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
            )
        )
    ]
    scheduler_identities = [
        {
            "group_index": index,
            "group_name": f"group_{index}",
            "identity": identity,
            "required_blocks": required,
            "assigned_blocks": assigned,
        }
        for index, (identity, required, assigned) in enumerate(
            zip(
                C128_PACKED_POOL_GROUP_IDENTITIES,
                C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS,
                C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS,
            )
        )
    ]
    contract = SimpleNamespace(
        plan=plan,
        expected_views=expected_views,
        rank_accounting=rank_accounting,
    )
    metadata = {
        "expert_parallel_size": 8,
        "required_group_block_quotas": list(C128_PACKED_POOL_REQUIRED_GROUP_QUOTAS),
        "assigned_group_block_quotas": list(C128_PACKED_POOL_ASSIGNED_GROUP_QUOTAS),
        "scheduler_group_identities": scheduler_identities,
        "groups": metadata_groups,
    }
    return contract, metadata


def test_production_startup_contract_pins_manifest_and_bytes() -> None:
    contract, metadata = _production_startup_contract()

    assert validate_c128_packed_startup_contract(contract, metadata) is contract
    assert len(contract.expected_views) == 168
    assert all(accounting.total_allocated_bytes == 4_034_920_448 for accounting in contract.rank_accounting)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda _contract, metadata: metadata.__setitem__(
                "required_group_block_quotas",
                [17, 2, 42, 42, 642, 165],
            ),
            "required_group_block_quotas",
        ),
        (
            lambda _contract, metadata: metadata.__setitem__(
                "assigned_group_block_quotas",
                [17, 3_280, 42, 42, 642, 166],
            ),
            "assigned_group_block_quotas",
        ),
        (
            lambda _contract, metadata: metadata.__setitem__(
                "expert_parallel_size",
                4,
            ),
            "expert_parallel_size",
        ),
        (
            lambda _contract, metadata: metadata["groups"][1].__setitem__(
                "identity",
                "wrong",
            ),
            r"groups\[1\]\.identity",
        ),
        (
            lambda contract, _metadata: setattr(
                contract.plan.groups[1].components[0],
                "page_size_bytes",
                65_536,
            ),
            "component signature",
        ),
        (
            lambda contract, _metadata: setattr(
                contract.rank_accounting[0],
                "total_allocated_bytes",
                4_215_275_519,
            ),
            "total_allocated_bytes",
        ),
    ],
)
def test_production_startup_contract_rejects_semantic_drift(
    mutation,
    message: str,
) -> None:
    contract, metadata = _production_startup_contract()
    mutation(contract, metadata)

    with pytest.raises(PackedArenaMetadataError, match=message):
        validate_c128_packed_startup_contract(contract, metadata)


def _plan() -> PackedPoolPlan:
    return PackedPoolPlan(
        global_block_capacity=16,
        tp_size=8,
        groups=(
            PackedPoolGroupSpec(
                name="group_0",
                logical_blocks=5,
                components=(
                    PackedPoolComponentSpec(
                        name="group_0_component_0",
                        bucket="wide",
                        page_size_bytes=PAGE_BYTES,
                        copies=1,
                        placement=PackedPlacement.REPLICATED,
                        allocation_granularity_bytes=(CANN_VMM_GRANULARITY_BYTES),
                    ),
                ),
            ),
            PackedPoolGroupSpec(
                name="group_1",
                logical_blocks=7,
                components=(
                    PackedPoolComponentSpec(
                        name="group_1_component_0",
                        bucket="wide",
                        page_size_bytes=PAGE_BYTES,
                        copies=2,
                        placement=PackedPlacement.C128_OWNER,
                        allocation_granularity_bytes=(CANN_VMM_GRANULARITY_BYTES),
                    ),
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=PAGE_BYTES,
                max_pages_per_rank=3,
                allocation_granularity_bytes=CANN_VMM_GRANULARITY_BYTES,
            ),
        ),
    )


def _metadata(
    *,
    runtime_ready: bool = True,
) -> dict[str, object]:
    plan = _plan()
    group_layers = {
        "group_0_component_0": ["model.layers.0.attn"],
        "group_1_component_0": [
            "model.layers.1.attn",
            "model.layers.2.attn",
        ],
    }
    groups = []
    for group_index, group in enumerate(plan.groups):
        logical_range = plan.group_range(group.name)
        components = []
        all_layers: list[str] = []
        for component in group.components:
            layer_names = group_layers[component.name]
            all_layers.extend(layer_names)
            segments = []
            for copy_index in range(component.copies):
                ranks = []
                for rank in range(plan.tp_size):
                    sentinel = plan.sentinel_address(
                        group.name,
                        component.name,
                        tp_rank=rank,
                        copy_index=copy_index,
                    )
                    ranks.append(
                        {
                            "rank": rank,
                            "segment_base_bytes": (sentinel.segment_base_bytes),
                            "segment_allocated_bytes": (sentinel.segment_allocated_bytes),
                            "sentinel_offset_bytes": (sentinel.physical_offset_bytes),
                        }
                    )
                segments.append(
                    {
                        "copy_index": copy_index,
                        "ranks": ranks,
                    }
                )
            components.append(
                {
                    "name": component.name,
                    "bucket": component.bucket,
                    "page_size_bytes": component.page_size_bytes,
                    "copies": component.copies,
                    "placement": component.placement.value,
                    "allocation_granularity_bytes": (component.allocation_granularity_bytes),
                    "layer_names": layer_names,
                    "segments": segments,
                }
            )
        groups.append(
            {
                "group_index": group_index,
                "name": group.name,
                "logical_blocks": group.logical_blocks,
                "logical_start": logical_range.start,
                "logical_stop": logical_range.stop,
                "layer_names": all_layers,
                "scheduler_shape": {
                    "partition_blocks": group.logical_blocks,
                },
                "components": components,
            }
        )

    buckets = [
        {
            "bucket": item.bucket,
            "page_size_bytes": item.page_size_bytes,
            "replicated_pages_per_rank": item.replicated_pages_per_rank,
            "owner_pages_by_rank": list(item.owner_pages_by_rank),
            "sentinel_pages_per_rank": item.sentinel_pages_per_rank,
            "scratch_pages_per_rank": item.scratch_pages_per_rank,
            "persistent_allocated_bytes_by_rank": list(item.persistent_allocated_bytes_by_rank),
            "scratch_region_bytes_by_rank": list(item.scratch_region_bytes_by_rank),
            "total_allocated_bytes_by_rank": list(item.total_allocated_bytes_by_rank),
        }
        for item in plan.bucket_accounting
    ]
    accounting = plan.bucket_accounting[0]
    scratch_bytes = (
        (3 * PAGE_BYTES + CANN_VMM_GRANULARITY_BYTES - 1) // CANN_VMM_GRANULARITY_BYTES * CANN_VMM_GRANULARITY_BYTES
    )
    scratch = [
        {
            "bucket": "wide",
            "page_size_bytes": PAGE_BYTES,
            "max_pages_per_rank": 3,
            "allocation_granularity_bytes": CANN_VMM_GRANULARITY_BYTES,
            "segments": [
                {
                    "rank": rank,
                    "segment_base_bytes": (accounting.total_allocated_bytes_by_rank[rank] - scratch_bytes),
                    "segment_allocated_bytes": scratch_bytes,
                }
                for rank in range(plan.tp_size)
            ],
        }
    ]
    return {
        "schema_version": 1,
        "profile": "dsv4_flash_prefill_8200_tokens_1out",
        "planner_only": not runtime_ready,
        "downstream_runtime_abi_ready": runtime_ready,
        "prompt_tokens": 8_200,
        "output_tokens": 1,
        "kv_slot_tokens": 8_200,
        "max_model_len": 8_201,
        "max_concurrent_requests": 1,
        "max_num_batched_tokens": 5_120,
        "max_num_scheduled_tokens": 5_120,
        "max_num_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "tp_size": 8,
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
        "aligned_quota_replicated_bytes_by_rank": list(plan.aligned_quota_replicated_bytes_by_rank()),
    }


class _FakeBackend:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.next_base = 0x1_0000_0000

    def allocation_granularity(self, *, device_index: int) -> int:
        self.events.append(("granularity", device_index))
        return CANN_VMM_GRANULARITY_BYTES

    def reserve_address(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
        device_index: int,
    ) -> int:
        base = self.next_base
        self.next_base += 0x1_0000_0000
        self.events.append(
            (
                "reserve",
                base,
                size_bytes,
                alignment_bytes,
                device_index,
            )
        )
        return base

    def allocate_physical(
        self,
        *,
        size_bytes: int,
        device_index: int,
    ) -> object:
        handle = object()
        self.events.append(("allocate", handle, size_bytes, device_index))
        return handle

    def map_physical(
        self,
        *,
        base_address: int,
        size_bytes: int,
        physical_handle: object,
        device_index: int,
    ) -> None:
        self.events.append(
            (
                "map",
                base_address,
                size_bytes,
                physical_handle,
                device_index,
            )
        )

    def zero_mapped(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self.events.append(("zero", base_address, size_bytes, device_index))

    def unmap(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self.events.append(("unmap", base_address, size_bytes, device_index))

    def free_physical(
        self,
        *,
        physical_handle: object,
        device_index: int,
    ) -> None:
        self.events.append(("free", physical_handle, device_index))

    def release_address(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> None:
        self.events.append(("release", base_address, size_bytes, device_index))


class _FakeBinding:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        token: object,
    ) -> None:
        self.events = events
        self.token = token

    def tensor(self) -> object:
        return self.token

    def close(self) -> None:
        self.events.append(("close_binding",))


class _FakeTensorFactory:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.token = object()

    def bind(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> _FakeBinding:
        self.events.append(("bind", base_address, size_bytes, device_index))
        return _FakeBinding(self.events, self.token)


def _open_runtime(
    events: list[tuple[object, ...]],
    *,
    quiesce: Callable[[], None] | None = None,
) -> PackedArenaRuntime:
    return PackedArenaRuntime.open_from_metadata(
        metadata=_metadata(),
        tp_rank=3,
        device_index=3,
        backend=_FakeBackend(events),
        tensor_factory=_FakeTensorFactory(events),
        arena_fence=lambda: events.append(("arena_fence",)),
        quiesce=(quiesce if quiesce is not None else lambda: events.append(("quiesce",))),
    )


def test_disabled_path_is_inert() -> None:
    class UnexpectedMapping(dict):
        def get(self, key: object, default: object = None) -> object:
            raise AssertionError("disabled path inspected metadata")

    assert (
        maybe_open_packed_arena_runtime(
            enabled=False,
            metadata=UnexpectedMapping(),
        )
        is None
    )


def test_planner_only_metadata_fails_closed_before_backend_use() -> None:
    events: list[tuple[object, ...]] = []
    with pytest.raises(
        PackedArenaMetadataError,
        match="planner_only=false",
    ):
        PackedArenaRuntime.open_from_metadata(
            metadata=_metadata(runtime_ready=False),
            tp_rank=0,
            device_index=0,
            backend=_FakeBackend(events),
            tensor_factory=_FakeTensorFactory(events),
            arena_fence=lambda: None,
            quiesce=lambda: None,
        )
    assert events == []


def test_planner_only_metadata_can_be_inspected_without_activation() -> None:
    contract = packed_arena_contract_from_metadata(
        _metadata(runtime_ready=False),
        require_runtime_ready=False,
    )

    assert contract.plan.used_logical_blocks == 12
    assert contract.accounting_for_rank(0).total_allocated_bytes > 0


def test_serialized_plan_round_trip_and_exact_rank_accounting() -> None:
    contract = packed_arena_contract_from_metadata(_metadata())
    plan = contract.plan
    accounting = contract.accounting_for_rank(3)

    assert plan.tp_size == 8
    assert plan.used_logical_blocks == 12
    assert tuple(view.key for view in contract.expected_views) == (EXPECTED_VIEW_KEYS)
    assert accounting.total_allocated_bytes == (plan.total_physical_bytes_by_rank()[3])
    assert accounting.as_metadata() == {
        "tp_rank": 3,
        "persistent_allocated_bytes": 6_291_456,
        "scratch_region_bytes": 2_097_152,
        "total_allocated_bytes": 8_388_608,
        "buckets": [
            {
                "bucket": "wide",
                "page_size_bytes": PAGE_BYTES,
                "persistent_allocated_bytes": 6_291_456,
                "scratch_region_bytes": 2_097_152,
                "total_allocated_bytes": 8_388_608,
            }
        ],
    }


def test_segment_tamper_is_rejected() -> None:
    metadata = deepcopy(_metadata())
    groups = metadata["groups"]
    assert isinstance(groups, list)
    component = groups[0]["components"][0]
    component["segments"][0]["ranks"][0]["segment_base_bytes"] += PAGE_BYTES

    with pytest.raises(
        PackedArenaMetadataError,
        match="segment_base_bytes",
    ):
        packed_arena_contract_from_metadata(metadata)


def test_integer_fields_reject_json_booleans() -> None:
    metadata = _metadata()
    metadata["schema_version"] = True

    with pytest.raises(
        PackedArenaMetadataError,
        match="schema_version",
    ):
        packed_arena_contract_from_metadata(metadata)


def test_aligned_quota_replicated_accounting_tamper_is_rejected() -> None:
    metadata = _metadata()
    aligned_bytes = metadata["aligned_quota_replicated_bytes_by_rank"]
    assert isinstance(aligned_bytes, list)
    aligned_bytes[3] += CANN_VMM_GRANULARITY_BYTES

    with pytest.raises(
        PackedArenaMetadataError,
        match="aligned_quota_replicated_bytes_by_rank",
    ):
        packed_arena_contract_from_metadata(metadata)


def test_close_quiesces_and_releases_views_before_lease() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)
    holder: dict[str, object | None] = {"view": None}

    def install_view(tensor: object) -> Callable[[], None]:
        holder["view"] = tensor

        def release_view() -> None:
            events.append(("release_view",))
            holder["view"] = None

        return release_view

    for view_key in EXPECTED_VIEW_KEYS[1:]:
        runtime.install_tensor_views(
            view_key,
            lambda _tensor, key=view_key: (lambda: events.append(("release_other_view", key))),
        )
    runtime.install_tensor_views(REPLICATED_VIEW_KEY, install_view)
    runtime.seal_views()
    runtime.publish()
    assert holder["view"] is not None
    events.clear()

    runtime.close()

    assert holder["view"] is None
    assert runtime.state is PackedArenaRuntimeState.CLOSED
    assert events[0:2] == [("quiesce",), ("release_view",)]
    assert events.index(("release_view",)) < events.index(("close_binding",))
    assert events.index(("close_binding",)) < events.index(("arena_fence",))
    assert events.index(("arena_fence",)) < next(index for index, event in enumerate(events) if event[0] == "unmap")
    with pytest.raises(PackedArenaRuntimeClosedError, match="closed"):
        runtime.install_tensor_views(REPLICATED_VIEW_KEY, install_view)


def test_composite_runtime_closes_views_then_peer_then_local_arena() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)

    class PeerLease:
        def close(self) -> None:
            events.append(("close_peer",))

    peer = runtime.open_peer_lease(lambda _owner: PeerLease())
    assert runtime.peer_lease is peer
    for view_key in EXPECTED_VIEW_KEYS:
        runtime.install_tensor_views(
            view_key,
            lambda _tensor, key=view_key: (
                lambda: events.append(("release_view", key))
            ),
        )
    runtime.seal_views()
    runtime.publish()
    events.clear()

    runtime.close()

    first_local_binding_close = events.index(("close_binding",))
    assert events[0] == ("quiesce",)
    assert max(
        index
        for index, event in enumerate(events)
        if event[0] == "release_view"
    ) < events.index(("close_peer",))
    assert events.index(("close_peer",)) < first_local_binding_close
    assert runtime.peer_lease is None
    assert runtime.state is PackedArenaRuntimeState.CLOSED


def test_composite_runtime_retains_failed_peer_before_local_close() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)

    class RetryPeerLease:
        attempts = 0

        def close(self) -> None:
            self.attempts += 1
            events.append(("close_peer", self.attempts))
            if self.attempts == 1:
                raise RuntimeError("peer cleanup failed once")

    peer = runtime.open_peer_lease(lambda _owner: RetryPeerLease())
    events.clear()

    with pytest.raises(
        PackedArenaRuntimeCleanupError,
        match="peer cleanup failed once",
    ):
        runtime.close()
    assert runtime.peer_lease is peer
    assert not any(event[0] == "close_binding" for event in events)

    runtime.close()

    assert runtime.peer_lease is None
    assert runtime.state is PackedArenaRuntimeState.CLOSED
    assert events.index(("close_peer", 2)) < events.index(("close_binding",))


def test_failed_view_install_does_not_publish_a_lease_pin() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)

    with pytest.raises(ValueError, match="unknown packed-arena view key"):
        runtime.install_tensor_views(
            "missing",
            lambda _tensor: lambda: None,
        )

    runtime.close()
    assert runtime.state is PackedArenaRuntimeState.CLOSED


def test_reentrant_view_install_is_rejected_without_owner_corruption() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)
    nested_error: list[str] = []

    def outer_install(_tensor: object) -> Callable[[], None]:
        try:
            runtime.install_tensor_views(
                OWNER_VIEW_KEY_0,
                lambda _inner: lambda: events.append(("inner_release",)),
            )
        except RuntimeError as error:
            nested_error.append(str(error))
        return lambda: events.append(("outer_release",))

    runtime.install_tensor_views(REPLICATED_VIEW_KEY, outer_install)
    runtime.close()

    assert nested_error == ["packed-arena view installation is not reentrant"]
    assert ("outer_release",) in events
    assert ("inner_release",) not in events
    assert runtime.state is PackedArenaRuntimeState.CLOSED


def test_seal_requires_the_exact_plan_view_manifest() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)

    with pytest.raises(RuntimeError, match="missing views"):
        runtime.seal_views()

    for view_key in EXPECTED_VIEW_KEYS:
        runtime.install_tensor_views(
            view_key,
            lambda _tensor: lambda: None,
        )
    runtime.seal_views()

    assert runtime.state is PackedArenaRuntimeState.SEALED
    runtime.publish()
    assert runtime.state is PackedArenaRuntimeState.PUBLISHED
    with pytest.raises(PackedArenaRuntimeClosedError, match="published"):
        runtime.install_tensor_views(
            REPLICATED_VIEW_KEY,
            lambda _tensor: lambda: None,
        )
    runtime.close()
    assert runtime.state is PackedArenaRuntimeState.CLOSED


def test_publish_rejects_unsealed_and_duplicate_transitions() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)

    with pytest.raises(RuntimeError, match="must be sealed"):
        runtime.publish()

    for view_key in EXPECTED_VIEW_KEYS:
        runtime.install_tensor_views(
            view_key,
            lambda _tensor: lambda: None,
        )
    runtime.seal_views()
    runtime.publish()

    with pytest.raises(RuntimeError, match="must be sealed"):
        runtime.publish()
    runtime.close()


def test_non_atomic_view_install_permanently_blocks_unmap() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)
    leaked: list[object] = []

    def failed_install(tensor: object) -> Callable[[], None]:
        leaked.append(tensor)
        raise RuntimeError("partial install")

    with pytest.raises(RuntimeError, match="partial install"):
        runtime.install_tensor_views(REPLICATED_VIEW_KEY, failed_install)
    assert runtime.state is PackedArenaRuntimeState.CLEANUP_FAILED

    with pytest.raises(
        PackedArenaRuntimeCleanupError,
        match="failure-atomically",
    ):
        runtime.close()

    assert leaked
    assert not any(event[0] == "unmap" for event in events)


def test_view_release_failure_retains_lease_for_retry() -> None:
    events: list[tuple[object, ...]] = []
    runtime = _open_runtime(events)
    attempts = 0
    holder: dict[str, object | None] = {"view": None}

    def release_view() -> None:
        nonlocal attempts
        attempts += 1
        events.append(("release_view", attempts))
        if attempts == 1:
            raise RuntimeError("busy view")
        holder["view"] = None

    def install_view(tensor: object) -> Callable[[], None]:
        holder["view"] = tensor
        return release_view

    runtime.install_tensor_views(REPLICATED_VIEW_KEY, install_view)
    events.clear()

    with pytest.raises(
        PackedArenaRuntimeCleanupError,
        match="release_view",
    ):
        runtime.close()
    assert runtime.state is PackedArenaRuntimeState.CLEANUP_FAILED
    assert not any(event[0] == "unmap" for event in events)
    assert holder["view"] is not None

    runtime.close()

    assert attempts == 2
    assert events.count(("quiesce",)) == 1
    assert holder["view"] is None
    assert runtime.state is PackedArenaRuntimeState.CLOSED
