# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU-only contract tests for the feature-off packed-pool prototype."""

from itertools import product

import pytest

from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    SENTINEL_BLOCK_ID,
    LogicalBlockRef,
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
    PackedPoolScratchSpec,
    SentinelBlockError,
    translate_group_block_ids,
)

pytestmark = pytest.mark.cpu_test

WIDE_PAGE_BYTES = 128 * 1024
NARROW_PAGE_BYTES = 16_640
VMM_GRANULARITY_BYTES = 2 * 1024 * 1024


def _component(
    name: str,
    *,
    placement: PackedPlacement,
    copies: int = 1,
    bucket: str = "wide",
    page_size_bytes: int = WIDE_PAGE_BYTES,
    allocation_granularity_bytes: int = 1,
) -> PackedPoolComponentSpec:
    return PackedPoolComponentSpec(
        name=name,
        bucket=bucket,
        page_size_bytes=page_size_bytes,
        copies=copies,
        placement=placement,
        allocation_granularity_bytes=allocation_granularity_bytes,
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


def test_group_ranges_are_deterministic_disjoint_and_bounded() -> None:
    groups = (
        _group(
            "prefill",
            3,
            _component("prefill_replicated", placement=PackedPlacement.REPLICATED),
        ),
        _group(
            "decode",
            5,
            _component("decode_replicated", placement=PackedPlacement.REPLICATED),
        ),
        _group(
            "mtp",
            2,
            _component("mtp_replicated", placement=PackedPlacement.REPLICATED),
        ),
    )
    first = PackedPoolPlan(
        global_block_capacity=12,
        tp_size=8,
        groups=groups,
    )
    second = PackedPoolPlan(
        global_block_capacity=12,
        tp_size=8,
        groups=groups,
    )

    expected = (
        ("prefill", 1, 4),
        ("decode", 4, 9),
        ("mtp", 9, 11),
    )
    assert (
        tuple(
            (logical_range.group_name, logical_range.start, logical_range.stop) for logical_range in first.group_ranges
        )
        == expected
    )
    assert first.group_ranges == second.group_ranges
    assert first.used_logical_blocks == 10
    assert first.unused_logical_blocks == 1

    with pytest.raises(ValueError, match="exceeds usable data capacity"):
        PackedPoolPlan(
            global_block_capacity=9,
            tp_size=8,
            groups=groups,
        )


@pytest.mark.parametrize("tp_size,logical_blocks", product((1, 2, 3, 8), (1, 2, 7, 17)))
def test_exhaustive_mapping_is_bijective_and_collision_free(
    tp_size: int,
    logical_blocks: int,
) -> None:
    """Every component/copy/block address has one exact inverse."""
    groups = (
        _group(
            "first",
            logical_blocks,
            _component(
                "first_replicated",
                placement=PackedPlacement.REPLICATED,
                copies=2,
            ),
            _component(
                "first_c128",
                placement=PackedPlacement.C128_OWNER,
                copies=3,
            ),
        ),
        _group(
            "second",
            logical_blocks + 2,
            _component(
                "second_replicated",
                placement=PackedPlacement.REPLICATED,
                copies=2,
            ),
            _component(
                "second_c128",
                placement=PackedPlacement.C128_OWNER,
                copies=2,
            ),
        ),
    )
    plan = PackedPoolPlan(
        global_block_capacity=2 * logical_blocks + 3,
        tp_size=tp_size,
        groups=groups,
    )

    physical_addresses: set[tuple[str, int, int]] = set()
    for group in groups:
        for component in group.components:
            for copy_index in range(component.copies):
                for block_id in range(1, group.logical_blocks + 1):
                    expected = LogicalBlockRef(group.name, block_id)
                    if component.placement is PackedPlacement.REPLICATED:
                        for rank in range(tp_size):
                            address = plan.map_replicated(
                                group.name,
                                component.name,
                                block_id,
                                tp_rank=rank,
                                copy_index=copy_index,
                            )
                            key = (
                                address.bucket,
                                rank,
                                address.physical_offset_bytes,
                            )
                            assert key not in physical_addresses
                            physical_addresses.add(key)
                            assert (
                                address.physical_offset_bytes + component.page_size_bytes
                                <= address.segment_base_bytes + address.segment_allocated_bytes
                            )
                            assert plan.unmap_replicated(
                                group.name,
                                component.name,
                                address.physical_slot,
                                tp_rank=rank,
                                copy_index=copy_index,
                            ) == (expected, copy_index)
                    else:
                        address = plan.map_c128(
                            group.name,
                            component.name,
                            block_id,
                            copy_index=copy_index,
                        )
                        assert address.owner_rank is not None
                        key = (
                            address.bucket,
                            address.owner_rank,
                            address.physical_offset_bytes,
                        )
                        assert key not in physical_addresses
                        physical_addresses.add(key)
                        assert (
                            address.physical_offset_bytes + component.page_size_bytes
                            <= address.segment_base_bytes + address.segment_allocated_bytes
                        )
                        assert plan.unmap_c128(
                            group.name,
                            component.name,
                            address.owner_rank,
                            address.physical_slot,
                            copy_index=copy_index,
                        ) == (expected, copy_index)


def test_tp8_owner_slots_are_dense_with_ceil_tail() -> None:
    total_blocks = 4_190
    logical_blocks = total_blocks - 1
    copies = 2
    plan = PackedPoolPlan(
        global_block_capacity=total_blocks,
        tp_size=8,
        groups=(
            _group(
                "flash",
                logical_blocks,
                _component(
                    "c128",
                    placement=PackedPlacement.C128_OWNER,
                    copies=copies,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
        ),
    )
    bucket = plan.bucket_accounting[0]

    # BlockPool(4190) reserves zero. Data IDs 1..4189 leave five tail blocks.
    expected_owner_pages = (523, 524, 524, 524, 524, 524, 523, 523)
    assert bucket.owner_pages_by_rank == tuple(count * copies for count in expected_owner_pages)

    slots_by_copy_and_owner = [[[] for _ in range(8)] for _ in range(copies)]
    for copy_index in range(copies):
        for block_id in range(1, logical_blocks + 1):
            address = plan.map_c128(
                "flash",
                "c128",
                block_id,
                copy_index=copy_index,
            )
            assert address.owner_rank == address.global_block_id % 8
            assert address.owner_rank is not None
            assert address.segment_base_bytes % VMM_GRANULARITY_BYTES == 0
            assert address.segment_allocated_bytes % VMM_GRANULARITY_BYTES == 0
            slots_by_copy_and_owner[copy_index][address.owner_rank].append(address.physical_slot)
    for slots_by_owner in slots_by_copy_and_owner:
        for rank, slots in enumerate(slots_by_owner):
            assert sorted(slots) == list(range(1, expected_owner_pages[rank] + 1))

    # Both 523- and 524-page payloads need 528 C128 slots at 2-MiB
    # granularity. Each C128 copy is an independently aligned segment.
    expected_allocated_bytes = (528 * WIDE_PAGE_BYTES * copies,) * 8
    assert bucket.persistent_bytes_by_rank == expected_allocated_bytes


def test_one_page_c128_quota_has_no_owner_saving_after_vmm_alignment() -> None:
    """Keep fixed-workload quota reduction separate from owner placement."""
    plan = PackedPoolPlan(
        global_block_capacity=4_190,
        tp_size=8,
        groups=(
            _group(
                "c128",
                1,
                _component(
                    "c128_owner",
                    placement=PackedPlacement.C128_OWNER,
                    copies=20,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=WIDE_PAGE_BYTES,
                max_pages_per_rank=65,
                allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
            ),
        ),
    )

    # A sentinel plus zero or one owned data page is below the 16-page VMM
    # granule. Every owner and replicated copy therefore consumes one 2-MiB
    # segment per rank, and the identical scratch does not change the delta.
    assert plan.total_physical_bytes_by_rank() == (plan.aligned_quota_replicated_bytes_by_rank())
    assert plan.total_physical_bytes_by_rank(include_scratch=False) == (
        plan.aligned_quota_replicated_bytes_by_rank(
            include_scratch=False,
        )
    )
    assert plan.quota_replicated_bytes_by_rank() != (
        plan.aligned_quota_replicated_bytes_by_rank(
            include_scratch=False,
        )
    )


def test_aligned_replicated_counterfactual_includes_bucket_alignment_gaps() -> None:
    """Mixed segment granularities must advance the shared bucket cursor."""
    plan = PackedPoolPlan(
        global_block_capacity=5,
        tp_size=1,
        groups=(
            _group(
                "first",
                1,
                _component(
                    "small_alignment",
                    placement=PackedPlacement.REPLICATED,
                    page_size_bytes=128,
                    allocation_granularity_bytes=256,
                ),
            ),
            _group(
                "second",
                1,
                _component(
                    "large_alignment",
                    placement=PackedPlacement.C128_OWNER,
                    page_size_bytes=128,
                    allocation_granularity_bytes=1_024,
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=128,
                max_pages_per_rank=1,
                allocation_granularity_bytes=1_024,
            ),
        ),
    )

    # first=[0,256), alignment gap=[256,1024), second=[1024,2048),
    # scratch=[2048,3072).
    assert plan.aligned_quota_replicated_bytes_by_rank() == (3_072,)
    assert plan.aligned_quota_replicated_bytes_by_rank(include_scratch=False) == (2_048,)


def test_six_group_same_b_counterfactual_isolates_owner_saving() -> None:
    """Assign the spare range to C128 so owner placement is the only delta."""
    workload_quotas = (17, 1, 65, 65, 642, 165)
    global_block_capacity = 4_190
    spare_blocks = global_block_capacity - 1 - sum(workload_quotas)
    same_b_quotas = (
        workload_quotas[0],
        workload_quotas[1] + spare_blocks,
        *workload_quotas[2:],
    )
    assert same_b_quotas == (17, 3_235, 65, 65, 642, 165)

    plan = PackedPoolPlan(
        global_block_capacity=global_block_capacity,
        tp_size=8,
        groups=(
            _group(
                "c4",
                same_b_quotas[0],
                _component(
                    "c4_narrow",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    bucket="narrow",
                    page_size_bytes=NARROW_PAGE_BYTES,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
                _component(
                    "c4_wide",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
            _group(
                "c128",
                same_b_quotas[1],
                _component(
                    "c128_owner",
                    placement=PackedPlacement.C128_OWNER,
                    copies=20,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
            _group(
                "dense_swa_a",
                same_b_quotas[2],
                _component(
                    "dense_swa_a",
                    placement=PackedPlacement.REPLICATED,
                    copies=22,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
            _group(
                "dense_swa_b",
                same_b_quotas[3],
                _component(
                    "dense_swa_b",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
            _group(
                "c4_state",
                same_b_quotas[4],
                _component(
                    "c4_state_narrow",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    bucket="narrow",
                    page_size_bytes=NARROW_PAGE_BYTES,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
                _component(
                    "c4_state_wide",
                    placement=PackedPlacement.REPLICATED,
                    copies=21,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
            _group(
                "c128_state",
                same_b_quotas[5],
                _component(
                    "c128_state",
                    placement=PackedPlacement.REPLICATED,
                    copies=20,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=WIDE_PAGE_BYTES,
                max_pages_per_rank=65,
                allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
            ),
        ),
    )

    assert plan.used_logical_blocks == global_block_capacity - 1
    owner_bytes = plan.total_physical_bytes_by_rank()
    aligned_replicated_bytes = plan.aligned_quota_replicated_bytes_by_rank()
    assert owner_bytes == (4_215_275_520,) * 8
    assert aligned_replicated_bytes == (11_639_193_600,) * 8
    assert all(
        owner < replicated
        for owner, replicated in zip(
            owner_bytes,
            aligned_replicated_bytes,
        )
    )
    assert len(set(aligned_replicated_bytes)) == 1


def test_global_id_roundtrip_across_group_ranges() -> None:
    groups = (
        _group(
            "c4",
            4,
            _component("c4_replicated", placement=PackedPlacement.REPLICATED),
        ),
        _group(
            "c128",
            7,
            _component("c128_owner", placement=PackedPlacement.C128_OWNER),
        ),
        _group(
            "swa",
            3,
            _component("swa_replicated", placement=PackedPlacement.REPLICATED),
        ),
    )
    plan = PackedPoolPlan(
        global_block_capacity=16,
        tp_size=8,
        groups=groups,
    )

    assert plan.decode_global_block_id(SENTINEL_BLOCK_ID) is None
    for group in groups:
        for group_block_id in range(1, group.logical_blocks + 1):
            global_block_id = plan.encode_group_block_id(
                group.name,
                group_block_id,
            )
            assert plan.decode_global_block_id(global_block_id) == LogicalBlockRef(
                group.name,
                group_block_id,
            )
            assert (
                plan.decode_for_group(
                    group.name,
                    global_block_id,
                )
                == group_block_id
            )

    with pytest.raises(ValueError, match="unassigned"):
        plan.decode_global_block_id(15)


def test_feature_off_translation_is_exact_identity_for_shared_pool_groups() -> None:
    """Feature-off instantiates no packed plan or quota validation."""
    full_pool_ids = tuple(range(8))
    for group_name in ("c4", "c128", "swa"):
        assert (
            translate_group_block_ids(
                None,
                group_name,
                full_pool_ids,
            )
            == full_pool_ids
        )


def test_sentinel_and_out_of_range_ids_fail_closed() -> None:
    plan = PackedPoolPlan(
        global_block_capacity=5,
        tp_size=2,
        groups=(
            _group(
                "flash",
                3,
                _component(
                    "replicated",
                    placement=PackedPlacement.REPLICATED,
                ),
                _component(
                    "c128",
                    placement=PackedPlacement.C128_OWNER,
                ),
            ),
        ),
    )
    assert plan.encode_group_block_id("flash", SENTINEL_BLOCK_ID) == 0
    with pytest.raises(SentinelBlockError, match="sentinel block 0"):
        plan.map_replicated("flash", "replicated", 0, tp_rank=0)
    with pytest.raises(SentinelBlockError, match="sentinel block 0"):
        plan.map_c128("flash", "c128", 0)
    first_data = plan.map_replicated(
        "flash",
        "replicated",
        1,
        tp_rank=0,
    )
    sentinel = plan.sentinel_address(
        "flash",
        "replicated",
        tp_rank=0,
    )
    assert sentinel.physical_slot == 0
    assert sentinel.physical_offset_bytes == 0
    assert first_data.physical_slot == 1
    assert first_data.physical_offset_bytes == WIDE_PAGE_BYTES
    assert sentinel.physical_offset_bytes != first_data.physical_offset_bytes
    with pytest.raises(SentinelBlockError, match="reserved sentinel"):
        plan.unmap_replicated(
            "flash",
            "replicated",
            0,
            tp_rank=0,
            copy_index=0,
        )
    with pytest.raises(ValueError, match="outside"):
        plan.encode_group_block_id("flash", -1)
    with pytest.raises(ValueError, match="outside"):
        plan.encode_group_block_id("flash", 4)
    with pytest.raises(ValueError, match="outside"):
        plan.decode_global_block_id(6)
    with pytest.raises(ValueError, match="unassigned"):
        plan.decode_global_block_id(4)
    with pytest.raises(ValueError, match="not c128_owner"):
        plan.map_c128("flash", "replicated", 1)
    with pytest.raises(ValueError, match="not replicated"):
        plan.map_replicated("flash", "c128", 1, tp_rank=0)


def test_representative_flash_shape_has_positive_accounting_delta() -> None:
    """A representative same-B counterfactual isolates an owner byte delta."""
    num_blocks = 4_190
    tp_size = 8
    c128_layers = 20
    other_wide_layers = 2
    narrow_layers = 21
    selected_8k_c128_pages = 65

    plan = PackedPoolPlan(
        global_block_capacity=num_blocks,
        tp_size=tp_size,
        groups=(
            _group(
                "flash",
                num_blocks - 1,
                _component(
                    "other_wide",
                    placement=PackedPlacement.REPLICATED,
                    copies=other_wide_layers,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
                _component(
                    "c128_history",
                    placement=PackedPlacement.C128_OWNER,
                    copies=c128_layers,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
                _component(
                    "narrow_state",
                    placement=PackedPlacement.REPLICATED,
                    copies=narrow_layers,
                    bucket="narrow",
                    page_size_bytes=NARROW_PAGE_BYTES,
                    allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
                ),
            ),
        ),
        scratch=(
            PackedPoolScratchSpec(
                bucket="wide",
                page_size_bytes=WIDE_PAGE_BYTES,
                max_pages_per_rank=selected_8k_c128_pages,
                allocation_granularity_bytes=VMM_GRANULARITY_BYTES,
            ),
        ),
    )

    baseline_bytes = num_blocks * (
        (c128_layers + other_wide_layers) * WIDE_PAGE_BYTES + narrow_layers * NARROW_PAGE_BYTES
    )
    assert plan.quota_replicated_bytes_by_rank() == (baseline_bytes,) * tp_size

    accounting = {bucket.bucket: bucket for bucket in plan.bucket_accounting}
    assert accounting["wide"].replicated_pages_per_rank == ((num_blocks - 1) * other_wide_layers)
    assert accounting["wide"].sentinel_pages_per_rank == (other_wide_layers + c128_layers)
    assert accounting["wide"].scratch_pages_per_rank == (selected_8k_c128_pages)
    assert accounting["narrow"].replicated_pages_per_rank == ((num_blocks - 1) * narrow_layers)
    assert accounting["narrow"].sentinel_pages_per_rank == narrow_layers

    def _rounded_segment_bytes(pages: int, page_size_bytes: int) -> int:
        payload_bytes = pages * page_size_bytes
        return (payload_bytes + VMM_GRANULARITY_BYTES - 1) // VMM_GRANULARITY_BYTES * VMM_GRANULARITY_BYTES

    candidate_bytes = plan.total_physical_bytes_by_rank()
    expected_worst_rank_bytes = (
        other_wide_layers * _rounded_segment_bytes(num_blocks, WIDE_PAGE_BYTES)
        + c128_layers * _rounded_segment_bytes(525, WIDE_PAGE_BYTES)
        + narrow_layers * _rounded_segment_bytes(num_blocks, NARROW_PAGE_BYTES)
        + _rounded_segment_bytes(
            selected_8k_c128_pages,
            WIDE_PAGE_BYTES,
        )
    )
    assert candidate_bytes == (expected_worst_rank_bytes,) * tp_size
    assert max(candidate_bytes) < baseline_bytes
    assert baseline_bytes - max(candidate_bytes) > 0
    assert all(
        candidate < replicated
        for candidate, replicated in zip(
            candidate_bytes,
            plan.aligned_quota_replicated_bytes_by_rank(),
        )
    )

    for component_name, copy_index in (
        ("other_wide", other_wide_layers - 1),
        ("narrow_state", narrow_layers - 1),
    ):
        address = plan.map_replicated(
            "flash",
            component_name,
            num_blocks - 1,
            tp_rank=7,
            copy_index=copy_index,
        )
        assert address.segment_base_bytes % VMM_GRANULARITY_BYTES == 0
        assert address.segment_allocated_bytes % VMM_GRANULARITY_BYTES == 0
        assert (
            address.physical_offset_bytes + (WIDE_PAGE_BYTES if component_name == "other_wide" else NARROW_PAGE_BYTES)
            <= address.segment_base_bytes + address.segment_allocated_bytes
        )
