import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.attention.context_parallel.c128_owner_cache import (
    C128OwnerShardCache,
    get_c128_owner_cache,
    register_c128_owner_cache,
)
from vllm_ascend.attention.context_parallel.c128_packed_pool import (
    PackedPlacement,
    PackedPoolComponentSpec,
    PackedPoolGroupSpec,
    PackedPoolPlan,
    PackedPoolScratchSpec,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        runner._c128_packed_layer_page_counts = {}
        runner._c128_packed_owner_layers = set()
        runner._c128_packed_layer_buckets = {}
        runner._c128_packed_scratch_raw_tensors = {}
        runner._c128_packed_owner_route_table = None
        runner._c128_packed_registered_owner_caches = {}
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerPackedArenaLifecycle(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.enable_c128_packed_vmm_arena = True
        runner._c128_packed_arena_runtime = None
        runner._c128_owner_stage_caches = {"stage": object()}
        runner._c128_packed_layer_page_counts = {}
        runner._c128_packed_owner_layers = set()
        runner._c128_packed_layer_buckets = {}
        runner._c128_packed_scratch_raw_tensors = {}
        runner._c128_packed_owner_route_table = None
        runner._c128_packed_registered_owner_caches = {}
        runner._c128_registered_owner_caches_by_layer = {}
        runner._c128_packed_block_table_translators = ()
        runner.device = SimpleNamespace(index=3)
        return runner

    @patch("vllm_ascend.worker.c128_packed_runtime." "packed_arena_contract_from_metadata")
    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    def test_install_validates_provenance_rank_and_device(
        self,
        mock_get_tp_group,
        mock_contract_from_metadata,
    ):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntime,
            PackedArenaRuntimeState,
        )

        runner = self._build_runner()
        runtime = PackedArenaRuntime.__new__(PackedArenaRuntime)
        runtime._state = PackedArenaRuntimeState.SEALED
        runtime.contract = SimpleNamespace(metadata_fingerprint="same")
        runtime.tp_rank = 3
        runtime.device_index = 3
        kv_cache_config = SimpleNamespace(c128_packed_pool_metadata={"schema_version": 1})
        mock_get_tp_group.return_value.rank_in_group = 3
        mock_contract_from_metadata.return_value = SimpleNamespace(metadata_fingerprint="same")

        runner._install_c128_packed_arena_runtime(
            runtime,
            kv_cache_config,
        )

        self.assertIs(runner._c128_packed_arena_runtime, runtime)
        self.assertEqual(
            runtime.state,
            PackedArenaRuntimeState.PUBLISHED,
        )

    @patch("vllm_ascend.worker.c128_packed_runtime." "packed_arena_contract_from_metadata")
    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    def test_failed_install_does_not_take_runtime_ownership(
        self,
        mock_get_tp_group,
        mock_contract_from_metadata,
    ):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntime,
            PackedArenaRuntimeState,
        )

        runner = self._build_runner()
        runtime = PackedArenaRuntime.__new__(PackedArenaRuntime)
        runtime._state = PackedArenaRuntimeState.SEALED
        runtime.contract = SimpleNamespace(metadata_fingerprint="same")
        runtime.tp_rank = 2
        runtime.device_index = 3
        kv_cache_config = SimpleNamespace(c128_packed_pool_metadata={"schema_version": 1})
        mock_get_tp_group.return_value.rank_in_group = 3
        mock_contract_from_metadata.return_value = SimpleNamespace(metadata_fingerprint="same")

        with self.assertRaisesRegex(ValueError, "TP rank"):
            runner._install_c128_packed_arena_runtime(
                runtime,
                kv_cache_config,
            )

        self.assertIsNone(runner._c128_packed_arena_runtime)

    @patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.shutdown")
    def test_shutdown_drops_model_aliases_before_runtime(
        self,
        mock_super_shutdown,
    ):
        runner = self._build_runner()
        events = []
        runtime = MagicMock()
        runtime.close.side_effect = lambda: events.append("runtime_close")
        runner._c128_packed_arena_runtime = runtime
        mock_super_shutdown.side_effect = lambda: events.append("super_shutdown")

        runner.shutdown()

        self.assertEqual(events, ["super_shutdown", "runtime_close"])
        self.assertEqual(runner._c128_owner_stage_caches, {})
        self.assertIsNone(runner._c128_packed_arena_runtime)

    @patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.shutdown")
    def test_shutdown_failure_retains_runtime_for_retry(
        self,
        mock_super_shutdown,
    ):
        runner = self._build_runner()
        runtime = MagicMock()
        runtime.close.side_effect = RuntimeError("retry")
        runner._c128_packed_arena_runtime = runtime

        with self.assertRaisesRegex(RuntimeError, "retry"):
            runner.shutdown()

        mock_super_shutdown.assert_called_once_with()
        self.assertIs(runner._c128_packed_arena_runtime, runtime)

    @patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.shutdown")
    def test_legacy_owner_shutdown_unregisters_after_alias_teardown(
        self,
        mock_super_shutdown,
    ):
        runner = self._build_runner()
        runner.enable_c128_packed_vmm_arena = False
        persistent = torch.zeros(3, 2, 4)
        stage = torch.empty(6, 2, 4)
        owner_cache = C128OwnerShardCache(
            persistent,
            stage,
            tp_size=2,
        )
        register_c128_owner_cache(owner_cache)
        runner._c128_registered_owner_caches_by_layer["legacy_attn"] = owner_cache
        runner._c128_owner_stage_caches = {
            "legacy": stage,
        }
        events = []

        def drop_model_aliases():
            self.assertIs(
                get_c128_owner_cache(persistent),
                owner_cache,
            )
            events.append("super_shutdown")

        mock_super_shutdown.side_effect = drop_model_aliases

        runner.shutdown()

        self.assertEqual(events, ["super_shutdown"])
        self.assertIsNone(get_c128_owner_cache(persistent))
        self.assertEqual(
            runner._c128_registered_owner_caches_by_layer,
            {},
        )
        self.assertEqual(runner._c128_owner_stage_caches, {})
        self.assertIsNone(runner._c128_packed_arena_runtime)

    @patch("vllm_ascend.worker.c128_packed_runtime." "packed_arena_contract_from_metadata")
    def test_production_activation_remains_fail_closed(
        self,
        mock_contract_from_metadata,
    ):
        runner = self._build_runner()
        kv_cache_config = SimpleNamespace(
            c128_packed_pool_metadata={"runtime_ready": True},
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "production activation remains disabled",
        ):
            runner.initialize_kv_cache(kv_cache_config)

        mock_contract_from_metadata.assert_called_once_with(kv_cache_config.c128_packed_pool_metadata)
        self.assertIsNone(runner._c128_packed_arena_runtime)


class _FakePackedArenaRuntime:
    def __init__(
        self,
        *,
        plan,
        tp_rank,
        roots,
        view_buckets,
    ):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntimeState,
        )

        self.contract = SimpleNamespace(plan=plan)
        self.tp_rank = tp_rank
        self.device_index = 0
        self.roots = roots
        self.view_buckets = view_buckets
        self.expected_keys = set(view_buckets)
        self.installed_keys = set()
        self.releasers = []
        self.state = PackedArenaRuntimeState.OPEN
        self.closed = False
        self.close_failures_remaining = 0
        accounting = plan.bucket_accounting
        self.accounting = SimpleNamespace(
            tp_rank=tp_rank,
            persistent_allocated_bytes=sum(bucket.persistent_allocated_bytes_by_rank[tp_rank] for bucket in accounting),
            scratch_region_bytes=sum(bucket.scratch_region_bytes_by_rank[tp_rank] for bucket in accounting),
            total_allocated_bytes=sum(bucket.total_allocated_bytes_by_rank[tp_rank] for bucket in accounting),
        )

    def install_tensor_views(self, view_key, install):
        if view_key not in self.expected_keys:
            raise ValueError(f"unknown view key: {view_key}")
        if view_key in self.installed_keys:
            raise ValueError(f"duplicate view key: {view_key}")
        release = install(self.roots[self.view_buckets[view_key]])
        self.releasers.append(release)
        self.installed_keys.add(view_key)

    def seal_views(self):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntimeState,
        )

        if self.installed_keys != self.expected_keys:
            raise RuntimeError("missing packed views")
        self.state = PackedArenaRuntimeState.SEALED

    def publish(self):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntimeState,
        )

        if self.state is not PackedArenaRuntimeState.SEALED:
            raise RuntimeError("runtime is not sealed")
        self.state = PackedArenaRuntimeState.PUBLISHED

    def close(self):
        from vllm_ascend.worker.c128_packed_runtime import (
            PackedArenaRuntimeState,
        )

        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            self.state = PackedArenaRuntimeState.CLEANUP_FAILED
            raise RuntimeError("synthetic cleanup failure")
        while self.releasers:
            self.releasers.pop()()
        self.state = PackedArenaRuntimeState.CLOSED
        self.closed = True


class TestNPUModelRunnerPackedAllocatorReshape(unittest.TestCase):
    PAGE_BYTES = 256
    ALIGNMENT_BYTES = 256

    def _plan_and_metadata(self):
        plan = PackedPoolPlan(
            global_block_capacity=16,
            tp_size=2,
            groups=(
                PackedPoolGroupSpec(
                    name="group_0",
                    logical_blocks=3,
                    components=(
                        PackedPoolComponentSpec(
                            name="group_0_component_0",
                            bucket="replicated",
                            page_size_bytes=self.PAGE_BYTES,
                            copies=1,
                            placement=PackedPlacement.REPLICATED,
                            allocation_granularity_bytes=(self.ALIGNMENT_BYTES),
                        ),
                    ),
                ),
                PackedPoolGroupSpec(
                    name="group_1",
                    logical_blocks=5,
                    components=(
                        PackedPoolComponentSpec(
                            name="group_1_component_0",
                            bucket="owner",
                            page_size_bytes=self.PAGE_BYTES,
                            copies=1,
                            placement=PackedPlacement.C128_OWNER,
                            allocation_granularity_bytes=(self.ALIGNMENT_BYTES),
                        ),
                    ),
                ),
            ),
            scratch=(
                PackedPoolScratchSpec(
                    bucket="owner",
                    page_size_bytes=self.PAGE_BYTES,
                    max_pages_per_rank=2,
                    allocation_granularity_bytes=self.ALIGNMENT_BYTES,
                ),
            ),
        )

        groups = []
        layer_names = ("replicated_attn", "owner_attn")
        logical_start = 1
        for group_index, (group, layer_name) in enumerate(zip(plan.groups, layer_names)):
            component = group.components[0]
            ranks = []
            for rank in range(plan.tp_size):
                sentinel = plan.sentinel_address(
                    group.name,
                    component.name,
                    tp_rank=rank,
                )
                ranks.append(
                    {
                        "rank": rank,
                        "segment_base_bytes": (sentinel.segment_base_bytes),
                        "segment_allocated_bytes": (sentinel.segment_allocated_bytes),
                        "sentinel_offset_bytes": (sentinel.physical_offset_bytes),
                    }
                )
            groups.append(
                {
                    "group_index": group_index,
                    "name": group.name,
                    "logical_blocks": group.logical_blocks,
                    "logical_start": logical_start,
                    "logical_stop": (logical_start + group.logical_blocks),
                    "layer_names": [layer_name],
                    "components": [
                        {
                            "name": component.name,
                            "bucket": component.bucket,
                            "page_size_bytes": (component.page_size_bytes),
                            "copies": 1,
                            "placement": component.placement.value,
                            "allocation_granularity_bytes": (component.allocation_granularity_bytes),
                            "layer_names": [layer_name],
                            "segments": [
                                {
                                    "copy_index": 0,
                                    "ranks": ranks,
                                }
                            ],
                        }
                    ],
                }
            )
            logical_start += group.logical_blocks

        owner_accounting = next(bucket for bucket in plan.bucket_accounting if bucket.bucket == "owner")
        scratch_bytes = 2 * self.PAGE_BYTES
        metadata = {
            "schema_version": 1,
            "planner_only": False,
            "downstream_runtime_abi_ready": True,
            "tp_size": plan.tp_size,
            "sentinel_block_id": 0,
            "global_block_capacity": plan.global_block_capacity,
            "usable_data_capacity": plan.usable_data_capacity,
            "used_logical_blocks": plan.used_logical_blocks,
            "groups": groups,
            "buckets": [
                {
                    "bucket": owner_accounting.bucket,
                    "page_size_bytes": (owner_accounting.page_size_bytes),
                    "persistent_allocated_bytes_by_rank": list(owner_accounting.persistent_allocated_bytes_by_rank),
                    "scratch_region_bytes_by_rank": list(owner_accounting.scratch_region_bytes_by_rank),
                    "total_allocated_bytes_by_rank": list(owner_accounting.total_allocated_bytes_by_rank),
                }
            ],
            "scratch": [
                {
                    "bucket": "owner",
                    "page_size_bytes": self.PAGE_BYTES,
                    "max_pages_per_rank": 2,
                    "allocation_granularity_bytes": (self.ALIGNMENT_BYTES),
                    "segments": [
                        {
                            "rank": rank,
                            "segment_base_bytes": (
                                owner_accounting.total_allocated_bytes_by_rank[rank] - scratch_bytes
                            ),
                            "segment_allocated_bytes": scratch_bytes,
                        }
                        for rank in range(plan.tp_size)
                    ],
                }
            ],
        }
        return plan, metadata

    def _build_runner(self, plan, metadata, *, tp_rank=1):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.enable_c128_packed_vmm_arena = True
        runner._c128_packed_arena_runtime = None
        runner._c128_owner_stage_caches = {}
        runner._c128_packed_layer_page_counts = {}
        runner._c128_packed_owner_layers = set()
        runner._c128_packed_layer_buckets = {}
        runner._c128_packed_scratch_raw_tensors = {}
        runner._c128_packed_owner_route_table = None
        runner._c128_packed_registered_owner_caches = {}
        runner._c128_registered_owner_caches_by_layer = {}
        runner._c128_packed_block_table_translators = ()
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
        )
        runner.may_reinitialize_input_batch = MagicMock()

        roots = {
            bucket.bucket: torch.zeros(
                bucket.total_allocated_bytes_by_rank[tp_rank],
                dtype=torch.uint8,
            )
            for bucket in plan.bucket_accounting
        }
        view_buckets = {}
        for group in plan.groups:
            for component in group.components:
                for copy_index in range(component.copies):
                    view_buckets[f"component/{group.name}/" f"{component.name}/{copy_index}"] = component.bucket
        view_buckets["scratch/owner"] = "owner"
        runtime = _FakePackedArenaRuntime(
            plan=plan,
            tp_rank=tp_rank,
            roots=roots,
            view_buckets=view_buckets,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=["replicated_attn"]),
                SimpleNamespace(layer_names=["owner_attn"]),
            ],
            c128_packed_pool_metadata=metadata,
        )
        return runner, runtime, kv_cache_config

    def _configure_representative_compressed_reshape(
        self,
        runner,
        kv_cache_config,
    ):
        replicated_spec = MLAAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=self.PAGE_BYTES,
            dtype=torch.uint8,
            compress_ratio=4,
            model_version="deepseek_v4",
        )
        owner_spec = MLAAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=self.PAGE_BYTES,
            dtype=torch.uint8,
            compress_ratio=128,
            model_version="deepseek_v4",
        )
        kv_cache_config.num_blocks = 9
        kv_cache_config.kv_cache_groups[0].kv_cache_spec = replicated_spec
        kv_cache_config.kv_cache_groups[1].kv_cache_spec = owner_spec
        runner.use_compress = True
        runner.enable_c128_owner_shard = False
        runner.enable_c128_owner_compact_allocation = False
        runner.runner_only_attn_layers = set()
        runner.attn_backend = SimpleNamespace(
            get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size: (
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
            )
        )
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                backend=runner.attn_backend,
                kv_cache_spec=replicated_spec,
                layer_names=["replicated_attn"],
            ),
            SimpleNamespace(
                backend=runner.attn_backend,
                kv_cache_spec=owner_spec,
                layer_names=["owner_attn"],
            ),
        ]
        runner.vllm_config.additional_config = {}
        return owner_spec

    def test_exact_component_and_scratch_aliases_replace_raw_allocation(self):
        plan, metadata = self._plan_and_metadata()
        runner, runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        captured = {}

        def reshape(_config, raw_tensors):
            captured.update(raw_tensors)
            return dict(raw_tensors)

        runner._reshape_kv_cache_tensors = reshape

        def install_runtime(installed_runtime, _config):
            installed_runtime.publish()
            runner._c128_packed_arena_runtime = installed_runtime

        runner._install_c128_packed_arena_runtime = install_runtime

        translators = (object(), object())
        with patch(
            "vllm_ascend.worker.packed_block_table." "packed_block_table_translators_from_metadata",
            return_value=translators,
        ) as build_translators:
            kv_caches = runner._initialize_kv_cache_from_c128_packed_arena(
                runtime,
                kv_cache_config,
            )

        build_translators.assert_called_once_with(
            kv_cache_config.c128_packed_pool_metadata,
            tp_rank=1,
            kv_cache_group_layer_names=(
                ("replicated_attn",),
                ("owner_attn",),
            ),
        )
        runner.may_reinitialize_input_batch.assert_called_once_with(
            kv_cache_config,
            packed_translators=translators,
        )

        replicated_component = plan.groups[0].components[0]
        replicated_sentinel = plan.sentinel_address(
            "group_0",
            replicated_component.name,
            tp_rank=1,
        )
        owner_component = plan.groups[1].components[0]
        owner_sentinel = plan.sentinel_address(
            "group_1",
            owner_component.name,
            tp_rank=1,
        )
        self.assertEqual(captured["replicated_attn"].numel(), 4 * self.PAGE_BYTES)
        self.assertEqual(captured["owner_attn"].numel(), 3 * self.PAGE_BYTES)
        self.assertEqual(
            captured["replicated_attn"].data_ptr(),
            runtime.roots["replicated"].data_ptr() + replicated_sentinel.segment_base_bytes,
        )
        self.assertEqual(
            captured["owner_attn"].data_ptr(),
            runtime.roots["owner"].data_ptr() + owner_sentinel.segment_base_bytes,
        )
        self.assertEqual(
            runner._c128_packed_scratch_raw_tensors["owner"].numel(),
            2 * self.PAGE_BYTES,
        )
        self.assertEqual(
            runner._c128_packed_layer_page_counts,
            {
                "replicated_attn": 4,
                "owner_attn": 3,
            },
        )
        self.assertEqual(
            runner._c128_packed_owner_layers,
            {"owner_attn"},
        )
        self.assertIs(
            runner._c128_packed_arena_runtime,
            runtime,
        )
        self.assertEqual(set(kv_caches), set(captured))
        self.assertFalse(runtime.closed)

    @patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.shutdown")
    def test_legacy_compact_owner_reshape_tracks_and_unregisters_cache(
        self,
        mock_super_shutdown,
    ):
        plan, metadata = self._plan_and_metadata()
        runner, _runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        owner_spec = self._configure_real_compressed_reshape(
            runner,
            kv_cache_config,
        )
        runner.enable_c128_packed_vmm_arena = False
        runner.enable_c128_owner_shard = True
        runner.enable_c128_owner_compact_allocation = True
        runner._c128_packed_owner_layers.clear()
        runner.device = torch.device("cpu")
        replicated_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        raw_tensors = {
            "replicated_attn": torch.zeros(
                kv_cache_config.num_blocks * replicated_spec.page_size_bytes,
                dtype=torch.uint8,
            ),
            "owner_attn": torch.zeros(
                5 * owner_spec.page_size_bytes,
                dtype=torch.uint8,
            ),
        }

        kv_caches = runner._reshape_kv_cache_tensors(
            kv_cache_config,
            raw_tensors,
        )
        persistent = kv_caches["owner_attn"][0]
        owner_cache = get_c128_owner_cache(persistent)
        self.assertIsNotNone(owner_cache)
        self.assertIs(
            runner._c128_registered_owner_caches_by_layer["owner_attn"],
            owner_cache,
        )

        def drop_model_aliases():
            kv_caches.clear()
            self.assertIs(
                get_c128_owner_cache(persistent),
                owner_cache,
            )

        mock_super_shutdown.side_effect = drop_model_aliases
        runner.shutdown()

        self.assertIsNone(get_c128_owner_cache(persistent))
        self.assertEqual(
            runner._c128_registered_owner_caches_by_layer,
            {},
        )
        self.assertEqual(runner._c128_owner_stage_caches, {})

    def test_manifest_mismatch_closes_runtime_without_publication(self):
        plan, metadata = self._plan_and_metadata()
        metadata["groups"][1]["components"][0]["layer_names"][0] = "wrong_owner_attn"
        runner, runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        runner._reshape_kv_cache_tensors = MagicMock()
        runner._install_c128_packed_arena_runtime = MagicMock()

        with self.assertRaises(ValueError):
            runner._initialize_kv_cache_from_c128_packed_arena(
                runtime,
                kv_cache_config,
            )

        self.assertTrue(runtime.closed)
        self.assertIsNone(runner._c128_packed_arena_runtime)
        self.assertEqual(
            runner._c128_packed_layer_page_counts,
            {},
        )
        self.assertEqual(
            runner._c128_packed_scratch_raw_tensors,
            {},
        )
        runner._reshape_kv_cache_tensors.assert_not_called()
        runner._install_c128_packed_arena_runtime.assert_not_called()

    def test_owner_cache_gets_exact_route_and_unregisters_on_close(self):
        plan, metadata = self._plan_and_metadata()
        runner, runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        self._configure_representative_compressed_reshape(
            runner,
            kv_cache_config,
        )

        def install_runtime(installed_runtime, _config):
            installed_runtime.publish()
            runner._c128_packed_arena_runtime = installed_runtime

        runner._install_c128_packed_arena_runtime = install_runtime
        translators = (object(), object())
        with patch(
            "vllm_ascend.worker.packed_block_table." "packed_block_table_translators_from_metadata",
            return_value=translators,
        ):
            kv_caches = runner._initialize_kv_cache_from_c128_packed_arena(
                runtime,
                kv_cache_config,
            )

        persistent_cache = kv_caches["owner_attn"][0]
        owner_cache = get_c128_owner_cache(persistent_cache)
        self.assertIsNotNone(owner_cache)
        assert owner_cache is not None
        self.assertEqual(
            owner_cache.packed_route.layer_name,
            "owner_attn",
        )
        self.assertIs(
            runner._c128_packed_registered_owner_caches["owner_attn"],
            owner_cache,
        )

        kv_caches.clear()
        runner._c128_owner_stage_caches.clear()
        runner._c128_packed_scratch_raw_tensors.clear()
        runtime.close()

        self.assertIsNone(get_c128_owner_cache(persistent_cache))
        self.assertEqual(
            runner._c128_packed_registered_owner_caches,
            {},
        )

    def test_post_reshape_failure_unregisters_owner_cache(self):
        plan, metadata = self._plan_and_metadata()
        runner, runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        self._configure_representative_compressed_reshape(
            runner,
            kv_cache_config,
        )
        captured = {}
        original_input_batch = object()
        original_kernel_block_sizes = [["original"]]
        runner.input_batch = original_input_batch
        runner.kernel_block_sizes = original_kernel_block_sizes

        def replace_input_batch(*_args, **_kwargs):
            runner.input_batch = object()
            runner.kernel_block_sizes = [["packed"]]

        runner.may_reinitialize_input_batch.side_effect = replace_input_batch

        def reject_install(_runtime, _config):
            owner_cache = runner._c128_packed_registered_owner_caches["owner_attn"]
            captured["persistent_cache"] = owner_cache.persistent_cache
            raise RuntimeError("synthetic publish failure")

        runner._install_c128_packed_arena_runtime = reject_install
        with (
            patch(
                "vllm_ascend.worker.packed_block_table." "packed_block_table_translators_from_metadata",
                return_value=(object(), object()),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "synthetic publish failure",
            ),
        ):
            runner._initialize_kv_cache_from_c128_packed_arena(
                runtime,
                kv_cache_config,
            )

        self.assertTrue(runtime.closed)
        self.assertIsNone(get_c128_owner_cache(captured["persistent_cache"]))
        self.assertEqual(
            runner._c128_packed_registered_owner_caches,
            {},
        )
        self.assertIs(
            runner.input_batch,
            original_input_batch,
        )
        self.assertIs(
            runner.kernel_block_sizes,
            original_kernel_block_sizes,
        )

    def test_cleanup_failure_restores_state_and_preserves_retry(self):
        plan, metadata = self._plan_and_metadata()
        runner, runtime, kv_cache_config = self._build_runner(
            plan,
            metadata,
        )
        self._configure_real_compressed_reshape(
            runner,
            kv_cache_config,
        )
        original_input_batch = object()
        original_kernel_block_sizes = [["original"]]
        original_stage_cache = object()
        original_page_counts = {"legacy": 9}
        original_owner_layers = {"legacy"}
        original_layer_buckets = {"legacy": "legacy_bucket"}
        original_scratch = {"legacy_bucket": torch.zeros(1)}
        original_route_table = object()
        original_translators = (object(),)
        runner.input_batch = original_input_batch
        runner.kernel_block_sizes = original_kernel_block_sizes
        runner._c128_owner_stage_caches = {"legacy": original_stage_cache}
        runner._c128_packed_layer_page_counts = original_page_counts
        runner._c128_packed_owner_layers = original_owner_layers
        runner._c128_packed_layer_buckets = original_layer_buckets
        runner._c128_packed_scratch_raw_tensors = original_scratch
        runner._c128_packed_owner_route_table = original_route_table
        runner._c128_packed_block_table_translators = original_translators

        def replace_input_batch(*_args, **_kwargs):
            runner.input_batch = object()
            runner.kernel_block_sizes = [["packed"]]

        runner.may_reinitialize_input_batch.side_effect = replace_input_batch
        captured = {}

        def reject_install(_runtime, _config):
            owner_cache = runner._c128_packed_registered_owner_caches["owner_attn"]
            captured["persistent_cache"] = owner_cache.persistent_cache
            raise RuntimeError("synthetic publish failure")

        runner._install_c128_packed_arena_runtime = reject_install
        runtime.close_failures_remaining = 1
        with (
            patch(
                "vllm_ascend.worker.packed_block_table." "packed_block_table_translators_from_metadata",
                return_value=(object(), object()),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "synthetic cleanup failure",
            ),
        ):
            runner._initialize_kv_cache_from_c128_packed_arena(
                runtime,
                kv_cache_config,
            )

        self.assertIs(runner.input_batch, original_input_batch)
        self.assertIs(
            runner.kernel_block_sizes,
            original_kernel_block_sizes,
        )
        self.assertEqual(
            runner._c128_owner_stage_caches,
            {"legacy": original_stage_cache},
        )
        self.assertIs(
            runner._c128_packed_layer_page_counts,
            original_page_counts,
        )
        self.assertIs(
            runner._c128_packed_owner_layers,
            original_owner_layers,
        )
        self.assertIs(
            runner._c128_packed_layer_buckets,
            original_layer_buckets,
        )
        self.assertIs(
            runner._c128_packed_scratch_raw_tensors,
            original_scratch,
        )
        self.assertIs(
            runner._c128_packed_owner_route_table,
            original_route_table,
        )
        self.assertIs(
            runner._c128_packed_block_table_translators,
            original_translators,
        )
        self.assertIs(
            runner._c128_packed_arena_runtime,
            runtime,
        )
        self.assertIn(
            "owner_attn",
            runner._c128_packed_registered_owner_caches,
        )
        self.assertIsNotNone(get_c128_owner_cache(captured["persistent_cache"]))

        runner._close_c128_packed_arena_runtime()

        self.assertIsNone(runner._c128_packed_arena_runtime)
        self.assertNotIn(
            "owner_attn",
            runner._c128_packed_registered_owner_caches,
        )
        self.assertIsNone(get_c128_owner_cache(captured["persistent_cache"]))


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.sampling_metadata.top_k = None
        input_batch.top_k_cpu = None
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])

    def test_placeholder_spec_tokens_are_sanitized_only_for_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, 33, -1])

    def test_placeholder_sanitization_is_scoped_to_current_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=2,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, -1])

    def test_mtp3_placeholder_metadata_is_preserved_before_sanitizing_forward(self):
        runner = self._build_runner()
        runner.pcp_size = 1
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner._arange_scratch = np.empty(8, dtype=np.int32)
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1, -1, -1]},
        )

        spec_decode_metadata = runner._calc_spec_decode_metadata(
            num_draft_tokens=np.array([3], dtype=np.int32),
            cu_num_scheduled_tokens=np.array([4], dtype=np.int32),
            num_pcp_pads=None,
        )
        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(spec_decode_metadata.draft_token_ids.tolist(), [-1, -1, -1])
        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 0, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, -1, -1])


class TestNPUModelRunnerModelForward(unittest.TestCase):
    def test_model_forward_keeps_input_ids_for_multimodal_embeds(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.model = MagicMock(return_value=torch.randn(3, 4))
        runner.supports_mm_inputs = True
        runner.enable_enpu = False
        runner.input_ids = SimpleNamespace(gpu=torch.arange(8))
        runner._update_full_graph_params_if_needed = MagicMock()

        forward_context = SimpleNamespace(flash_comm_v1_enabled=False)
        positions = torch.arange(3)
        inputs_embeds = torch.randn(3, 4)

        with patch(
            "vllm_ascend.worker.model_runner_v1.get_forward_context",
            return_value=forward_context,
        ):
            runner._model_forward(
                3,
                input_ids=None,
                positions=positions,
                inputs_embeds=inputs_embeds,
                mm_kwargs="kept",
            )

        call_kwargs = runner.model.call_args.kwargs
        torch.testing.assert_close(call_kwargs["input_ids"], torch.arange(3))
        self.assertIs(call_kwargs["positions"], positions)
        self.assertIs(call_kwargs["inputs_embeds"], inputs_embeds)
        self.assertEqual(call_kwargs["mm_kwargs"], "kept")


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


if __name__ == "__main__":
    unittest.main()
