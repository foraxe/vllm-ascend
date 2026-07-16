#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Mock heavy dependencies (torch, vllm, etc.) for ascend_store unit tests.

IMPORTANT: This module MUST be imported before any vllm_ascend or vllm
imports in each test file.

Usage at the top of each test file:
    import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
"""

import logging
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

sys.modules.setdefault("regex", re)

# ---------------------------------------------------------------------------
# Mock torch / torch_npu
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.Tensor = MagicMock  # type: ignore[attr-defined]
    _torch.bool = "bool"  # type: ignore[attr-defined]
    _torch.float16 = "float16"  # type: ignore[attr-defined]
    _torch.zeros = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    _torch.sum = MagicMock(return_value=0)  # type: ignore[attr-defined]
    _torch.device = MagicMock()  # type: ignore[attr-defined]
    _torch.distributed = MagicMock()  # type: ignore[attr-defined]
    _npu = MagicMock()
    _npu.Event = MagicMock
    _npu.current_device = MagicMock(return_value=0)
    _npu.set_device = MagicMock()
    _torch.npu = _npu  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch
    sys.modules["torch.distributed"] = _torch.distributed  # type: ignore[attr-defined]

if "torch_npu" not in sys.modules:
    sys.modules["torch_npu"] = MagicMock()
    sys.modules["torch_npu._inductor"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock vllm modules
# ---------------------------------------------------------------------------
_vllm_mock_modules = [
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.kv_events",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.factory",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.parallel_state",
    "vllm.envs",
    "vllm.forward_context",
    "vllm.logger",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.linear",
    "vllm.model_executor.layers.quantization",
    "vllm.platforms",
    "vllm.utils",
    "vllm.utils.hashing",
    "vllm.utils.math_utils",
    "vllm.utils.network_utils",
    "vllm.v1",
    "vllm.v1.attention",
    "vllm.v1.attention.backend",
    "vllm.v1.core",
    "vllm.v1.core.block_pool",
    "vllm.v1.core.kv_cache_manager",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "vllm.v1.core.single_type_kv_cache_manager",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.kv_cache_spec_registry",
    "vllm.v1.outputs",
    "vllm.v1.request",
    "vllm.v1.serial_utils",
]
for _mod_name in _vllm_mock_modules:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

sys.modules["vllm.utils.math_utils"].cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
sys.modules["vllm.logger"].logger = logging.getLogger("vllm")  # type: ignore[attr-defined]

_base_mod = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
_base_mod.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {"__init__": lambda self, **kw: None})  # type: ignore[attr-defined]
_base_mod.KVConnectorMetadata = type("KVConnectorMetadata", (), {})  # type: ignore[attr-defined]
_base_mod.SupportsHMA = type("SupportsHMA", (), {})  # type: ignore[attr-defined]
_base_mod.KVConnectorRole = MagicMock()  # type: ignore[attr-defined]
_base_mod.KVConnectorRole.SCHEDULER = "SCHEDULER"
_base_mod.KVConnectorRole.WORKER = "WORKER"

_events_mod = sys.modules["vllm.distributed.kv_events"]
_events_mod.KVCacheEvent = type("KVCacheEvent", (), {})  # type: ignore[attr-defined]
_events_mod.KVConnectorKVEvents = type("KVConnectorKVEvents", (), {})  # type: ignore[attr-defined]


class _FakeAggregator:
    def __init__(self, *args, **kwargs):
        self._mock = MagicMock()

    def __getattr__(self, name):
        return getattr(self._mock, name)


_events_mod.KVEventAggregator = _FakeAggregator  # type: ignore[attr-defined]
_events_mod.BlockStored = type(  # type: ignore[attr-defined]
    "BlockStored",
    (),
    {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
)

_kv_cache_utils_mod = sys.modules["vllm.v1.core.kv_cache_utils"]
_kv_cache_utils_mod.BlockHash = bytes  # type: ignore[attr-defined]
_kv_cache_utils_mod.BlockHashList = list  # type: ignore[attr-defined]
_kv_cache_utils_mod.maybe_convert_block_hash = lambda x: x  # type: ignore[attr-defined]


class _BlockHashListWithBlockSize(list):
    def __init__(self, block_hashes, hash_block_size, block_size):
        scale = block_size // hash_block_size
        super().__init__(block_hashes[::scale])


_kv_cache_utils_mod.BlockHashListWithBlockSize = _BlockHashListWithBlockSize  # type: ignore[attr-defined]


@dataclass
class _KVCacheBlock:
    block_id: int = 0
    block_hash: bytes | None = None

    @property
    def is_null(self) -> bool:
        return self.block_id == 0


_kv_cache_utils_mod.KVCacheBlock = _KVCacheBlock  # type: ignore[attr-defined]

_block_pool_mod = sys.modules["vllm.v1.core.block_pool"]
_block_pool_mod.BlockPool = type("BlockPool", (), {})  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _FakeKVCacheSpec:
    block_size: int = 16
    compress_ratio: int = 1
    model_version: str | None = None
    manager_cls: Any = field(default=None, compare=False, repr=False)

    def copy_with_new_block_size(self, block_size: int):
        values = self.__dict__.copy()
        values["block_size"] = block_size
        return type(self)(**values)


@dataclass(frozen=True)
class _FakeFullAttentionSpec(_FakeKVCacheSpec):
    pass


@dataclass(frozen=True)
class _FakeSlidingWindowSpec(_FakeKVCacheSpec):
    sliding_window: int = 32


@dataclass(frozen=True)
class _FakeMambaSpec(_FakeKVCacheSpec):
    mamba_cache_mode: str | None = None


@dataclass(frozen=True)
class _FakeUniformTypeKVCacheSpecs(_FakeKVCacheSpec):
    kv_cache_specs: dict[str, _FakeKVCacheSpec] = field(default_factory=dict)


@dataclass
class _FakeKVCacheGroupSpec:
    layer_names: list[str] = field(default_factory=list)
    kv_cache_spec: _FakeKVCacheSpec = field(default_factory=_FakeFullAttentionSpec)
    is_eagle_group: bool = False


@dataclass
class _FakeKVCacheConfig:
    num_blocks: int = 1
    kv_cache_tensors: list[Any] = field(default_factory=list)
    kv_cache_groups: list[_FakeKVCacheGroupSpec] = field(default_factory=list)


class _FakeSingleTypeKVCacheManager:
    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        num_prompt_tokens=None,
    ):
        return None

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        del alignment_tokens, dcp_world_size, pcp_world_size
        computed = tuple([] for _ in kv_cache_group_ids)
        max_blocks = max_length // kv_cache_spec.block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached, strict=True):
                blocks.append(block)
        if drop_eagle_block and computed and computed[0]:
            for blocks in computed:
                blocks.pop()
        return computed


class _FakeSlidingWindowManager(_FakeSingleTypeKVCacheManager):
    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        num_prompt_tokens=None,
    ):
        del use_eagle, retention_interval, num_prompt_tokens
        if alignment_tokens is None:
            return None
        per_segment = max(alignment_tokens // kv_cache_spec.block_size, 1)
        return [(idx + 1) % per_segment == 0 for idx in range(start_block, end_block)]


_single_type_mod = sys.modules["vllm.v1.core.single_type_kv_cache_manager"]
_single_type_mod.SingleTypeKVCacheManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.FullAttentionManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.SlidingWindowManager = _FakeSlidingWindowManager  # type: ignore[attr-defined]
_single_type_mod.MambaManager = _FakeSingleTypeKVCacheManager  # type: ignore[attr-defined]
_single_type_mod.spec_manager_map = {  # type: ignore[attr-defined]
    _FakeFullAttentionSpec: _FakeSingleTypeKVCacheManager,
    _FakeSlidingWindowSpec: _FakeSlidingWindowManager,
    _FakeMambaSpec: _FakeSingleTypeKVCacheManager,
}

_kv_interface_mod = sys.modules["vllm.v1.kv_cache_interface"]
_kv_interface_mod.KVCacheSpec = _FakeKVCacheSpec  # type: ignore[attr-defined]
_kv_interface_mod.FullAttentionSpec = _FakeFullAttentionSpec  # type: ignore[attr-defined]
_kv_interface_mod.SlidingWindowSpec = _FakeSlidingWindowSpec  # type: ignore[attr-defined]
_kv_interface_mod.MambaSpec = _FakeMambaSpec  # type: ignore[attr-defined]
_kv_interface_mod.UniformTypeKVCacheSpecs = _FakeUniformTypeKVCacheSpecs  # type: ignore[attr-defined]
_kv_interface_mod.KVCacheGroupSpec = _FakeKVCacheGroupSpec  # type: ignore[attr-defined]
_kv_interface_mod.KVCacheConfig = _FakeKVCacheConfig  # type: ignore[attr-defined]


class _KVCacheSpecRegistry:
    @staticmethod
    def get_manager_class(spec):
        if isinstance(spec, _FakeSlidingWindowSpec):
            return _FakeSlidingWindowManager
        if isinstance(spec, (_FakeFullAttentionSpec, _FakeMambaSpec)):
            return _FakeSingleTypeKVCacheManager
        return getattr(spec, "manager_cls", None)


sys.modules["vllm.v1.kv_cache_spec_registry"].KVCacheSpecRegistry = _KVCacheSpecRegistry  # type: ignore[attr-defined]

_sched_output_mod = sys.modules["vllm.v1.core.sched.output"]
_sched_output_mod.NewRequestData = MagicMock  # type: ignore[attr-defined]

sys.modules["vllm.envs"].VLLM_RPC_BASE_PATH = "/tmp/vllm_rpc"  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Mock external backends
# ---------------------------------------------------------------------------
for _mod_name in [
    "mooncake",
    "mooncake.engine",
    "mooncake.store",
    "memcache_hybrid",
    "yr",
    "yr.datasystem",
    "yr.datasystem.hetero_client",
    "yr.datasystem.kv_client",
    "yr.datasystem.object_client",
    "zmq",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# ---------------------------------------------------------------------------
# Mock vllm_ascend transitive imports
# ---------------------------------------------------------------------------


def _make_pkg(name, path=""):
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name  # type: ignore[attr-defined]
    return mod


for _pkg in ["vllm_ascend", "vllm_ascend.distributed"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_pkg(_pkg)

_kv_transfer_init = _make_pkg("vllm_ascend.distributed.kv_transfer")
_kv_transfer_init.register_connector = MagicMock()  # type: ignore[attr-defined]
sys.modules["vllm_ascend.distributed.kv_transfer"] = _kv_transfer_init

_kv_utils_pkg = _make_pkg("vllm_ascend.distributed.kv_transfer.utils")
sys.modules["vllm_ascend.distributed.kv_transfer.utils"] = _kv_utils_pkg
sys.modules["vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine"] = MagicMock()

_kv_pool_pkg = _make_pkg("vllm_ascend.distributed.kv_transfer.kv_pool")
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool"] = _kv_pool_pkg

_ascend_store_real_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "vllm_ascend",
    "distributed",
    "kv_transfer",
    "kv_pool",
    "ascend_store",
)
_ascend_store_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
    os.path.abspath(_ascend_store_real_path),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"] = _ascend_store_pkg

_backend_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend",
    os.path.join(os.path.abspath(_ascend_store_real_path), "backend"),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"] = _backend_pkg

if "vllm_ascend.utils" not in sys.modules or not hasattr(sys.modules["vllm_ascend.utils"], "AscendDeviceType"):
    _ascend_utils = MagicMock()
    _ascend_utils.AscendDeviceType = MagicMock()
    _ascend_utils.get_ascend_device_type = MagicMock()
    sys.modules["vllm_ascend.utils"] = _ascend_utils
