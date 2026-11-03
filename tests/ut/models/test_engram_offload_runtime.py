# SPDX-License-Identifier: Apache-2.0
"""Offload ownership and native configuration tests without device allocation."""

import contextlib
import ctypes
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    pytest.importorskip("torch_npu")
    package = types.ModuleType("_engram_offload_ut")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module("_engram_offload_ut.shared_host")
    monkeypatch.setattr(torch.npu, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    calls, storage = [], []

    def malloc(size):
        block = ctypes.create_string_buffer(size)
        storage.append(block)
        calls.append(("malloc", size))
        return ctypes.addressof(block)

    offload = types.SimpleNamespace(
        OffloadConfig=types.SimpleNamespace,
        Scene=types.SimpleNamespace(SHARED="shared"),
        initialize=lambda config: calls.append(("initialize", config)) or 0,
        uninitialize=lambda: calls.append(("uninitialize",)),
        malloc=malloc,
        free=lambda ptr: calls.append(("free", ptr)),
        get_dva=lambda ptr: ptr,
    )
    monkeypatch.setitem(
        sys.modules,
        "memfabric_hybrid",
        types.SimpleNamespace(offload=offload, initialize=lambda: 0, get_last_err_msg=lambda: ""),
    )
    yield module, calls
    for name in list(sys.modules):
        if name.startswith("_engram_offload_ut."):
            sys.modules.pop(name)


def test_owner_suballocations_share_initialization(backend):
    module, calls = backend
    kwargs = dict(owner=True, capacity=module.GIB, rank=0)
    runtime = module.MemfabricRuntime.acquire("generation", "tcp://127.0.0.1:9000", 2, "npu:0", **kwargs)
    again = module.MemfabricRuntime.acquire("generation", "tcp://127.0.0.1:9000", 2, "npu:0", **kwargs)
    assert runtime is again and runtime.references == 2
    config = calls[0][1]
    assert (config.scene, config.alloc_size, config.reserve_size, config.rank_id) == (
        "shared",
        module.GIB,
        module.GIB,
        0,
    )
    codes = module.MappedHostBuffer(runtime, 0, 2, 256, torch.int8, owner=True)
    scales = module.MappedHostBuffer(runtime, 1, 2, 8, torch.float32, owner=True)
    data = torch.arange(512).to(torch.int8).reshape(2, 256)
    codes.publish(0, data)
    assert ctypes.string_at(codes.gva, 512) == data.numpy().tobytes()
    runtime.close()
    with pytest.raises(RuntimeError, match="Close Engram mappings"):
        runtime.close()
    with pytest.raises(ValueError, match="outside"):
        codes.device_pointer(512)
    scales.close()
    codes.close()
    runtime.close()
    assert sum(call[0] == "initialize" for call in calls) == 1
    assert sum(call[0] == "free" for call in calls) == 2
    assert sum(call[0] == "uninitialize" for call in calls) == 1


def test_consumer_maps_without_allocating_or_freeing(backend):
    module, calls = backend
    runtime = module.MemfabricRuntime.acquire(
        "generation", "tcp://127.0.0.1:9000", 2, "npu:1", owner=False, capacity=module.GIB, rank=1
    )
    config = calls[0][1]
    assert config.alloc_size == 0 and config.reserve_size == module.GIB
    buffer = module.MappedHostBuffer(runtime, 0, 2, 256, torch.int8, owner=False, address=4096)
    assert buffer.device_pointer(256, 256) == 4352
    with pytest.raises(RuntimeError, match="Only a live owner"):
        buffer.publish(0, torch.zeros((2, 256), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="only one"):
        module.MemfabricRuntime.acquire(
            "another", "tcp://127.0.0.1:9000", 2, "npu:1", owner=False, capacity=module.GIB, rank=1
        )
    buffer.close()
    runtime.close()
    assert not any(call[0] in ("malloc", "free") for call in calls)


def test_sparse_kv_pool_conflict_rejected_before_initialization():
    from vllm_ascend.ascend_config import AscendConfig

    config = types.SimpleNamespace(
        engram_shared_host_dir="/private/generation",
        sparse_kv_offload_config=types.SimpleNamespace(enabled=True),
    )
    with pytest.raises(ValueError, match="cannot own separate pools"):
        AscendConfig._validate_sparse_packed_kv_offload_compatibility(config)
