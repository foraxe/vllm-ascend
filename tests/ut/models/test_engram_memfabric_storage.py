# SPDX-License-Identifier: Apache-2.0
"""Offload instance, allocation and release invariants with a fake backend."""

import contextlib
import ctypes
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def storage(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/shared_host.py"
    spec = importlib.util.spec_from_file_location("_memfabric_storage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda: None), raising=False
    )
    return module


@pytest.fixture
def runtime():
    data = ctypes.create_string_buffer(65536)
    calls: list[object] = []

    offload = SimpleNamespace(
        malloc=lambda size: calls.append(("malloc", size)) or ctypes.addressof(data),
        get_dva=lambda ptr: ptr,
        free=lambda ptr: calls.append(("free", ptr)) or 0,
    )

    def check(result, operation):
        if result:
            raise RuntimeError(f"{operation}: {result}")

    return SimpleNamespace(
        offload=offload, device=torch.device("cpu"), rank=0, objects=set(), data=data, calls=calls, check=check
    )


@pytest.mark.parametrize("owner", [True, False])
def test_only_owner_allocates(storage, runtime, owner):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=owner, address=0x10000000)
    assert runtime.calls == ([("malloc", 8192)] if owner else [])
    buffer.close()
    assert not runtime.objects
    assert len(runtime.calls) == (2 if owner else 0)


def test_publication_uses_owner_cpu_mapping_and_preserves_bounds(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=True)
    source = torch.arange(256, dtype=torch.int16).to(torch.int8).reshape(1, 256)
    buffer.publish(7, source)
    actual = torch.frombuffer(runtime.data, dtype=torch.int8)[7 * 256 : 8 * 256]
    torch.testing.assert_close(actual, source.flatten())
    with pytest.raises(ValueError, match="outside"):
        buffer.publish(32, source)
    buffer.close()


def test_live_views_block_release_and_failed_synchronize_is_retryable(storage, runtime, monkeypatch):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, address=0x10000000)
    view = buffer.view(8, 24)
    assert view.tensor.device.type == "meta"
    assert view.ptrs.tolist() == [0x10000000 + 8 * 256]
    with pytest.raises(RuntimeError, match="slices"):
        buffer.close()
    view.close()

    def fail_sync():
        raise RuntimeError("synchronize failed")

    monkeypatch.setattr(torch.npu, "synchronize", fail_sync)
    with pytest.raises(RuntimeError, match="synchronize"):
        buffer.close()
    assert not buffer.closed and runtime.objects == {0}
    assert not runtime.calls
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    buffer.close()
    assert buffer.closed and not runtime.calls


def test_consumer_cannot_publish_or_duplicate_live_object(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, address=0x10000000)
    with pytest.raises(RuntimeError, match="owner"):
        buffer.publish(0, torch.zeros(1, 256, dtype=torch.int8))
    with pytest.raises(ValueError, match="Duplicate"):
        storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=False, address=0x10000000)
    buffer.close()


def test_nonidentity_owner_address_frees_allocation(storage, runtime):
    runtime.offload.get_dva = lambda ptr: ptr + 4096
    with pytest.raises(RuntimeError, match="identity-mapped"):
        storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=True)
    assert runtime.calls[-1][0] == "free"
    assert not runtime.objects


def test_failed_free_retains_mapping_for_retry(storage, runtime):
    buffer = storage.MappedHostBuffer(runtime, 0, 32, 256, torch.int8, owner=True)
    free = runtime.offload.free
    runtime.offload.free = lambda ptr: -1
    with pytest.raises(RuntimeError, match="Offload free"):
        buffer.close()
    assert not buffer.closed and runtime.objects == {0}
    runtime.offload.free = free
    buffer.close()
    assert buffer.closed and not runtime.objects


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

    pool = types.SimpleNamespace(
        OffloadConfig=types.SimpleNamespace,
        Scene=types.SimpleNamespace(SHARED="shared"),
        initialize=lambda config: calls.append(("initialize", config)) or 0,
        uninitialize=lambda: calls.append(("uninitialize",)),
        malloc=malloc,
        free=lambda ptr: calls.append(("free", ptr)) or 0,
        get_dva=lambda ptr: ptr,
    )
    # No default-instance allocation API: accidental module-level use must fail.
    offload = types.SimpleNamespace(Offload=lambda: pool, OffloadConfig=pool.OffloadConfig, Scene=pool.Scene)
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
    kwargs = dict(owner=True, capacity=module.GIB, rank=0, local_world_size=2)
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
        "generation", "tcp://127.0.0.1:9000", 2, "npu:1", owner=False, capacity=module.GIB, rank=1, local_world_size=2
    )
    config = calls[0][1]
    assert config.alloc_size == 0 and config.reserve_size == module.GIB
    buffer = module.MappedHostBuffer(runtime, 0, 2, 256, torch.int8, owner=False, address=4096)
    assert buffer.device_pointer(256, 256) == 4352
    with pytest.raises(RuntimeError, match="Only a live owner"):
        buffer.publish(0, torch.zeros((2, 256), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="only one"):
        module.MemfabricRuntime.acquire(
            "another", "tcp://127.0.0.1:9000", 2, "npu:1", owner=False, capacity=module.GIB, rank=1, local_world_size=2
        )
    buffer.close()
    runtime.close()
    assert not any(call[0] in ("malloc", "free") for call in calls)


def test_sparse_kv_pool_can_coexist():
    from vllm_ascend.ascend_config import AscendConfig

    config = types.SimpleNamespace(
        engram_shared_host_dir="/private/generation",
        sparse_kv_offload_config=types.SimpleNamespace(enabled=True),
        uses_packed_sfa_main_cache=False,
    )
    AscendConfig._validate_sparse_packed_kv_offload_compatibility(config)


def test_cross_node_config_and_topology_reuse(backend):
    module, calls = backend
    kwargs = dict(owner=False, capacity=module.GIB, rank=2, local_world_size=2)
    runtime = module.MemfabricRuntime.acquire("generation", "tcp://owner:9000", 4, "npu:0", **kwargs)
    config = calls[0][1]
    assert (config.world_size, config.local_world_size, config.rank_id, config.alloc_size) == (4, 2, 2, 0)
    with pytest.raises(RuntimeError, match="only one"):
        module.MemfabricRuntime.acquire(
            "generation", "tcp://owner:9000", 4, "npu:0", **{**kwargs, "local_world_size": 4}
        )
    runtime.close()


@pytest.mark.parametrize("local_world_size", [0, 3, 5])
def test_invalid_topology_rejected_before_native_init(backend, local_world_size):
    module, calls = backend
    with pytest.raises(ValueError, match="configuration"):
        module.MemfabricRuntime.acquire(
            "generation",
            "tcp://owner:9000",
            4,
            "npu:0",
            owner=True,
            capacity=module.GIB,
            rank=0,
            local_world_size=local_world_size,
        )
    assert calls == []


def test_missing_multi_instance_api_fails_explicitly(backend, monkeypatch):
    module, calls = backend
    monkeypatch.delattr(sys.modules["memfabric_hybrid"].offload, "Offload")
    with pytest.raises(RuntimeError, match="independent offload.Offload API"):
        module.MemfabricRuntime.acquire(
            "generation",
            "tcp://owner:9000",
            2,
            "npu:0",
            owner=True,
            capacity=module.GIB,
            rank=0,
            local_world_size=2,
        )
    assert calls == []
