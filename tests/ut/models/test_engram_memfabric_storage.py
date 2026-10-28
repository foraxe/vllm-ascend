# SPDX-License-Identifier: Apache-2.0
"""Allocation/publication/release invariants with a fake Offload pool, no NPU imports."""

import ctypes
import importlib.util
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
        free=lambda ptr: calls.append(("free", ptr)),
    )
    return SimpleNamespace(offload=offload, device=torch.device("cpu"), rank=0, objects=set(), data=data, calls=calls)


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
