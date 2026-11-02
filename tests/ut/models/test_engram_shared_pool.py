# SPDX-License-Identifier: Apache-2.0
"""Real TCP lifecycle tests with a fake allocation backend; no NPU needed."""

import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def pool_module(monkeypatch):
    package = types.ModuleType("_shared_pool_ut")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module("_shared_pool_ut.shared_pool")

    class Buffer:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.fail_release = False

        def view(self, start, end):
            return types.SimpleNamespace(close=lambda: None)

        def close(self):
            if self.fail_release:
                raise RuntimeError("device release failed")
            self.closed = True

    monkeypatch.setattr(module, "MappedHostBuffer", Buffer)

    class Runtime:
        @staticmethod
        def acquire(*args, **kwargs):
            return types.SimpleNamespace(rank=0, close=lambda: None)

    monkeypatch.setattr(module, "MemfabricRuntime", Runtime)
    yield module
    for name in list(sys.modules):
        if name.startswith("_shared_pool_ut."):
            sys.modules.pop(name)


@pytest.fixture
def owner(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(tmp_path / "pool", "model-a", device="npu:0")
    owner.add("codes", 32, 256, torch.int8)
    owner.ready()
    yield owner
    owner.close(timeout=2)


def test_identity_rejection_does_not_hold_lease(pool_module, owner):
    with pytest.raises(ValueError, match="model-compatible"):
        pool_module.SharedPoolClient(owner.root, "model-b", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_drain_waits_for_consumer_and_rejects_new_attach(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    finally:
        client.close()
    owner.close(timeout=1)
    assert owner.buffers["codes"].closed


def test_failed_device_release_retains_lease_until_retry(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    buffer = client.buffers["codes"]
    try:
        buffer.fail_release = True
        with pytest.raises(RuntimeError, match="device release failed"):
            client.close()
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
    finally:
        buffer.fail_release = False
        client.close()
    owner.close(timeout=1)


def test_layout_rejected_before_mapping(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="layout mismatch"):
            client.view("codes", 0, 16, rows=33, width=256, dtype=torch.int8)
        assert not client.buffers
    finally:
        client.close()


def test_consumer_reattaches_without_reloading_owner(pool_module, owner):
    original = owner.buffers["codes"]
    for _ in range(2):
        client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
        client.close()
    assert owner.buffers["codes"] is original
    assert not original.closed


def test_private_directory_required(pool_module, owner):
    owner.root.chmod(0o755)
    with pytest.raises(ValueError, match="mode0700"):
        pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    owner.root.chmod(0o700)


def test_old_schema_rejected_and_lease_released(pool_module, owner):
    owner.manifest["schema"] = 1
    with pytest.raises(ValueError, match="manifest"):
        pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_invalid_bm_object_rejected_before_mapping(pool_module, owner):
    owner.manifest["tensors"]["codes"]["object_id"] = -1
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="object id"):
            client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
        assert not client.buffers
        assert client.runtime is None
    finally:
        client.close()


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_launcher_ut", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_network_bootstrap_local_directory_attaches(pool_module, owner, launcher, tmp_path):
    owner.network_bootstrap = True
    local = tmp_path / "consumer-local"
    host, port = owner.manifest["endpoint"]
    launcher.bootstrap_local(local, "model-a", host, port, 5)
    assert local.stat().st_mode & 0o777 == 0o700
    assert json.loads((local / "manifest.json").read_text())["backend"] == "memfabric_bm"
    client = pool_module.SharedPoolClient(local, "model-a", device="npu:0")
    try:
        client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
        with pytest.raises(TimeoutError):
            owner.close()
    finally:
        client.close()


def test_network_bootstrap_requires_opt_in_and_matching_identity(owner, launcher, tmp_path):
    host, port = owner.manifest["endpoint"]
    local = tmp_path / "consumer-local"
    with pytest.raises(ValueError, match="rejected bootstrap"):
        launcher.bootstrap_local(local, "model-a", host, port, 5)
    assert not local.exists()
    owner.network_bootstrap = True
    with pytest.raises(ValueError, match="rejected bootstrap"):
        launcher.bootstrap_local(local, "wrong-model", host, port, 5)
    assert not local.exists()


def test_old_network_capability_is_rejected(pool_module, owner, launcher, tmp_path):
    owner.network_bootstrap = True
    local = tmp_path / "consumer-local"
    host, port = owner.manifest["endpoint"]
    launcher.bootstrap_local(local, "model-a", host, port, 5)
    replacement = pool_module.SharedPoolOwner(tmp_path / "replacement", "model-a", device="npu:0")
    replacement.add("codes", 32, 256, torch.int8)
    replacement.ready()
    try:
        stale = json.loads((local / "manifest.json").read_text())
        stale["endpoint"] = replacement.manifest["endpoint"]
        (local / "manifest.json").write_text(json.dumps(stale))
        with pytest.raises(ValueError, match="Invalid Engram capability"):
            pool_module.SharedPoolClient(local, "model-a", device="npu:0")
        assert not replacement._leases
    finally:
        replacement.close(timeout=1)


def test_consumer_launcher_bootstraps_in_separate_process(owner, launcher, tmp_path):
    owner.network_bootstrap = True
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (model / name).write_text("{}")
    owner.manifest["identity"] = launcher.model_identity(model)
    owner._publish_manifest()
    local = tmp_path / "consumer-local"
    host, port = owner.manifest["endpoint"]
    result = subprocess.run(
        [
            sys.executable,
            launcher.__file__,
            "--role",
            "consumer",
            "--model",
            str(model),
            "--pool-dir",
            str(local),
            "--owner-host",
            host,
            "--owner-port",
            str(port),
            "--",
            sys.executable,
            "-c",
            "print('service-started')",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "service-started"
    assert json.loads((local / "manifest.json").read_text())["identity"] == owner.manifest["identity"]
