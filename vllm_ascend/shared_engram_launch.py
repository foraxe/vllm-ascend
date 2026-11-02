# SPDX-License-Identifier: Apache-2.0
"""Launch an Engram owner or consumer before an existing service command."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def model_identity(model):
    digest = hashlib.sha256()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((Path(model) / name).read_bytes())
    return digest.hexdigest()


def wait_ready(pool_dir, identity, timeout, owner=None, stopped=lambda: False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("Engram startup was interrupted")
        if owner is not None and owner.poll() is not None:
            raise RuntimeError(f"Engram owner exited with status {owner.returncode}")
        try:
            root = Path(pool_dir)
            stat = root.stat()
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                raise ValueError("Shared Engram directory must be uid-owned and mode0700")
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["identity"] != identity:
                raise ValueError("Shared Engram checkpoint identity mismatch")
            if manifest["schema"] != 2 or manifest.get("backend") != "memfabric_bm":
                raise ValueError("Unsupported shared Engram manifest schema")
            if manifest["state"] == "ready":
                with socket.create_connection(tuple(manifest["endpoint"]), timeout=5) as conn:
                    conn.settimeout(5)
                    conn.sendall((json.dumps({"op": "ping", "token": manifest["token"]}) + "\n").encode())
                    with conn.makefile("rb") as wire:
                        response = json.loads(wire.readline(4096))
                    if response.get("alive") is True:
                        return
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Engram owner was not ready within {timeout} seconds")


def bootstrap_local(pool_dir, identity, host, port, timeout, stopped=lambda: False):
    """Fetch the BM discovery manifest into a private local directory."""
    deadline = time.monotonic() + timeout
    response = None
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("Engram startup was interrupted")
        try:
            with socket.create_connection((host, port), timeout=5) as conn:
                conn.settimeout(5)
                request = {"op": "bootstrap", "identity": identity}
                conn.sendall((json.dumps(request) + "\n").encode())
                with conn.makefile("rb") as wire:
                    response = json.loads(wire.readline((1 << 20) + 1))
            if "error" not in response:
                break
            if response["error"] != "Engram pool is not ready":
                raise ValueError("Engram owner rejected bootstrap")
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    else:
        raise TimeoutError(f"Engram owner was not ready within {timeout} seconds")

    manifest = response["manifest"]
    if (
        manifest.get("schema") != 2
        or manifest.get("backend") != "memfabric_bm"
        or manifest.get("identity") != identity
        or manifest.get("state") != "ready"
    ):
        raise ValueError("Incompatible shared Engram manifest")
    if not isinstance(manifest.get("tensors"), dict) or not manifest["tensors"]:
        raise ValueError("Invalid Engram tensor manifest")
    root = Path(pool_dir)
    root.mkdir(mode=0o700)
    try:
        fd = os.open(root / "manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(manifest, file)
    except BaseException:
        shutil.rmtree(root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("owner", "consumer"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--listen-host", help="Reachable owner address; required for the owner role")
    parser.add_argument("--owner-host", help="Owner address for network bootstrap")
    parser.add_argument("--owner-port", type=int, help="Fixed owner port for network bootstrap")
    parser.add_argument("--store-url", help="Fixed MemFabric BM store URL for the owner")
    parser.add_argument("--max-consumers", type=int, help="Maximum simultaneous consumer workers")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Service command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or (args.role == "owner" and not args.listen_host):
        parser.error("A service command, positive timeout, and owner listen address are required")
    network = args.owner_host is not None or args.owner_port is not None
    if network and (not args.owner_port or (args.role == "consumer" and not args.owner_host)):
        parser.error("Network bootstrap requires an owner port, plus owner host on consumers")
    if args.owner_port is not None and not 1 <= args.owner_port <= 65535:
        parser.error("Owner port must be between 1 and 65535")
    if args.max_consumers is not None and args.max_consumers < 1:
        parser.error("max-consumers must be positive")
    if args.role != "owner" and (args.store_url is not None or args.max_consumers is not None):
        parser.error("store-url and max-consumers are owner options")

    owner = None
    service = None
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        if service is not None and service.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service.pid, signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        identity = model_identity(args.model)
        if args.role == "owner":
            if Path(args.pool_dir).exists():
                raise ValueError("Owner requires a new, nonexistent pool directory")
            owner_command = [
                sys.executable,
                "-m",
                "vllm_ascend.shared_engram",
                "--model",
                args.model,
                "--pool-dir",
                args.pool_dir,
                "--device",
                str(args.device),
                "--listen-host",
                args.listen_host,
            ]
            if network:
                owner_command += ["--port", str(args.owner_port), "--network-bootstrap"]
            if args.store_url is not None:
                owner_command += ["--store-url", args.store_url]
            if args.max_consumers is not None:
                owner_command += ["--max-consumers", str(args.max_consumers)]
            owner = subprocess.Popen(owner_command)
        if network and args.role == "consumer":
            bootstrap_local(args.pool_dir, identity, args.owner_host, args.owner_port, args.timeout, lambda: stop)
        else:
            wait_ready(args.pool_dir, identity, args.timeout, owner, lambda: stop)
        if stop:
            return 1
        service = subprocess.Popen(command, start_new_session=True)
        if stop:
            request_stop(signal.SIGTERM, None)
        while service.poll() is None:
            if owner is not None and owner.poll() is not None:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait()
                raise RuntimeError("Engram owner exited while the service was running")
            time.sleep(1)
        return service.returncode
    finally:
        if owner is not None and owner.poll() is None:
            owner.terminate()
            # The owner intentionally waits for all remote leases to drain.
            owner.wait()


if __name__ == "__main__":
    sys.exit(main())
