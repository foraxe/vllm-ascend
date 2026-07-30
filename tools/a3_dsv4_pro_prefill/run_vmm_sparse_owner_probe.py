#!/usr/bin/env python3
"""Launch the standalone two-rank A3 sparse-owner VMM probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REQUIRED_SYMBOLS = (
    "aclrtReserveMemAddress",
    "aclrtMemGetAllocationGranularity",
    "aclrtMallocPhysical",
    "aclrtMapMem",
    "aclrtUnmapMem",
    "aclrtMemExportToShareableHandleV2",
    "aclrtMemSetPidToShareableHandleV2",
    "aclrtMemImportFromShareableHandleV2",
    "aclrtDeviceCanAccessPeer",
    "aclrtDeviceEnablePeerAccess",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--per-rank-budget-mib", type=int, default=64)
    return parser.parse_args()


def check_abi() -> list[str]:
    library = ctypes.CDLL("libascendcl.so")
    return [name for name in REQUIRED_SYMBOLS if not hasattr(library, name)]


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_rank_json(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def classify_rank_results(
    ranks: list[dict[str, object] | None], returncodes: list[int | None]
) -> str:
    statuses = [rank.get("status") if rank else None for rank in ranks]
    if statuses == ["PASS", "PASS"] and returncodes == [0, 0]:
        exact_counts = all(
            rank
            and rank.get("logical_pages") == 4
            and rank.get("owned_physical_allocations") == 2
            and rank.get("imported_aliases") == 2
            and rank.get("local_mismatches") == 0
            and rank.get("peer_read_mismatches") == 0
            and rank.get("remote_write_mismatches") == 0
            and rank.get("imports_released_barrier") is True
            for rank in ranks
        )
        return "PASS" if exact_counts else "FAIL_ACCOUNTING"
    if (
        statuses == ["BLOCKED_CAPABILITY", "BLOCKED_CAPABILITY"]
        and returncodes == [3, 3]
    ):
        return "BLOCKED_CAPABILITY"
    if any(status and status.startswith("BLOCKED") for status in statuses):
        return "FAIL_INCOHERENT_CAPABILITY"
    return "FAIL"


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    devices = [int(value) for value in args.devices.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    socket_path = f"/tmp/dsa-g27-{os.getpid()}.sock"
    missing_symbols = check_abi()
    started_at = time.time()
    report: dict[str, object] = {
        "gate": "G27_sparse_owner_vmm_v2",
        "status": "BLOCKED_ABI" if missing_symbols else "RUNNING",
        "devices": devices,
        "logical_pages": 4,
        "owner_formula": "logical_page % 2",
        "local_page_formula": "logical_page // 2",
        "missing_symbols": missing_symbols,
        "timeout_seconds": args.timeout,
        "per_rank_budget_mib": args.per_rank_budget_mib,
        "binary": str(binary),
    }
    if len(devices) != 2 or devices[0] == devices[1]:
        report["status"] = "FAIL_INVALID_ARGUMENT"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 2
    if missing_symbols:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 3

    rank_logs = [
        args.output.with_name(f"{args.output.stem}_rank{rank}{args.output.suffix}")
        for rank in range(2)
    ]
    processes: list[subprocess.Popen[str]] = []
    streams: list[tuple[object, object]] = []
    try:
        for rank in range(2):
            stdout_path = rank_logs[rank].with_suffix(".stdout.log")
            stderr_path = rank_logs[rank].with_suffix(".stderr.log")
            stdout_stream = stdout_path.open("w")
            stderr_stream = stderr_path.open("w")
            streams.append((stdout_stream, stderr_stream))
            command = [
                str(binary),
                str(rank),
                socket_path,
                str(devices[rank]),
                str(devices[1 - rank]),
                str(args.per_rank_budget_mib * 1024 * 1024),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    text=True,
                    start_new_session=True,
                )
            )

        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                report["status"] = "FAIL_TIMEOUT"
                break
            time.sleep(0.1)
    finally:
        for process in processes:
            stop_process(process)
        for stdout_stream, stderr_stream in streams:
            stdout_stream.close()
            stderr_stream.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass

    ranks: list[dict[str, object] | None] = []
    for rank in range(2):
        stdout_path = rank_logs[rank].with_suffix(".stdout.log")
        ranks.append(parse_rank_json(stdout_path.read_text()))
    report["ranks"] = ranks
    report["returncodes"] = [process.returncode for process in processes]
    report["elapsed_seconds"] = time.time() - started_at

    if report["status"] != "FAIL_TIMEOUT":
        report["status"] = classify_rank_results(ranks, report["returncodes"])

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
