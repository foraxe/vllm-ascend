#!/usr/bin/env python3
"""Two-process Ascend VMM remote-pointer torch NPU kernel gate."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HANDLE_TYPE = "ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT"
INITIAL_OFFSET = 11.0
WRITE_VALUE = 37.0


class ProbeFailure(RuntimeError):
    """A completed operation produced the wrong tensor value."""


class Bridge:
    def __init__(self, library_path: str) -> None:
        self.library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        lib = self.library
        lib.dsa_vmm_last_error.argtypes = []
        lib.dsa_vmm_last_error.restype = ctypes.c_char_p
        lib.dsa_vmm_v2_handle_size.argtypes = []
        lib.dsa_vmm_v2_handle_size.restype = ctypes.c_size_t
        lib.dsa_vmm_get_bare_tgid.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.dsa_vmm_get_bare_tgid.restype = ctypes.c_int
        lib.dsa_vmm_enable_peer.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.dsa_vmm_enable_peer.restype = ctypes.c_int
        lib.dsa_vmm_create_local.argtypes = [
            ctypes.c_int32,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.dsa_vmm_create_local.restype = ctypes.c_int
        lib.dsa_vmm_export_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.dsa_vmm_export_v2.restype = ctypes.c_int
        lib.dsa_vmm_authorize_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.dsa_vmm_authorize_v2.restype = ctypes.c_int
        lib.dsa_vmm_import_v2.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.dsa_vmm_import_v2.restype = ctypes.c_int
        lib.dsa_vmm_set_local_access.argtypes = [ctypes.c_void_p]
        lib.dsa_vmm_set_local_access.restype = ctypes.c_int
        lib.dsa_vmm_region_pointer.argtypes = [ctypes.c_void_p]
        lib.dsa_vmm_region_pointer.restype = ctypes.c_uint64
        lib.dsa_vmm_region_size.argtypes = [ctypes.c_void_p]
        lib.dsa_vmm_region_size.restype = ctypes.c_size_t
        lib.dsa_vmm_destroy_region.argtypes = [ctypes.c_void_p]
        lib.dsa_vmm_destroy_region.restype = ctypes.c_int

    def _check(self, result: int) -> None:
        if result == 0:
            return
        error = self.library.dsa_vmm_last_error()
        message = error.decode("utf-8", errors="replace") if error else "unknown"
        raise RuntimeError(message)

    @property
    def handle_size(self) -> int:
        return int(self.library.dsa_vmm_v2_handle_size())

    def get_bare_tgid(self, device_id: int) -> int:
        bare_tgid = ctypes.c_int32()
        self._check(self.library.dsa_vmm_get_bare_tgid(device_id, ctypes.byref(bare_tgid)))
        return int(bare_tgid.value)

    def enable_peer(self, device_id: int, peer_device_id: int) -> int:
        can_access = ctypes.c_int32()
        self._check(self.library.dsa_vmm_enable_peer(device_id, peer_device_id, ctypes.byref(can_access)))
        return int(can_access.value)

    def create_local(self, device_id: int, requested_size: int) -> ctypes.c_void_p:
        region = ctypes.c_void_p()
        self._check(self.library.dsa_vmm_create_local(device_id, requested_size, ctypes.byref(region)))
        return region

    def export_v2(self, region: ctypes.c_void_p) -> bytes:
        handle = ctypes.create_string_buffer(self.handle_size)
        self._check(self.library.dsa_vmm_export_v2(region, ctypes.byref(handle), self.handle_size))
        return bytes(handle.raw)

    def authorize_v2(self, handle: bytes, bare_tgid: int) -> None:
        handle_buffer = ctypes.create_string_buffer(handle, len(handle))
        self._check(self.library.dsa_vmm_authorize_v2(ctypes.byref(handle_buffer), len(handle), bare_tgid))

    def import_v2(self, device_id: int, handle: bytes, mapped_size: int) -> ctypes.c_void_p:
        handle_buffer = ctypes.create_string_buffer(handle, len(handle))
        region = ctypes.c_void_p()
        self._check(
            self.library.dsa_vmm_import_v2(
                device_id,
                ctypes.byref(handle_buffer),
                len(handle),
                mapped_size,
                ctypes.byref(region),
            )
        )
        return region

    def set_local_access(self, region: ctypes.c_void_p) -> None:
        self._check(self.library.dsa_vmm_set_local_access(region))

    def pointer(self, region: ctypes.c_void_p) -> int:
        return int(self.library.dsa_vmm_region_pointer(region))

    def size(self, region: ctypes.c_void_p) -> int:
        return int(self.library.dsa_vmm_region_size(region))

    def destroy(self, region: ctypes.c_void_p | None) -> None:
        if region is not None and region.value:
            self._check(self.library.dsa_vmm_destroy_region(region))


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _element_size(dtype_name: str) -> int:
    return 4 if dtype_name == "float32" else 2


def _expected_values(torch: Any, elements: int, dtype_name: str, *, device: str | None = None) -> Any:
    dtype = _torch_dtype(torch, dtype_name)
    values = torch.arange(elements, dtype=torch.int32, device=device) % 127
    return values.to(dtype) + INITIAL_OFFSET


def _construct_tensor(
    pointer: int,
    mapped_size: int,
    device_id: int,
    elements: int,
    dtype_name: str,
) -> tuple[Any, Any]:
    import torch
    import torch_npu

    device = torch.device(f"npu:{device_id}")
    storage = torch_npu._C._construct_storage_from_data_pointer(pointer, device, mapped_size)
    metadata = {
        "data_ptr": pointer,
        "device": device,
        "nbytes": mapped_size,
        "dtype": _torch_dtype(torch, dtype_name),
        "size": (elements,),
        "stride": (1,),
        "storage_offset": 0,
    }
    tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
    if tensor.data_ptr() != pointer:
        raise ProbeFailure(f"tensor data_ptr mismatch: expected {pointer:#x}, " f"observed {tensor.data_ptr():#x}")
    return storage, tensor


def _initialize_npu(device_id: int) -> tuple[Any, str, str]:
    import torch
    import torch_npu

    torch.npu.set_device(device_id)
    torch.npu.synchronize()
    return torch, torch.__version__, torch_npu.__version__


def _send_error(connection: Any, role: str, error: BaseException) -> None:
    status = "FAIL" if isinstance(error, ProbeFailure) else "BLOCKED"
    with contextlib.suppress(BrokenPipeError, EOFError, OSError):
        connection.send(
            {
                "stage": "error",
                "role": role,
                "status": status,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def _importer(
    connection: Any,
    library_path: str,
    device_id: int,
    peer_device_id: int,
    elements: int,
    dtype_name: str,
    row_width: int,
    set_access: bool,
) -> None:
    bridge: Bridge | None = None
    region: ctypes.c_void_p | None = None
    storage = None
    tensor = None
    torch = None
    try:
        torch, torch_version, torch_npu_version = _initialize_npu(device_id)
        bridge = Bridge(library_path)
        can_access = bridge.enable_peer(device_id, peer_device_id)
        bare_tgid = bridge.get_bare_tgid(device_id)
        connection.send(
            {
                "stage": "importer_ready",
                "pid": os.getpid(),
                "bare_tgid": bare_tgid,
                "can_access_peer": can_access,
                "torch_version": torch_version,
                "torch_npu_version": torch_npu_version,
            }
        )

        command = connection.recv()
        if command.get("op") != "import":
            raise RuntimeError(f"unexpected importer command: {command!r}")
        region = bridge.import_v2(device_id, command["handle"], command["mapped_size"])
        if set_access:
            bridge.set_local_access(region)
        pointer = bridge.pointer(region)
        mapped_size = bridge.size(region)
        storage, tensor = _construct_tensor(
            pointer,
            mapped_size,
            device_id,
            elements,
            dtype_name,
        )

        observed = tensor.clone()
        torch.npu.synchronize()
        observed_cpu = observed.cpu()
        expected = _expected_values(torch, elements, dtype_name)
        if not torch.equal(observed_cpu, expected):
            mismatch = int((observed_cpu != expected).sum().item())
            raise ProbeFailure(f"remote clone mismatch_count={mismatch}")

        selected_row_indices: list[int] = []
        selected_row_checksum: float | None = None
        if row_width:
            if row_width <= 0 or elements % row_width:
                raise ValueError("row_width must divide elements")
            row_count = elements // row_width
            selected_row_indices = sorted({0, row_count // 2, row_count - 1})
            selected_index = torch.tensor(
                selected_row_indices,
                dtype=torch.int64,
                device=f"npu:{device_id}",
            )
            selected_rows = tensor.view(row_count, row_width).index_select(
                0,
                selected_index,
            )
            torch.npu.synchronize()
            selected_rows_cpu = selected_rows.cpu()
            expected_selected = expected.view(row_count, row_width)[selected_row_indices]
            if not torch.equal(selected_rows_cpu, expected_selected):
                mismatch = int((selected_rows_cpu != expected_selected).sum().item())
                raise ProbeFailure(f"remote selected-row mismatch_count={mismatch}")
            selected_row_checksum = float(selected_rows_cpu.float().sum().item())

        tensor.fill_(WRITE_VALUE)
        torch.npu.synchronize()
        importer_after_write = tensor.clone()
        torch.npu.synchronize()
        importer_after_write_cpu = importer_after_write.cpu()
        if not torch.equal(
            importer_after_write_cpu,
            torch.full(
                (elements,),
                WRITE_VALUE,
                dtype=_torch_dtype(torch, dtype_name),
            ),
        ):
            raise ProbeFailure("importer fill_ self-read mismatch")

        connection.send(
            {
                "stage": "importer_kernel_done",
                "pointer": pointer,
                "mapped_size": mapped_size,
                "read_checksum": float(observed_cpu.sum().item()),
                "read_first": float(observed_cpu[0].item()),
                "read_last": float(observed_cpu[-1].item()),
                "selected_row_indices": selected_row_indices,
                "selected_row_checksum": selected_row_checksum,
                "write_value": WRITE_VALUE,
                "set_access": set_access,
            }
        )

        command = connection.recv()
        if command.get("op") != "cleanup":
            raise RuntimeError(f"unexpected importer command: {command!r}")
        tensor = None
        storage = None
        observed = None
        observed_cpu = None
        selected_index = None
        selected_rows = None
        selected_rows_cpu = None
        expected_selected = None
        importer_after_write = None
        importer_after_write_cpu = None
        gc.collect()
        torch.npu.synchronize()
        bridge.destroy(region)
        region = None
        connection.send({"stage": "importer_unmapped"})
    except BaseException as error:
        _send_error(connection, "importer", error)
    finally:
        tensor = None
        del storage
        gc.collect()
        if bridge is not None and region is not None:
            try:
                if torch is not None:
                    torch.npu.synchronize()
                bridge.destroy(region)
            except BaseException:
                pass
        connection.close()


def _exporter(
    connection: Any,
    library_path: str,
    device_id: int,
    peer_device_id: int,
    elements: int,
    dtype_name: str,
    set_access: bool,
) -> None:
    bridge: Bridge | None = None
    region: ctypes.c_void_p | None = None
    storage = None
    tensor = None
    torch = None
    try:
        torch, torch_version, torch_npu_version = _initialize_npu(device_id)
        bridge = Bridge(library_path)
        can_access = bridge.enable_peer(device_id, peer_device_id)
        connection.send(
            {
                "stage": "exporter_ready",
                "pid": os.getpid(),
                "can_access_peer": can_access,
                "torch_version": torch_version,
                "torch_npu_version": torch_npu_version,
            }
        )

        command = connection.recv()
        if command.get("op") != "create":
            raise RuntimeError(f"unexpected exporter command: {command!r}")
        requested_size = elements * _element_size(dtype_name)
        region = bridge.create_local(device_id, requested_size)
        if set_access:
            bridge.set_local_access(region)
        shareable_handle = bridge.export_v2(region)
        bridge.authorize_v2(shareable_handle, command["importer_bare_tgid"])
        pointer = bridge.pointer(region)
        mapped_size = bridge.size(region)
        storage, tensor = _construct_tensor(
            pointer,
            mapped_size,
            device_id,
            elements,
            dtype_name,
        )

        source = _expected_values(
            torch,
            elements,
            dtype_name,
            device=f"npu:{device_id}",
        )
        tensor.copy_(source)
        torch.npu.synchronize()
        initialized = tensor.clone()
        torch.npu.synchronize()
        initialized_cpu = initialized.cpu()
        expected = _expected_values(torch, elements, dtype_name)
        if not torch.equal(initialized_cpu, expected):
            raise ProbeFailure("exporter copy_ initialization mismatch")
        connection.send(
            {
                "stage": "export_ready",
                "handle": shareable_handle,
                "handle_size": len(shareable_handle),
                "handle_type": HANDLE_TYPE,
                "pointer": pointer,
                "mapped_size": mapped_size,
                "requested_size": requested_size,
                "initial_checksum": float(initialized_cpu.sum().item()),
            }
        )

        command = connection.recv()
        if command.get("op") != "validate":
            raise RuntimeError(f"unexpected exporter command: {command!r}")
        after_remote_write = tensor.clone()
        torch.npu.synchronize()
        after_remote_write_cpu = after_remote_write.cpu()
        expected_after_write = torch.full(
            (elements,),
            WRITE_VALUE,
            dtype=_torch_dtype(torch, dtype_name),
        )
        if not torch.equal(after_remote_write_cpu, expected_after_write):
            mismatch = int((after_remote_write_cpu != expected_after_write).sum().item())
            raise ProbeFailure(f"exporter post-write mismatch_count={mismatch}")
        connection.send(
            {
                "stage": "exporter_validated",
                "post_write_checksum": float(after_remote_write_cpu.sum().item()),
                "post_write_first": float(after_remote_write_cpu[0].item()),
                "post_write_last": float(after_remote_write_cpu[-1].item()),
            }
        )

        command = connection.recv()
        if command.get("op") != "cleanup":
            raise RuntimeError(f"unexpected exporter command: {command!r}")
        tensor = None
        storage = None
        source = None
        initialized = None
        initialized_cpu = None
        after_remote_write = None
        after_remote_write_cpu = None
        gc.collect()
        torch.npu.synchronize()
        bridge.destroy(region)
        region = None
        connection.send({"stage": "exporter_freed"})
    except BaseException as error:
        _send_error(connection, "exporter", error)
    finally:
        tensor = None
        del storage
        gc.collect()
        if bridge is not None and region is not None:
            try:
                if torch is not None:
                    torch.npu.synchronize()
                bridge.destroy(region)
            except BaseException:
                pass
        connection.close()


def _recv(connection: Any, stage: str, timeout_seconds: float) -> dict[str, Any]:
    if not connection.poll(timeout_seconds):
        raise TimeoutError(f"timeout waiting for {stage} after {timeout_seconds:.1f}s")
    message = connection.recv()
    if message.get("stage") == "error":
        error_type = ProbeFailure if message.get("status") == "FAIL" else RuntimeError
        raise error_type(f"{message['role']} {message['error']}\n{message['traceback']}")
    if message.get("stage") != stage:
        raise RuntimeError(f"expected stage {stage!r}, received {message!r}")
    printable = {key: f"<{len(value)} bytes>" if isinstance(value, bytes) else value for key, value in message.items()}
    print(json.dumps(printable, sort_keys=True), flush=True)
    return message


def _stop_process(process: mp.Process, action: dict[str, Any]) -> None:
    if process.pid is None:
        return
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        action["terminated"].append(process.pid)
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        action["killed"].append(process.pid)
        process.join(timeout=5)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exporter-device", type=int, default=0)
    parser.add_argument("--importer-device", type=int, default=1)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--row-width",
        type=int,
        default=0,
        help=(
            "When positive, view the remote tensor as rows of this width and "
            "index_select representative rows before the write-back check."
        ),
    )
    parser.add_argument("--stage-timeout", type=float, default=60.0)
    parser.add_argument(
        "--set-access",
        action="store_true",
        help="Call aclrtMemSetAccess(...READWRITE) after each map.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "hypothesis": (
            "An importer-side ordinary torch NPU kernel can read an exporter "
            "V2-mapped tensor and materialize selected rows locally without "
            "a communication collective."
        ),
        "config": {
            "exporter_device": args.exporter_device,
            "importer_device": args.importer_device,
            "elements": args.elements,
            "dtype": f"torch.{args.dtype}",
            "row_width": args.row_width,
            "initial_offset": INITIAL_OFFSET,
            "write_value": WRITE_VALUE,
            "set_access": args.set_access,
            "stage_timeout_seconds": args.stage_timeout,
            "library": str(Path(args.library).resolve()),
            "git_sha": os.environ.get("PROBE_GIT_SHA", "unknown"),
        },
        "api": {
            "export": ("aclrtMemExportToShareableHandleV2(handle, flags, " "shareType, shareableHandle)"),
            "authorize": ("aclrtMemSetPidToShareableHandleV2(shareableHandle, " "shareType, pid, pidNum)"),
            "import": ("aclrtMemImportFromShareableHandleV2(shareableHandle, " "shareType, flags, handle)"),
            "torch_storage": ("torch_npu._C._construct_storage_from_data_pointer(" "data_ptr, device, nbytes)"),
            "torch_tensor": ("torch_npu._C." "_construct_NPU_Tensor_From_Storage_And_Metadata(" "metadata, storage)"),
        },
        "cleanup": {"terminated": [], "killed": []},
        "stages": {},
    }

    ctx = mp.get_context("spawn")
    importer_parent, importer_child = ctx.Pipe()
    exporter_parent, exporter_child = ctx.Pipe()
    importer = ctx.Process(
        name="vmm-importer-npu1",
        target=_importer,
        args=(
            importer_child,
            args.library,
            args.importer_device,
            args.exporter_device,
            args.elements,
            args.dtype,
            args.row_width,
            args.set_access,
        ),
    )
    exporter = ctx.Process(
        name="vmm-exporter-npu0",
        target=_exporter,
        args=(
            exporter_child,
            args.library,
            args.exporter_device,
            args.importer_device,
            args.elements,
            args.dtype,
            args.set_access,
        ),
    )

    try:
        importer.start()
        importer_child.close()
        result["stages"]["importer_ready"] = _recv(importer_parent, "importer_ready", args.stage_timeout)

        exporter.start()
        exporter_child.close()
        result["stages"]["exporter_ready"] = _recv(exporter_parent, "exporter_ready", args.stage_timeout)

        exporter_parent.send(
            {
                "op": "create",
                "importer_bare_tgid": result["stages"]["importer_ready"]["bare_tgid"],
            }
        )
        export_ready = _recv(exporter_parent, "export_ready", args.stage_timeout)
        result["stages"]["export_ready"] = {key: value for key, value in export_ready.items() if key != "handle"}

        importer_parent.send(
            {
                "op": "import",
                "handle": export_ready["handle"],
                "mapped_size": export_ready["mapped_size"],
            }
        )
        result["stages"]["importer_kernel_done"] = _recv(importer_parent, "importer_kernel_done", args.stage_timeout)

        exporter_parent.send({"op": "validate"})
        result["stages"]["exporter_validated"] = _recv(exporter_parent, "exporter_validated", args.stage_timeout)

        importer_parent.send({"op": "cleanup"})
        result["stages"]["importer_unmapped"] = _recv(importer_parent, "importer_unmapped", args.stage_timeout)
        importer.join(timeout=args.stage_timeout)
        if importer.exitcode != 0:
            raise RuntimeError(f"importer exitcode after unmap: {importer.exitcode}")

        exporter_parent.send({"op": "cleanup"})
        result["stages"]["exporter_freed"] = _recv(exporter_parent, "exporter_freed", args.stage_timeout)
        exporter.join(timeout=args.stage_timeout)
        if exporter.exitcode != 0:
            raise RuntimeError(f"exporter exitcode after free: {exporter.exitcode}")
        result["status"] = "PASS"
    except ProbeFailure as error:
        result["status"] = "FAIL"
        result["error"] = str(error)
    except BaseException as error:
        result["status"] = "BLOCKED"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        _stop_process(importer, result["cleanup"])
        _stop_process(exporter, result["cleanup"])
        importer_parent.close()
        exporter_parent.close()
        result["elapsed_seconds"] = time.time() - started_at
        result["child_exitcodes"] = {
            "importer": importer.exitcode,
            "exporter": exporter.exitcode,
        }
        _write_json(Path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(124))
    raise SystemExit(main())
