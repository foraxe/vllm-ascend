# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Lazy non-owning Torch-NPU tensor aliases for packed C128 VMM arenas."""

from __future__ import annotations

import importlib
from typing import Any

from .c128_packed_acl_backend import PackedArenaAdapterUnavailable


class TorchNpuTensorBindingError(RuntimeError):
    """A mapped pointer could not be represented by the requested tensor."""


class TorchNpuPackedArenaTensorBinding:
    """Strong references to one external storage alias and its root tensor.

    The ACL arena owns the virtual and physical memory.  Closing this binding
    only drops Python references; it never frees or unmaps the external
    address.
    """

    def __init__(self, *, storage: object, root_tensor: object) -> None:
        self._storage: object | None = storage
        self._root_tensor: object | None = root_tensor

    def tensor(self) -> object:
        if self._root_tensor is None:
            raise RuntimeError("packed-arena tensor binding is closed")
        return self._root_tensor

    def close(self) -> None:
        # Tensor must go first because it references the storage object.
        self._root_tensor = None
        self._storage = None


class TorchNpuPackedArenaTensorFactory:
    """Construct byte tensors over external mapped NPU virtual addresses."""

    def __init__(
        self,
        *,
        torch_module: object | None = None,
        torch_npu_module: object | None = None,
    ) -> None:
        self._torch_module = torch_module
        self._torch_npu_module = torch_npu_module

    def _modules(self) -> tuple[object, object]:
        try:
            torch_module = self._torch_module or importlib.import_module("torch")
            torch_npu_module = self._torch_npu_module or importlib.import_module("torch_npu")
        except ImportError as error:
            raise PackedArenaAdapterUnavailable(f"Torch-NPU tensor binding is unavailable: {error}") from error
        return torch_module, torch_npu_module

    @staticmethod
    def _constructors(torch_npu_module: object) -> tuple[Any, Any]:
        extension = getattr(torch_npu_module, "_C", None)
        storage_constructor = getattr(
            extension,
            "_construct_storage_from_data_pointer",
            None,
        )
        tensor_constructor = getattr(
            extension,
            "_construct_NPU_Tensor_From_Storage_And_Metadata",
            None,
        )
        if not callable(storage_constructor) or not callable(tensor_constructor):
            raise PackedArenaAdapterUnavailable("torch_npu lacks the external-storage tensor constructors")
        return storage_constructor, tensor_constructor

    def bind(
        self,
        *,
        base_address: int,
        size_bytes: int,
        device_index: int,
    ) -> TorchNpuPackedArenaTensorBinding:
        if base_address <= 0:
            raise ValueError("base_address must be positive")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if device_index < 0:
            raise ValueError("device_index must be non-negative")

        torch_module, torch_npu_module = self._modules()
        storage_constructor, tensor_constructor = self._constructors(torch_npu_module)
        try:
            device = torch_module.device(f"npu:{device_index}")
            dtype = torch_module.uint8
        except (AttributeError, TypeError, RuntimeError) as error:
            raise PackedArenaAdapterUnavailable(f"cannot construct local NPU device metadata: {error}") from error

        storage: object | None = None
        root_tensor: object | None = None
        metadata = {
            "data_ptr": base_address,
            "device": device,
            "nbytes": size_bytes,
            "dtype": dtype,
            "size": (size_bytes,),
            "stride": (1,),
            "storage_offset": 0,
        }
        try:
            storage = storage_constructor(
                base_address,
                device,
                size_bytes,
            )
            root_tensor = tensor_constructor(metadata, storage)
            if int(root_tensor.data_ptr()) != base_address:
                raise TorchNpuTensorBindingError(
                    "external tensor data_ptr mismatch: expected "
                    f"{base_address:#x}, observed "
                    f"{int(root_tensor.data_ptr()):#x}"
                )
            if int(root_tensor.numel()) != size_bytes:
                raise TorchNpuTensorBindingError("external tensor numel does not span the mapped range")
            if int(root_tensor.element_size()) != 1:
                raise TorchNpuTensorBindingError("external packed-arena tensor is not byte-addressable")
            if root_tensor.dtype != dtype:
                raise TorchNpuTensorBindingError("external packed-arena tensor dtype is not uint8")
            if root_tensor.device != device:
                raise TorchNpuTensorBindingError("external tensor uses the wrong local NPU device tag")
        except BaseException:
            # These aliases do not own the mapped pointer. Clearing both
            # references is the complete failure rollback for this seam.
            root_tensor = None
            storage = None
            raise
        return TorchNpuPackedArenaTensorBinding(
            storage=storage,
            root_tensor=root_tensor,
        )
