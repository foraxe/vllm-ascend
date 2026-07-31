from collections.abc import Sequence

import numpy as np
import torch
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
from vllm.v1.worker.cp_utils import get_total_cp_world_size

from vllm_ascend.attention.context_parallel.c128_packed_pool import PackedPlacement
from vllm_ascend.worker.packed_block_table import PackedBlockTableTranslator


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_sizes: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        num_speculative_tokens: int = 0,
        kv_cache_group: KVCacheGroupSpec = None,
        packed_translator: PackedBlockTableTranslator | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        compress_ratio = 1
        if (
            kv_cache_group is not None
            and hasattr(kv_cache_group, "kv_cache_spec")
            and hasattr(kv_cache_group.kv_cache_spec, "compress_ratio")
        ):
            compress_ratio = kv_cache_group.kv_cache_spec.compress_ratio
        max_num_blocks_per_req = max(cdiv(max_num_blocks_per_req, compress_ratio), 1)
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.physical_block_size = block_size
        self.packed_translator = packed_translator

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_world_size > 1 else 0
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0

        # If kernel_sizes is None or [0], use physical block size (no splitting)
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = 1
            self.use_hybrid_blocks = False
        else:
            # Find the first kernel size that divides physical_block_size evenly
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break

            if selected_kernel_size is None:
                raise ValueError(
                    f"None of the kernel sizes {kernel_sizes} can divide "
                    f"physical block size {self.physical_block_size} evenly"
                )

            self.block_size = selected_kernel_size
            self.logical_block_size = selected_kernel_size
            self.blocks_per_phys_block = self.physical_block_size // self.logical_block_size
            if self.blocks_per_phys_block > 1:
                self.use_hybrid_blocks = True
            else:
                self.use_hybrid_blocks = False
        if (
            self.use_hybrid_blocks
            and self.packed_translator is not None
            and self.packed_translator.placement is PackedPlacement.C128_OWNER
        ):
            raise ValueError("packed C128 owner translation requires matching physical " "and kernel block sizes")

        if self.use_hybrid_blocks:
            logical_table_size = max_num_blocks_per_req * self.blocks_per_phys_block
        else:
            logical_table_size = max_num_blocks_per_req

        duplicate_size = 1
        if self.pcp_world_size * self.dcp_world_size > 1:
            duplicate_size += num_speculative_tokens
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        slot_mapping_capacity = self.max_num_batched_tokens + 2 * self.pcp_world_size * self.max_num_reqs
        self.slot_mapping = self._make_buffer(slot_mapping_capacity, dtype=torch.int32)
        # Packed C128 owner placement rewrites the normal slot mapping in
        # place: locally owned rows become owner-local slots and remote rows
        # become padding. Preserve the pre-translation packed-global domain on
        # device for the current-row overlay path. Other placements do not pay
        # for this buffer and retain their established behavior.
        self.packed_global_slot_mapping = (
            torch.full(
                (slot_mapping_capacity,),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device=self.device,
            )
            if self.packed_translator is not None and self.packed_translator.placement is PackedPlacement.C128_OWNER
            else None
        )
        self.packed_global_slot_mapping_cpu = (
            np.full(
                (slot_mapping_capacity,),
                PAD_SLOT_ID,
                dtype=np.int32,
            )
            if self.packed_global_slot_mapping is not None
            else None
        )

        self.kernel_sizes = kernel_sizes
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids,
        row_idx: int,
    ) -> None:
        prepared = self.prepare_block_ids(block_ids)
        self.append_prepared_row(prepared, row_idx)

    def prepare_block_ids(self, block_ids) -> np.ndarray:
        """Translate one scheduler row without mutating table state."""
        if len(block_ids) == 0:
            return np.empty((0,), dtype=np.int32)
        block_ids = np.array(block_ids)
        if self.packed_translator is not None:
            block_ids = self.packed_translator.translate_group_block_ids(block_ids)
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)
        return block_ids

    def append_prepared_row(
        self,
        block_ids: np.ndarray,
        row_idx: int,
    ) -> None:
        self.validate_prepared_row(
            block_ids,
            row_idx,
            append=True,
        )
        num_blocks = len(block_ids)
        if num_blocks == 0:
            return
        start = self.num_blocks_per_row[row_idx]

        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] += num_blocks

    def add_prepared_row(
        self,
        block_ids: np.ndarray,
        row_idx: int,
    ) -> None:
        self.validate_prepared_row(
            block_ids,
            row_idx,
            append=False,
        )
        self.num_blocks_per_row[row_idx] = 0
        self.append_prepared_row(block_ids, row_idx)

    def validate_prepared_row(
        self,
        block_ids: np.ndarray,
        row_idx: int,
        *,
        append: bool,
    ) -> None:
        """Validate a prepared write without mutating the destination row."""
        if not 0 <= row_idx < self.max_num_reqs:
            raise IndexError(f"row_idx {row_idx} is outside [0, {self.max_num_reqs})")
        start = int(self.num_blocks_per_row[row_idx]) if append else 0
        final_num_blocks = start + len(block_ids)
        logical_capacity = self.block_table.np.shape[1]
        if final_num_blocks > logical_capacity:
            operation = "append" if append else "add"
            raise ValueError(
                f"{operation} requires {final_num_blocks} logical blocks " f"but row capacity is {logical_capacity}"
            )

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        prepared = self.prepare_block_ids(block_ids)
        self.add_prepared_row(prepared, row_idx)

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
        self._preserve_packed_global_slots(num_tokens)
        if self.packed_global_slot_mapping_cpu is not None:
            # This GPU-only path has no authoritative CPU source. Never leave
            # a certificate from the preceding request visible.
            self.packed_global_slot_mapping_cpu.fill(PAD_SLOT_ID)
        self._translate_packed_slots(num_tokens)

    def compute_slot_mapping_draft(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.

        if self.dcp_world_size * self.pcp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % pcp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size

            # IMPORTANT: In hybrid mode, positions are in logical block space,
            # but we need to map them to the correct logical block table indices
            logical_block_idx = positions // virtual_block_size

            # Account for the expanded logical table
            # (always needed with unified tensor)
            # Each physical block is split into multiple logical blocks
            # The logical table has been expanded to accommodate this
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            self.current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
            mask = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size % (self.dcp_world_size * self.pcp_world_size)
                == self.current_rank
            )
            # Calculate local block_offsets
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.pcp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(mask, slot_mapping, -1)
        else:
            assert self.kernel_sizes is not None
            assert self.block_size == self.kernel_sizes[0]
            # IMPORTANT: In hybrid mode, positions are in logical block space,
            # but we need to map them to the correct logical block table indices
            logical_block_idx = positions // self.block_size

            # Account for the expanded logical table
            # (always needed with unified tensor)
            # Each physical block is split into multiple logical blocks
            # The logical table has been expanded to accommodate this
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.np[: req_indices.shape[0]])
            if self.packed_translator is None:
                self.slot_mapping.copy_to_gpu(req_indices.shape[0])
        if self.packed_translator is not None:
            num_slots = req_indices.shape[0]
            if self.packed_global_slot_mapping is not None:
                # The compressed mapping is produced on CPU today. Reuse its
                # established H2D transfer, then snapshot and translate on
                # device. Keep the CPU mirror translated as before without an
                # additional host/device copy.
                self.slot_mapping.copy_to_gpu(num_slots)
                self._preserve_packed_global_slots(num_slots)
                self._translate_packed_slots(num_slots)
                self._preserve_packed_global_slots_cpu(num_slots)
                self._translate_packed_slots(num_slots, cpu_source=True)
            else:
                self._translate_packed_slots(num_slots, cpu_source=True)
                self.slot_mapping.copy_to_gpu(num_slots)

    def _preserve_packed_global_slots(self, num_slots: int) -> None:
        if self.packed_global_slot_mapping is None:
            return
        # Clear the complete reusable buffer before publishing the new active
        # prefix so a shorter batch cannot expose rows from the prior step.
        self.packed_global_slot_mapping.fill_(PAD_SLOT_ID)
        self.packed_global_slot_mapping[:num_slots].copy_(self.slot_mapping.gpu[:num_slots])

    def _preserve_packed_global_slots_cpu(self, num_slots: int) -> None:
        """Snapshot the owner write domain before its in-place CPU rewrite."""
        if self.packed_global_slot_mapping_cpu is None:
            return
        self.packed_global_slot_mapping_cpu.fill(PAD_SLOT_ID)
        self.packed_global_slot_mapping_cpu[:num_slots] = (
            self.slot_mapping.np[:num_slots]
        )

    def _translate_packed_slots(
        self,
        num_slots: int,
        *,
        cpu_source: bool = False,
    ) -> None:
        if self.packed_translator is None:
            return
        slots = self.slot_mapping.cpu[:num_slots] if cpu_source else self.slot_mapping.gpu[:num_slots]
        self.packed_translator.translate_slot_mapping_(
            slots,
            logical_block_size=self.block_size,
            physical_block_size=self.physical_block_size,
            blocks_per_phys_block=self.blocks_per_phys_block,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)
        if self.packed_global_slot_mapping is not None:
            self.packed_global_slot_mapping.fill_(PAD_SLOT_ID)
        if self.packed_global_slot_mapping_cpu is not None:
            self.packed_global_slot_mapping_cpu.fill(PAD_SLOT_ID)

    def _convert_physical_to_logical_blocks(
        self,
        physical_blocks: np.ndarray,
    ) -> np.ndarray:
        """Convert physical block IDs to logical block IDs."""
        if not self.use_hybrid_blocks:
            return physical_blocks

        # Create logical block IDs by splitting each physical block
        logical_blocks: list[int] = []
        for phys_block in physical_blocks:
            # Convert physical block to multiple logical blocks
            # Physical block 1 becomes logical blocks
            # [1*split_ratio, 1*split_ratio+1, ...]
            # But we need to account for the fact that block 0 is special
            base_logical = phys_block * self.blocks_per_phys_block
            logical_blocks.extend(range(base_logical, base_logical + self.blocks_per_phys_block))

        return np.array(logical_blocks, dtype=np.int32)

    def get_device_tensor(self) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu

    def get_packed_global_slot_mapping(self) -> torch.Tensor | None:
        """Return the pre-owner-translation device slots, when available."""
        return self.packed_global_slot_mapping

    def get_packed_global_slot_mapping_cpu(self) -> np.ndarray | None:
        """Return CPU current rows captured before owner-local translation."""
        return self.packed_global_slot_mapping_cpu

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(self, *size: int | torch.SymInt, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: KVCacheGroupSpec = None,
        packed_translators: Sequence[PackedBlockTableTranslator | None] | None = None,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        # Ensure kernel_sizes matches block_sizes length
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * total_cp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        if packed_translators is None:
            packed_translators = [None] * len(block_sizes)
        elif len(packed_translators) != len(block_sizes):
            raise ValueError(
                "packed_translators length "
                f"({len(packed_translators)}) must match block_sizes length "
                f"({len(block_sizes)})"
            )

        # Use zip to pair block_sizes with kernel_sizes one-to-one
        if kv_cache_groups is not None:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                    kv_cache_group,
                    packed_translator,
                )
                for (
                    block_size,
                    kernel_size_list,
                    max_num_blocks_per_req,
                    kv_cache_group,
                    packed_translator,
                ) in zip(
                    block_sizes,
                    kernel_sizes,
                    max_num_blocks,
                    kv_cache_groups,
                    packed_translators,
                )
            ]
        else:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                    packed_translator=packed_translator,
                )
                for (
                    block_size,
                    kernel_size_list,
                    max_num_blocks_per_req,
                    packed_translator,
                ) in zip(
                    block_sizes,
                    kernel_sizes,
                    max_num_blocks,
                    packed_translators,
                )
            ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        prepared_rows = self._prepare_rows(
            block_ids,
            row_idx,
            append=True,
        )
        for block_table, prepared in zip(
            self.block_tables,
            prepared_rows,
        ):
            block_table.append_prepared_row(prepared, row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        prepared_rows = self._prepare_rows(
            block_ids,
            row_idx,
            append=False,
        )
        for block_table, prepared in zip(
            self.block_tables,
            prepared_rows,
        ):
            block_table.add_prepared_row(prepared, row_idx)

    def _prepare_rows(
        self,
        block_ids: tuple[list[int], ...],
        row_idx: int,
        *,
        append: bool,
    ) -> list[np.ndarray]:
        if len(block_ids) != len(self.block_tables):
            raise ValueError(
                f"block_ids group count ({len(block_ids)}) must match " f"block table count ({len(self.block_tables)})"
            )
        prepared_rows = [
            block_table.prepare_block_ids(group_block_ids)
            for block_table, group_block_ids in zip(
                self.block_tables,
                block_ids,
            )
        ]
        for block_table, prepared in zip(
            self.block_tables,
            prepared_rows,
        ):
            block_table.validate_prepared_row(
                prepared,
                row_idx,
                append=append,
            )
        return prepared_rows

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray,
        positions: np.ndarray,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping_draft(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
