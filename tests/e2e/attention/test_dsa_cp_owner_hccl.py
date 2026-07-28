# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Real-HCCL gate for C128 owner placement and local staging.

Run this on an eight-NPU Ascend host.  It deliberately invokes the production
``scatter_prepared`` and ``materialize_for_attention`` methods rather than a
CPU model: every TP rank observes the same compressor output, writes only its
canonical logical pages, and then HCCL stages the union of pages requested by
the ranks' block tables.  The staged cache must reconstruct the original
compressed rows exactly.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.attention.context_parallel.c128_owner_cache import C128OwnerShardCache


_WORLD_SIZE = 8
_PAGES_PER_OWNER = 3
_PAGE_SIZE = 4
_KV_SHAPE = (1, 2)


def _worker(rank: int, port: int, result_queue) -> None:
    """Execute the production owner cache path for one rank."""
    try:
        torch_npu.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=_WORLD_SIZE,
            init_method=f"tcp://127.0.0.1:{port}",
        )

        total_pages = _WORLD_SIZE * _PAGES_PER_OWNER
        total_rows = total_pages * _PAGE_SIZE
        persistent = torch.zeros((_PAGES_PER_OWNER, _PAGE_SIZE, *_KV_SHAPE), dtype=torch.float32).npu()
        stage = torch.empty((total_pages, _PAGE_SIZE, *_KV_SHAPE), dtype=torch.float32).npu()
        owner_cache = C128OwnerShardCache(persistent, stage, tp_size=_WORLD_SIZE)

        # A deterministic stand-in for the current compressor output. Every
        # rank receives the same global rows, as the DSA compressor does after
        # its required producer synchronization.
        compressed_kv = torch.arange(total_rows * 2, dtype=torch.float32).reshape(total_rows, *_KV_SHAPE).npu()
        page_ids = torch.arange(total_pages, dtype=torch.int64).repeat_interleave(_PAGE_SIZE).npu()
        offsets = torch.arange(_PAGE_SIZE, dtype=torch.int64).repeat(total_pages).npu()
        slot_mapping = torch.stack((page_ids, offsets), dim=1)

        plan = owner_cache.prepare_owned_scatter(slot_mapping, tp_rank=rank)
        owner_cache.scatter_prepared(compressed_kv, *plan, tp_rank=rank)

        # Each rank requests three different pages. Their union covers all
        # logical pages, exercising rank-varying block tables and all owners.
        local_pages = torch.arange(rank, total_pages, _WORLD_SIZE, dtype=torch.int64).npu()
        block_table = torch.full((_PAGES_PER_OWNER, _PAGES_PER_OWNER), -1, dtype=torch.int64).npu()
        block_table[:, 0] = local_pages
        staged, remapped = owner_cache.materialize_for_attention(
            block_table,
            tp_rank=rank,
            group=dist.group.WORLD,
        )
        torch.npu.synchronize()

        expected = compressed_kv.reshape(total_pages, _PAGE_SIZE, *_KV_SHAPE)
        torch.testing.assert_close(staged.cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(remapped[:, 0].cpu(), local_pages.cpu(), rtol=0, atol=0)
        result_queue.put((rank, "PASS"))
    except Exception as error:  # pragma: no cover - failure is returned to parent
        result_queue.put((rank, f"FAIL: {type(error).__name__}: {error}"))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def test_c128_owner_scatter_and_stage_hccl() -> None:
    """HCCL staging exactly reconstructs the producer's compressed C128 cache."""
    mp.set_start_method("fork", force=True)
    result_queue = mp.SimpleQueue()
    port = 29501 + random.randint(0, 10000)
    workers = [mp.Process(target=_worker, args=(rank, port, result_queue)) for rank in range(_WORLD_SIZE)]
    for worker in workers:
        worker.start()
    results = [result_queue.get() for _ in workers]
    for worker in workers:
        worker.join(timeout=180)
    failures = [result for result in results if result[1] != "PASS"]
    deadlocked = [worker.pid for worker in workers if worker.is_alive()]
    assert not deadlocked, f"owner HCCL workers did not terminate: {deadlocked}"
    assert not failures, f"owner HCCL failures: {failures}"
    assert all(worker.exitcode == 0 for worker in workers)
