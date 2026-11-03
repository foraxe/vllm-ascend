# SPDX-License-Identifier: Apache-2.0
"""Four-rank correctness probe: two DRAM contributors and two lookup consumers.

This tests the native Offload pool and the unchanged vLLM lookup kernel. It
does not extend SharedPoolOwner's production control protocol or checkpoint
loader. Run with four free A3 devices visible in the selected environment.
Only queried rows are initialized; this is not a bandwidth benchmark.
"""

import argparse
import ctypes
import json
import multiprocessing as mp
import socket
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

GIB = 1 << 30
SLOT_BYTES = 2 * GIB
CHUNK_ROWS = 1 << 22
WIDTH = 256
GROUP = 32
WORLD_SIZE = 4
OWNER_COUNT = 2
ROW_COUNT = CHUNK_ROWS * OWNER_COUNT
WAIT_SECONDS = 120
GRAPH_REPLAYS = 3


def query_cases(torch, rank):
    boundary = torch.tensor(
        [[0, CHUNK_ROWS - 1, CHUNK_ROWS], [CHUNK_ROWS + 1, ROW_COUNT - 1, 17], [-1, ROW_COUNT, CHUNK_ROWS - 2]],
        dtype=torch.int64,
    )
    cases = [("boundary", boundary)]
    for tokens in (32, 128, 512):
        generator = torch.Generator().manual_seed(rank * 1000 + tokens)
        ids = torch.randint(0, ROW_COUNT, (tokens, 24), generator=generator)
        # Every case must read both owners, independent of the random seed.
        ids[0, :4] = torch.tensor([CHUNK_ROWS - 1, CHUNK_ROWS, 0, ROW_COUNT - 1])
        cases.append((f"tokens-{tokens}", ids))
    return cases


def row_values(torch, rows, epoch):
    owner = rows[:, None] // CHUNK_ROWS
    local = rows[:, None] % CHUNK_ROWS
    codes = (local * 17 + torch.arange(WIDTH)[None, :] * 7 + owner * 29 + epoch * 13) % 255 - 127
    scales = (local * 3 + torch.arange(WIDTH // GROUP)[None, :] + owner * 5 + epoch * 7) % 16 + 1
    return codes.to(torch.int8), scales.float() / 16


def expected_values(torch, ids, epoch):
    rows = ids.flatten()
    valid = (rows >= 0) & (rows < ROW_COUNT)
    codes, scales = row_values(torch, rows.clamp(0, ROW_COUNT - 1), epoch)
    result = (codes.float() * scales.repeat_interleave(GROUP, 1)).bfloat16()
    result[~valid] = 0
    return result


def publish_rows(torch, rank, code_address, scale_address, epoch):
    selected = torch.cat([ids.flatten() for consumer in (2, 3) for _, ids in query_cases(torch, consumer)])
    selected = torch.unique(selected[(selected >= rank * CHUNK_ROWS) & (selected < (rank + 1) * CHUNK_ROWS)])
    codes, scales = row_values(torch, selected, epoch)
    for index, global_row in enumerate(selected.tolist()):
        local_row = global_row - rank * CHUNK_ROWS
        ctypes.memmove(code_address + local_row * WIDTH, codes.data_ptr() + index * WIDTH, WIDTH)
        ctypes.memmove(
            scale_address + local_row * (WIDTH // GROUP) * 4,
            scales.data_ptr() + index * (WIDTH // GROUP) * 4,
            (WIDTH // GROUP) * 4,
        )
    return len(selected)


def run_rank(rank, store_url, addresses, barrier, consumer_done, second_owner_done, output_dir):
    log = Path(output_dir) / f"rank-{rank}.log"
    # Native libraries and Python share the same per-process diagnostic stream.
    import os
    import sys

    with log.open("w", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            import memfabric_hybrid
            import torch
            import torch_npu  # noqa: F401
            from memfabric_hybrid import offload

            import vllm_ascend.ops  # noqa: F401
            from vllm_ascend.models.deepseek_v41.engram.npu import gather_dequantize_host_uva

            torch.npu.set_device(rank)
            config = offload.OffloadConfig()
            config.device_id = rank
            config.rank_id = rank
            config.world_size = WORLD_SIZE
            config.local_world_size = WORLD_SIZE
            config.reserve_size = SLOT_BYTES
            config.alloc_size = SLOT_BYTES if rank < OWNER_COUNT else 0
            config.scene = offload.Scene.SHARED
            config.store_url = store_url
            print(
                json.dumps(
                    {
                        "stage": "INIT",
                        "rank": rank,
                        "alloc_size": config.alloc_size,
                        "reserve_size": config.reserve_size,
                        "package": memfabric_hybrid.__file__,
                    }
                ),
                flush=True,
            )
            assert offload.initialize(config) == 0, f"Offload initialization failed on rank {rank}"
            code_address = scale_address = None
            if rank < OWNER_COUNT:
                code_address = offload.malloc(CHUNK_ROWS * WIDTH)
                scale_address = offload.malloc(CHUNK_ROWS * (WIDTH // GROUP) * 4)
                assert code_address and scale_address
                assert offload.get_dva(code_address) == code_address
                assert offload.get_dva(scale_address) == scale_address
                addresses[rank * 2] = code_address
                addresses[rank * 2 + 1] = scale_address
            barrier.wait(WAIT_SECONDS)
            assert addresses[2] - addresses[0] == SLOT_BYTES, "owners must occupy distinct native rank slots"
            print(json.dumps({"stage": "MAPPED", "rank": rank, "addresses": list(addresses)}), flush=True)

            graphs = []
            summaries = []
            if rank >= OWNER_COUNT:
                codes = SimpleNamespace(
                    tensor=torch.empty((ROW_COUNT, WIDTH), dtype=torch.int8, device="meta"),
                    ptrs=torch.tensor([addresses[0], addresses[2]], dtype=torch.int64, device=f"npu:{rank}"),
                )
                scales = SimpleNamespace(
                    tensor=torch.empty((ROW_COUNT, WIDTH // GROUP), dtype=torch.float32, device="meta"),
                    ptrs=torch.tensor([addresses[1], addresses[3]], dtype=torch.int64, device=f"npu:{rank}"),
                )

            for epoch in (0, 1):
                if rank < OWNER_COUNT:
                    count = publish_rows(torch, rank, code_address, scale_address, epoch)
                    print(json.dumps({"stage": "PUBLISHED", "rank": rank, "epoch": epoch, "rows": count}), flush=True)
                barrier.wait(WAIT_SECONDS)
                if rank >= OWNER_COUNT:
                    for case_index, (name, ids_cpu) in enumerate(query_cases(torch, rank)):
                        expected = expected_values(torch, ids_cpu, epoch)
                        if epoch == 0:
                            ids = ids_cpu.to(f"npu:{rank}")
                            actual = gather_dequantize_host_uva(codes, scales, ids, local_heads=ids.shape[1])
                            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                            graph = torch.npu.NPUGraph()
                            with torch.npu.graph(graph):
                                result = gather_dequantize_host_uva(codes, scales, ids, local_heads=ids.shape[1])
                            graphs.append((graph, result, ids))
                        graph, result, _ = graphs[case_index]
                        for _ in range(GRAPH_REPLAYS):
                            graph.replay()
                        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
                        summary = {
                            "rank": rank,
                            "epoch": epoch,
                            "case": name,
                            "lookup_rows": ids_cpu.numel(),
                            "owners_read": [0, 1],
                            "pass": True,
                        }
                        summaries.append(summary)
                        print(json.dumps(summary), flush=True)
                barrier.wait(WAIT_SECONDS)

            if rank >= OWNER_COUNT:
                torch.npu.synchronize()
                graphs.clear()
                del codes, scales, graph, result, ids
                offload.uninitialize()
                consumer_done[rank - OWNER_COUNT].set()
                Path(output_dir, f"consumer-{rank}.json").write_text(json.dumps(summaries, indent=2) + "\n")
                print("TWO_OWNERS_LOOKUP_GRAPH_UPDATE_PASS", flush=True)
            else:
                for done in consumer_done:
                    assert done.wait(WAIT_SECONDS), "consumer did not release mappings"
                if rank == 0:
                    assert second_owner_done.wait(WAIT_SECONDS), "owner 1 did not release its pool"
                offload.free(scale_address)
                offload.free(code_address)
                offload.uninitialize()
                if rank == 1:
                    second_owner_done.set()
                print("OWNER_RELEASED_PASS", flush=True)
            loaded = sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "memfabric_hybrid" in line and ".so" in line
                }
            )
            Path(output_dir, f"libraries-{rank}.json").write_text(json.dumps(loaded, indent=2) + "\n")
            sys.stdout.flush()
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="A new private directory for results")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(mode=0o700)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        store_url = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
    context = mp.get_context("spawn")
    addresses = context.Array("Q", OWNER_COUNT * 2, lock=False)
    barrier = context.Barrier(WORLD_SIZE)
    consumer_done = [context.Event() for _ in range(WORLD_SIZE - OWNER_COUNT)]
    second_owner_done = context.Event()
    processes = [
        context.Process(
            target=run_rank, args=(rank, store_url, addresses, barrier, consumer_done, second_owner_done, str(output))
        )
        for rank in range(WORLD_SIZE)
    ]
    deadline = time.monotonic() + 240
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("A rank failed; inspect rank logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Four-rank test exceeded its deadline")
            time.sleep(0.2)
        codes = [process.exitcode for process in processes]
        assert codes == [0] * WORLD_SIZE, codes
        result = {
            "result": "PASS",
            "exit_codes": codes,
            "world_size": WORLD_SIZE,
            "owner_alloc_bytes": [SLOT_BYTES, SLOT_BYTES],
            "consumer_alloc_bytes": [0, 0],
            "logical_rows": ROW_COUNT,
            "queried_rows_only_initialized": True,
            "virtual_reserve_bytes": WORLD_SIZE * SLOT_BYTES,
            "epochs": 2,
            "graph_replays_per_case_per_epoch": GRAPH_REPLAYS,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for process in processes:
            if process.pid and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


if __name__ == "__main__":
    main()
