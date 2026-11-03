# Shared host-backed Engram with MemFabric Offload

This experimental backend uses the same `OffloadConfig` and `Scene.SHARED`
initialization as sparse KV Offload. A separate owner is rank 0 and contributes
physical DRAM. Every consumer sets `alloc_size=0`, with the same `reserve_size`,
`world_size`, and store URL. Only the owner's `offload.malloc` suballocates
codes and scales; there is one native pool per process, shared by all layers.

The existing vLLM `gather_dequantize_host_uva` operator reads the shared VMM
addresses directly. MemFabric's lookup/copy operators are not used. Consumers
keep metadata-only table tensors, small device pointer arrays, and selected
BF16 output rows. This does not change model collectives or KV transfer.

## Requirements and limitations

- Single-host A3 with working SHARED VMM imports and matching GVA fingerprints.
  This adapter does not configure multi-host `local_world_size` or rank placement.
- MemFabric with nonuniform SHARED allocations, `get_dva`, and the compiled
  `libmf_hybm_accoffload.so` extension. An empty wheel placeholder is insufficient,
  even though Engram does not invoke its lookup kernel. Provision kernels in the
  selected environment before starting workers.
- INT8 Engram codes of width 256, group 32 scales, and a complete ModelSlim
  checkpoint index. FP32 and E8M0 checkpoint scales are loaded as FP32.
- A fresh mode-0700 directory under the same UID, accessible to every process.
  Its manifest contains a capability token and addresses; do not publish it.
- Enough physical DRAM for all tables, aligned to 64 bytes per allocation and
  1 GiB for the pool. SHARED also reserves one capacity-sized virtual slot per
  rank, including zero-allocation consumers. Capacity times world size must fit
  the deployed MemFabric virtual address limit; small-pool success does not
  establish full-model capacity.
- **Sparse KV Offload and shared Engram Offload cannot be enabled together.**
  They would initialize the same native singleton with different configurations.
  Configuration validation rejects this combination. Other direct Offload API
  callers must also not initialize a separate pool in the same process.

## Start and attach

Unlike the former BM backend, Offload requires fixed membership. Set the exact
number of consumer **worker processes**, across all model instances, excluding
the owner. All ranks must initialize concurrently.

```bash
python -m vllm_ascend.shared_engram \
  --model /models/immutable-checkpoint \
  --pool-dir /shared/private/engram-generation-001 \
  --device 0 --consumers 2 --listen-host 127.0.0.1
```

After `POOL_JOINING`, start all configured workers with:

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
--additional-config '{"engram_shared_host_dir":"/shared/private/engram-generation-001"}'
```

Merge these options with the model's other required configuration. Do not wait
for `POOL_READY` before starting workers: initialization needs their participation.
The owner allocates and loads tables after all ranks initialize; consumers wait
for publication before constructing their views. The existing AICPU scheduler
warmup still precedes attachment. The store defaults to an available port;
`--store-url tcp://127.0.0.1:PORT` selects one explicitly.

`--consumers` replaces the former `--max-consumers`. Dynamic membership and
independent worker restart are not supported. Keep the session alive for all
layers. A final session release retires that worker's rank; create a fresh
generation to restart workers.

## Shutdown and recovery

Stop request admission, then all consumers, then the owner. Consumers synchronize
before releasing views and the native pool. Only the owner frees allocations;
it waits for all leases before uninitializing. Unexpected disconnects retain
orphaned leases. Verify that every consumer has stopped before terminating an
orphaned owner, then start a fresh generation. Owner failure during an active
graph remains unvalidated.

## Component validation

The CPU tests cover configuration, ownership, publication bounds, reference
counting, startup rendezvous, and lease teardown. The A3 component benchmark
uses two layers/four tables, native lookup, CPU/HBM reference comparisons, graph
replay, and continued lookup after one layer detaches.

Run these commands concurrently in a configured environment, using free devices:

```bash
python benchmarks/benchmark_shared_engram.py owner \
  --root /shared/private/engram-offload-smoke --device 0 --consumers 2
# Start after POOL_JOINING, in separate terminals:
python benchmarks/benchmark_shared_engram.py consumer \
  --root /shared/private/engram-offload-smoke --device 1 --tag consumer-1
python benchmarks/benchmark_shared_engram.py consumer \
  --root /shared/private/engram-offload-smoke --device 2 --tag consumer-2
```

After both consumers exit, create `/shared/private/engram-offload-smoke/stop`.
Use a new root for another run. Small warm-table timings are not full-model
throughput or DRAM bandwidth measurements. Earlier BM/VMM full-model results
do not validate this Offload backend; complete model and P/D serving remain
separate validation steps.
