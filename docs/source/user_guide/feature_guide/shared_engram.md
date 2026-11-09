# Shared host-backed Engram with MemFabric Offload

This experimental backend shares an independent `offload.Offload()` instance
across Engram layers. Rank 0 owns the table DRAM; consumers use `alloc_size=0`
and matching `reserve_size`, `world_size` and store URL with `Scene.SHARED`.
The default Offload instance used by sparse KV remains independently owned.

The existing `gather_dequantize_host_uva` operator reads shared VMM addresses
directly. Consumers retain metadata, small NPU pointer arrays and selected BF16
rows. Model collectives and KV transfer are unchanged. Without
`engram_shared_host_dir`, existing table placement remains unchanged.

## Requirements and limitations

- A3 with working SHARED VMM imports and matching GVA fingerprints. Cross-node
  sharing requires a supported MemFabric fabric path between the nodes; TCP
  reachability alone is insufficient. Each physical node must have the same
  number of Offload ranks, **including the owner on node 0**.
- MemFabric with the independent `offload.Offload` API, nonuniform SHARED
  allocations, `get_dva`, and a compiled `libmf_hybm_accoffload.so` extension.
  Empty wheel placeholders are insufficient; older APIs fail explicitly.
- INT8 Engram codes of width 256, group 32 scales, and a complete ModelSlim
  checkpoint index. FP32 and E8M0 checkpoint scales are loaded as FP32.
- A fresh mode-0700 directory under the same UID, accessible to every process.
  Its manifest contains a capability token and addresses; do not publish it.
- Enough physical DRAM for all tables and alignment padding. SHARED also reserves
  one capacity-sized virtual slot per rank, including zero-allocation consumers.
  MemFabric validates the total reservation against its native VA region and
  available space during initialization. All Offload instances share this process
  VA region. Small-pool success does not establish full-model capacity.
- A distinct store URL for each SHARED instance. Pool isolation does not
  establish end-to-end sparse KV/model compatibility for every configuration.

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

Merge these options with the model's other configuration. Start workers after
`POOL_JOINING`, before `POOL_READY`: the owner loads tables only after all ranks
initialize. Consumers retain the existing AICPU warmup before attachment and
wait for publication before constructing views. The store defaults to an
available port; `--store-url tcp://127.0.0.1:PORT` selects one explicitly.

`--consumers` replaces the former `--max-consumers`. Dynamic membership and
independent worker restart are not supported. Keep the session alive for all
layers. A final session release retires that worker's rank; create a fresh
generation to restart workers.

### Fixed-member cross-node P/D sharing

P and D may attach to the same owner from different nodes. Their existing model
process groups and KV transfer configuration remain separate from the Engram
Offload group. This does not enable cross-node TP/DP within a model replica.

For two nodes with two Offload ranks each, run the owner and one consumer on
node 0, and two consumers on node 1 (`world_size=4`, `local_world_size=2`):

```bash
# Node 0: owner counts as rank 0; use a reachable address and an unused port.
python -m vllm_ascend.shared_engram \
  --model /models/immutable-checkpoint \
  --pool-dir /shared/private/engram-generation-002 \
  --device 0 --consumers 3 --local-world-size 2 \
  --listen-host OWNER_ADDRESS --store-url tcp://OWNER_ADDRESS:29701
```

Set `engram_shared_host_node_rank` to `0` for workers on the owner's physical
node and `1` for workers on the second node, alongside the shared directory:

```bash
# Node 1, merged with each model instance's other additional configuration:
--additional-config '{"engram_shared_host_dir":"/shared/private/engram-generation-002","engram_shared_host_node_rank":1}'
```

All workers on a physical node use the same node rank, independently of P/D,
TP or DP rank. Offload assigns a contiguous rank block per node and rejects
over-subscription. Count the owner in node 0's block: adding it to equal-sized
P and D worker groups makes the node counts unequal. Omitting
`--local-world-size` retains single-node behavior.

Consumers need the private discovery manifest after `POOL_JOINING`. A shared
directory is simplest; a securely copied snapshot also works because subsequent
state and table metadata come from the owner control connection. Both that
endpoint and the MemFabric store must be reachable from every node.

## Shutdown and recovery

Stop request admission, then all consumers, then the owner. Consumers synchronize
before releasing views and the native pool. Only the owner frees allocations;
it waits for all leases before uninitializing. Unexpected disconnects retain
orphaned leases. Verify that every consumer has stopped before terminating an
orphaned owner, then start a fresh generation. Owner failure during an active
graph remains unvalidated.

## Component validation

The component benchmark checks two layers/four tables, exact CPU/HBM references,
graph replay and continued reads after one layer detaches. Use free A3 devices:

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
