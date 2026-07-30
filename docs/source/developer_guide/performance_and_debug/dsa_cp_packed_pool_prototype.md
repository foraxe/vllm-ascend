# DSA-CP packed-pool planning prototype

Status: feature-gated planner/config propagation is implemented; scheduler and
worker consumption remain disabled.

## Hypothesis and gate

The fallback to Ascend VMM is a packed physical pool.  A valid packed plan can
give each scheduler cache group a deterministic logical range, retain block
zero as padding, map ordinary cache pages to dense local slots, map C128 pages
to a unique TP owner and dense owner-local slot, and account every persistent
page plus a bounded materialization workspace.

The baseline is the existing fully replicated Flash layout.  The first gate is
CPU-only:

- configuration: fixed group ranges, TP sizes 1/2/3/8, one-based data IDs,
  sentinel block 0, replicated and C128 owner components, measured 2-MiB VMM
  alignment, and bounded scratch;
- metric: exact mapping round-trip, physical-byte collision count, aligned
  segment size, and bytes per rank;
- pass: every valid ID has one inverse, collision count is zero, invalid IDs
  fail closed, feature-off translation is identity, and the real Flash-shaped
  example saves physical bytes;
- kill: any mapping ambiguity, unbounded scratch, or a capacity calculation
  that retains the old full C128 backing.

The contract lives in
`vllm_ascend/attention/context_parallel/c128_packed_pool.py`.  It intentionally
imports neither torch nor vLLM, so it cannot affect model bootstrap or a hot
path.

## Fixed Flash 8K/1 planner slice

`patch_kv_cache_utils._ascend_get_kv_cache_configs` now builds a packed plan
only when the vLLM JSON `additional_config` contains:

```json
{"enable_c128_packed_pool_planner": true}
```

This first profile is deliberately narrow:

- model: DeepSeek-V4-Flash;
- topology: TP8, PP1, upstream DCP1/PCP1; DSA layer sharding remains the
  model's TP-local DSA-CP mechanism;
- scheduler: `max_num_seqs=1`, chunked prefill on,
  `max_num_batched_tokens=5120`, effective
  `max_num_scheduled_tokens=5120`, one partial prefill, and long-prefill
  threshold zero;
- admission bound: `max_model_len=8201`, exactly the 8200 prompt slots plus
  one sampled output, so a longer request cannot overrun a fixed group range;
- client shape: 8192 repetitions of `" hello"`, which the validated client
  request represents as 8200 prompt tokens, followed by one sampled output;
- MTP and prefix caching: disabled;
- chunk schedule: exactly 5120 then 3080 prompt tokens.

The output token is sampled by the final prefill invocation and does not need a
new KV slot.  The planner mirrors both existing cache-manager contracts.
Compressed MLA first floors by `compress_ratio` and then takes the block
ceiling.  Sliding-window groups are simulated one scheduler chunk at a time,
including whole-block reclamation before the next chunk.  Their partition also
honors `can_fit_full_sequence`:

```text
admission_cap =
  ceil(min(window - 1 + max_num_batched_tokens, max_model_len) / block_size)
  + 1
admission_blocks = min(ceil(prompt_tokens / block_size), admission_cap)
partition_blocks = max(peak_live_blocks, admission_blocks)
```

For the representative six-group Flash fixture, the exact one-request quotas
under that pinned scheduler are:

| group | scheduler shape | live peak | admission | partition |
|---|---|---:|---:|---:|
| C4 MLA | block 128, compression 4 | 17 | 17 | 17 |
| C128 MLA | block 128, compression 128 | 1 | 1 | 1 |
| C4 compressor state | block 8, window 8 | 640 | 642 | 642 |
| C128 compressor state | block 32, window 128 | 160 | 165 | 165 |
| dense SWA A | block 128, window 4096 | 57 | 65 | 65 |
| dense SWA B | block 128, window 4096 | 57 | 65 | 65 |

Their partition sum is 955 positive data IDs.  At the measured B0 block count
`B=4190`, declaration order assigns ranges `[1,18)`, `[18,19)`,
`[19,661)`, `[661,826)`, `[826,891)`, and `[891,956)`, leaving 3234
unassigned data IDs after the global block-zero sentinel.

The `window=4096` and two dense-SWA rows in this table remain fixture evidence,
not a captured live `config.json`/group dump.  Runtime planning reads every
group's actual block size, compression ratio, window, and layer order from the
resolved `KVCacheConfig`; it will recompute or fail closed rather than apply
these fixture constants.  The A3 activation gate must preserve the emitted
group dump next to the result.

The planner runs after vLLM clamps all worker configs to the minimum final
`num_blocks`; it cannot serialize a stale per-rank pre-clamp capacity.  It
attaches one JSON-safe `c128_packed_pool_metadata` object to every worker
`KVCacheConfig`.  `initialize_from_config` sends those configs to already
spawned workers through the multiprocess RPC; that attribute is the declared
worker ABI.  Deep-copying the worker config into the scheduler config preserves
the metadata.  The later write to EngineCore's
`vllm_config.additional_config` is diagnostic only and is not a worker
propagation mechanism.  Every worker independently rebuilds the plan, and
startup fails if group schemas, ranges, or final block counts differ.

The schema includes group ranges; component placement; the exact ordered
`layer_names` mapping copy index to layer; every copy/rank's aligned segment
base and size; bucket accounting; explicit per-rank scratch base, allocated
size, and granularity; and exact bytes per rank.

This slice is metadata-only.  It leaves `num_blocks`, every
`KVCacheTensor.size`, and `shared_by` unchanged and marks the metadata
`planner_only=true` and `downstream_runtime_abi_ready=false`.  A consumer must
fail closed until the scheduler, block-table, worker allocator, and C128
scatter/materialize ABI land together.

The experiment gate is:

- hypothesis: fixed group quotas plus owner-local C128 placement fit inside the
  existing final block capacity and produce deterministic JSON metadata;
- baseline: feature disabled, using the existing shared `BlockPool` and raw
  `KVCacheTensor` layout;
- metric: exact quota/range values, JSON round trip, post-clamp block count,
  and unchanged baseline object/tensor identity;
- pass: the quota sum is at most `B-1`, all ranges are disjoint, every worker
  emits the same schema, metadata serializes, and feature-off returns the
  original config objects without mutation;
- fail: any quota/range/accounting mismatch or non-JSON metadata;
- kill: unsupported topology/workload state, quota sum above `B-1`, missing
  C128 group, mixed manager semantics inside a group, or any attempted tensor
  resize before the downstream ABI exists.

## Logical and physical contract

`global_block_capacity` is the full `BlockPool` size, including block zero.
Each `PackedPoolGroupSpec` receives `N_g` one-based data IDs.  Declaration
order assigns disjoint global intervals starting at 1, and
`sum(N_g) <= global_block_capacity - 1`.  Every component copy/rank segment
reserves physical slot zero as a dummy page, so padding can never alias the
first live data page.  Data slots begin at one.  Direct data mapping rejects
zero; `sentinel_address()` resolves the explicit local dummy page.

A group can drive multiple physical components.  This models the real Flash
layout without inventing independent logical capacities for layers that share
the same scheduler block IDs:

- replicated components map every group ID and copy to a dense per-rank slot;
- C128 components use `owner = global_id % TP` and map every copy to a dense
  owner-local slot;
- every component copy and rank receives a disjoint byte segment whose start
  and allocated size honor its declared allocation granularity;
- one fixed `max_pages_per_rank` scratch bound is accounted per bucket.

The G27 probe on `.204` measured a 2-MiB VMM allocation granule:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g27_sparse_owner_20260730_121939/
```

A 128-KiB C128 page is therefore one sixteenth of the physical mapping
granule.  The original `block_id % TP` logical interleave cannot be sparsely
mapped one 128-KiB page at a time.  The plan translates it into a contiguous
owner-local ordinal, rounds every per-copy/per-rank C128 segment to 16 pages,
and returns the aligned segment base plus exact byte offset that the worker
block table must use.  Non-C128 and scratch segments use the same byte-rounding
rule when they are placed in the VMM arena; page size does not need to divide
the granule.

Both mapping directions are part of the contract.  A runtime implementation
must not replace exact metadata with a hash or dynamically sized staging
allocation.

## Real Flash-shaped accounting example

The CPU gate uses the measured B0 shape:

```text
B = 4190 total BlockPool slots
data slots = 4189
TP = 8
wide page = 131072 bytes
narrow page = 16640 bytes
wide replicated copies = 2
C128 owner copies = 20
narrow replicated copies = 21
8K selected-page scratch bound = 65 wide pages per rank
VMM allocation granularity = 2097152 bytes
```

The replicated baseline is exactly:

```text
B * (22 * 131072 + 21 * 16640)
```

The packed candidate keeps the two ordinary wide copies and all narrow copies,
allocates only each rank's 523 or 524 C128 data pages plus one sentinel page
for every C128 layer, and adds 65 wide scratch payload pages.  Every component
copy and the scratch region is rounded independently to 2 MiB.  Thus the
524- or 525-page occupied C128 segments both allocate 528 pages, while the
65-page scratch allocates 80 pages.  The test compares exact aligned bytes
against the baseline.  It does not count a full `B`-page stage and therefore
cannot reproduce the G26 false capacity claim.

## Runtime seams still unresolved

The prototype is feature-off until all four seams below are implemented and
validated together.  Feature-off means no `PackedPoolPlan` is constructed;
`translate_group_block_ids(None, ...)` is an unconditional identity over the
existing shared-pool IDs.

1. Planner (partial):
   `vllm_ascend.patch.platform.patch_kv_cache_utils` now chooses the fixed
   Flash `N_g` quotas after final block-count clamping and serializes the
   range/component metadata.  It deliberately does not size tensors from
   `BucketPhysicalBytes`; that activation belongs with the worker allocator
   ABI.  Feature-off returns the original configs without metadata mutation.
2. Scheduler:
   `vllm_ascend.patch.platform.patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator.__init__`
   currently constructs one shared `BlockPool(kv_cache_config.num_blocks)`.
   The upstream
   `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots` returns block
   objects by cache group.  The runtime must preserve prefix-cache lifetime and
   block-zero padding while encoding each group's positive local ID into its
   assigned range.  Whether this uses range-aware views of one pool or separate
   per-group pools is still a scheduler design decision.
3. Worker block tables:
   `vllm_ascend.worker.block_table.MultiGroupBlockTable.append_row` and
   `add_row` currently copy scheduler IDs directly, and
   `BlockTable.compute_slot_mapping` treats those values as physical page
   numbers.  The worker must translate group IDs once and select the correct
   component address without changing padding, CP interleave, or hybrid-block
   expansion semantics.  A raw `global_id * page_size` address is invalid for
   C128 because one 2-MiB mapping granule contains 16 C128 pages; the table must
   use the plan's owner-local slot and aligned segment base.
4. Worker allocation and C128 consumption:
   `NPUModelRunner._allocate_kv_cache_tensors` and
   `_reshape_kv_cache_tensors` currently assume raw tensors contain the
   planner's full page count.
   `NPUModelRunner._get_c128_owner_stage_cache` allocates a full-page execution
   view.  They must allocate the packed component segments, expose ordinary
   dense views to non-C128 kernels, bind C128 persistent tensors to the
   owner-local segment, and replace the full stage with the declared bounded
   scratch before `AscendDSACPImpl` materializes selected rows.
   `C128OwnerShardCache.prepare_owned_scatter`, `scatter_owned`,
   `materialize_for_attention`, and `materialize_selected_for_attention`
   currently use `c128_local_page(page, TP) = page // TP`.  That formula is
   valid only for the old zero-based full logical range.  Compressor scatter,
   selected-page reads, and inverse materialization must all use this plan's
   group-range-aware owner slot (including the reserved dummy page); changing
   only `BlockTable.compute_slot_mapping` would address the wrong page.

The next runtime gate is not TTFT.  It is a feature-on/off allocator proof at a
fixed block count: deduplicate raw storage pointers, compare allocated bytes to
this plan, reconstruct every component from its inverse mapping, and then prove
continuation/cache equivalence.  Only after that gate may the service-capacity
and 8K/one-output TTFT comparisons run.

## Packed physical-arena lease slice

`vllm_ascend/attention/context_parallel/c128_packed_arena.py` implements the
CPU-testable lifetime boundary for seam 4 without connecting it to the worker.
It has no torch import and no concrete ACL adapter.  Production modules do not
import it, so feature-off bootstrap and hot paths are unchanged.

For one TP rank, the lease:

- requires every component and scratch segment to use the measured 2-MiB VMM
  granularity;
- reserves one contiguous VA range for each page-size bucket, using exactly
  `BucketPhysicalBytes.total_allocated_bytes_by_rank[rank]`;
- queries `aclrtMemGetAllocationGranularity`, rejects anything other than the
  plan's measured 2 MiB, and allocates/maps/zeros one locally owned physical
  handle per range;
- binds one opaque, non-owning local-device tensor over the complete range;
- resolves plan addresses only when `address.tp_rank` equals the lease's local
  rank and the requested bytes fit both the component segment and bucket;
- derives scratch from the aligned allocation at the arena tail and reports
  payload and allocated bytes separately;
- closes explicitly in the G28-proven order: stop exposing the lease, drop
  owned tensor/storage aliases, synchronize queued NPU work, unmap, free the
  physical handle, and release the VA.

There is deliberately no `__del__` driver cleanup.  A failed unmap retains its
physical handle and VA so an explicit `close()` retry cannot create a
use-after-free.  `close()` changes the lease to `CLOSING` before calling the
binding or fence, so reentrant code cannot borrow a new tensor/view while
teardown is in progress.  Already borrowed tensor views must be quiesced by
the integration before close; enforcing that pin count in a concrete C++/torch
binding is still a runtime gate.  If construction fails after a partial
allocation, the same stage-aware close path rolls back every completed bucket.
If rollback also fails, `PackedArenaOpenError.lease` keeps the remaining
handles reachable for an explicit retry.

`PackedArenaBackend` maps directly to the validated ACL lifecycle:

```text
reserve_address   -> aclrtReserveMemAddress
allocation_granularity -> aclrtMemGetAllocationGranularity
allocate_physical      -> aclrtMallocPhysical
map_physical      -> aclrtMapMem
zero_mapped       -> an ACL memset on the mapped local VA
unmap             -> aclrtUnmapMem
free_physical     -> aclrtFreePhysical
release_address   -> aclrtReleaseMemAddress
```

This lease does not import peer physical handles.  A shared-handle lease needs
an explicit owner/importer role, owner-outlives-importers coordination, and a
policy that prevents an importer from zeroing already-published owner data.
That protocol is a later gate.

`PackedArenaTensorFactory.bind` is the only torch_npu seam.  The G28 probe
proved that the installed runtime can wrap a mapped pointer with
`_construct_storage_from_data_pointer` and
`_construct_NPU_Tensor_From_Storage_And_Metadata`, then run ordinary clone and
fill kernels.  G28 also proved that an imported peer mapping must still use
`npu:<local rank>`; this local-owning lease follows the same device-tag rule.
The tensor is non-owning: the lease owns ACL resources, while
`PackedArenaTensorBinding.close` must discard every
tensor/storage alias owned by the binding before the injected NPU fence runs.
`zero_mapped` must also complete before it returns, so no tensor is exposed
while initialization is still pending.
The canonical G28 artifact is:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g28_remote_tensor_20260730_043455/
```

### Line-anchored integration design

The anchors below are for integration base
`7355d108f96bda578440a196bf7136140e010427`; re-resolve the symbols after a
rebase.

1. Feature gate and lifetime owner:
   `vllm_ascend/worker/model_runner_v1.py:271-300` reads the current C128
   `additional_config` flags and initializes owner-stage state.  A later
   integration adds a default-false `enable_c128_packed_vmm_arena` there and a
   nullable `_c128_packed_arena_lease`.  Import the arena module only inside
   the enabled branch.  The runner or worker must expose an explicit shutdown
   hook that drops every derived KV view and calls `lease.close()` before
   device reset; do not put ACL calls in `__del__`.
2. Allocation switch:
   `model_runner_v1.py:3515-3537` owns KV-cache initialization, and
   `model_runner_v1.py:3575-3588` currently allocates then reshapes raw
   tensors.  The enabled branch must construct the serialized
   `PackedPoolPlan`, create the rank-local lease, and hand root bucket tensors
   to the reshape adapter.  Commit the lease to the runner only after all
   bucket bindings succeed.  Feature-off must continue directly into the
   existing calls.
3. Raw allocation replacement:
   `model_runner_v1.py:3695-3720` is the exact allocation entry, while
   `model_runner_v1.py:3748-3768` creates the current compressed/C128 raw
   tensor.  The enabled branch replaces only plan-owned components with
   byte/shape views over the lease bucket tensor.  It must not allocate a
   second `torch.zeros` backing for those components.  The existing path
   remains untouched when disabled.
4. Shape ABI:
   `model_runner_v1.py:3926-3975` derives `num_blocks` from raw-tensor bytes and
   assumes one contiguous component.  A packed integration must instead use
   the plan's segment base/size for each component copy, then call the existing
   layout adapter on that segment.  It must not infer a bucket-wide block count
   from the root arena tensor.
5. C128 execution scratch:
   `model_runner_v1.py:3882-3895` allocates a full `num_blocks` stage cache, and
   `model_runner_v1.py:3995-4002` installs it in `C128OwnerShardCache`.
   Replace this only after the lease's bounded scratch view is shaped for the
   attention backend.  `c128_owner_cache.py:257-284` and
   `c128_owner_cache.py:421-455` still use the legacy zero-based owner-local
   formula; scheduler/block-table integration must pass plan-aware owner slots
   before packed persistent storage can be enabled.

The next `.204` experiment is an allocator/lifetime gate, not an end-to-end
TTFT claim:

```text
baseline: feature off, existing torch allocations
candidate: packed feature on, fixed plan and block count
metric: unique backing bytes, base/segment/page offsets, zero initialization,
        exact tensor read/write, and cleanup event order
PASS: measured bytes equal BucketPhysicalBytes on every rank; every segment
      and scratch view round-trips; no NPU process or VMM mapping remains
FAIL: wrong bytes/offset/value, alias after close, or teardown-order violation
BLOCKED: missing concrete ACL/tensor adapter or worker shutdown hook
kill: first incorrect address/value or any 60-second lifecycle-stage timeout
```
