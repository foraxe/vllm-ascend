# DSA-CP packed-pool planning prototype

Status: the feature-gated planner/config propagation, concrete ACL arena
adapter, worker lifecycle, global-ID block-table/owner-route contract,
synthetic allocator/reshape transaction, and packed DSA materialization seam
are implemented.  Production packed allocation remains disabled:
the planner still emits `planner_only=true` and
`downstream_runtime_abi_ready=false`, and the normal model-runner startup
rejects the feature before opening CANN VMM.  This milestone proves component
composition and cleanup, not allocator replacement, KV-capacity saving, cache
equivalence, or TTFT.

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
  fail closed, feature-off translation is identity, and a same-`B`
  owner-versus-replicated plan saves physical bytes after VMM alignment;
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

For the reconciled six-group live Flash layout, the exact declaration order,
component copies, and one-request quotas under that pinned scheduler are:

| group | component copies | scheduler shape | live peak | admission | partition |
|---|---|---|---:|---:|---:|
| C4 MLA | 21 narrow + 21 wide replicated | block 128, compression 4 | 17 | 17 | 17 |
| C128 MLA | 20 wide owner | block 128, compression 128 | 1 | 1 | 1 |
| dense SWA A | 22 wide replicated | block 128, window 4096 | 57 | 65 | 65 |
| dense SWA B | 21 wide replicated | block 128, window 4096 | 57 | 65 | 65 |
| C4 compressor state | 21 narrow + 21 wide replicated | block 8, window 8 | 640 | 642 | 642 |
| C128 compressor state | 20 wide replicated | block 32, window 128 | 160 | 165 | 165 |

Their partition sum is 955 positive data IDs.  At the measured B0 block count
`B=4190`, declaration order assigns ranges `[1,18)`, `[18,19)`,
`[19,84)`, `[84,149)`, `[149,791)`, and `[791,956)`, leaving 3234
unassigned data IDs after the global block-zero sentinel.

This fixed-workload plan does not prove owner-shard capacity saving.  Its C128
MLA quota is one page.  Each of its 20 component copies contains a sentinel
plus zero or one owned data page, and both the owner and replicated
counterfactual round that copy to one 2-MiB VMM segment on every rank.
Consequently:

```text
fixed 955-quota owner bytes
  = fixed 955-quota aligned-replicated bytes
  = 3,166,699,520 bytes/rank
```

The large difference between either fixed-quota number and the historical
`B=4190` raw allocation is workload-envelope reduction.  It must not be
reported as DSA owner-shard capacity saving.

There is a second replacement gate in the legacy allocator.  DeepSeek-V4
groups layers by page-size bucket and aliases one full-`B` raw tensor across
the same tuple index in several cache groups.  The 128-KiB tuple containing a
C128 attention layer also contains non-C128 consumers, including the
corresponding C128 compressor-state `SlidingWindowMLASpec` layer (and, for
some tuple indices, other state/SWA layers).  Splitting the C128 attention
layer and adding an owner tensor leaves the non-C128 full-`B` backing alive.
That is the additive-sidecar configuration measured in G30, not replacement.

An allocator replacement must move every live consumer of that shared tensor
to its own plan component view in one transaction.  For the C128
compressor-state layer, this also requires block-table/state-continuation
equivalence across both prefill chunks; moving only the attention history
cannot reclaim the original backing.  A later distributed/local compressor
path additionally needs the prefix-scan or chunk-boundary state handoff gate.

The fixture reflects the reconciled live group dump and has no MTP group.
Runtime planning still reads every group's actual block size, compression
ratio, window, and layer order from the resolved `KVCacheConfig`; it will
recompute or fail closed rather than apply fixture constants.  The A3
activation gate must preserve the emitted group dump next to the result.

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

## Same-B owner-placement accounting oracle

The raw B0 allocator has 43 physical tuple tensors per rank: 21 narrow
16,640-byte pages and 22 wide 131,072-byte pages.  MTP is disabled.  Its exact
unaligned allocation is:

```text
B = 4190 total BlockPool slots
B * (22 * 131072 + 21 * 16640)
  = 13,546,370,560 bytes/rank
```

The packed plan disaggregates those aliased raw tuples into the six live
scheduler groups shown above.  Its same-`B` oracle keeps that exact group and
component inventory, assigns all 3,234 spare data IDs to C128, and changes
only the 20 C128-wide copies between owner and replicated placement.  It also
adds the bounded 65-wide-page materialization scratch.  Every component copy
and scratch region is rounded independently to the measured 2-MiB VMM
granularity.

This oracle proves that owner placement can save bytes when a C128 logical
range is large enough to amortize 2-MiB per-copy granularity.  It is not the
current scheduler plan: the production fixed-quota scheduler has six disjoint
group ranges and only one C128 MLA page.  A capacity experiment therefore
needs a new same-service-capacity plan with a C128 quota above one VMM granule
per owner, or a scheduler/allocator design that retains the shared `B=4190`
service envelope while routing C128 physical pages to owners.

The CPU suite pins one explicit accounting candidate by assigning all 3234
currently unused fixed-profile IDs to C128:

```text
group order = [C4, C128, SWA-A, SWA-B, C4-state, C128-state]
workload quotas = [17, 1, 65, 65, 642, 165]
same-B accounting quotas = [17, 3235, 65, 65, 642, 165]
sum = 4189 data IDs, plus sentinel 0 => B=4190
same-B owner = 4,215,275,520 bytes/rank
same-B aligned-replicated = 11,639,193,600 bytes/rank
```

For that exact six-group plan, the aligned owner allocation is smaller than
the aligned replicated counterfactual on every rank.  This isolates a real
C128 placement delta without comparing against a 955-ID envelope.  It is
still accounting-only: locking all spare shared-pool capacity to C128 changes
the general multi-request borrowing semantics, so it may be used for the
pinned one-request profile only after the scheduler and state-continuation
gates pass.  Until that contract exists, runtime metadata remains
`planner_only=true/downstream_runtime_abi_ready=false`.

A different fixed-profile service-capacity experiment can preserve the same
aggregate `B=4190` ID domain and place the slack in one ordinary replicated
group:

```text
[17, 1, 65, 3299, 642, 165]
```

This satisfies `FixedQuotaBlockPool`'s exact `sum(quotas) == B - 1` contract
and the pinned single-request 8200/1 admission requirements.  It is a
quota/repacking experiment, not an owner-sharding result: the C128 quota is
still one and therefore contributes zero aligned owner saving.  It also
removes arbitrary cross-group borrowing, so the assigned slack group and the
distinction between required versus assigned quota must be explicit in
metadata.  The current planner does not emit this policy or attach
`ascend_kv_cache_group_block_quotas`; this layout is representation-ready but
not activated.

## Implemented mechanism seam and remaining activation gates

Feature-off still means no `PackedPoolPlan` is constructed;
`translate_group_block_ids(None, ...)` remains an unconditional identity over
the existing shared-pool IDs.  Feature-on production startup still fails
before CANN/Torch allocation.  The following pieces are composable only
through an internal synthetic transaction used by unit tests:

1. Planner:
   `vllm_ascend.patch.platform.patch_kv_cache_utils` chooses the fixed Flash
   quotas after final block-count clamping and serializes the complete
   component/view manifest.  It deliberately leaves both activation bits
   false and does not resize live tensors.
2. Scheduler and block IDs:
   `FixedQuotaBlockPool` emits already-global IDs.  The packed translator
   validates their one-dimensional signed-integer domain and assigned group
   range and preserves sentinel zero.  Replicated groups then store
   component-local execution IDs (`global - group_start + 1`) so compressor
   state and SWA read the same pages written by their slot mappings.  C128
   owner groups retain packed-global table IDs until materialization and
   translate only their write slots to owner-local pages.
   `MultiGroupBlockTable` preflights all groups before mutating a row, and
   `NPUInputBatch` can accept an explicit translator tuple.  No production
   scheduler path currently installs the fixed quotas or constructs these
   translators.  A same-service-capacity owner experiment should instead keep
   the existing shared `BlockPool` ID domain unless borrowing and prefix-cache
   semantics are separately proven.
3. Allocation and lifecycle:
   `_initialize_kv_cache_from_c128_packed_arena` accepts an already-open,
   strictly validated synthetic runtime, installs exact component-copy and
   scratch aliases, reshapes each layer using its local page count, attaches
   owner routes, seals and publishes only after all views succeed, and closes
   the runtime on failure.  Runtime view releasers unregister the exact owner
   cache before dropping aliases; teardown is quiesce, reverse-order view
   release, backing lease close.  Normal `initialize_kv_cache` does not call
   this transaction and still rejects packed allocation.
4. C128 consumption:
   Packed scatter consumes the worker-translated one-based owner-local slot
   without applying ownership twice.  Full and selected materialization accept
   packed-global IDs, use the immutable owner route to locate persistent
   pages, and remap results into zero-based bounded scratch for attention.
   Sentinel/padding pages cannot be written.  The new route conversion and
   scatter-preparation helpers contain no tensor-value-to-host branch.
   Existing HCCL materialization still performs host-visible count/list
   conversion while constructing variable-size collective payloads; removing
   those synchronizations is a separate performance gate.

The decisive next gate is not TTFT.  It is a same-`B` feature-on/off allocator
replacement proof:

- capture the live `KVCacheTensor.shared_by` families and require every C128
  attention/compressor-state consumer of a replaced raw backing to move
  together;
- preserve the existing scheduler ID domain and prove prefix-cache/block
  lifetime semantics;
- open the concrete ACL arena from normal startup, install every declared
  view, and prove the legacy full-`B` backing and full-`B` stage are absent;
- measure authoritative ACL physical bytes, not only PyTorch storage
  metadata;
- prove cache bytes, compressor continuation across both prefill chunks, and
  output/logit equivalence before measuring 8K/one-output TTFT.

Only after those gates pass may capacity saving or TTFT be claimed.  The
fixed-955-quota transaction in this milestone remains a mechanism/correctness
fixture and must not be used as the capacity result.

## Packed physical-arena lease slice

`vllm_ascend/attention/context_parallel/c128_packed_arena.py` implements the
CPU-testable lifetime boundary for seam 4.  It has no torch import and no
concrete ACL adapter.  The worker-side
`vllm_ascend/worker/c128_packed_runtime.py` reconstructs and independently
checks the serialized plan, exposes exact rank/bucket allocation accounting,
and owns the lease across all registered derived views.  The model runner
imports that module only inside the enabled branch, so feature-off bootstrap
and hot paths are unchanged.

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

### Concrete ACL and Torch-NPU adapters

The default-off runtime seam is implemented by:

- `c128_packed_acl_backend.py`: `AscendAclPackedArenaBackend`;
- `c128_packed_torch_npu.py`: `TorchNpuPackedArenaTensorFactory`.

Importing either module does not load `libascendcl.so`, import `torch` or
`torch_npu`, select a device, allocate memory, or create a tensor. The
integration must construct both adapters only after its packed-arena feature
gate passes. Construction fails closed when a required public symbol is
absent. Tensor binding fails closed when either of the two G28-proven
Torch-NPU external-storage constructors is absent.

The ACL backend uses the public CANN 9.0 lifecycle and checks every return
code:

```text
aclrtSetDevice
aclrtMemGetAllocationGranularity == 2 MiB
aclrtReserveMemAddress(alignment=0) -> verify returned VA is 2-MiB aligned
aclrtMallocPhysical(2-MiB multiple)
aclrtMapMem
aclrtMemSetAccess(READWRITE, local device)
aclrtMemset
...
aclrtUnmapMem
aclrtFreePhysical
aclrtReleaseMemAddress
```

CANN 9.0 documents the `aclrtReserveMemAddress` `alignment` argument as
reserved and requires zero. The adapter therefore passes zero to ACL while
requiring the arena contract to request 2-MiB alignment and checking the
returned VA. An unaligned non-null VA is immediately released.

`aclrtMemSetAccess` is failure-atomic with respect to the lease: if it fails
after `aclrtMapMem` succeeds, the adapter calls `aclrtUnmapMem` before
propagating the access error. If that rollback also fails,
`AclVmmRollbackError` retains both errors and the backend keeps the mapping in
its own registry. The lease rollback's `free_physical` call retries that
unmap; physical free and VA release refuse to run while the mapping remains.
If the retry also fails, `PackedArenaOpenError.lease` remains reachable and a
later explicit `close()` can retry the same ordered cleanup.

The Torch-NPU factory binds a root `uint8` tensor spanning the complete mapped
range. It uses a local `npu:<device_index>` tag, verifies pointer, byte count,
element size, dtype, and device, and retains the external storage object for
exactly as long as the root tensor. Its binding never calls ACL or allocator
free APIs. `close()` only drops the tensor reference followed by the storage
reference, leaving the arena lease as the sole VMM owner.

CPU mock coverage is in
`tests/ut/attention/test_c128_packed_acl_adapter.py`. It proves canonical
argument values, 2-MiB range checks, missing-capability failure, map/access
rollback, non-owning binding behavior, and the complete
`alias close -> fence -> unmap -> free physical -> release VA` order.

The `.204` runtime gate is intentionally separate from model-runner
integration:

1. with no serving process using the selected device, create one 2-MiB lease;
2. verify the root tensor pointer equals the reserved VA;
3. run `zero -> fill -> clone -> compare` through ordinary Torch-NPU kernels;
4. discard every derived view, close the lease, and synchronize;
5. verify HBM returns to the pre-allocation level and no mapping or process is
   left behind;
6. inject one post-map access failure in a dedicated process and verify the
   immediate-unmap rollback before enabling a worker feature flag.

The success path passed on 2026-07-30 using logical NPU0:

```text
artifact = /a3_inference/nyx/dsv4_dsa_cp/runs/204/
           20260730_g35_packed_acl_adapter_smoke_84f3da22/
arena_bytes = 2097152
tensor_data_ptr == reserved_va
fill(37) -> clone -> compare = PASS
tracked reservations after close = 0
tracked physical handles after close = 0
tracked mappings after close = 0
```

`npu_smi_before.txt` records eight idle NPUs. The controlling session's
immediate post-run `npu-smi` also showed no running processes and NPU0 HBM use
at 3130 MiB versus 3132 MiB before the run. The later redirected
`npu_smi_after.txt` captured another session's newly started vLLM workers and
is not the smoke teardown sample; the artifact README preserves that timing
boundary. The real fault-injection step remains mock-proven only and must use
a dedicated idle process before worker integration. This success-path result
does not prove model cache equivalence, capacity, or TTFT.

### Worker/model-runner lifecycle seam

The current mechanism integration has these exact boundaries:

1. `enable_c128_packed_vmm_arena` is a default-false JSON boolean.  Normal
   `initialize_kv_cache` validates the delivered metadata and rejects both the
   planner-only schema and any prematurely runtime-ready schema before opening
   CANN or allocating Torch cache tensors.
2. `PackedArenaRuntime.open_from_metadata` reconstructs the serialized plan
   and opens the concrete backend/tensor-factory lease.  The internal
   `_initialize_kv_cache_from_c128_packed_arena` transaction is test-only: it
   receives an already-open runtime, installs every component-copy and scratch
   alias, derives each cache shape from the installed payload pages, attaches
   block-table translators and owner routes, then seals and publishes.
3. A failed transaction drops shaped aliases, runs registered view releasers,
   identity-safely unregisters owner caches, closes the runtime, and restores
   the previous input-batch/kernel-block state.  A failed cleanup stage keeps
   the runtime and remaining callbacks reachable for retry.
4. `packed_arena_contract_from_metadata` still accepts only schema version 1
   and the fixed Flash TP8 8200/1 profile.  It recomputes every range, segment,
   view key, bucket, aligned owner/replicated comparator, and per-rank total
   before backend access.
5. The packed block table validates scheduler-global IDs, then stores
   component-local execution IDs for replicated groups and packed-global IDs
   for C128 owner groups.  Packed C128 slot mappings alone become owner-local;
   scatter consumes those slots, while materialization uses the retained
   global table plus the immutable owner route and returns a zero-based
   scratch-local table.

These mechanisms do not make packed KV tensors production-runnable.  The
remaining blocker is the complete live replacement transaction: normal
startup must capture the real shared-backing families, move every consumer,
including C128 compressor state, open/install the arena instead of the legacy
raw allocation, and prove continuation and cache equivalence.

The next A3 experiment is an allocator/lifetime gate, not an end-to-end TTFT
claim:

```text
baseline: feature off, existing torch allocations
candidate: packed feature on, fixed plan and block count
metric: unique backing bytes, base/segment/page offsets, zero initialization,
        exact tensor read/write, and cleanup event order
PASS: measured bytes equal BucketPhysicalBytes on every rank; every segment
      and scratch view round-trips; no NPU process or VMM mapping remains
FAIL: wrong bytes/offset/value, alias after close, or teardown-order violation
BLOCKED: live shared-view manifest or compressor-continuation proof unavailable
kill: first incorrect address/value or any 60-second lifecycle-stage timeout
```
