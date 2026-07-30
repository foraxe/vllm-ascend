# DSA-CP packed-pool planning prototype

Status: CPU contract only; no runtime path imports this module.

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

1. Planner:
   `vllm_ascend.patch.platform.patch_kv_cache_utils._get_kv_cache_config_deepseek_v4`
   currently derives one `num_blocks` from the fully replicated layer-tuple
   denominator and emits `KVCacheTensor(size=page_size * num_blocks)`.
   It must choose the `N_g` quotas, serialize the range/component metadata,
   size tensors from `BucketPhysicalBytes`, and keep the old output bit-for-bit
   when the feature is disabled.
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
