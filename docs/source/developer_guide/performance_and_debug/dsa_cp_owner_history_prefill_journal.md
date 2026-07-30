# DSA-CP owner-history prefill journal

Owner: Codex session `019fa36a-956a-7372-aceb-19d91f08f990`.

## Scope and acceptance gates

This is a prefill-only DSV4-Pro experiment. Its objective is to reduce
persistent DSA KV storage without adding a remote-pointer dependency to the
stock attention kernel or regressing TTFT.

- Frozen service baseline: `codex/a3-dsv4-pro-prefill-019fa36a` at `cd2803e`.
- Correctness: owner-sharded, selected-row materialization equals the current
  replicated C128 history for the same logical rows.
- Capacity: count persistent C128 pages only; the BF16 local materialization
  workspace is bounded and transient.
- TTFT gate: 8K input, one output token, FusedMC2-on baseline, p50 must not
  regress by more than 1%. Production runs are not yet valid without the Pro
  checkpoint.

## C128 prefill contract

For an already-produced compressed history page `p` on CP size `N`:

```text
owner_rank = p % N
owner_page = p // N

owner-sharded quantized C128 page
  -> selected rows only
  -> bounded local BF16 materialization workspace
  -> npu_sparse_attn_sharedkv
  -> release workspace
```

The sparse-attention operator therefore consumes local rows. VMM is used to
make the owner page addressable; it is not evidence that the existing fused
attention operator can safely consume a remote pointer directly.

SWA remains local/replicated hot state. C4/C128 compressor production remains
a separate ordered-state problem: CP segments need a prefix-state scan or
boundary-state handoff before their owner pages can be treated as equivalent to
the replicated baseline.

## Evidence

### G1: CANN VMM peer map — PASS

On `2026-07-28`, pod `dsv4-dsa-prefill-204-nyx` was pinned to
`33.215.119.204` with image
`hcr.meta-wulan01.hw-wulan.local/antsys/vllm:release_0.20.2_0601_202607271124_aarch64`.
The pod reports `npu-smi 25.5.1` and CANN `9.0.0`; all 16 NPUs were idle.

The standalone two-process NPU0/NPU1 probe completed:

```text
aclrtDeviceCanAccessPeer / aclrtDeviceEnablePeerAccess: PASS
aclrtMemExportToShareableHandleV2: PASS
aclrtMemSetPidToShareableHandleV2: PASS
aclrtMemImportFromShareableHandleV2: PASS
aclrtMapMem and exporter/importer pattern roundtrip: PASS
```

Artifacts are durable under:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/g1_vmm/
```

This supersedes only the old iTask VMM `207000` result. It does not prove a
PyTorch tensor binding, an AscendC peer accessor, visibility fencing, or
stock-attention remote-row support.

### G2: IPC peer import — INVALID

The first rerun did not reach `aclrtIpcMemImportByKey`: the old probe passed a
1024-byte key buffer, while CANN requires the IPC key length to be exactly 65
bytes. `aclrtIpcMemSetAttr` rejected it with `107000 Invalid_Argument`.
Do not classify IPC as blocked or supported until the 65-byte-key probe is
rerun.

### G3: owner-sharded C128 materialization oracle — PASS

`tests/ut/attention/test_dsa_cp_owner_placement_reference.py` now verifies
rank-major owner-page translation, selected-row quantized dequantization into
a local workspace, and equality with a replicated C128 oracle. It exercises
CP sizes 2, 4, and 16. For the C128 quantized-page-plus-scale family, owner
storage is exactly `1 / CP` of persistent replicated storage; the temporary
materialization workspace is excluded from that capacity count. On the `.204`
image:

```text
VLLM_VERSION=0.20.2 python3 -m pytest -q test_dsa_cp_owner_placement_reference.py
8 passed
```

### G4: real DSV4-Flash cold-prefill B0 — PASS

On `2026-07-28`, the same pod ran the actual checkpoint
`/mnt/deepseek/models/DeepSeek-V4-Flash-w8a8-mtp` with DSA-CP enabled,
FusedMC2 enabled, no Mooncake connector, no prefix caching, and a direct
standalone service. Flash has `o_groups=8`; TP16 is invalid because
`wo_a` would compute `n_local_groups = 8 // 16 = 0`. The launcher now rejects
that topology before model load. The valid B0 was TP8/EP8 on NPU 0--7:

```text
TP_SIZE=8
A3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENABLE_DSA_LAYER_SHARDING=0
ENABLE_MOONCAKE_KV_CONNECTOR=0
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0
ENABLE_FUSED_MC2=1
SAFETENSORS_LOAD_STRATEGY=lazy
```

`lazy` was selected only after the historical forced-NFS `prefetch` run failed
before a shard completed. It is a load-path choice, not a TTFT A/B. The valid
run loaded all 70 files in 271.62 s, created the cache/warmed the engine in
15.81 s, and reached `GET /health = 200`.

The fixed streaming client sent 8192 repetitions of `" hello"`, a unique
suffix per request, `max_tokens=1`, one warmup, and five timed samples. Its
metric is time to the first nonempty streamed text delta:

| Metric | Result |
|---|---:|
| TTFT median | 597.93 ms |
| TTFT mean | 599.89 ms |
| TTFT p90 | 606.02 ms |
| Timed requests with a first text token | 5 / 5 |
| Warmup TTFT | 20.337 s |

The live allocator reported 12.68--12.69 GiB current KV-cache memory per
NPU, or about 101.5 GiB across TP8. This is the service's total current KV
cache allocation, not a C128-only owner-shard saving. The C128 oracle remains
the capacity proof: for a CP8 owner shard, persistent quantized C128 pages
plus scales are `1/8` (12.5%) of eight replicated copies; the selected-row
BF16 workspace is transient and excluded.

Durable raw evidence is on NFS:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_b0/
  log_single_node_prefill_flash_tp8_lazy_nomooncake_fmc2_8k.log
  results/flash_tp8_lazy_nomooncake_fmc2_8k_ttft.json
  results/flash_tp8_lazy_nomooncake_fmc2_8k_kv_and_load.txt
  results/flash_tp8_lazy_nomooncake_fmc2_8k_vllm_server.log
```

Artifact checksums are respectively
`f66a465447d7025db3dd185a605caf8b65923f0fda007bfe6c98b8831447416f`
(TTFT JSON) and
`17b8d3e8f6189deb3d02c804a1bb32484ec0c99de154a27bb358c73c908f6634`
(KV/load excerpt).

This is a valid Flash B0 and a correctness/capacity gate for C128. It is not
yet an owner-sharded allocator implementation, a DSA-CP TTFT comparison, or
a DeepSeek-V4-Pro result.

### G5: production C128 DSA-CP seam — identified

The normal prefill path already supports C128 DSA-CP. The
`compressor_ratio <= 1` assertion is limited to
`AscendDSACPMetadataBuilder.build_for_drafting()` (MTP/speculative drafting),
which is disabled for the Flash B0; it does not govern
`AscendDSACPMetadataBuilder.build()`.

For normal C128 prefill, `AscendDSACPImpl._forward()` currently all-gathers
the hidden states, runs the compressor over that gathered sequence, then
scatters every compressed row into a full local `compress_kv_cache` before
calling `npu_sparse_attn_sharedkv`. In parallel,
`model_runner_v1.py::_allocate_kv_cache_tensors` allocates that full local
compressed-attention tensor for every rank. Therefore a correct feature-on
C128 implementation must replace this coupled replication path with all of
the following together:

1. C128 owner-page allocation plus logical-page-to-owner metadata in the
   worker allocator.
2. Compressor prefix-state handoff/scan before a rank publishes its owned
   C128 pages.
3. HCCL-staged selected-row materialization into a bounded local workspace;
   the existing sparse-attention kernel must continue to receive local rows.
4. A feature-off replicated fallback, followed by an identical Flash TP8
   8K/one-output candidate measurement.

VMM remote pointers remain an R&D transport alternative after the staged path
is correct; they are not required for this next feature-on gate.

## Current blocker

The old cloudide iTask pod holding
`/a3_inference/itask/workdir/models/DeepSeek-V4-Pro-w4a8-mtp` no longer
exists, and that checkpoint is not mounted in the `.204` pod. Therefore no
Pro-model 8K/one-output baseline or TTFT claim has been made from this pod.

### G6: feature-on C128 owner allocation — initialization PASS, request INVALID

The opt-in implementation is on branch
`codex/a3-dsv4-pro-prefill-019fa36a` (`fc82ac44533b055c36b37fd70eba3b87f8dbf36f`).
It isolates C128 `MLAAttentionSpec(compress_ratio=128)` entries from the
mixed C128-attention/compressor-state raw bucket, stores only owner pages
(`page_id % TP`), and materializes a local attention view through HCCL.

Two early attempts were invalid before model execution: the first exposed the
mixed raw-storage bucket and the second had a planner `NameError`. The latter
was fixed by deriving C128 names from all actual KV-cache specs, not by
assuming a cache-group index. The fixed r3 run loaded 70/70 Flash shards and
returned `GET /health = 200`; the owner-page reference unit passed on the
image (`10 passed`).

Its first real 8K/one-output request produced no text SSE delta and the
service exited, so it is **INVALID** for correctness, capacity, and TTFT. No
feature-on TTFT number exists. The raw run directory is:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r3_fmc2_8k.log
  bench_r3.out
  bench_r3.rc
  logs/ascend/run/
```

This failure also invalidates the current full-stage design as a performance
candidate. A stage tensor sized to all allocator pages recreates a replicated
C128 execution view; additionally, the baseline aliases C128 attention with
compressor state in one raw bucket, so splitting the tensor adds a full state
bucket before the owner shard is counted. Consequently the unit-oracle
`1 / TP` persistent-C128 result must not be reported as total service KV
reduction. The next implementation gate is a bounded per-request page stage
with explicit lifetime/capacity accounting, followed by a true selected-row
or VMM peer-read kernel path. Do not run another 8K TTFT comparison until the
feature-on request returns an output and its allocator accounting is proven.

### G7: tensor-ABI retry and overlap-only B1

The r3/r4 feature-on runs had put a Python `C128OwnerShardCache` wrapper into
the model's `static_forward_context` cache slot. That slot is consumed by the
model as a Tensor before `AscendDSACPImpl._forward()` executes. Commit
`dbdea8f93fbfb5380ee8e79179067290b50056cb` keeps the model-visible cache as
the persistent Tensor and resolves owner metadata through an out-of-band
Tensor-data-pointer registry at the DSA-CP seam. The updated reference suite
on the `.204` image passed `11 passed`.

The r5 Flash TP8 service loaded 70/70 shards and reached `GET /health = 200`,
but its first 128-token, one-output request again ended without a text SSE
delta and the service exited. The service log contains no C128 scatter, stage,
or sparse-attention trace; CANN device/PLOG searches did not expose a Python,
HCCL, or kernel exception. Therefore r5 is **INVALID** and establishes only a
fault boundary: failure is before the instrumented C128 DSA-CP scatter/stage
seam. It is not evidence against HCCL staging, VMM, or Mooncake (which was
disabled). Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r5_tensorabi_fmc2_8k.log
  bench_r5_smoke.out
  bench_r5_smoke.rc
  test_owner_stage_r5.log
  source_sha256.txt
```

The r6 retry added a rank-0 warning at the DSA-CP handoff and again reached
health after 70/70 shards. Its 128-token request exited without a text delta
and without the handoff warning; rank/device logs show normal HCCL teardown,
not an attributable HCCL or kernel failure. Static inspection then found the
concrete ABI violation: ordinary DeepSeek-V4 cache entries are a one-element
`[Tensor]` list, while the first owner implementation replaced that list with
a bare Tensor. `static_forward_context` wraps the value once more, so this
changes the nesting consumed by the model before DSA-CP. The next retry
restores `kv_caches[layer_name] = [persistent_tensor]` while retaining the
out-of-band owner registry. This is a code-derived root-cause hypothesis until
the next feature-on request returns a handoff trace or text.

The r7 retry at `cb8577b749892a8948c21f0287e149bed818e623` applied that
container fix, passed the same 11 reference tests, loaded 70/70 shards, and
reached health. Its first 128-token request still ended without a text SSE and
again emitted no handoff warning or attributable CANN error. The container
fix is therefore a necessary ABI correction but not the sole fault. Do not
continue treating the failed path as a HCCL/VMM experiment: no instrumented
DSA-CP seam has run. The next diagnostic must instrument the model/executor
boundary before `AscendDSACPImpl.forward()`, or split compact allocation from
owner-placement execution, rather than repeat another cold feature-on smoke.

In parallel, B1 isolated the existing implementation's
`prefill_comm_compute_overlap` switch. It retained TP8/EP8, Flash, 8K input,
one output token, FusedMC2, lazy weight loading, no Mooncake, and the
replicated C128 path; the only change from B0 was
`ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=1`. It passed correctness (5/5 timed
requests delivered nonempty text), but did not improve TTFT:

| Run | Median TTFT | Relative to B0 |
|---|---:|---:|
| B0 overlap=0 | 597.932 ms | baseline |
| B1 overlap=1 | 601.532 ms | -0.60% |

The >8% target is at most 550.097 ms against this B0. B1 is consequently a
valid negative result, not a production optimization. Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_overlap/
  experiment_b1.txt
  log_single_node_prefill_flash_tp8_overlap_fmc2_8k.log
  bench_b1_smoke.out
  bench_b1_8k.out
  results/flash_tp8_overlap_b1_fmc2_8k_ttft.json
```

## Next implementation gate

Before another owner-shard e2e attempt, instrument the model-to-DSA-CP cache
handoff at warning level with cache Tensor shape, data pointer, and owner
registry resolution. The gate is a 128-token prefill that reaches either the
owner scatter or a captured exception. Keep C4/SWA and Mooncake out of this
gate. Only after a feature-on request returns text may the implementation
replace the full local stage with a bounded selected-page workspace and make a
capacity or TTFT claim.

### G8: C128 owner-write localization — in progress

The model/executor trace corrected the earlier fault boundary. In r9, every
rank completed C4 and entered C128 layer 3 with the compact persistent tensor
shape `[524, 128, 1, 512]`, where `524 = ceil(4190 / 8)`. It then completed Q,
SWA KV write, and the stateful C128 compressor. Thus neither the cache
container ABI, current-KV path, nor compressor prefix-state execution is the
immediate failure point.

r10 was **INVALID** before model loading because the command accidentally
omitted `ENABLE_DSA_LAYER_SHARDING=0`; the inherited P-only layer-sharding
configuration is rejected by a single-node run. r11 restored that B0 setting.
The one-request, 128-word smoke test reached `scatter_owned` on all eight
ranks, but no rank completed it. HCCL materialization and sparse attention
were never entered. Mooncake remained disabled, so this is neither a Mooncake
nor a VMM result.

r12 replaced the masked two-dimensional CANN-9 `npu_scatter_nd_update_v2`
write with a flattened `torch_npu.npu_scatter_nd_update_` address:

```text
compact_row = (logical_page // TP) * page_size + in_page
```

The single-NPU flattened write and empty-write probes both passed. The full
model still stopped in `scatter_owned`, but all worker processes remained
alive. That showed the remaining failure was the host scalar synchronization
in diagnostics (`owner_mask.sum().item()` / `bool(owner_mask.any())`), not a
new device exception. Commit `0f751cdf` removes those synchronizations and
always invokes the device scatter, including a zero-row owner subset; it also
filters negative padded page/offset entries before ownership calculation.
The owner-placement reference suite still passes `11 passed` on the A3 image.

The active r13 gate is deliberately only a 128-word, one-output smoke test
with TP8/EP8, FusedMC2 on, overlap off, layer sharding off, and Mooncake off.
It must show `c128_owner_scatter_ready` on all ranks before any statement
about HCCL stage, selected-row materialization, capacity, or TTFT can be made.
Raw evidence and launch output remain under:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r9_customop_trace_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r11_c128_stages_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r12_flat_scatter_fmc2_8k.log
  launch_r10.out
  launch_r11.out
  launch_r12.out
```

### G9: allocation-layout dependency — confirmed

r13 and r14 retained the compact owner allocation and both reached C128
compressor submission on all ranks, but no rank reached the first
`owner_cache_mask_ready` marker. Since `compressor_ready` means only that the
asynchronous NPU compressor was queued, the next dependent NPU operation is
blocked before any owner-mask, scatter, HCCL, or sparse-attention operation.
It is therefore incorrect to attribute that behavior to the masked scatter.

Commit `c97a8c0b` introduced the explicit
`enable_c128_owner_compact_allocation` gate so owner placement can be tested
with a full tensor independently of the `1/TP` allocator. The r15 command
used that gate (`0`) and otherwise retained the B0 topology/workload options.
It is **INVALID** at engine initialization: separating the C128 attention
cache from the state-cache bucket and retaining all 4190 pages needs an
additional 524 MiB per NPU. At fixed B0 `gpu_memory_utilization=0.9`, the
workers had only 111--347 MiB free, and every rank raised `torch_npu.memory:
NPU out of memory. Tried to allocate 524.00 MiB`.

This establishes a real coupling, not a Mooncake/HCCL/VMM blocker: the old
aliasing layout is needed to fit Flash at B0 capacity, while the compact
allocation avoids that 524 MiB cost but changes the asynchronous compressor
execution contract. The next valid experiment is a reduced allocator capacity
that still admits the 128-word request, with full allocation and owner
placement enabled. It is a correctness/layout gate only; it must not be
compared against B0 TTFT. After it proves the owner path, restore the B0
capacity and repair the compact-cache/compressor ABI before attempting a TTFT
candidate.

The corresponding raw log is:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r15_fullalloc_layout_fmc2_8k.log
```

### G10: bounded full-layout admission gate — in progress

r16 increased no feature setting; it only lowered `gpu_memory_utilization` to
`0.88` while retaining the full C128 allocation. It remained **INVALID**:
the additional C128 state bucket requested 474 MiB and the ranks had only
29--266 MiB free. This does not alter the G9 conclusion.

r17 instead used the new launcher controls with full C128 allocation,
`MAX_MODEL_LEN=32768`, and `NUM_GPU_BLOCKS_OVERRIDE=256`. It was also
**INVALID**, but for the upstream planner admission check rather than an NPU
allocation or owner-path operation: the 32K request requires 2.9 GiB of KV
cache while the forced 256-block cache exposes 0.77 GiB (estimated maximum
length 1432). No prefill ran, and r17 must not be used to assess compact
allocation or owner placement.

The first G10 retry, r18, set `MAX_MODEL_LEN=8192` and
`NUM_GPU_BLOCKS_OVERRIDE=64` while retaining full allocation and owner
placement. It was **INVALID** at the same upstream planner admission gate:
an 8K request needs 2.75 GiB of KV cache while the process has 0.19 GiB after
model and cache-layout initialization. That is a capacity result, not an
owner-write result. Full layout cannot be the 8K correctness gate on this
image.

The next single-variable layout gate is intentionally reduced to
`MAX_MODEL_LEN=256`, `MAX_NUM_BATCHED_TOKENS=256`, and
`NUM_GPU_BLOCKS_OVERRIDE=2`. It admits one sub-128-token request and tests
only whether the separated full-layout compressor state and owner-placement
ABI can reach the owner-write trace. It is never a TTFT or 8K capacity
candidate. Mooncake stays disabled for every G10 run.

r19 executed the 256-token/2-block full-layout gate before the launcher batch
envelope parameterization was deployed (the command still showed the
irrelevant hard-coded `--max-num-batched-tokens 5120`). It is nevertheless
decisively **INVALID** on the capacity gate: after full-layout initialization,
only 0.01 GiB remained, while one request at `max_model_len=256` requires
0.15 GiB. There is no full-layout request size that is useful for this pod;
do not spend another launch attempting to prove owner placement through that
layout.

The launcher now exposes and validates `MAX_NUM_BATCHED_TOKENS`, and its
remote `PRINT_CONFIG_ONLY=1` expansion verified `256` for both model and
batch envelopes. That is a reproducibility improvement, not a performance
change.

The implementation gate returns to compact allocation: determine the exact
compressor state/attention raw-storage alias required by the CANN compressor,
then retain that state layout while allocating only the canonical owner pages
for persistent attention. The existing compact-path stop after
`compressor_ready` is the next causal target. HCCL staging, VMM, and TTFT
remain out of scope until that gate returns one response.

### G11: compact owner placement reaches attention — PASS

r20 added a layout trace and reproduced the compact-path failure with a
64-word/one-output request. On every rank the C128 persistent attention cache
was `[524, 128, 1, 512]`, while compressor state was
`[4190, 32, 1, 1024]`, at distinct data pointers. The earlier raw-storage
alias hypothesis is therefore refuted. The trace stopped after
`c128_owner_scatter_begin`, before the first owner-cache marker.

r21 moved only owner-mask/address construction before the stateful compressor.
All ranks reached `owner_prepare_ready`, but stopped at the same post-compressor
boundary before `prepared_select_begin`. The remaining operation was a dynamic
`compressed_kv.shape[0]` inspection. r22 removed that inspection and made the
first dependent work `compressed_kv.index_select(precomputed_owner_rows)`.

r22 is **PASS** for the compact owner-placement execution gate:

* the A3 reference suite passed `12 passed`;
* a 64-word/one-output request returned nonempty text (diagnostic TTFT
  1.495 s, not a performance number);
* a single 8K-word/one-output request returned nonempty text (1.066 s with
  debug tracing, not comparable with B0);
* trace evidence spans all TP ranks and multiple C128 layers through
  `prepared_select`, owner scatter, HCCL materialization, and sparse attention;
* the process remains healthy (`GET /health = 200`).

This proves only the feature-on path can execute. It does **not** prove
numerical equivalence, cache equivalence across multi-request history,
total-memory reduction, or an 8% TTFT gain. The immediate next gates are a
feature-on/feature-off numerical oracle and allocator accounting, then a clean
8K paired benchmark with tracing disabled.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r20_compact_layout_fmc2_smoke.log
  log_single_node_prefill_flash_tp8_c128_owner_r21_prepared_scatter_fmc2_smoke.log
  log_single_node_prefill_flash_tp8_c128_owner_r22_direct_select_fmc2_smoke.log
  bench_r22_smoke64.out
  bench_r22_smoke8k.out
  results/flash_tp8_c128_owner_r22_direct_select_smoke64.json
  results/flash_tp8_c128_owner_r22_direct_select_smoke8k.json
```

### G12: clean paired 8K TTFT — FAIL / correctness not proven

The r23 feature-off control and r24 compact-owner candidate both used Flash
TP8/EP8, FusedMC2, no Mooncake, overlap off, 8K words, one output token,
warmup 1, and five timed requests. r24 additionally disabled every owner
debug print, so this is the first comparable owner candidate. Its result is a
clear regression:

| Run | Median TTFT | Relative to r23 B0 |
|---|---:|---:|
| r23 B0 feature-off | 593.847 ms | baseline |
| r24 compact owner | 940.926 ms | -58.45% |

The >8% threshold against r23 is at most 546.339 ms. r24 is not a candidate.
The current HCCL implementation materializes a full selected-page execution
view at every C128 layer, so it adds synchronization and copy work before
sparse attention; it is a correctness prototype, not a TTFT optimization.

The initial single repetition text check used identical prompt/output
(`"你好"`). However, r24 timed repetition 2 emitted `"Hello"` while r23
emitted `"你好"` for its corresponding deterministic prompt. Treat this as a
correctness failure until a logits/cache oracle resolves whether it is a model
nondeterminism artifact or a C128 placement error. Do not run a sweep or
report capacity/TTFT gains from the owner feature. The next implementation
task is a per-layer persistent-cache and attention-output numerical oracle;
only then may the materialization path be optimized to a bounded selected-row
workspace or direct peer placement.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  bench_b0_r23_8k.out
  bench_r24_clean_8k.out
  results/flash_tp8_b0_r23_fmc2_8k_ttft.json
  results/flash_tp8_c128_owner_r24_clean_fmc2_8k_ttft.json
```

### G13: current-row cache oracle — in progress

The branch experimentally added a debug-only in-forward oracle for the direct
producer/consumer invariant: after HCCL materialization, every current
`compressed_kv` row would equal the staged row addressed by its compressor
slot mapping. It used no replicated persistent cache.

r25 was **INVALID** because the launcher passed the oracle value to `jq` but
omitted the key from `additional-config`; no oracle code ran. That launcher
bug is fixed. r26 carried the flag correctly but exited at the first
`materialize_begin`, before an oracle result or an attributable runtime error.
The optional third staging return was then removed and the original two-value
ABI restored. r27 was still **INVALID** at the same boundary. r28 also made
the materialize call common to the debug and production branches and added a
debug-only NPU synchronization before verification; it failed identically.
Therefore an NPU verifier containing dynamic gathers cannot currently share
this model forward graph on the target CANN stack. These runs do not implicate
the placement protocol, and no additional internal-oracle retry is justified
without a different graph-isolation mechanism.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r16_fullalloc_u88_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r17_fullalloc_256blk_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r18_fullalloc_64blk_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r19_fullalloc_2blk_fmc2_smoke.log
  log_single_node_prefill_flash_tp8_c128_owner_r27_oracle_fixedabi_fmc2_8k.log
  bench_flash_tp8_c128_owner_r27_oracle_fixedabi_fmc2_8k.out
  log_single_node_prefill_flash_tp8_c128_owner_r28_oracle_sync_fmc2_8k.log
  bench_flash_tp8_c128_owner_r28_oracle_sync_fmc2_8k.out
```

### G14: isolated production HCCL placement gate — PASS

`tests/e2e/attention/test_dsa_cp_owner_hccl.py` runs the production
`prepare_owned_scatter`, `scatter_prepared`, and
`materialize_for_attention` methods under a real single-node TP8 HCCL process
group. Each rank owns only 3 of 24 persistent C128 pages. Every rank requests
a distinct three-page local block table; the global union contains all 24
pages. The staged cache reconstructed all 24 producer pages bit-exactly on
all eight ranks, and the remapped local block tables matched their original
logical pages.

The first invocation was **INVALID** because inherited
`HCCL_INTRA_PCIE_ENABLE=1` conflicted with the required
`HCCL_INTRA_ROCE_ENABLE=1` (`EI0001`). The valid command explicitly sets
`HCCL_INTRA_PCIE_ENABLE=0`, matching the Flash launcher:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
HCCL_IF_IP=33.215.119.204 HCCL_INTRA_PCIE_ENABLE=0 \\
HCCL_INTRA_ROCE_ENABLE=1 HCCL_BUFFSIZE=1024 VLLM_VERSION=0.20.2 \\
python3 -m pytest -q -s tests/e2e/attention/test_dsa_cp_owner_hccl.py
```

This proves the real NPU/HCCL owner-scatter and full-stage transport
mechanism. It does not prove the model's stateful compressor history or
attention/logit equivalence, and it does not make full-union staging a viable
TTFT optimization.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  hccl_owner_tp8_r2_pass.out
  hccl_owner_tp8_r2_pass.rc
  hccl_owner_tp8_final_8ea73c42.out
  hccl_owner_tp8_final_8ea73c42.rc
```

### G15: selected-page HCCL all-to-all transport — PASS

The full-union stage is functionally correct but sent every rank the union of
all ranks' requested C128 pages. A gated `materialize_selected_for_attention`
path now first all-gathers only variable-length page-id requests, then uses
HCCL `all_to_all_single` with variable splits to send each owner page only to
the rank that requested it. Each receiver stages its own sorted local-page
view and remaps only its own block table.

The TP8 NPU test uses 24 logical pages with three distinct local requests per
rank. It passed bit-exact reconstruction: every rank reported
`local_pages=3 sent_pages=3 received_pages=3`, whereas its full-union stage
would contain all 24 pages. The launcher gate is
`ENABLE_C128_OWNER_SELECTIVE_STAGE=1`; it is off by default until Flash
model-output and TTFT gates pass.

### G16: selective-stage Flash 8K candidate — FAIL / correctness not proven

r29 proved the selected-page path executes in the actual Flash model: 64-word
and 8K/one-output debug smoke requests completed, with C128 traces reaching
`materialize_ready` and `sparse_attn_ready` through multiple layers. r30 then
ran the clean paired workload (Flash TP8/EP8, FusedMC2 on, Mooncake off,
overlap off, 8K words, one output, warmup 1, runs 5). It is a valid candidate
but not an improvement:

| Run | Median TTFT | Relative to r23 B0 |
|---|---:|---:|
| r23 B0 feature-off | 593.847 ms | baseline |
| r24 full-union owner stage | 940.926 ms | -58.45% |
| r30 selected-page all-to-all stage | 967.903 ms | -62.99% |

The selected payload optimization alone is insufficient. Variable request
metadata, variable-split HCCL, and per-layer staging/reassembly remain on the
critical prefill path. r30 also produced `"Hello"` for repetition 2 while r23
produced `"你好"`, the same unresolved output mismatch seen in r24. Therefore
neither owner candidate establishes model-output equivalence or qualifies for
TTFT comparison claims.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r29_selective_smoke_fmc2.log
  bench_flash_tp8_c128_owner_r29_selective_smoke_fmc2_64.out
  bench_flash_tp8_c128_owner_r29_selective_smoke_fmc2_8k.out
  results/flash_tp8_c128_owner_r29_selective_smoke_fmc2_64.json
  results/flash_tp8_c128_owner_r29_selective_smoke_fmc2_8k.json
  log_single_node_prefill_flash_tp8_c128_owner_r30_selective_clean_fmc2_8k.log
  bench_flash_tp8_c128_owner_r30_selective_clean_fmc2_8k.out
  results/flash_tp8_c128_owner_r30_selective_clean_fmc2_8k.json
```

### G17: CP-local C128 producer alignment and output-collective gate — BLOCKED

The Flash TP8 layout probe (Mooncake disabled) established that the first
5,120-token prefill chunk is CP-aligned for C128: every rank receives 640
tokens, exactly five C128 groups. The global compressed slot rows are
contiguous per rank (`[0:5]` through `[35:40]`); for the observed request all
40 rows address logical page 11. The following 3,080-token chunk splits to
385 tokens per rank, which is not C128-aligned and must retain the
gathered-hidden/compressor fallback.

The gated local-producer implementation computes the aligned chunk from
`hidden_states_local`, uses `start_pos + CP-local offset` (not tokenizer
positions) to certify alignment, and keeps the existing WKV/SWA path
unchanged. CPU reference tests passed (13 tests): the 5,120 / TP8 plan yields
eight five-row slices and the 3,080 tail is rejected.

r33 was **INVALID**: the first planner used tokenizer position origins and
therefore never enabled the local branch. r34 confirmed the corrected plan on
all eight ranks, then terminated without a Python or HCCL error after the
first local `compressor` launch and before `c128_compressor_ready`; the client
stream ended before a text token and the API/worker processes exited. This is
the same target-runtime class of restriction already seen for dynamic work
after `compressor`: a host-side distributed collective over the asynchronous
local output is not a viable bridge. Do not repeat the list-based dynamic
`dist.all_gather` variant. The next viable implementation must consume the
compressor output in a graph-safe fused/direct-placement operator or an
explicitly validated static HCCL output buffer.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  c128_cp_layout_r31.log
  log_single_node_prefill_flash_tp8_c128_owner_r32_cp_slots_fmc2.log
  log_single_node_prefill_flash_tp8_c128_owner_r33_local_c128_fmc2.log
  bench_flash_tp8_c128_owner_r33_local_c128_fmc2_smoke_8k.out
  log_single_node_prefill_flash_tp8_c128_owner_r34_local_c128_alignment_fmc2.log
  bench_flash_tp8_c128_owner_r34_local_c128_alignment_fmc2_smoke_8k.out
```

### G18: static C128 result buffer integration — BLOCKED at server initialization

r35 replaces the prohibited Python-list `all_gather` bridge with a fixed,
TP-major buffer allocated before the C128 `compressor`, followed by
`all_to_all_single`.  For the validated 5,120-token / TP8 chunk it exchanges
only 40 compressed rows and preserves the existing rank-major global-slot
order expected by owner scatter.  It has not reached that branch yet.

The Flash server remained unready for more than eight minutes after launch:
the API port 7100 did not accept `/health`, the log stopped at 104,711 bytes
at `2026-07-28T21:54:55Z`, and no `c128_static_collective_*` marker was
emitted.  All eight workers remained alive, each with roughly 57,085 MB HBM
allocated and sustained host CPU use, while NPU AICore utilization was zero.
This is an initialization-state blocker, not evidence that the static copy or
HCCL collective is invalid.  No TTFT, cache-equivalence, or output-equivalence
claim follows from r35; do not use it as a performance result.

The CPU layout oracle passed on the target image (14 tests): it verifies that
the broadcast TP-major send buffer produces the rank-major compressed-slot
order expected after equal-split `all_to_all_single`.  The unready server was
then stopped with `SIGTERM`; all r35 API, EngineCore, and worker PIDs were
absent eight seconds later.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r35_local_c128_static_a2a_fmc2.log
```

### G19: static C128 output-buffer HCCL transport — PASS

The r35 startup failure does not reproduce in the isolated TP8 HCCL gate.  On
the target image, each rank writes its five local C128 rows into all eight
equal-sized TP-major destination chunks, then `all_to_all_single` restores the
40 rows in source-rank-major global-slot order.  The NPU test passed bit-exact
on all eight ranks in 9.45 seconds with the Flash RoCE settings
(`HCCL_INTRA_PCIE_ENABLE=0`, `HCCL_INTRA_ROCE_ENABLE=1`).

This validates the static buffer's HCCL and NPU-copy semantics, but it does
not validate the asynchronous compressor-to-buffer handoff or model output.
The next model gate may therefore use this implementation; it must retain the
existing 8K/one-output correctness-first smoke criterion before any TTFT run.

### G20: CP-local compressor RoPE boundary — FIXED, integration pending

The r34 CANN diagnostic identified the actual first local-compressor failure:
for each 640-token TP8 shard, `Compressor` requires
`min(tokenSize, tokenSize/cmpRatio + batchSize) = 6` RoPE rows.  The previous
local implementation sliced only its five emitted C128 rows and therefore
failed tiling on every rank with `ropeSin shape dim 0 ... should be ... 6, but
got 5` (`errno[561002]`).  This precedes the list-based collective and explains
why no `c128_compressor_ready` trace was emitted.

The local RoPE adapter now preserves the five shard rows and appends the
pre-existing padded compressed-position-zero row.  It is the same padding
metadata contract produced by `_get_padded_compressed_position`; no new RoPE
values are synthesized.  The target-image reference suite passes 15 tests,
including a rank-7 `[35:40] + padding` oracle.  The static HCCL gate remains
separately passed (G19).  Actual model compressor/attention/output validation
is still pending.

r36 (full capacity) and r37 (`MAX_MODEL_LEN=8192`, correctness-only) both
stopped at the pre-API DSA warmup/initialization boundary before a request and
without a new Ascend error.  They do not test this fix and have no TTFT result.

### G21: r38 real-Flash integration launch — BLOCKED before request dispatch

r38 used the corrected six-row local RoPE adapter, static TP8 C128 result
exchange, owner compact allocation, selected staging, FusedMC2, Mooncake off,
and the unchanged 1M-capacity / 8K-one-output workload.  It loaded all 70
checkpoint shards, then remained pre-API for more than 12 minutes: port 7100
never served `/health`, the vLLM log stopped at 104,170 bytes at
`2026-07-28T22:25:23Z`, and no `c128_local_compressor`,
`c128_compressor_ready`, or `c128_static_collective_*` marker was emitted.
There was no new Python, HCCL, or Ascend error in this run.

The service and all workers were stopped with `SIGTERM`; `npu-smi` confirmed
no remaining VLLM workers.  Thus r38 does not validate compressor handoff,
model output, cache equivalence, capacity reduction, or TTFT.  It is a
bootstrap blocker distinct from the r34 Compressor shape failure fixed in G20.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r38_local_c128_ropepad_static_a2a_fmc2.log
```

### G22: feature-off B0 bootstrap control — BLOCKED, environment confirmed

r39 is an otherwise identical Flash TP8/EP8 control with every C128 owner,
selective-stage, and local-compressor gate disabled.  It also loaded all 70
checkpoint shards, emitted the same three `rope_parameters` warnings, and
then never opened port 7100.  Its post-load log stopped at 19,226 bytes; no
DSA-CP request-path code can run in this configuration.  After the diagnostic
comparison, it was stopped with `SIGTERM` and `npu-smi` confirmed no residual
VLLM workers.

The r38 pre-API failure is therefore not attributable to the local C128
RoPE/static-HCCL changes.  The current pod/image cannot produce a new valid
B0 or candidate TTFT result until its generic vLLM bootstrap reaches API
readiness.  The previously valid r23 B0 measurement remains the only valid
baseline; r39 is launcher diagnosis only.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_b0_r39_bootstrap_fmc2.log
```

### G23: resumed `.204` feature-off bootstrap gate — PASS

Hypothesis: the r38/r39 post-load bootstrap stall was transient pod runtime
state rather than a deterministic Flash launcher or C128-owner failure.  A
clean `.204` pod with no vLLM/NPU processes should reproduce the real-Flash B0
API-ready state without changing the model, serving topology, or request.

Pinned configuration:

- image:
  `hcr.meta-wulan01.hw-wulan.local/antsys/vllm:release_0.20.2_0601_202607271124_aarch64`;
- source: `2fca35c1`, with the installed `c128_owner_cache.py` and `dsa_cp.py`
  hashes matching the branch;
- model: `/mnt/deepseek/models/DeepSeek-V4-Flash-w8a8-mtp`;
- TP8/EP8 on devices `0,1,2,3,4,5,6,7`;
- FusedMC2 on, Mooncake/MTP/layer-sharding/prefill-overlap/C128-owner/local
  compressor/selective stage off;
- unchanged 1M capacity controls and `prefetch` safetensor loading.

The first metric is bootstrap readiness, not TTFT.  `PASS` requires
`GET /health` to return HTTP 200 and the effective configuration in the launch
log to match the pinned values.  After that, one fixed 8,192-word/one-output
request must return a nonempty first text token.  The run is `FAIL` if the
process exits with an attributable error, `BLOCKED` if it repeats the
post-load/no-listener state without an attributable feature error, and
`INVALID` if any pinned control differs.  Do not start the C128-owner candidate
until this gate passes.

r40 passed.  All 70 shards loaded in 2m27s and the API reached HTTP 200 about
5m17s after launch.  The fixed 8K/one-output correctness request returned
`你好`.  After one warmup, five measured requests produced 596.680 ms median
client TTFT, 600.762 ms p90, and the same completion in every sample.  The
same-session 8% target is therefore 548.945 ms.  The B0 process group was
terminated after capture; port 7100 closed and all eight NPUs reported no
workload.

This narrows the historical conclusion: r38/r39 remain invalid, but they do
not prove the `.204` pod or image is intrinsically unable to bootstrap.  The
three `rope_parameters` warnings also appear in this successful run and are
not a failure signature.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_b0_r40_resume_fmc2.log
/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730T031115_b0_2fca35c1/
  b0_smoke_8k_1out.json
  b0_ttft_8k_1out_w1_r5.json
```

### G24: corrected local-compressor owner integration — FAIL at padded output copy

Hypothesis: with the six-row local RoPE contract and static result exchange,
the current owner candidate reaches the real CANN compressor, selected owner
materialization, sparse attention, and a nonempty output for the fixed request.

This is an integration/correctness gate, not a performance A/B.  Relative to
r40 it enables owner shard, compact allocation, CP-local compressor, selected
stage, and diagnostic markers together so every boundary can be localized.
FusedMC2 remains on; Mooncake, MTP, layer sharding, prefill overlap, profiling,
and synthetic weights remain off.  The model, TP8/EP8 topology, allocator
capacity, safetensor strategy, and 8K/one-output request remain unchanged.

Before launch, the current target-image reference suite must pass 15 tests and
the static TP8 HCCL result exchange must pass.  The model gate is `PASS` only
if all expected owner markers complete, the 3,080-token tail takes the
replicated fallback, and the completion matches B0 `你好`.  A boundary-specific
exception is `FAIL`; API readiness without a completed request is
`BLOCKED`; a changed control is `INVALID`.  No candidate timing follows until
this correctness gate and persistent-capacity accounting pass.

r41 reached HTTP 200 in 5m04s and accepted the fixed request, but every rank
failed on the aligned 5,120-token chunk after `c128_compressor_ready` and
`c128_static_collective_copy_begin`.  The CANN merged-token compressor returns
six local rows: five mapped C128 data rows followed by the padded batch row
required by its ABI.  The preallocated TP send view is `[8,5,512]`; copying the
unsliced `[6,512]` result failed with `aclnnInplaceCopy` error 161002 and CANN
errno 561000.  No rank entered the static HCCL collective, so HCCL is not
implicated.

The unaligned 3,080-token fallback independently reached compressor, owner
scatter, selected materialization, and sparse-attention completion on all 20
C128 layers and all eight ranks.  The request remains invalid because its
aligned chunk killed EngineCore.  The client correctly recorded no result
JSON and reported `stream ended without an SSE event containing a text token`.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r41_resume_correctness_fmc2.log
/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730T032110_c128_r41_2fca35c1/
/home/admin/logs/vllm/vllm_server.log
```

### G25: discard CANN output padding before the static TP exchange — PASS

Hypothesis: retaining the six-row RoPE/operator input while slicing the
compressor output to the five mapped prefix rows fixes r41 without adding a
post-operator host synchronization.  Commit `f44323d3` adds that static prefix
slice before the TP send copy and a CPU ABI regression.  It deliberately does
not inspect the asynchronous output shape.

The installed five-file owner runtime manifest matches the integration branch,
and the expanded target-image reference suite passes 16 tests.  The r42
configuration and request are otherwise byte-for-byte the r41 gate.  `PASS`
requires every rank to complete static copy, HCCL exchange, owner scatter,
selected materialization, sparse attention, and return B0 completion `你好`.
Another boundary failure is `FAIL`; no output or TTFT claim follows from an
incomplete request.

r42 reached HTTP 200 in 5m20s.  Its fixed 8K/one-output request returned the B0
completion `你好`.  For the aligned 5,120-token chunk, all 20 C128 layers and
all eight ranks reached local compressor, static copy, static HCCL collective,
owner scatter, selected-row materialization, and sparse attention: 160
begin/ready pairs at each stage.  The 3,080-token tail used replicated fallback
on all 160 layer/rank pairs and correctly emitted no static-copy or
static-collective markers.  The eight worker logs contain no `ERROR`, `EE`,
compressor-shape failure, CANN 561xxx failure, or HCCL failure.

The cold debug TTFT was 2.193209s and is diagnostic only.  Debug markers and a
cold first request make this run ineligible as a performance candidate.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r42_drop_pad_correctness_fmc2.log
/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730T034010_c128_r42_f44323d3/
  c128_smoke_8k_1out.json
```

### G26: account actual service-capacity bytes — FAIL

Hypothesis: the compact owner tensor reduces persistent C128 allocation and
therefore increases the block count at a fixed GPU utilization.  This gate
counts physical storage, not logical owner-tensor elements.

At the real Flash block count `B=4190`, TP8, 20 C128 layers, and 128-KiB C128
pages, the baseline raw KV allocation per rank is:

```text
B * (22 * 131072 + 21 * 16640) = 12.6160 GiB
```

The candidate retains those baseline allocations and adds:

```text
20 * ceil(B / 8) * 131072 = 1.2793 GiB  # compact owner tensors
B * 131072               = 0.5115 GiB  # full shared staging tensor
```

The result is 14.4068 GiB per rank, a 1.7908-GiB or 14.19% increase.  The
compact tensor alone holds 524 owner pages instead of 4,190 replicated pages
and therefore reduces that tensor's logical C128 elements by 87.494%, but it
does not release the aliased baseline raw storage.  The planner also retains
the baseline per-block denominator, so neither allocatable blocks nor service
capacity improve.  G26 is `FAIL`; r42 proves the execution path, not persistent
capacity reduction.

### G27: sparse owner-backed Ascend VMM pages — PASS

Hypothesis: Ascend VMM sparse backing can preserve the existing B-page virtual
cache ABI while physically backing only the pages owned by the local TP rank.
The first gate is an isolated same-process reserve/map/access probe on the .204
image; the second is a shareable-handle IPC map/read/write probe.  `PASS`
requires correct peer-visible data and physical allocation proportional to
local owner pages.  Missing APIs, unsupported handle export/import, or a driver
failure is `BLOCKED` with the exact failing API and status.

If the installed HDK/CANN cannot pass that gate, the fallback design is a
packed physical owner pool plus explicit logical-block-to-owner-slot
indirection.  It must remove the baseline C128 backing and the full B-page
staging allocation before any TTFT benchmark is admissible.

The `.204` pod runs CANN 9.0.0 and HDK/driver 25.5.1.  Its system
`acl_rt.h` and `libascendcl.so` provide the canonical `aclrtMem*` reserve,
physical allocation, map, V2 shareable-handle, PID authorization, import, and
unmap APIs.  G27 maps four logical VMM pages per rank while allocating only
the two pages selected by `owner(logical_page) = logical_page % 2`; the peer
imports those two handles into the remaining VA holes.

The two-rank NPU0/NPU1 run passed:

```text
allocation granularity/rank = 2,097,152 bytes
logical VA/rank             = 4 pages = 8,388,608 bytes
owned physical/rank         = 2 pages = 4,194,304 bytes
imported peer aliases/rank  = 2 pages = 4,194,304 virtual bytes
local/read/remote-write mismatches = 0/0/0 on both ranks
```

Both ranks acknowledged imported-alias release before owner allocations were
freed, exited zero, and left NPU0/NPU1 idle.  Imported aliases did not consume
another 4 MiB of HBM.  This is a physical-allocation and peer-alias `PASS`,
not yet a Torch or attention-kernel result.

The measured 2-MiB minimum granularity is 16 C128 pages of 128 KiB.  Therefore
single C128 pages cannot be independently sparse-mapped at their current
interleaved logical addresses.  A capacity implementation must pack
owner-local C128 slots contiguously, align each segment to 2 MiB, and remap
block-table IDs; VMM alone cannot preserve the current page-level layout.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g27_sparse_owner_20260730_121939/
```

### G28: consume a remote VMM mapping as a Torch NPU tensor — PASS

Hypothesis: an imported V2 VMM address can be wrapped as a non-owning Torch NPU
tensor and consumed by ordinary NPU kernels, rather than only by
`aclrtMemcpy`.

The probe uses the current image's private, version-pinned Torch-NPU
constructors:

```text
_construct_storage_from_data_pointer
_construct_NPU_Tensor_From_Storage_And_Metadata
```

NPU0 initialized 4,096 FP32 elements with ordinary `copy_`.  NPU1 imported
the 2-MiB VMM handle, wrapped the mapped address, and `clone()` read every
element exactly.  NPU1 then used ordinary `fill_(37)` on the remote tensor;
NPU0's subsequent `clone()` observed all 4,096 values as 37.  The child
processes exited `0/0`, no timeout signal was required, and the importer
unmapped before the exporter freed its allocation.

This proves the remote tensor data path required by an owner-pool prototype.
The wrapper is non-owning: production must retain an explicit VMM lease,
synchronize every accessing stream, drop all tensor aliases, then unmap and
free in that order.  It does not by itself fix G26's shared-pool capacity
accounting.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g28_remote_tensor_20260730_043455/
```

### G29: exact-shape local-compressor prefix-state oracle — PASS

Hypothesis: eight sequential TP-local 640-token compressor calls for the
5,120-token aligned prefix, followed by the replicated 3,080-token tail,
produce the same cache-bearing rows and semantically live state as the global
compressor path.

The one-NPU oracle uses the production `[8200,7168]` hidden shape, C128
weights, a 258-block FP32 state cache, logical-to-physical state IDs `1..257`,
production zero-position RoPE padding, and all eight TP start positions.  It
compares only completed C128 rows; the custom operator's final per-batch row is
ABI padding and is undefined even between repeated global calls.

All eight rank histories passed.  The five valid local prefix rows per rank
matched the global rows with worst `max_abs=0.0078125` and
`max_rel=0.0075758`.  All 24 cache-bearing tail rows were bit-exact.  The live
state at positions `8072:8200` and the future continuation state at
`8096:8224` were byte-identical on every simulated rank.  The target-image run
reported `3 passed` in 7.71s.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730T042131Z_prefix_oracle_r44_nonzero_blocks/
```

### G30: fixed-block runtime allocation accounting — PASS measurement, FAIL capacity

Hypothesis: the one-shot PyTorch storage accounting at fixed `B=4190`
reproduces G26's static conclusion.  r44 is the real-Flash feature-off B0;
r45 changes only owner shard, compact owner allocation, and selective stage.
Both keep the local compressor off, FusedMC2 on, and use the same fixed
8K/one-output correctness request.  This is an allocation gate, not a TTFT
candidate.

`PASS` for the measurement requires eight rank records, identical deploy
hashes and block count, B0 output `你好`, and a clean stop.  The current layout's
capacity hypothesis passes only if r45 `total_unique_bytes` is below r44.
An equal or larger total is a valid experiment whose capacity hypothesis
fails.

r44 passed the measurement gate.  All eight ranks reported:

```text
num_blocks=4190
baseline_raw_bytes=13546370560
compact_owner_bytes=0
c128_stage_bytes=0
total_unique_bytes=13546370560
```

Its 8K/one-output request returned `你好` and cleanup left no NPU worker.

The first owner run, r45, was `INVALID_ACCOUNTING_STAGE`: its helper flattened
lists and tuples but not the `dict_values` object holding reusable C128 stage
tensors, so `c128_stage_bytes=0` was a measurement bug.  The corrected helper
treats Torch tensors as atomic, recursively flattens finite mappings and
iterables without device reads, and passed five local plus five target-image
tests.

r47 reran the same owner configuration with only that helper corrected.  All
eight ranks emitted the same record:

```text
num_blocks=4190
baseline_raw_bytes=13546370560
compact_owner_bytes=1373634560
c128_stage_bytes=549191680
total_unique_bytes=15469196800
unique_storages=64
duplicate_references=104
cross_category_aliases=0
```

The fixed 8K/one-output request returned `你好`, the exact launcher log had no
fatal signature, TERM completed within 30 seconds, and all eight NPUs were
idle afterward.  The runtime result matches G26 exactly: the current owner
layout adds 1,922,826,240 bytes or 14.1944% per rank.  Runtime accounting is
`PASS`; persistent service-capacity saving is `FAIL`.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r44_b0_fixed4190_accounting_634ef01f_r2/
  20260730_r45_owner_fixed4190_accounting_634ef01f/
  20260730_r47_owner_fixed4190_accounting_stagefix_7355d108/
```

### G31: group-partitioned packed-pool contract — PASS CPU, runtime pending

Hypothesis: disjoint scheduler-group ranges and packed physical component
segments can remove the retained baseline C128 backing while preserving
sentinel block zero, deterministic inverse mapping, TP ownership, and the
measured 2-MiB VMM granularity.

The CPU planning contract assigns one-based disjoint logical ranges, reserves
one dummy physical page per component, converts interleaved C128 IDs into
contiguous owner-local slots, aligns each segment independently, and accounts
a bounded scratch region instead of the current full `B`-page stage.
Feature-off constructs no plan and translates block IDs identically.

For the exact 8,200-token fixed request, the admission-aware planner assigns
these six scheduler-group quotas:

```text
[17, 1, 642, 165, 65, 65]  # plus sentinel block 0
```

The totals include continuation/admission headroom and therefore differ from
the earlier required-now live-block trace.  They sum to 955 data blocks and
map to deterministic one-based ranges:

```text
[1,18), [18,19), [19,661), [661,826), [826,891), [891,956)
```

This is a pinned single-request planner contract, not yet a measured capacity
claim.  The exact persistent/scratch allocation bytes must come from the
serialized plan consumed by the target runtime.  General service still needs
dynamic quota borrowing or lifecycle-aware remapping.

The isolated planning prototype passed 22 tests across TP1/2/3/8, mapping
round trips, collision checks, alignment, invalid IDs, feature-off identity,
and Flash-shaped capacity accounting.  Runtime still needs coherent planner
metadata, scheduler partitioning, worker block-table translation, packed
allocation, and bounded C128 materialization before an NPU capacity claim.

### G32: prefill overlap launcher audit — INVALID configuration

Hypothesis: setting `prefill_comm_compute_overlap=1` alone improves real Flash
DSA-CP 8K/one-output TTFT by at least 8%, with B0, FusedMC2, and all owner
features unchanged.

r46 completed one warmup and five timed requests.  Its observed median was
602.496 ms versus the canonical r40 median of 596.680 ms.  That number is not
an overlap verdict: source inspection showed that
`prefill_comm_compute_overlap` is read only by the non-CP implementation in
`attention/dsa_v1.py`.  Real Flash DSA-CP runs
`attention/context_parallel/dsa_cp.py`, whose hidden-state all-gather overlap
is controlled by `multistream_dsa_preprocess`.

r46 is therefore `INVALID_CONFIGURATION`; the intended independent variable
never reached the measured path.  The versioned launcher now exposes and logs
three separate controls:

```text
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP
ENABLE_MULTISTREAM_DSA_PREPROCESS
ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT
```

The next A/Bs change only one of the two controls that the real Flash path
actually consumes.  Raw r46 evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r46_b0_overlap_on_634ef01f/
```

### G33: shared-expert multistream overlap — PASS signal, MISS target

Hypothesis: enabling only `multistream_overlap_shared_expert` overlaps the
shared-expert projections with the routed FusedMC2 MoE lane and improves
real-Flash 8K/one-output TTFT by at least 3%, without changing the model,
fixed block count, or DSA path.

r48 changed only that switch from `0` to `1`; r49 returned it to `0`.
FusedMC2 remained enabled and every prefill, DSA-owner, Mooncake, MTP,
accounting, profiling, and layer-sharding switch remained disabled.  Three
matched candidate/control sequences produced:

```text
candidate r48          control r49          median gain
585.530 ms             599.208 ms              2.282%
573.194 ms             586.727 ms              2.306%
559.924 ms             569.910 ms              1.752%
```

The sign is positive in all three repetitions, so the lane is retained for a
later combination.  It misses the independent 3% pass threshold and is not
the 8% TTFT result.  Greedy one-token text varied among `你好`, `Hello`, and
`I` in both feature-on and feature-off runs; that service-level variability is
not evidence of an overlap-specific correctness defect.  Cache/state oracles
and a nonempty completion remain the correctness gate.

Both services were stopped by their exact process groups and all eight NPUs
were idle after cleanup.  Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r48_b0_shared_expert_overlap_75b54627/
  20260730_r49_b0_a_after_shared_overlap_75b54627/
```

### G34: E3 local-current-KV — PASS mechanism, PASS <5% TTFT budget

Hypothesis: the aligned 5,120-token C128 prefix computes WKV, norm, RoPE, and
compressor rows from each TP-local hidden shard, exchanges the narrow
RoPE-complete KV and compressor outputs into the established replicated
caches, and leaves the unaligned 3,080-token tail on the full-hidden fallback.

The deployed E3 sources and launcher matched integration commit `84f3da22`
byte-for-byte.  The target-image reference suite passed `27/27`; effective
configuration proved `dsa_cp_local_current_kv=1`, FusedMC2 enabled, and every
overlap, owner-capacity, Mooncake, MTP, accounting, profiling, and
layer-sharding switch disabled.  The real Flash TP8 service loaded `70/70`
shards, returned HTTP 200, and its first fixed 8K/one-output request returned
`你好`.

The launch log contained no `DSA-CP local-current-KV active` record.  That
record is an INFO-level rank-0 `logger.info_once` emitted only after all
admission predicates pass.  Its absence cannot distinguish a rejected gate
from worker-log filtering or once-only sink behavior.  The request therefore
proves service correctness but does not prove that E3 ran; its cold
`1,624.761 ms` TTFT is not a performance sample.

The run is classified `INCONCLUSIVE_TELEMETRY`.  The next diagnostic is
default-off and reports all scalar admission inputs plus ordered rejection
reasons from every rank without reading device data.  Only an explicit
`accepted` record permits a matched TTFT run.

The exact process group terminated cleanly, port 7100 closed, and all eight
NPUs reported no running process.  Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r50_e3_deploy_84f3da22/
```

r51 resolved that telemetry gap.  One fixed request emitted exactly 320
admission records:

```text
8 TP ranks x 20 C128 layers x 2 chunks = 320
5,120-token prefix: 160 admitted
3,080-token tail:   160 rejected, local_compressor_plan_missing
```

Every admitted record had five C128 rows per 640-token local shard and exact
unpadded global metadata (`5120 == num_tokens_pad == num_input_tokens ==
num_actual_tokens`).  Every rank admitted all 20 C128 layers for the prefix
and rejected all 20 for the tail.  The output was `你好`, health remained 200,
and the exact process group stopped cleanly.

r52 then restarted the same E3 code and configuration with debug disabled and
no extra smoke request.  Against the matched r49 control sequence:

```text
window             r49 control     r52 E3        median regression
w1/r10 first        599.208 ms      601.494 ms        0.382%
w1/r10 repeat       586.727 ms      591.629 ms        0.835%
w5/r10 steady       569.910 ms      572.990 ms        0.540%
```

The steady p90 was 583.646 ms versus 575.609 ms, a 1.396% regression.  E3
does not improve TTFT, but it stays within the project's revised hard budget
of less than 5% TTFT regression.  It is retained as a mechanism dependency for
the packed owner-capacity path, not as a standalone TTFT optimization.
All 30 measured completions were nonempty, post-benchmark health was 200,
the fatal scan was empty, TERM was clean, and a retry snapshot showed all
eight NPUs idle.

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r51_e3_admission_diag_6419fa00/
  20260730_r52_e3_ttft_clean_6419fa00/
```
