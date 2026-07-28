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

Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/20260728_prefill_owner/flash_c128_owner/
  log_single_node_prefill_flash_tp8_c128_owner_r16_fullalloc_u88_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r17_fullalloc_256blk_fmc2_8k.log
  log_single_node_prefill_flash_tp8_c128_owner_r18_fullalloc_64blk_fmc2_8k.log
```
