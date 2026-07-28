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

The current runtime cannot turn G3 into a feature flag by changing only cache
allocation. `vllm_ascend/attention/context_parallel/dsa_cp.py` explicitly
asserts `compressor_ratio <= 1` in its DSA metadata builder, so its CP path is
SWA-only. In parallel, `model_runner_v1.py::_allocate_kv_cache_tensors` still
allocates a full local compressed-attention tensor for every rank. Therefore a
correct feature-on C128 implementation must add all of the following together:

1. C128 owner-page allocation plus logical-page-to-owner metadata in the
   worker allocator.
2. Compressor prefix-state handoff/scan before a rank publishes its owned
   C128 pages.
3. HCCL-staged selected-row materialization into a bounded local workspace;
   the existing sparse-attention kernel must continue to receive local rows.
4. C128-capable DSA metadata and a feature-off replicated fallback, followed
   by an identical Flash TP8 8K/one-output candidate measurement.

VMM remote pointers remain an R&D transport alternative after the staged path
is correct; they are not required for this next feature-on gate.

## Current blocker

The old cloudide iTask pod holding
`/a3_inference/itask/workdir/models/DeepSeek-V4-Pro-w4a8-mtp` no longer
exists, and that checkpoint is not mounted in the `.204` pod. Therefore no
Pro-model 8K/one-output baseline or TTFT claim has been made from this pod.

## Next implementation gate

Implement a feature-off-by-default C128-only allocation and page-translation
layer. It must retain the existing replicated path, expose the above mapping,
and materialize selected owner rows into a local buffer before attention. Do
not wire C4 or SWA into this gate; first prove compressor prefix-state
equivalence and a tensor/kernel consumer for the VMM view.
