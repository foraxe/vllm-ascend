# DSA-CP single-node prefill reproduction and task map

This note records the 16-NPU A3 single-node reproduction surface used for
DeepSeek-V4-Pro DSA-CP prefill work. It separates a synthetic path-performance
experiment from a production-model result.

## Scope and current capacity boundary

The historical `P0/start.sh` launch is a 4P2D, DP4 x TP16 deployment. Its EP
group therefore has 64 ranks. On one 16-NPU node the matching DSA-CP layout is
TP16/DP1 and EP16, which owns 24 routed experts per rank instead of six.

With the W4A8 production checkpoint, TP16/EP16 fails during base MoE weight
construction before serving starts:

```text
torch.OutOfMemoryError: Tried to allocate 254.00 MiB
NPU 0; 61.27 GiB total; 60.40 GiB already allocated; 257.34 MiB free
```

Disabling MTP does not change that boundary because the failure occurs before
the MTP draft layer is constructed. Do not report any 16-NPU synthetic result
as a 384-routed-expert model result.

## Files and pod

Local experiment root:

```text
/Users/nyx/projects/wip_and_temp/vllm_dsa_cp
```

Remote role directory:

```text
/a3_inference/itask/workdir/shared/zhaomingchu/aiworker/codex/pro-debug/P0
```

The root can instead be under `/a3_inference/shared/...`; resolve it first on
the target pod. Preserve `P0/start.sh`; use the independent
`start_single_node.sh` launcher.

For disposable iTask capability gates, preserve the evidence outside the pod
workdir. The 2026-07-28 DSA owner/VMM run uses:

```text
/a3_inference/itask/workdir/shared/nyx/dsv4_dsa_cp/20260728_owner_vmm_gates/
```

Create an otherwise idle 16-NPU gate pod with the same image family, then
verify the resulting Pod before copying probes or starting a service:

```bash
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml \
  /Users/nyx/bin/itask -n cloudide create \
  --name nyx-dsv4-dsa-struct-YYYYMMDD \
  --user hx02481871 --type a3 --16card \
  --image hcr.meta-wulan01.hw-wulan.local/antsys/vllm:release_0.20.2_0601_202607271124_aarch64 \
  --skip-sync --no-model-download 'sleep infinity'

rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml \
  kubectl --context=a3 -n cloudide get pod <created-pod> -o wide
```

## Deploy the launcher and the test-only compatibility fix

From the local experiment root, set the current pod name once and copy the two
artifacts. The `rtk` prefix is required in this environment.

```bash
DSA_POD=ide-run-task-nyx-vllm-a3-20260727-16npu-vllm-dsa-cp-hx0248w2rk7
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  start_single_node.sh \
  "${DSA_POD}:/a3_inference/itask/workdir/shared/zhaomingchu/aiworker/codex/pro-debug/P0/start_single_node.sh"
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py \
  "${DSA_POD}:/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/attention/context_parallel/dsa_cp.py"
```

`dsa_cp.py` accepts both production-materialized 3-D `wo_a` weights and the
2-D parameter left by `DummyModelLoader`, materializing a view only in the
latter case before `npu_transpose_batchmatmul`.

## Launch a synthetic DSA-CP baseline

The synthetic path uses 64 routed experts, disables the checkpoint-shaped hash
router, and uses dummy weights. It preserves TP16, DSA-CP, FlashComm1, the
Mooncake producer configuration, top-k=6, and the request shape. It is valid
only for path performance and regression checks.

```bash
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide exec "${DSA_POD}" -- sh -c '
  cd /a3_inference/itask/workdir/shared/zhaomingchu/aiworker/codex/pro-debug/P0 &&
  chmod +x start_single_node.sh &&
  RUN_ID=synthetic64_overlap0 \
  SYNTHETIC_ROUTED_EXPERTS=64 \
  ALLOW_SYNTHETIC_WEIGHTS=1 \
  ENABLE_DSA_LAYER_SHARDING=1 \
  ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0 \
  nohup ./start_single_node.sh > launcher_synthetic64_overlap0.log 2>&1 &
'
```

Wait for `health_http=200` before submitting work. Dummy initialization uses
all worker CPUs and can take several minutes.

```bash
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide exec "${DSA_POD}" -- sh -c \
  'curl -sS -o /dev/null -w "health_http=%{http_code}\\n" http://127.0.0.1:7100/health'
```

Run the fixed workload after copying `bench_prefill_only.py` into the same
remote role directory:

```bash
python3 bench_prefill_only.py \
  --words 8192 --warmup 1 --runs 3 --timeout 600 \
  --output results/b0_8k_perf.json
```

For the only valid immediate A/B, restart with
`ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=1`, retain all other fields, and write
`results/overlap1_8k_perf.json`. Compare median prompt tokens/s only after
both files contain three successful timed samples. The 2026-07-27 B0 result
with this exact workload is `6191.205 tok/s` median (range `6108.545` to
`6245.094`); it is synthetic path evidence only.

The first A/B result is `FAIL`: overlap=1 reached `6037.137 tok/s` median
(`-2.4885%` versus B0), with all B0 and overlap=1 validity fields unchanged.
Do not tune this flag further for this workload without a changed code path or
a profile proving a different overlap opportunity.

## FusedMC2 8K streaming-TTFT A/B/A (`PASS`)

The profile's largest communication kernel sum is the three variable-size MoE
AllToAllV legs (activation-scale dispatch, activation dispatch, and expert
return). On A3 the existing `FusedMC2CommImpl` replaces that prefill path with
the fused `dispatch_ffn_combine` operator. This is a configuration-only
experiment: it does not alter DSA-CP cache semantics, model shape, or request
shape.

Use the same synthetic baseline, but change only `ENABLE_FUSED_MC2`:

```bash
RUN_ID=fmc2_8k_ttft \
SYNTHETIC_ROUTED_EXPERTS=64 \
ALLOW_SYNTHETIC_WEIGHTS=1 \
ENABLE_DSA_LAYER_SHARDING=1 \
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0 \
ENABLE_FUSED_MC2=1 \
ENABLE_MTP=0 \
ENABLE_TORCH_PROFILER=0 \
bash ./start_single_node.sh
```

`start_single_node.sh` emits `"enable_fused_mc2": 1` in
`--additional-config`; verify that exact field in the launch log before
benchmarking. Wait for `health_http=200`, copy `bench_ttft_stream.py` to the
role directory, then run the fixed streaming contract:

```bash
python3 bench_ttft_stream.py \
  --endpoint http://127.0.0.1:7100/v1/chat/completions \
  --words 8192 --warmup 1 --runs 10 --timeout 600 \
  --output results/fmc2_8k_ttft_stream.json
```

The metric is client time to the first nonempty SSE text delta, with one output
token, `temperature=0`, a unique prompt suffix, and prefix caching disabled.
For the 2026-07-27 TP16/EP16 dummy-model A/B/A, all timed responses had the
same four-character completion length:

| Configuration | 10-run TTFT P50 | TTFT change vs post-run B0 |
|---|---:|---:|
| B0 recheck (`enable_fused_mc2=0`) | 1.326215 s | baseline |
| FusedMC2 first distribution | 1.211960 s | -8.61% |
| FusedMC2 independent repeat | 1.185368 s | -10.62% |

The corresponding 8K non-streaming cross-check was `6955.178 tok/s` median
versus the prior B0 `6191.205 tok/s` (+12.33%). Preserve the raw JSON outputs
and restore `ENABLE_FUSED_MC2=0` after the candidate run to execute the B0
recheck. These are synthetic path-performance results, not a production
384-expert accuracy or throughput claim.

## DSA-CP optimization tasks from the design notes

### Track A: immediate measurement and low-risk implementation

1. Establish a clean B0 synthetic baseline, then A/B
   `prefill_comm_compute_overlap`. This is the existing pure-prefill overlap
   path; `multistream_dsa_preprocess` is decode-only and is not a prefill A/B.
   The B0 profile shows the largest communication kernel-sum is variable-size
   MoE `alltoallv` dispatch/return; this A/B tests whether it is hidden.
2. Add a current-KV execution view: attention consumes WKV/compressed/indexer
   artifacts before their paged-cache scatter completes; page publication and
   P/D/store persistence become asynchronous.
3. Fuse WKV, norm, RoPE, compression/quantization, final-page placement, and
   epoch publication. Do not materialize temporary contiguous KV followed by a
   separate scatter.
4. Fuse attention-output exchange with `o_proj`: replace
   pack -> contiguous -> AllToAll -> receive buffer -> o_proj with an
   object-aware A2A/GEMM/direct-reduce path.

Acceptance for every task: same prompt/batch shape, per-layer timing,
end-to-end prefill tokens/s, peak HBM, fabric bytes, and attention-start time.

### Track B: eliminate redundant current-KV production

5. For stateless/SWA layers, keep hidden states sequence-sharded, compute WKV
   once at the token owner, and directly place the required-now KV into the
   consumer cache slots. First validate full 16-way fan-out with the existing
   replicated attention cache.
6. Replace full fan-out with causal required-now fan-out and defer the reverse
   triangle of replicas. On the final prompt chunk, elide P-side durable
   replicas unless prefix admission requires a canonical copy.
7. Remove the materialized TP16 hidden AllGather. A lower-risk phase performs
   tiled remote reads into cache production; the higher-value phase is owner
   compute plus direct placement. Measure both rather than assuming remote
   reads beat HCCL AllGather.

The first full prototype is E1/E2 on an SWA layer with a batch-1 5120-token
chunk (320 tokens/rank, five 64-token blocks). It changes
`hidden AllGather + 16x WKV` to `one owner WKV + direct-out` without changing
the existing attention kernel.

Before that device prototype, run the CPU semantic oracle:

```bash
python -m pytest tests/ut/attention/test_dsa_cp_owner_placement_reference.py -q
```

It proves owner-local stateless WKV plus direct target-slot placement matches
the gathered-hidden reference cache, rejects colliding target slots, and
explicitly demonstrates why stateful C4/C128 compression needs a prefix-state
protocol instead of this first shortcut.

### Track C: cold-history capacity after TTFT path is sound

8. Owner-shard C128 compressed history by logical page modulo 16; keep Q,
   softmax state, accumulator, and hot SWA local. The attention kernel must
   fetch selected C128 pages into on-chip tiles rather than gather them to HBM.
9. C4 V1: owner-shard C4 KV while retaining the replicated indexer. Group
   selected pages by owner and fetch only selected pages.
10. C4 V2: colocate C4 index and KV shards, generate local candidates, merge
    global Top-K candidates, then fetch selected owner pages. A controlled
    replication progression is 16 -> 4 -> 1.
11. Treat C4/C128 compressor state separately. The strong form is local
    segment transforms plus a 16-rank prefix scan of compact states; ordered
    boundary-state handoff is a fallback, not a free parallel solution.

Do not owner-shard the dense SWA hot window first: it is repeatedly reused and
remote reads would put fabric latency on the critical attention path.

## Ascend VMM and peer-memory feasibility gate

CUDA VMM cannot be called from this NPU implementation. The CANN counterparts
exist, but they are a separate R&D gate rather than a launch flag:

```text
aclrtReserveMemAddress       reserve virtual address space
aclrtMemGetAllocationGranularity
aclrtMallocPhysical          create physical allocation
aclrtMapMem / aclrtUnmapMem  map/unmap an allocation
aclrtMemSetAccess            set access permission
aclrtMemExportToShareableHandleV2 / aclrtMemImportFromShareableHandleV2
aclrtDeviceCanAccessPeer / aclrtDeviceEnablePeerAccess
```

`vllm_ascend/csrc/camem_allocator.cpp` already uses local reserve/physical
allocation/map for its allocator. It does **not** export/import peer mappings,
and `dsa_cp.py` passes ordinary local cache tensors to
`npu_sparse_attn_sharedkv`. A VMM map alone therefore does not prove that the
stock fused attention operator can dereference remote rows.

Run a separate two-process probe before modifying DSA-CP:

1. On the intended rank edges, verify the CANN symbols and that
   `aclrtDeviceCanAccessPeer` returns `1`.
2. Export a 64 MiB physical allocation from rank 0, import/map it in rank 1 at
   an allocation-granular VA, and verify deterministic rank-1 read/write with
   explicit synchronization and exporter lifetime held.
3. Only then prove an AscendC accessor or a supported `HcclBatchGet/Put` path
   can consume it. Do not assume a PyTorch tensor or the stock sparse-attention
   kernel accepts a mapped remote pointer.

For immediate TTFT, use HCCL staged owner/fan-out experiments first. VMM is
the enabling path for the later owner-direct-placement prototype, not for MoE
`alltoallv` dispatch/return.

### 2026-07-28 capability result: VMM/SHMEM `BLOCKED` on the iTask image

The exact two-process NPU0/NPU1 probe passes
`aclrtDeviceCanAccessPeer` and `aclrtDeviceEnablePeerAccess`, but the corrected
V2 ABI uses `aclrtMemFabricHandle` (128 bytes) and still receives
`ACL_ERROR_RT_FEATURE_NOT_SUPPORT (207000)` from
`aclrtMemImportFromShareableHandleV2`. The test intentionally omits HBM
`aclrtMemSetAccess`, as CANN SHMEM's HBM path does.

This is independently reproduced by CANN SHMEM commit `341b46e`: the isolated
wheel builds and imports under CANN 9.0.0, but its two-rank symmetric HBM heap
fails at `aclrtMemExportToShareableHandleV2` with `ret: 207000` on both ranks.
Therefore `get_peer_buffer`, remote-pointer attention, and direct peer-tensor
cache placement are unavailable on this runtime. Keep the first owner-current
SWA implementation as staged HCCL KV placement into ordinary local cache
views; do not make VMM a prerequisite for the TTFT lane.

IPC is independently `BLOCKED` on the same image. The two-process probe
successfully creates a P2P HBM allocation, exports an IPC key, sets the
importer TGID and HCCS access link, but
`aclrtIpcMemImportByKey(..., ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS)`
returns `207000 feature not support`. Thus this result is a driver/runtime
boundary, not a conclusion that IPC cannot replace VMM on HDK 25.5/CANN 9;
rerun the same gates there before selecting a direct-pointer transport.

The companion HCCL semantic gate passed at `[320, 1024] -> [320, 512]`:
`local WKV -> all-gather(KV) -> shuffled cache slots` matches the current
`all-gather(hidden) -> WKV` reference. This is an oracle result only, not a
collective timing measurement.

## Result labels

- `PASS`: health check plus five successful timed samples under an unchanged
  workload.
- `FAIL`: launch, model load, or request failure; retain the exact error.
- `INVALID`: workload, model shape, cache placement, or request path changed.
- `BLOCKED`: required capacity or hardware capability is absent.

The production 384-routed-expert TP16/EP16 run is currently `BLOCKED` by HBM
capacity. The 64-expert dummy run is a separate `synthetic` evidence lane.
