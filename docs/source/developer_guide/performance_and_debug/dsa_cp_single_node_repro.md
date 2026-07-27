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
  ENABLE_DSA_LAYER_SHARDING=0 \
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
  --words 4096 --warmup 2 --runs 5 --timeout 600 \
  --output result_synthetic64_prefill_overlap0.json
```

For the only valid immediate A/B, restart with
`ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=1`, retain all other fields, and write
`result_synthetic64_prefill_overlap1.json`. Compare median prompt tokens/s
only after both files contain five successful samples.

## DSA-CP optimization tasks from the design notes

### Track A: immediate measurement and low-risk implementation

1. Establish a clean B0 synthetic baseline, then A/B
   `prefill_comm_compute_overlap`. This is the existing pure-prefill overlap
   path; `multistream_dsa_preprocess` is decode-only and is not a prefill A/B.
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

## Result labels

- `PASS`: health check plus five successful timed samples under an unchanged
  workload.
- `FAIL`: launch, model load, or request failure; retain the exact error.
- `INVALID`: workload, model shape, cache placement, or request path changed.
- `BLOCKED`: required capacity or hardware capability is absent.

The production 384-routed-expert TP16/EP16 run is currently `BLOCKED` by HBM
capacity. The 64-expert dummy run is a separate `synthetic` evidence lane.
