# E3: DSA-CP local current-KV experiment

## Status

`CODE_READY_NPU_BLOCKED`: the feature-off path and CPU/reference gates are
implemented. The worktree was not deployed to an A3 pod, so correctness,
collective shape, and TTFT remain NPU gates.

## Hypothesis

For one C128-aligned prefill request, each TP rank can run the replicated WKV
projection, KV norm, RoPE, and C128 compressor on its local sequence shard.
Exchanging the RoPE-complete SWA KV rows and compressed C128 rows reconstructs
the existing replicated consumer caches while removing the full hidden-state
AllGather.

This feature does not require `C128OwnerShardCache`. C4, decode, mixed-request
batches, unaligned tails, and calls without an SP gather retain the established
hidden-state gather.

## Fixed experiment

- Model: `/mnt/deepseek/models/DeepSeek-V4-Flash-w8a8-mtp`
- Hardware: one A3 node, TP8/EP8, devices `0,1,2,3,4,5,6,7`
- Workload: one 8192-token input, one output token
- Sampling: one warmup plus five recorded streaming requests
- Required output: every request emits the same nonempty first token as B0
- Baseline: real-Flash B0 median TTFT `596.6797508299351 ms`
- Baseline artifact:
  `/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730T031115_b0_2fca35c1`
- TTFT target: median at most `548.9453707635403 ms` (`8%` below B0)

Keep the following controls fixed:

```bash
A3_MODEL_PATH=/mnt/deepseek/models/DeepSeek-V4-Flash-w8a8-mtp
TP_SIZE=8
A3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENABLE_DSA_LAYER_SHARDING=0
ENABLE_MOONCAKE_KV_CONNECTOR=0
ENABLE_C128_OWNER_SHARD=0
ENABLE_C128_OWNER_LOCAL_COMPRESSOR=0
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0
ENABLE_FUSED_MC2=1
ENABLE_MTP=0
ENABLE_TORCH_PROFILER=0
ENABLE_DSA_CP_LOCAL_CURRENT_KV=1
```

The causal variable is only `ENABLE_DSA_CP_LOCAL_CURRENT_KV`.

## Pass, fail, and kill criteria

`PASS` requires all of the following:

1. the service reaches health `200` and rank 0 logs
   `DSA-CP local-current-KV active`;
2. one 8K/one-output correctness request completes with the B0 first token;
3. an NPU trace shows no full-hidden AllGather in the aligned C128 layer path,
   and shows a TP gather over the narrower RoPE-complete KV rows plus the
   fixed C128 result exchange;
4. the one-warmup/five-run median TTFT is at most
   `548.9453707635403 ms`.

`FAIL` is a completed experiment that violates correctness, structural, or
TTFT criteria. `INVALID` covers a launch/config mismatch, missing request
records, or a changed causal control. Kill the candidate immediately on a
collective-shape mismatch, rank hang, cache-scatter error, nonfinite output, or
any request output mismatch. An aligned-run TTFT regression is sufficient to
stop this variant before a broader SWE-bench run.

## Mechanism and fallback

Feature off passes the original `need_gather_q_kv` value to
`maybe_all_gather_and_maybe_unpad`, preserving the established path.

Feature on is admitted only when
`can_use_c128_local_current_kv(...)` sees prefill, C128, a required SP gather,
and the single-request aligned `C128LocalCompressorPlan`. The WKV projection,
contiguous request positions, an exact unpadded local-hidden shape,
`num_input_tokens == num_actual_tokens`, and zero global token padding. The WKV
projection, KV norm, and RoPE consume local hidden rows and local position tables. A TP
gather reconstructs global KV row order before the unchanged SWA slot scatter.
The already-validated fixed C128 result exchange reconstructs global
compressor-slot order before the unchanged replicated C128 scatter.

## Admission diagnostics

For one correctness-only launch, set `ENABLE_C128_OWNER_DEBUG=1`. Every C128
invocation prints a `DSA_OWNER_TRACE c128_local_current_kv_admission` line with
the admission result, every boolean/scalar gate input, and all rejection
reasons. The report reads only Python booleans, integer metadata, tensor shape
metadata, and the CPU-built compressor plan; it does not inspect NPU tensor
values or call `.item()`, `.cpu()`, or a synchronization API.

Keep this debug gate off for TTFT measurements. Debug off performs no admission
logging. The older `logger.info_once("DSA-CP local-current-KV active ...")`
line is positive evidence when present, but its absence alone is inconclusive:
the admission gate may have rejected every invocation, the INFO sink may be
filtered or routed elsewhere, and `info_once` emits from rank 0 only.

The r50 deployment of integration commit `84f3da22` passed its 27 target-image
unit tests, effective-config check (`enable_dsa_cp_local_current_kv=1`),
service health, and output `你好`, but its recursive log scan found no active
marker. This is a correctness/configuration `PASS` and an E3 structural
`UNPROVEN`: the old positive-only instrumentation cannot recover which
admission input rejected the invocation, or distinguish rejection from an
unobserved INFO message. Rerun one correctness request with the debug report;
do not use that run for TTFT.
