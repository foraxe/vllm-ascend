# E3: DSA-CP local current-KV experiment

## Status

`RUNTIME_SELECTION_PASS`, `PROFILE_GATE_PENDING`: the aligned-prefix
admission, unaligned-tail fallback, target-image unit tests, real-Flash
output, and matched TTFT budget passed on the `.204` A3 node.  A causal NPU
trace has not yet proved removal of the full-hidden collective or the exact
replacement payloads, so the experiment does not yet satisfy the full
structural `PASS` criteria below.

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
- Workload: 8,192 repeated `hello` words plus a unique suffix, which the
  service reports as an 8,200-prompt-token request; one output token
- Sampling: matched `w1/r10`, repeated `w1/r10`, then steady `w5/r10`
- Required output: every request emits a nonempty token; matched B0 already
  varies among `你好`, `Hello`, and `I`
- Baseline: matched r49 medians `599.207691`, `586.726767`, and
  `569.910270 ms`
- Baseline artifact:
  `/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730_r49_b0_a_after_shared_overlap_75b54627`
- TTFT budget: less than `5%` matched median regression; the same budget is
  applied to steady p90

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

Full structural `PASS` requires all of the following:

1. the service reaches health `200` and the debug-only admission run proves
   every TP rank admits the aligned prefix and rejects the tail;
2. one 8K/one-output correctness request completes with a nonempty token;
3. an NPU trace shows no full-hidden AllGather in the aligned C128 layer path,
   and shows a TP gather over the narrower RoPE-complete KV rows plus the
   fixed C128 result exchange;
4. matched median and steady p90 TTFT regress by less than `5%`.

`FAIL` is a completed experiment that violates correctness, structural, or
TTFT criteria. `INVALID` covers a launch/config mismatch, missing request
records, or a changed causal control. Kill the candidate immediately on a
collective-shape mismatch, rank hang, cache-scatter error, nonfinite output,
empty output, or a `5%` or larger matched TTFT regression.

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

r51, with debug enabled, emitted exactly 320 admission records: 8 ranks times
20 C128 layers times two chunks. All 160 5,120-token prefix calls were
admitted with five C128 rows per 640-token local shard. All 160 3,080-token
tail calls rejected with `local_compressor_plan_missing`, proving fallback.

r52 restarted the same feature-on configuration with debug disabled and no
extra request. Its matched medians were `601.494392`, `591.628510`, and
`572.990280 ms`, regressions of `0.381621%`, `0.835439%`, and `0.540438%`.
Steady p90 regressed `1.396%`. The service remained healthy, every completion
was nonempty, the fatal scan was empty, and teardown left all eight NPUs idle.
Raw evidence:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/
  20260730_r51_e3_admission_diag_6419fa00/
  20260730_r52_e3_ttft_clean_6419fa00/
```
