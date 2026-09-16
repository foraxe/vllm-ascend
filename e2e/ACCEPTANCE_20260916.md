# Engram VMM full-model acceptance — 2026-09-16

Result: pure-host VMM tile16 serves the real model with about13.95GiB less
observed HBM per NPU and no measured throughput regression in this workload.
These are single-run observations, not a statistically established speedup.

Follow-up: [native CPU-offload three-way comparison](NATIVE_CPU_COMPARISON.md)
is now available. PR16544's fused CPU implementation also maintains roughly
HBM-level throughput while saving HBM; VMM is not the only host-backed option.

## Scope and controls

DSV4.1 Flash W8A8 on two dedicated Guian A3 nodes14037/14084, TP8/DP4/EP32,
DSpark5, fixed8GiB KV/NPU, max-batched4096, prefix cache disabled. Same image,
model, nonce A, seed1234, varied short prompts,64 output tokens and64 requests
per concurrency. C1/C4/C8 are separate closed-loop workloads with warmup.
Actual prompt token counts are recorded in each response's usage.

Metrics are HTTP token-ID-delta timing: TTFT is first delta arrival; TPOT is
average first-to-last delta time divided by completion tokens minus one.
This is not the distribution of individual inter-token arrival gaps.
Profile workloads are diagnostic-only and excluded from these measurements.

Acceptance requires completed requests, independent kernel correctness, exact
serial message/finish/token checks, measured HBM, and explicit regression
reporting. Concurrent generated-token equality is reported separately because
batching variability also affects the HBM control. Full service semantic
equivalence is not inferred from six prompts alone.

## Before restart: HBM owner vs scalar VMM direct

Same server processes and persistent model buffers; hot-switch after all four
DP engines drain, markers updated on both nodes before any next request.
HBM shards are retained in BOTH modes: this comparison does not save HBM.

| Mode | C | Output tok/s | Median TTFT ms | Median TPOT ms | Median E2E s |
|---|---:|---:|---:|---:|---:|
| HBM owner | 1 | 51.13 | 225.10 | 15.98 | 1.241 |
| scalar VMM | 1 | 53.91 | 218.85 | 15.03 | 1.186 |
| HBM owner | 4 | 136.35 | 322.60 | 23.80 | 1.836 |
| scalar VMM | 4 | 135.16 | 303.06 | 23.36 | 1.776 |
| HBM owner | 8 | 221.53 | 537.07 | 28.27 | 2.297 |
| scalar VMM | 8 | 219.59 | 544.53 | 28.18 | 2.350 |

All192 requests/mode completed,64 output tokens each. Six serial correctness
prompts matched message, finish reason and token IDs. Perf token sequences
matched64/64 at C1,53/64 at C4,41/64 at C8. The earlier large C8 regression did
not reproduce; its disappearance must not be credited to tile16.

Raw results: [HBM](results/acceptance-20260916/hbm-before.json),
[scalar VMM](results/acceptance-20260916/vmm-scalar.json).

## After restart: matched HBM vs tiled VMM, then pure host

| Mode | C | Output tok/s | Median TTFT ms | Median TPOT ms | Median E2E s |
|---|---:|---:|---:|---:|---:|
| HBM owner after restart | 1 | 51.04 | 227.68 | 15.99 | 1.246 |
| tiled VMM, HBM retained | 1 | 54.19 | 218.56 | 15.05 | 1.171 |
| pure-host tiled VMM | 1 | 54.56 | 218.60 | 14.91 | 1.161 |
| HBM owner after restart | 4 | 137.03 | 290.02 | 24.65 | 1.847 |
| tiled VMM, HBM retained | 4 | 138.41 | 302.59 | 24.15 | 1.838 |
| pure-host tiled VMM | 4 | 144.13 | 350.70 | 23.15 | 1.771 |
| HBM owner after restart | 8 | 205.11 | 573.61 | 29.85 | 2.465 |
| tiled VMM, HBM retained | 8 | 213.63 | 577.31 | 28.74 | 2.407 |
| pure-host tiled VMM | 8 | 226.65 | 557.61 | 26.78 | 2.301 |

All192requests/mode completed; six serial checks exactly matched original
HBM message/finish/token IDs. Pure-host C1 workload64/64 token sequences
matched; C4/C8 matched48/64 and42/64. HBM-after versus HBM-before itself matches
55/64 and46/64 at C4/C8. Concurrent differences are therefore reported, not
hidden and not automatically attributed to VMM. Comprehensive model-quality
equivalence beyond these requests remains unproven.

C8 HBM throughput varied221.53→205.11 between runs. User accepts current
variance; no extra repetitions were made. Pure-host throughput is higher than
both measured HBM runs, but TTFT is NOT uniformly better: C4 pure350.70ms
versus original322.60ms and restarted290.02ms. Do not claim every latency wins.
Likewise, tiling's incremental E2E effect is not isolated across residency and
restart changes; its isolated unit speedup is separately documented.

## Actual HBM savings

| Post-startup mode | Mean chip HBM,32NPUs | Difference |
|---|---:|---:|
| tiled switch, HBM tables retained | 52,273.8 MiB (51.05 GiB) | control |
| pure-host VMM | 37,993.1 MiB (37.10 GiB) | -14,280.7 MiB (-13.95 GiB)/NPU |

These are npu-smi total-chip snapshots, not torch parameter-only memory.
Logical Engram shards are approximately12.88GiB/NPU; the rest of the observed
reduction can include allocation/temporary-buffer differences. Same8GiB KV
quota, same graph capture sizes and model. Two complete host tables occupy
208GiB rounded physical memory per node, shared across that node's16consumers.
This is startup VMM allocation, not per-request registration.

Fresh MODE=direct constructs meta placeholders and substitutes VMM aliases;
it does not retain the switch-mode HBM control tables. All32NPUs healthy after
the workload; service health200 and allfourDP running/waiting counters0.
Allocator counters remain4allocations/0frees perworker through>2400prepares.
Servers remain running in pure-host mode in the two dedicated E2E pods.

## Profile conclusion

One short active C8 profile, parsed offline. Ten fused Engram kernels total
0.418ms (median40.45us), while a584.866ms captured device gap overlaps
584.208ms of host `gloo:all_reduce`. The trace is nonstationary and potentially
perturbed by collection; it does not establish the production cause of that
wait. No speculative prefetch/double-buffer change was made.
See [profile report](PROFILE_ACCEPTANCE.md) and
[architecture report](model_architecture_report_acceptance.md).

## Evidence and reproducibility

Remote artifacts/code backups:
`/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/e2e-code/acceptance-20260916/`.
One lightweight journal: [progress.md](../progress.md).

```bash
python3 bench_http.py --input-sizes 128 --concurrencies 1 4 8 \
  --requests 64 --nonce A --output RESULTS.json
# Subsequent modes additionally use --baseline hbm-before.json.
python3 summarize_acceptance.py results/acceptance-20260916
```

Current pure-host launch (node0; node1 uses rank2/localIP10.30.14.84):

```bash
ENGRAM_E2E_BACKEND=vmm ENGRAM_E2E_REUSE=0 \
ENGRAM_E2E_RUN=acceptance-pure-20260916 \
bash launch.sh 0 10.30.14.37 10.30.14.37 direct
```

Use a NEW runID when restarting; existing VMM descriptors are deliberately
not reused. Launch provenance and results are preserved in the raw artifacts.
Validated summary: [summary.json](results/acceptance-20260916/summary.json).
The older per-request paging failure and7x preparation-only results remain
historical evidence, not descriptions of this completed pure-host run.
