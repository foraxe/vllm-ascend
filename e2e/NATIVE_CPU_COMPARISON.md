# HBM vs native CPU-offload vs VMM

2026-09-16. Native CPU-offload throughput is approximately HBM-level on this
workload. VMM has a modest single-run lead, not a demonstrated large speedup.
Both host-backed paths save substantial HBM; this is not unique to VMM.

## Throughput

Same model/checkpoint and32A3,TP8/DP4/EP32,8GiBKV/NPU,DSpark5,FlashComm1off,
prefix cacheoff.64requests perC1/C4/C8,64outputtokens each, same nonceA/seed1234
and identical short varied prompts/client script hash. Earlier HBM/VMM results
are reused; no additional variance runs were performed.

| Concurrency | HBM owner tok/s | Native CPU-offload tok/s | VMM direct tok/s | VMM vs CPU |
|---:|---:|---:|---:|---:|
| 1 | 51.13 | 50.81 | 54.56 | +7.4% |
| 4 | 136.35 | 141.29 | 144.13 | +2.0% |
| 8 | 221.53 | 219.47 | 226.65 | +3.3% |

Small differences remain subject to run/batching variation. The earlier HBM
repeat had C8=205.11tok/s, so do not market these percentages as guaranteed.

## Latency

Each cell is median TTFT / average-per-request TPOT, milliseconds. TPOT uses
first-to-last HTTP token-ID-delta arrivals divided by output tokens minus one.

| Concurrency | HBM | Native CPU-offload | VMM |
|---:|---:|---:|---:|
| 1 | 225.10 / 15.98 | 229.27 / 16.12 | 218.60 / 14.91 |
| 4 | 322.60 / 23.80 | 299.70 / 23.68 | 350.70 / 23.15 |
| 8 | 537.07 / 28.27 | 522.67 / 28.14 | 557.61 / 26.78 |

Native CPU median E2E:1.247/1.785/2.298s. VMM does not improve every latency
metric: its C4/C8 TTFT is higher than this CPU baseline.

## HBM snapshots

| Mode | Observed mean chip HBM/NPU |
|---|---:|
| HBM control, with inactive VMM mapping retained | 51.05GiB |
| Native CPU-offload | 38.14GiB |
| Pure-host VMM | 37.10GiB |

These are total-chip snapshots on separate starts, not parameter-only figures
or a proven intrinsic1GiB difference between the host paths. Logical Engram
shards are approximately12.88GiB/NPU. Both host modes move those shards out
of HBM. VMM has208GiB rounded shared physical tables/node; CPU offload keeps
ordinary CPU shards totaling approximately206GiB/node. KV quota is unchanged.

## Exactly which native implementation?

[Official vllm-ascend PR16544](https://github.com/vllm-project/vllm-ascend/pull/16544),
pinned head `ecb641ec3a47f74bf5173b88491a2aa65e0aec06`:

- `engram_hbm.py`, blob `452b144ba13c2632b8c814f08eb34decc808da54`.
- `csrc/engram_cpu.cpp`, blob `ef4ed44d1ea51bfa9c66868d069fa5c3234a80e2`.
- Both files unchanged; fused ARMNEON INT8 lookup/dequant compiled standalone
  against installed PyTorch with `-O3 -fopenmp`. Threads1 per serving worker,
  same CPU binding configuration as the controls.
- Only constructor wiring sets `cpu_offload=True`; no lookup, staging/event,
  loader or routing algorithm edits. Full tables are ordinary CPU memory;
  selected BF16 rows use double-slot pinned staging and asynchronous H2D.

This is an **Engram-only backport into the previously measured e43cf1 image**,
not a full deployment/validation of PR16544's newer main framework/vLLM pair.
The old image has only FP8-native CPU offload and lacks the INT8 flag; changing
the checkpoint to FP8 would invalidate this matched-precision comparison.
The original image's model input preparation remains constant across modes.

PR39 was also checked but is NOT the measured backend; its older lookup used
PyTorch CPU operators rather than the fused CPU operator in PR16544.

## Correctness and evidence

Native CPU completed192/192performance requests,64outputtokens each. Six serial
message/finish/token-ID checks exactly matched HBM. C1 performance token IDs
matched64/64; C4/C8 matched54/64 and53/64. HBM self-repeat also has concurrent
variation, so exhaustive concurrent output equivalence is not claimed.
CPU table/pinned output/H2D/event-reuse unit checks passed first.32NPUs healthy.

- [CPU raw results](cpu-offload/results/native-cpu.json) and [log](cpu-offload/results/bench.log).
- [CPU build](cpu-offload/results/build-pr16544.log), [unit](cpu-offload/results/unit-pr16544.log).
- CPU memory snapshots: `cpu-offload/results/cpu-rank{0,1}-npu.txt`.
- [HBM/VMM original acceptance](ACCEPTANCE_20260916.md).
- [Adapter and reproduction](cpu-offload/README.md).

One INVALID_SETUP launch lost the CANN PYTHONPATH and failed before model load;
excluded entirely. The corrected launcher preserves SDK paths. No performance
claim uses that failure or the older failed per-request HostRegister prototype.

Remote evidence root:
`/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/cpu-offload-reference-20260916/`.
Temporary CPU footer was removed and the exact delivery source checksum
restored on both nodes. VMM service restored with fresh run ID
`delivery-after-cpu-20260916`; all six message/finish/token-ID checks passed
again,32NPUs healthy. See `cpu-offload/results/restored-vmm.json` and
`restored-rank{0,1}-npu.txt`. The temporary CPU baseline is no longer active.
