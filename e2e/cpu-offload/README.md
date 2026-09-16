# Native CPU-offload reference benchmark

Reference: [official PR16544](https://github.com/vllm-project/vllm-ascend/pull/16544),
pinned head `ecb641ec3a47f74bf5173b88491a2aa65e0aec06`.

The measured image's e43cf1 source lacks an INT8 offload configuration flag.
To hold the model/framework/checkpoint constant, this harness imports the
official PR's Engram class and CPU operator unchanged and sets the constructor's
`cpu_offload=True`. It does NOT upgrade the whole PR framework or claim full
validation of that PR head. Original model preparation/routing call sites remain.

Unmodified source verification against GitHub blobs:

| File | Git blob |
|---|---|
| pr16544/engram_hbm.py | 452b144ba13c2632b8c814f08eb34decc808da54 |
| pr16544/engram_cpu.cpp | ef4ed44d1ea51bfa9c66868d069fa5c3234a80e2 |

`build_cpu.py` compiles only the CPU translation unit using installed PyTorch,
`-O3 -fopenmp`, and loads its existing `_C_ascend` registration. No edits to its
NEON conversion, parallel thresholds, pinned staging, event fencing, or lookup
algorithm. Worker CPU binding/thread configuration stays unchanged.

PR39 was inspected first. Its implementation is preserved in vendor_engram.py,
blob5eb12a6e8bb62269ffab588df91c53d8f6c961ac, but is NOT the measured E2E backend:
official PR16544 additionally fuses CPU lookup/dequant and fixes staging fencing.

Runtime gate: `test_reference.py` validates actual CPU INT8/FP32 tables, pinned
BF16 results, H2D copies/event reuse and exact independent CPU references for
empty/variable counts. Both old PR39 and selected PR16544 unit gates passed;
only selected PR16544 is used for the full-model comparison.

Benchmark entry uses the existing delivery launcher with fourth argument `hbm`
(disables VMM), `ENGRAM_NATIVE_CPU_REFERENCE=1`, and this directory on PYTHONPATH.
`patch_model.py` temporarily enables the constructor adapter; `--restore`
restores the exact previous delivery source checksum. Nothing is pushed upstream.

Workload: same checkpoint/32A3/TP8DP4EP32,8GiBKV,DSpark5,FlashComm1off,
prefix cacheoff,64requests perC1/C4/C8,64outputtokens,nonceA andseed1234.
Use existing `e2e/bench_http.py` and the saved HBM baseline. No extra repeats
for variance.
Completed result: [three-way comparison](../NATIVE_CPU_COMPARISON.md).
Native CPU C1/C4/C8 throughput50.81/141.29/219.47tok/s,192/192requests complete,
six exact serial checks pass. The explicit same-image adaptation boundary above
still applies; this is not full official-PR-head validation.

Launch in the tested shared directory:

```bash
python3 build_cpu.py
python3 patch_model.py --repo /vllm-workspace/vllm-ascend
bash launch.sh 0 10.30.14.37 10.30.14.37  # rank0 node
bash launch.sh 2 10.30.14.84 10.30.14.37  # rank1 node
```

The first launch accidentally replaced PYTHONPATH and lost the CANN `acl`
module. It is INVALID_SETUP, not a CPU-offload performance failure; the checked
launcher now prefixes the existing SDK path. No native implementation change
was needed. After benchmarking, stop workers and use `patch_model.py --restore`
before restarting the delivered VMM service with a fresh run ID.
