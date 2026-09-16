# Host-backed Engram delivery

Opt-in shared-DRAM Engram for the validated A3 DSV4.1 fork. Model-side change:
[model.patch](model.patch), seven lines. Runtime: [engram_vmm](engram_vmm/).
Benchmark/switching/paging code is not a runtime dependency.

Validated base: vLLM-Ascend `e43cf1e9f5d9bead076853aa6bcacb671465de94`,
vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, torch-npu2.10.0.post4,
CANN9.1.0, HDK25.5.1.1, A3. This is a pinned-fork patch, not an upstream release.
Requires INT8 width256/group32 FP32 scales, FlashComm1 off, expandable NPU
allocator, sufficient host huge-page memory, trusted same-UID local workers.
Validated serving topology: TP8/DP4/EP32 over two nodes,8GiB KV per NPU.

## Install and run

Stop/drain the service before changing source. Build/install on the A3 host:

```bash
bash build.sh
python3 install.py --repo /vllm-workspace/vllm-ascend --check
python3 install.py --repo /vllm-workspace/vllm-ascend
# On both nodes use the SAME fresh run ID; examples use the tested node IPs.
bash launch_a3.sh 0 10.30.14.37 10.30.14.37 fresh-run-id  # node0
bash launch_a3.sh 2 10.30.14.84 10.30.14.37 fresh-run-id  # node1
```

`install.py` refuses other revisions or unknown source edits. Replacing the
exact earlier experiment requires `--replace-experiment`; its source is backed
up as `model.py.engram-experiment.bak`. Repeated installation is idempotent.
The launcher adds this directory to PYTHONPATH. Other deployments can instead
set PYTHONPATH and `VLLM_ASCEND_ENGRAM_VMM_RUN=<fresh-id>` with their normal CLI.
Run IDs permit only letters/numbers/underscore/hyphen and must be shared across
ranks. Do not reuse old process handles or enable the legacy E2E switches.

Use fourth launcher argument `hbm` to run the original HBM backend. To remove
the source hook entirely, stop workers then run `install.py --repo ... --restore`.
Package/native updates take effect only in newly started workers. Native builds
use atomic replacement, never truncate a mapped shared library.

## Data path and ownership

Startup: checkpoint → temporary local HBM → persistent shared host table.
Inference: IDs → direct mapped-host loads → grouped INT8 dequant → final BF16
HBM buffer. No per-request mapping or explicit owner routing. CPU hashing,
ID transfer and final BF16 HBM buffers remain.

- `native.cpp`: Supermem M7-derived host VMM allocation/import, ABI2 ownership
  tokens, serialized registry, exact failure propagation and retry state.
- `mapping.py`: ctypes/tensor adapter, startup publication, tensor-view guard,
  constructor rollback and idempotent close.
- `table.py`: sharded checkpoint loading into node-shared tables.
- `lookup.py`: previously measured tile16 kernel and persistent outputs.
- `integration.py`: opt-in model preparation/load and worker shutdown hooks.

Each mapping's descriptor is created exclusively; existing live/stale handles
are never overwritten. Run directories must be owned by the current UID and
mode0700; descriptor files use0600. Fabric handles are capabilities, not a
multi-tenant isolation mechanism. The export uses disabled PID validation for
same-job peers; do not expose the run directory to untrusted workers.

## Close, failure and restart contract

Stop new inference and graph replay before closing. Close input objects, drop
table parameter/views, then close mappings. The adapter rejects close while
tensor aliases remain; a failed close is retryable. Tensor wrappers do not own
the physical allocation themselves. Direct use of cached raw addresses after
close is unsupported.

Normal worker shutdown closes registered models and retries failed allocation
rollbacks. Normal interpreter exit has a synchronized native cleanup fallback.
Owner close removes only descriptors it created; imported consumers keep their
own physical references. Partial cleanup preserves state/token for retry—even
when a freed VA is reused. Combined construction/release failures move to the
pending cleanup list rather than losing ownership.

SIGKILL/crashes and forced multi-process termination can bypass Python cleanup:
device resources then rely on driver process cleanup, and stale capability
files may remain. This was observed during the service restart test. Restart
with a NEW run ID. Never delete/reuse a run directory while any old worker is
alive. Cleanup errors are reported, never treated as successful release.

After every old worker has exited, run this separately on each node:

```bash
python3 cleanup_run.py --run the-stopped-run-id
```

This tool rejects active VLLM/library users, unfamiliar filenames, symlinks,
or unexpected permissions. It only unlinks the eight known table descriptor
files and removes the empty run directory; it never recursively deletes data.
It is an offline, dedicated-container operation: prevent concurrent restarts
and ensure no other PID namespace/container can access the same shm directory.
It is not a distributed lock or a multi-tenant garbage collector. Automatic
stale-run deletion during startup is deliberately not implemented.

## Reproduce checks

```bash
export PYTHONPATH="$PWD:/vllm-workspace/vllm-ascend"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 tests/test_lifecycle.py
python3 tests/test_exit.py
torchrun --standalone --nproc_per_node=2 tests/test_lookup.py \
  --full --root /dev/shm/fresh-engram-test
```

Release-failure injection is TEST ONLY, never preload into serving:

```bash
g++ -shared -fPIC -std=c++17 -pthread tests/release_faults.cpp \
  -I/usr/local/Ascend/ascend-toolkit/latest/include -ldl -o tests/release_faults.so
LD_PRELOAD="$PWD/tests/release_faults.so" ENGRAM_TEST_RELEASE_FAILURES=1 \
  python3 tests/test_lifecycle.py
```

HTTP stability gate (six-prompt baseline JSON from the acceptance benchmark):

```bash
python3 tests/stability_http.py --baseline /path/to/hbm-before.json \
  --output stability.json
```

It checks serial exact messages/token IDs, variable-length serial requests,
short concurrent requests, then20seconds idle and exact checks again. It is not
another throughput sweep. Known concurrent long-input `_pool_kernel` failures
in the original model remain outside this patch's claim.

## Evidence

Earlier performance/capacity result: [acceptance](../e2e/ACCEPTANCE_20260916.md).
Approximately14GiB less HBM/NPU and comparable throughput on the tested load;
not a universal latency or hardware-ceiling claim. Hardening acceptance is
recorded in [VALIDATION.md](VALIDATION.md), with raw logs in `results/`.
Remote tested bundle:
`/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/engram-delivery-20260916`.
