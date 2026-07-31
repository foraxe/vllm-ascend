# Ascend VMM remote-tensor kernel probe

This is the follow-on gate after the raw `.204` V2 VMM map/roundtrip probe.
It tests whether ordinary torch NPU kernels can consume a remote physical
allocation after the importer maps it into its own virtual address space.
It does not modify vLLM allocation or DSA-CP execution.

## Experiment contract

- Hypothesis: an NPU1 torch tensor built over an NPU0 V2 imported/mapped
  virtual address can read through `clone()` and write through `fill_()`.
- Configuration: two spawned Python processes, exporter NPU0, importer NPU1,
  4,096 contiguous FP32 values, default same-server V2 share-handle type, PID
  whitelist enabled.
- Baseline: the preceding raw G27 V2 map/bidirectional roundtrip must be
  `PASS` on the same pod and NPU0/1 must be idle before this probe starts.
- Metric: exact equality of all 4,096 values after exporter `copy_`, importer
  `clone`, importer `fill_`, and exporter post-write `clone`.
- `PASS`: both ordinary-kernel reads and writes are exact; importer tensors
  and storage are dropped and its VA is unmapped before exporter physical
  memory is freed.
- `FAIL`: an ordinary NPU kernel completes but any value differs.
- `BLOCKED`: bootstrap, V2 API, constructor, device-kernel, synchronization,
  or timeout failure. The JSON records the exact failing boundary.
- Kill criterion: 60 seconds per coordination stage or 180 seconds total.
  The parent terminates, then kills, importer before exporter.

## ABI used

The C wrapper compiles against the pod's system
`/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h` and links
`libascendcl.so`. The same-server share lifecycle is:

```text
aclrtMemGetAllocationGranularity(prop, option, granularity)
aclrtReserveMemAddress(va, size, alignment, expect, flags)
aclrtMallocPhysical(handle, size, prop, flags)
aclrtMapMem(va, size, offset, handle, flags)
aclrtMemExportToShareableHandleV2(handle, flags, shareType, shareHandle)
aclrtMemSetPidToShareableHandleV2(shareHandle, shareType, pid, pidNum)
aclrtMemImportFromShareableHandleV2(shareHandle, shareType, flags, handle)
aclrtUnmapMem(va)
aclrtFreePhysical(handle)
aclrtReleaseMemAddress(va)
```

`aclrtMemFabricHandle` is the 128-byte V2 handle container. This probe uses
`ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT`; Fabric is a separate cross-server gate.
The importer sends `aclrtDeviceGetBareTgid`, not its namespace-visible
`os.getpid()`, for the V2 whitelist.

The mapped pointer is wrapped through the installed private torch_npu APIs:

```python
storage = torch_npu._C._construct_storage_from_data_pointer(
    data_ptr, torch.device("npu:<rank>"), mapped_bytes
)
tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
    {
        "data_ptr": data_ptr,
        "device": torch.device("npu:<rank>"),
        "nbytes": mapped_bytes,
        "dtype": torch.float32,
        "size": (4096,),
        "stride": (1,),
        "storage_offset": 0,
    },
    storage,
)
```

This is a non-owning tensor. Every tensor/storage alias is discarded and the
current NPU stream is synchronized before unmap. C++ owns every ACL handle and
VA.

## Reproduce on `.204`

From the repository worktree on the Mac:

```bash
POD=dsv4-dsa-prefill-204-nyx
KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml
REMOTE_SOURCE=/tmp/vmm_remote_tensor_probe
ARTIFACT_DIR=/a3_inference/nyx/dsv4_dsa_cp/runs/204/<run-id>
PROBE_GIT_SHA="$(git rev-parse HEAD)"

rtk proxy env KUBECONFIG="${KUBECONFIG}" kubectl --context a3 -n default \
  cp tools/a3_dsv4_pro_prefill/vmm_remote_tensor_probe \
  "${POD}:${REMOTE_SOURCE}"

rtk proxy env KUBECONFIG="${KUBECONFIG}" kubectl --context a3 -n default \
  exec "${POD}" -- bash -lc \
  "PROBE_GIT_SHA=${PROBE_GIT_SHA} \
   bash ${REMOTE_SOURCE}/run_probe.sh ${ARTIFACT_DIR}"
```

Preserve `build.log`, `symbols.log`, `run.log`, `result.json`, `exit_code`,
the copied source, and a before/after `npu-smi info` snapshot together in the
artifact directory. Do not run this probe concurrently with a vLLM service or
another NPU0/1 VMM probe.

## `.204` result

Canonical run `g28_remote_tensor_20260730_043455` is `PASS` from signed commit
`11b275512aeba5e855668176f2c22b5d35c20342`:

```text
mapped size:             2,097,152 bytes
requested tensor bytes:     16,384 bytes
exporter initial values: 11.0 ... 4106.0, checksum 8,431,616
importer remote clone:    exact equality across all 4,096 values
importer remote fill_:    37.0 across all 4,096 values
exporter final clone:     37.0 ... 37.0, checksum 151,552
child exit codes:         importer=0, exporter=0
forced cleanup:           terminated=[], killed=[]
wrapper exit code:        0
```

The importer emitted `importer_unmapped` before the exporter emitted
`exporter_freed`. Before and after snapshots show no process on NPU0 or NPU1.
The durable artifact is:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g28_remote_tensor_20260730_043455/
```

Two preceding attempts are retained because they test the harness itself:

- `g28_remote_tensor_20260730_042433` is `INVALID`: V2 export completed, but
  parent logging tried to JSON-serialize the raw 128-byte handle before
  forwarding it to the importer. Both children were terminated; no kernel
  result was produced.
- `g28_remote_tensor_20260730_042932` contains a functional `PASS` with exact
  values and orderly cleanup, but its shell wrapper recorded exit code `2`
  despite `result.json` being `PASS`. It is not the canonical reproduction.
  Commit `11b27551` replaced pipeline status extraction and the canonical run
  confirms wrapper exit code `0`.

## G46 production-size wide-bucket gate on `.32`

Hypothesis: one exact `1777 * 2 MiB = 3,726,639,104` byte
(`3.470703125 GiB`) physical allocation on NPU0 can be V2-exported,
PID-authorized, imported and mapped on NPU1, then bound as non-owning BF16
Torch-NPU storage. Ordinary kernels must read and update bounded canaries at
the first, middle and last offsets without a collective or a physical HBM
allocation on the importer.

The production mode binds a logical BF16 tensor over the complete mapping but
does not construct a full-size source, clone, fill, or CPU reference. It
touches only three 256-element (512-byte) views:

```text
requested bytes:  3,726,639,104
mapped pages:      1,777 x 2 MiB
canary offsets:    0
                   1,863,319,552
                   3,726,638,592
distinct bytes per validation/write pass: 1,536
```

Baseline: the existing 4,096-element remote-tensor gate must pass first on the
same pod. Production `PASS` requires:

- exact requested and mapped byte count;
- one exporter-side `aclrtMallocPhysical` and no importer-side physical
  allocation, proven by process-local bridge call counters;
- importer HBM free-byte delta recorded across import/map/bind as an advisory
  allocator metric;
- exact BF16 canary values before and after importer `add_(3)`;
- exporter sees the three remote updates;
- importer unmaps before exporter frees; both child exit codes are zero;
- wrapper exit code zero and NPU0/1 process-free before and after.

Any completed value or size mismatch is `FAIL`. An API, allocation, kernel,
bootstrap, or timeout failure is `BLOCKED`. Use 120 seconds per coordination
stage and a 300-second outer timeout; terminate then kill importer before
exporter on timeout.

Run the small regression first, then the production gate:

```bash
POD=dsv4-dsa-prefill-032-nyx
KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml
REMOTE_SOURCE=/tmp/g46_vmm_full_wide_bucket
ARTIFACT_ROOT=/a3_inference/nyx/dsv4_dsa_cp/runs/032/20260731_g46_vmm_full_wide_bucket

kubectl --kubeconfig "${KUBECONFIG}" --context a3 -n default cp \
  tools/a3_dsv4_pro_prefill/vmm_remote_tensor_probe \
  "${POD}:${REMOTE_SOURCE}"

kubectl --kubeconfig "${KUBECONFIG}" --context a3 -n default exec "${POD}" \
  -- bash -c \
  "bash ${REMOTE_SOURCE}/run_probe.sh ${ARTIFACT_ROOT}/small_regression"

kubectl --kubeconfig "${KUBECONFIG}" --context a3 -n default exec "${POD}" \
  -- bash -c \
  "STAGE_TIMEOUT_SECONDS=120 TOTAL_TIMEOUT_SECONDS=300 \
   bash ${REMOTE_SOURCE}/run_probe.sh ${ARTIFACT_ROOT}/production \
     --dtype bfloat16 --mapped-bytes 3726639104 \
     --canary-elements 256"
```

G46 passed on 2026-07-31 in `dsv4-dsa-prefill-032-nyx` with CANN/HDK
`25.5.1`. The durable evidence root is:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/032/20260731_g46_vmm_full_wide_bucket/
```

Both `small_regression/result.json` and `production/result.json` report
`PASS`, wrapper exit code zero, child exit codes zero, and no terminated or
killed process. Production recorded the exact requested, mapped, and
exporter HBM delta as `3,726,639,104` bytes. Its bridge counters were
`physical_allocation_calls=1, import_calls=0` on the exporter and
`physical_allocation_calls=0, import_calls=1` on the importer. The three
BF16 canaries at byte offsets `0`, `1,863,319,552`, and `3,726,638,592`
matched before the importer update and after `add_(3)` on both processes.
The advisory importer free-HBM delta was `-110,592` bytes. All eight logical
NPUs were process-free before and after the runs.
