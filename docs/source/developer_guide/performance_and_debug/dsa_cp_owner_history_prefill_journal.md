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
