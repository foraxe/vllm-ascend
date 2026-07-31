# A3 G27 sparse-owner VMM probe

This standalone two-rank probe validates allocator and transport semantics
without loading vLLM or constructing a Torch tensor.

For four logical pages and two ranks, page ownership is:

```text
owner_rank = logical_page % 2
owner_page = logical_page // 2
```

Each rank reserves four pages of virtual address space, allocates physical HBM
only for its two owner pages, and maps imported peer handles into the two
holes. The full-granule deterministic checks cover local VMM map/read/write,
peer reads, remote writes observed through the owner's local mapping, and
ordered teardown. Imported mappings are released and acknowledged before an
owner frees its physical handles.

Build and run inside an idle A3 pod with CANN 9. These are in-pod commands:

```bash
ASCEND_ROOT=/usr/local/Ascend/cann-9.0.0/aarch64-linux

g++ -std=c++17 -O2 -Wall -Wextra \
  tools/a3_dsv4_pro_prefill/vmm_sparse_owner_probe.cpp \
  -I"${ASCEND_ROOT}/include" -L"${ASCEND_ROOT}/lib64" \
  -Wl,-rpath,"${ASCEND_ROOT}/lib64" -lascendcl \
  -o /tmp/vmm_sparse_owner_probe

timeout 70s python3 \
  tools/a3_dsv4_pro_prefill/run_vmm_sparse_owner_probe.py \
  --binary /tmp/vmm_sparse_owner_probe \
  --devices 0,1 --timeout 60 --per-rank-budget-mib 64 \
  --output /a3_inference/nyx/dsv4_dsa_cp/runs/204/<run-id>/result.json
```

`PASS` requires both ranks to report exactly two physical owner allocations,
two imported aliases, zero mismatches in all three data checks, a successful
import-release barrier, and exit code zero. `207000` on a peer/V2 capability
API is `BLOCKED_CAPABILITY` only when both ranks report that result and exit
with the capability code. A one-rank capability result is
`FAIL_INCOHERENT_CAPABILITY`. Any payload mismatch, unexpected API failure,
timeout, or incomplete cleanup is not a pass.

This proves that one physical backing page can remain on its canonical owner
while another rank maps it into a logical cache view. It does not prove that
PyTorch, an AscendC kernel, or `npu_sparse_attn_sharedkv` can dereference the
mapped address. That is a separate consumer gate.

## 2026-07-30 `.204` result: PASS

The four-page symmetric run passed on pod `dsv4-dsa-prefill-204-nyx`,
NPU0/NPU1, `npu-smi 25.5.1`, and CANN 9.0.0. Durable artifacts:

```text
/a3_inference/nyx/dsv4_dsa_cp/runs/204/g27_sparse_owner_20260730_121939/
```

Both ranks returned zero. The measured allocation contract was:

```text
allocation granularity:          2 MiB
logical VA per rank:             4 pages = 8 MiB
owned physical HBM per rank:     2 allocations = 4 MiB
imported peer aliases per rank:  2 aliases = 4 MiB
local / peer-read / remote-write mismatches: 0 / 0 / 0
import-release barrier:          PASS on both ranks
post-run NPU0/NPU1 process state: idle
```

Rank 0's ACL free-HBM counter returned exactly to its pre-allocation value.
Rank 1 was 64 KiB below its pre-allocation counter at the final in-process
sample; after process exit, `npu-smi` reported no running process on either
NPU. The capacity claim therefore comes from the exact physical-allocation
count plus alias roundtrip, not from treating the advisory HBM counter as
byte-exact allocator accounting.

The rejected first harness attempt is preserved separately at
`g27_sparse_owner_20260730_121811/`. It is `INVALID`: the relative binary path
was passed to `subprocess.Popen` without a slash, so no NPU process started.
The versioned harness resolves the binary to an absolute path.
