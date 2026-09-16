# Delivery validation

## Completed unit/lifecycle gates

All tests below use the actual A3 CANN9.1.0 runtime, not mocked allocation.
The release-fault shim only intercepts selected API returns in the test
process; no invalid device kernels and no serving-process preload are used.

| Gate | Result | Evidence |
|---|---|---|
| EOF descriptor returns EBADMSG, not success | PASS | [lifecycle](results/lifecycle-final.log) |
| Duplicate owner cannot overwrite descriptor | PASS | same |
| Tensor alias prevents premature unmap | PASS | same |
| Idempotent close, three same-path reuse cycles | PASS | same |
| Python wrapping failure rolls back native allocation | PASS | same |
| Imported mapping survives owner close | PASS | same |
| Unmap/address/physical-free failures report and retry | PASS | same |
| New mapping survives retrying partially freed old mapping | PASS | same |
| Native allocation + rollback failure retained and retried | PASS | same |
| Python wrapping + release failure retained and retried | PASS | same |
| Two full208GiB consumers, CPU-reference outputs, empty/grow/shrink | PASS | [full unit](results/full-unit.log) |
| Full-test close leaves zero native handles and no descriptor files | PASS | same |
| Two normal-exit processes clean descriptors and restart same path | PASS | [exit](results/exit.log) |
| Opt-in absent retains original HBM class | PASS | [import](results/hbm-import.log) |

VA reuse is allowed by the failure test, not forced; the token-keyed registry
is independently reviewed to remove address-reuse ownership ambiguity.
Normal process exit is tested; SIGKILL cannot run the Python cleanup path.
The restart contract for hard termination is fresh run IDs plus driver cleanup,
not automatic stale-descriptor deletion. These gates are not a long-duration
leak/soak or multi-tenant security certification.

Invalid setup attempts excluded: direct VMM->CPU copy in the first test fixture
(fixed by staging through HBM); bare model import before Ascend operator
registration (upstream circular import, corrected to normal import order).

## Full-service gate

Independent delivery package installed in both dedicated E2E pods with guarded
source checksum and a7-line opt-in footer. Both full-model boots passed:

| Gate | Result | Evidence |
|---|---|---|
| First boot:44 requests, variable lengths, C4 short requests,20s idle/resume | PASS | [A](results/stability-a.json) |
| Fresh-process restart: same44-request gate | PASS | [B](results/stability-b.json) |
| Six exact serial prompts before/after idle on BOTH boots | PASS | same |
| Native lifecycle counters during requests | 4 allocations / 0 frees per worker | service-a/b-rank*.log |
| Offline cleanup after all processes exit | 8 stale files/node removed | cleanup-a-rank*.log |
| Cleanup refuses active worker | PASS | [guard](results/cleanup-live-guard.log) |
| Postflight | 32NPUs OK,HTTP200,all4DP drained | service-b-rank*-npu.txt |

Total88requests completed. The exact checks compare full message, finish reason
and token IDs to the original HBM baseline. Concurrent variable responses are
checked for completion/token counts, not asserted bitwise equal across batching.
Long inputs are serial because the baseline model has a known concurrent-long
`_pool_kernel` failure. This is short reliability acceptance, not long-duration
soak or exhaustive model-quality validation.

Observed mean chip HBM:37.10GiB first startup,37.32GiB after the restarted test;
native table mapping counts stayed constant. Different phases/allocator caches
prevent interpreting this as an exact leak measurement. Previous throughput
acceptance remains the performance evidence; no new speedup is claimed here.

Forced simultaneous multi-process SIGTERM did not invoke every worker's Python
cleanup: owner capability files remained. After all workers were confirmed dead,
the offline tool removed only the eight allowlisted files/node; new boot used a
fresh run ID. README documents this boundary, rather than claiming SIGTERM
always cleans descriptors. No checkpoint or unrelated run directory was removed.

Serving remains running on `dsv41-engram-e2e-rank0`/`rank1` in VMM mode with run
ID `delivery-b-20260916`. The HBM implementation is retained behind absent
opt-in / launcher argument `hbm`. Package source, native ABI and evidence are
pinned by the delivered SHA256SUMS manifest.
