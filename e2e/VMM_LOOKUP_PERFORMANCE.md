# VMM grouped lookup integration

The VMM consumer now uses groups of 16 rows, with FP32 multiplication and
BF16 output. It reads the contiguous VMM base directly with INT64 row offsets.
The old scalar/chunked implementation remains available with `tile=1`;
legacy registered-chunk consumers still default to that implementation.
Metadata copies, full output-tail clearing, masks and mapping lifetime are
unchanged. No new graph capture or routing changes are included.

This is the direct-consumer path (`VmmInputs`), not a replacement for
`NodeShardedEngram.lookup_local` in the owner-routed path. The older 7x
preparation experiment already used a tiled consumer kernel; its improvement
cannot be multiplied by the new numbers.

## Measurement

Validated paired msprof medians (microseconds, one table):

| Buffer capacity | Active tokens | Scalar | Tile16 | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 128 | 29.890 | 16.470 | 1.81x |
| 4096 | 1 | 16.620 | 15.490 | 1.07x |
| 4096 | 32 | 21.860 | 18.180 | 1.20x |
| 4096 | 128 | 40.110 | 27.069 | 1.48x |

All four cases passed exact BF16/mask comparisons. Raw forty-sample arrays:
[summary.json](results/vmm-tiled/summary.json). The 1-token result is a small
within-run improvement, not an independently replicated speedup claim.
No updated full-model E2E result has been measured for this change.

Full-capacity correctness also PASS: two NPUs import the same 208 GiB rounded
physical allocation for both production-size tables. The existing `test_vmm.py
--full` exercised table ends, owner boundaries, former chunk boundaries and
0/1/17/128/3/0 tokens against independent CPU references. Both ranks reported
`VMM_PROBE_PASS`; allocator counters stayed 4 allocations / 0 frees through
lookup. See [full2.log](results/vmm-tiled/full2.log). This proves large-offset
addressing and cross-consumer correctness, not full-capacity random bandwidth.

`profile_vmm_lookup.py` compares tile1 and tile16 in the same process on the
same VMM backing, alternating order. Seven correctness cases precede five
warmup and forty measured pairs. `summarize_vmm_lookup.py` requires a PASS
marker and exactly 104 kernel launches before extracting those forty pairs.
Reported duration is msprof `Task Duration(us)` for one table's fused lookup,
padding clear and mask kernel, not CPU preparation latency or model E2E.

Table shape: 4,194,321 x 256 INT8 plus eight FP32 scales per row; 3 GiB physical
allocation after huge-page rounding. Sampled rows span the table, include both
sides of the former chunk boundary, and use random non-power-of-two scales.
This is not a full-table bandwidth scan. Each method is checked against an
independent CPU BF16 reference; publication is checked by reading back codes
and scales. Cases include empty/grow/shrink and output/mask padding.

Environment: existing DSV4.1 image, CANN9.1.0, node14037. The model server
remained resident (preflight AICore utilization zero); these are isolated
test-process task durations, not an exclusive-node hardware-ceiling claim.
The live serving code was not replaced or restarted.

Remote raw evidence and tested overlay:
`/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/vmm-tiled-20260916/`.

Reproduce inside the tested overlay with its existing PYTHONPATH and
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`; use a fresh handle root:

```bash
msprof --output=profile-new --application="python3 profile_vmm_lookup.py --root /dev/shm/engram-unique-run --capacity 4096 --tokens 128"
python3 summarize_vmm_lookup.py .
```

Discarded attempts: vector-of-chunk-pointers triggered a Triton compiler
assertion. Test-only `index_copy_` publication failed independent readback;
the valid runs use the already proven `VmmMapping.publish` staging path.
