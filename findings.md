# Findings

- Integration base: `84f3da22`.
- The existing `PackedArenaLease` already owns the required teardown order:
  alias close, injected fence, unmap, free physical handle, release VA.
- CANN 9.0 requires `aclrtReserveMemAddress(..., alignment=0, ...)`; the
  adapter must still require and verify the measured 2-MiB granularity.
- `aclrtMemSetAccess` follows `aclrtMapMem`. If access setup fails, the
  adapter must undo the mapping before returning because the lease records a
  mapping only after `map_physical` returns.
- G28 used
  `_construct_storage_from_data_pointer` and
  `_construct_NPU_Tensor_From_Storage_And_Metadata` with a local `npu:<rank>`
  device tag. The adapter must keep both imports and capability checks lazy.
- The `.204` success path passed with one 2-MiB arena: pointer alias exact,
  Torch-NPU fill/clone/compare correct, explicit fence/close complete, all
  backend ownership registries empty, and the experiment process absent
  immediately afterward. Exact artifacts are under
  `/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730_g35_packed_acl_adapter_smoke_84f3da22/`.
- A delayed `npu_smi_after.txt` captured a concurrently launched vLLM service
  on NPU0-3. It is not teardown evidence; the artifact README records the
  timing boundary.
