# Progress

- 2026-07-30: Created isolated branch `codex/c128-packed-acl-adapter` at
  `84f3da22`; inspected the arena protocol, CANN headers/docs, and G27/G28
  probes.
- 2026-07-30: Implemented lazy ACL and Torch-NPU adapters, access-map rollback,
  strict 2-MiB validation, and focused mock coverage.
- 2026-07-30: `ruff check` and `compileall` passed. Focused CPU run passed
  `19` tests with `--confcutdir=tests/ut/attention`; the normal repository
  conftest is unavailable on the Mac because `torch_npu` is not installed.
- 2026-07-30: Read-only `.204` capability check passed with PyTorch `2.10.0`,
  torch_npu `2.10.0`, all ten required ACL symbols, and both external-storage
  tensor constructors. No ACL/NPU operation was invoked.
- 2026-07-30: Independent review found one stale cleanup paragraph; corrected
  it. Final verdict: `APPROVE`, no P0/P1/P2 findings.
- 2026-07-30: Bounded real-NPU success path passed on idle NPU0: one 2-MiB
  arena, pointer exact, fill/clone/compare correct, explicit cleanup complete,
  and all backend registries empty. Durable artifacts:
  `/a3_inference/nyx/dsv4_dsa_cp/runs/204/20260730_g35_packed_acl_adapter_smoke_84f3da22/`.
