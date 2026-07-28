# A3 DeepSeek-V4-Pro single-node prefill reproduction

Versioned reproduction surface for the TP16/EP16 A3 single-node DSA-CP
prefill experiment. Owner: Codex session
`019fa36a-956a-7372-aceb-19d91f08f990`.

This directory contains the exact launcher and fixed clients used for the
synthetic 8K/one-output prefill and TTFT A/Bs:

- `start_single_node.sh`: independent TP16/DP1 launcher derived from
  `P0/start.sh`; it does not modify the 4P2D launcher.
- `bench_prefill_only.py`: non-streaming prompt-throughput client.
- `bench_ttft_stream.py`: streaming TTFT client that measures first
  nonempty text delta.

The detailed topology, deployment, validity boundaries, results, and DSA-CP
task map are in
`docs/source/developer_guide/performance_and_debug/dsa_cp_single_node_repro.md`.
The test scripts are path-performance tooling: synthetic 64-routed-expert
results are not DeepSeek-V4-Pro 384-expert production results.

Mooncake KV transfer is disabled by default. The single-node cold-prefill
benchmark does not save or restore external KV, so starting its master would
introduce a separate process and port without exercising the measured path.
Use `ENABLE_MOONCAKE_KV_CONNECTOR=1` only for an explicit KV-transfer test.

`ENABLE_C128_OWNER_SHARD=1` is a separate, prefill-only DSA-CP experiment. It
keeps compressor state production unchanged, stores each C128 page on one TP
owner, and HCCL-stages only the block-table pages required by attention into a
temporary local view. It requires TP > 1, forbids a KV-transfer connector, and
keeps the replicated C128 path as the default. Do not combine it with an
unrelated DSA overlap or Mooncake A/B.

`layer_sharding` is a PD-disaggregated prefill-role option in this vLLM
release. The historical Pro P-side reproduction leaves it enabled by default;
a direct standalone DSV4-Flash service must use
`ENABLE_DSA_LAYER_SHARDING=0`.

## Deploy to an A3 iTask pod

Run these commands from the repository root after resolving the role directory
on the pod. The usual directory is
`/a3_inference/itask/workdir/shared/zhaomingchu/aiworker/codex/pro-debug/P0`;
some environments instead use `/a3_inference/shared/.../P0`.

```bash
DSA_POD=<pod-name>
A3_ROLE_DIR=/a3_inference/itask/workdir/shared/zhaomingchu/aiworker/codex/pro-debug/P0

rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  tools/a3_dsv4_pro_prefill/start_single_node.sh "${DSA_POD}:${A3_ROLE_DIR}/start_single_node.sh"
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  tools/a3_dsv4_pro_prefill/bench_prefill_only.py "${DSA_POD}:${A3_ROLE_DIR}/bench_prefill_only.py"
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  tools/a3_dsv4_pro_prefill/bench_ttft_stream.py "${DSA_POD}:${A3_ROLE_DIR}/bench_ttft_stream.py"
rtk proxy env KUBECONFIG=/Users/nyx/.kube/wulan-htest4.yaml kubectl --context=a3 -n cloudide cp \
  vllm_ascend/attention/context_parallel/dsa_cp.py \
  "${DSA_POD}:/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/attention/context_parallel/dsa_cp.py"
```

## Fixed TTFT experiment

Hypothesis: enabling existing A3 FusedMC2 removes exposed W4A8 MoE
dispatch/return work without changing DSA-CP cache semantics. Baseline and
candidate differ only in `ENABLE_FUSED_MC2`. Use a 64-routed-expert dummy
model solely for path performance.

```bash
cd "${A3_ROLE_DIR}"
RUN_ID=fmc2_8k_ttft \
SYNTHETIC_ROUTED_EXPERTS=64 \
ALLOW_SYNTHETIC_WEIGHTS=1 \
ENABLE_DSA_LAYER_SHARDING=1 \
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0 \
ENABLE_FUSED_MC2=1 \
ENABLE_MTP=0 \
ENABLE_TORCH_PROFILER=0 \
bash ./start_single_node.sh

python3 bench_ttft_stream.py \
  --endpoint http://127.0.0.1:7100/v1/chat/completions \
  --words 8192 --warmup 1 --runs 10 --timeout 600 \
  --output results/fmc2_8k_ttft_stream.json
```

## Real DSV4-Flash standalone B0

This is the valid 8-NPU, TP8/EP8 cold-prefill baseline for the current Flash
proxy experiment. Flash has `o_groups=8`; TP16 produces zero local output
groups and fails in the Ascend `wo_a` loader. It uses real Flash W8A8 weights
and measures one 8K-input / one-output request. It is not a replacement for a
final DSV4-Pro claim.

```bash
cd "${A3_ROLE_DIR}"
RUN_ID=flash_b0_nomooncake_fmc2_8k \
A3_MODEL_PATH=/mnt/deepseek/models/DeepSeek-V4-Flash-w8a8-mtp \
TP_SIZE=8 \
A3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ENABLE_DSA_LAYER_SHARDING=0 \
ENABLE_MOONCAKE_KV_CONNECTOR=0 \
ENABLE_C128_OWNER_SHARD=0 \
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=0 \
ENABLE_FUSED_MC2=1 \
ENABLE_MTP=0 \
ENABLE_TORCH_PROFILER=0 \
bash ./start_single_node.sh

python3 bench_ttft_stream.py \
  --endpoint http://127.0.0.1:7100/v1/chat/completions \
  --words 8192 --warmup 1 --runs 5 --timeout 600 \
  --output results/flash_b0_nomooncake_fmc2_8k_ttft.json
```

This run is valid only after `GET /health` returns `200`, the launcher log
contains `dsa_layer_sharding=0`, `mooncake_kv_connector=0`, and every recorded
request has a nonzero TTFT and a first text token. Keep its JSON next to the
launch log; do not compare it with a synthetic 64-expert row.

The historical Pro role uses `SAFETENSORS_LOAD_STRATEGY=prefetch`. If that
specific NFS prefetch path fails before any shard loads, rerun the identical
configuration once with `SAFETENSORS_LOAD_STRATEGY=lazy`; this is a model-load
gate, not a DSA-CP performance A/B. Record the selected strategy in the
launch log and do not compare load time with TTFT.

Before measuring, verify `health_http=200` and that the launch log contains
`"enable_fused_mc2": 1`. Do not enable an OTLP endpoint unless intentionally
needed: `OTLP_TRACES_ENDPOINT` defaults to empty in this versioned launcher.
