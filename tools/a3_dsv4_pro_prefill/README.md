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

Before measuring, verify `health_http=200` and that the launch log contains
`"enable_fused_mc2": 1`. Do not enable an OTLP endpoint unless intentionally
needed: `OTLP_TRACES_ENDPOINT` defaults to empty in this versioned launcher.
