#!/usr/bin/env bash
set -euo pipefail
rank=${1:?DP start rank}
local_ip=${2:?local IP}
master_ip=${3:?master IP}
mode=${4:?baseline or direct}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ENGRAM_E2E_KV_ARENA=0
export HCCL_BUFFSIZE=512
export ENGRAM_E2E_BOUNDED=${ENGRAM_E2E_BOUNDED:-1}
export ENGRAM_E2E_BACKEND=${ENGRAM_E2E_BACKEND:-paged}
if [[ "$ENGRAM_E2E_BACKEND" != paged && "$ENGRAM_E2E_BACKEND" != vmm ]]; then
  echo 'Unsupported Engram backend' >&2
  exit 2
fi
if [[ "$ENGRAM_E2E_BACKEND" == vmm ]]; then
  export ENGRAM_E2E_BOUNDED=0
fi
if [[ "$ENGRAM_E2E_BACKEND" == paged && "$mode" != baseline && "$ENGRAM_E2E_BOUNDED" != 1 ]]; then
  echo 'Full-table registration is blocked: full-model runs hit driver allocation failures.' >&2
  exit 2
fi
export HCCL_IF_IP="$local_ip" GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export ENGRAM_E2E_MODE="$mode"
export ENGRAM_E2E_RUN=${ENGRAM_E2E_RUN:-dsv41-direct-v1}
export PYTHONPATH=/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/e2e-code:/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/engram-unit-20260914/delivery/engram-host-backed:/vllm-workspace/vllm-ascend:${PYTHONPATH:-}
headless=()
if [[ "$rank" != 0 ]]; then headless=(--headless); fi
exec vllm serve /home/admin/model-csi/model \
  --host 0.0.0.0 --port 8000 "${headless[@]}" \
  --data-parallel-address "$master_ip" --data-parallel-rpc-port 13399 \
  --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-start-rank "$rank" --tensor-parallel-size 8 \
  --enable-expert-parallel --served-model-name deepseek-v41 \
  --max-model-len 1048576 --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --block-size 128 --no-enable-prefix-caching \
  --kv-cache-memory-bytes 8589934592 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy --quantization ascend \
  --additional-config '{"enable_engram":true,"engram_storage":"int8","enable_cpu_binding":true,"mc2_comm_alg":"hierarchy","ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
