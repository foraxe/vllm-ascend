#!/usr/bin/env bash
set -euo pipefail

# Isolated single-node launcher derived from P0/start.sh.  It intentionally
# leaves the original 4P2D role script untouched.
ROLE_NAME="single_node_prefill"
DP_RANK="0"

ROLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${ROLE_DIR}/.." && pwd)
NETWORK_INTERFACE=${NETWORK_INTERFACE:-eth0}
A3_MODEL_PATH=${A3_MODEL_PATH:-/a3_inference/itask/workdir/models/DeepSeek-V4-Pro-w4a8-mtp}
# Optional observability endpoint. Leave unset for a self-contained benchmark.
OTLP_TRACES_ENDPOINT=${OTLP_TRACES_ENDPOINT:-}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${ROLE_DIR}/log_single_node_prefill_${RUN_ID}.log
PID_FILE=${ROLE_DIR}/.vllm_pids_single_node
# Non-CP DSA prefill overlap gate. AscendDSACPImpl does not read this switch
# in this release; use ENABLE_MULTISTREAM_DSA_PREPROCESS for DSA-CP.
ENABLE_PREFILL_COMM_COMPUTE_OVERLAP=${ENABLE_PREFILL_COMM_COMPUTE_OVERLAP:-0}
# DSA-CP reads this switch for its hidden-state all-gather / local-Q overlap.
# Keep it separate from the non-CP prefill_comm_compute_overlap control.
ENABLE_MULTISTREAM_DSA_PREPROCESS=${ENABLE_MULTISTREAM_DSA_PREPROCESS:-0}
# Run the shared expert on a separate stream while routed FusedMC2 executes.
ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT=${ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT:-0}
# C128 only: retain canonical compressed pages on one TP owner and stage the
# pages needed by prefill attention through HCCL.  This is independent of
# Mooncake and stays off until its replicated fallback has been compared.
ENABLE_C128_OWNER_SHARD=${ENABLE_C128_OWNER_SHARD:-0}
# Run owner placement with the historical compact C128 allocation by default.
# Set to 0 only for the causal layout-ABI gate; that keeps a full persistent
# tensor while exercising the same owner write/materialization code.
ENABLE_C128_OWNER_COMPACT_ALLOCATION=${ENABLE_C128_OWNER_COMPACT_ALLOCATION:-1}
# Emits per-layer direct DSA custom-op boundary markers. Keep this off for
# performance experiments; it exists only to localize feature-on failures.
ENABLE_C128_OWNER_DEBUG=${ENABLE_C128_OWNER_DEBUG:-0}
# Exchange only pages requested by each rank's local C128 block table. This is
# experimental; 0 retains the established full-union staging fallback.
ENABLE_C128_OWNER_SELECTIVE_STAGE=${ENABLE_C128_OWNER_SELECTIVE_STAGE:-0}
# Compute C128 compressor rows from a CP-local, 128-aligned shard. The
# gathered producer remains the fallback for a non-aligned tail or multi-request batch.
ENABLE_C128_OWNER_LOCAL_COMPRESSOR=${ENABLE_C128_OWNER_LOCAL_COMPRESSOR:-0}
# Compute current SWA KV and C128 compressor rows from each rank's local,
# C128-aligned prefill shard, then exchange the narrower results into the
# existing replicated caches. This does not require C128 owner sharding.
ENABLE_DSA_CP_LOCAL_CURRENT_KV=${ENABLE_DSA_CP_LOCAL_CURRENT_KV:-0}
# One-shot physical backing-storage accounting after KV cache allocation.
# This diagnostic stays out of every forward path and is disabled by default.
ENABLE_KV_CACHE_ALLOCATION_ACCOUNTING=${ENABLE_KV_CACHE_ALLOCATION_ACCOUNTING:-0}
# `layer_sharding` is accepted only by a PD-disaggregated prefill (P) role in
# this vLLM release. Keep the historical P-side default, but set this to 0 for
# a direct standalone service such as the DSV4-Flash single-node baseline.
ENABLE_DSA_LAYER_SHARDING=${ENABLE_DSA_LAYER_SHARDING:-1}
# A3 fused-MC2 prefill experiment.  At 8K, mode 0 selects the three-stage
# alltoallv MoE path; mode 1 selects the existing W4A8 dispatch_ffn_combine
# implementation.  Keep it independent from the DSA overlap A/B.
ENABLE_FUSED_MC2=${ENABLE_FUSED_MC2:-0}
# Development-only trace gate.  Profiling is armed at launch and starts only
# after POST /start_profile, so model load and ordinary warmup stay out of a
# fixed-request trace.
ENABLE_TORCH_PROFILER=${ENABLE_TORCH_PROFILER:-0}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${ROLE_DIR}/profiling/${RUN_ID}}
# MTP is a decode-time draft model.  Keep it out of the single-node,
# prefill-only experiment so it does not consume one extra MoE layer of HBM.
ENABLE_MTP=${ENABLE_MTP:-0}
# A single-node cold-prefill benchmark neither saves nor restores external KV.
# Keep Mooncake out of the default process tree so its master, ports, and
# connector initialization cannot affect TTFT.  Set this only when explicitly
# exercising KV-transfer behavior.
ENABLE_MOONCAKE_KV_CONNECTOR=${ENABLE_MOONCAKE_KV_CONNECTOR:-0}
# The historical Pro role prefetches safetensors on NFS. Keep that default,
# but make the policy explicit so a standalone model-load failure can be
# isolated without changing any DSA-CP or serving setting.
SAFETENSORS_LOAD_STRATEGY=${SAFETENSORS_LOAD_STRATEGY:-prefetch}
# Keep B0's 0.9 by default. A lower value is a correctness-only allocator
# capacity gate and must never be compared as a TTFT candidate.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
# These two controls form an explicit reduced-capacity correctness gate. Keep
# both unset for B0/candidate TTFT measurements.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-5120}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-}
# Explicit synthetic-model gate for capacity and DSA-CP path experiments.
# A reduced routed-expert count changes gate/hash tensor shapes, so it must
# never be paired with the production checkpoint weights.
SYNTHETIC_ROUTED_EXPERTS=${SYNTHETIC_ROUTED_EXPERTS:-0}
ALLOW_SYNTHETIC_WEIGHTS=${ALLOW_SYNTHETIC_WEIGHTS:-0}

for boolean_name in ENABLE_PREFILL_COMM_COMPUTE_OVERLAP ENABLE_MULTISTREAM_DSA_PREPROCESS ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT ENABLE_C128_OWNER_SHARD ENABLE_C128_OWNER_COMPACT_ALLOCATION ENABLE_C128_OWNER_DEBUG ENABLE_C128_OWNER_SELECTIVE_STAGE ENABLE_C128_OWNER_LOCAL_COMPRESSOR ENABLE_DSA_CP_LOCAL_CURRENT_KV ENABLE_KV_CACHE_ALLOCATION_ACCOUNTING \
    ENABLE_DSA_LAYER_SHARDING ENABLE_FUSED_MC2 ENABLE_MTP \
    ENABLE_TORCH_PROFILER ENABLE_MOONCAKE_KV_CONNECTOR; do
    boolean_value=${!boolean_name}
    [[ "${boolean_value}" == 0 || "${boolean_value}" == 1 ]] || {
        echo "${boolean_name} must be 0 or 1, got ${boolean_value}" >&2
        exit 2
    }
done
case "${SAFETENSORS_LOAD_STRATEGY}" in
    lazy|eager|prefetch) ;;
    *)
        echo "SAFETENSORS_LOAD_STRATEGY must be lazy, eager, or prefetch, got ${SAFETENSORS_LOAD_STRATEGY}" >&2
        exit 2
        ;;
esac
python3 - "${GPU_MEMORY_UTILIZATION}" <<'PY'
import sys

value = float(sys.argv[1])
if not 0 < value <= 1:
    raise SystemExit(f"GPU_MEMORY_UTILIZATION must be in (0, 1], got {value}")
PY
[[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_MODEL_LEN must be a positive integer, got ${MAX_MODEL_LEN}" >&2
    exit 2
}
[[ "${MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_NUM_BATCHED_TOKENS must be a positive integer, got ${MAX_NUM_BATCHED_TOKENS}" >&2
    exit 2
}
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    [[ "${NUM_GPU_BLOCKS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || {
        echo "NUM_GPU_BLOCKS_OVERRIDE must be a positive integer when set" >&2
        exit 2
    }
fi

resolve_local_ip() {
    python3 - "${NETWORK_INTERFACE}" <<'PY'
import fcntl
import socket
import struct
import sys

interface = sys.argv[1].encode("ascii")[:15]
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    request = struct.pack("256s", interface)
    address = fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24]
print(socket.inet_ntoa(address))
PY
}

ensure_not_running() {
    local pid
    [[ -f "${PID_FILE}" ]] || return 0
    while read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        if kill -0 "${pid}" 2>/dev/null; then
            echo "${ROLE_NAME} is already running: pid=${pid}" >&2
            exit 1
        fi
    done < "${PID_FILE}"
}

LOCAL_IP=$(resolve_local_ip)
[[ -n "${LOCAL_IP}" ]] || {
    echo "Unable to resolve IPv4 address on ${NETWORK_INTERFACE}" >&2
    exit 1
}
# Prefill process placement. Keep the Pro P-side default, while allowing a
# checkpoint whose attention output groups require a smaller TP width.
VISIBLE_DEVICES=${A3_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
VLLM_PORT=7100
DP_SIZE=1
DP_ADDRESS="${LOCAL_IP}"
DP_RPC_PORT=14435
TP_SIZE=${TP_SIZE:-16}

[[ "${TP_SIZE}" =~ ^[0-9]+$ ]] && (( TP_SIZE > 0 )) || {
    echo "TP_SIZE must be a positive integer, got ${TP_SIZE}" >&2
    exit 2
}
IFS=',' read -r -a visible_device_array <<<"${VISIBLE_DEVICES}"
(( ${#visible_device_array[@]} == TP_SIZE )) || {
    echo "A3_VISIBLE_DEVICES has ${#visible_device_array[@]} entries but TP_SIZE=${TP_SIZE}" >&2
    exit 2
}

validate_model_parallelism() {
    python3 - "${A3_MODEL_PATH}" "${TP_SIZE}" <<'PY'
import json
import pathlib
import sys

model_path = pathlib.Path(sys.argv[1])
tp_size = int(sys.argv[2])
config_path = model_path / "config.json"
if not config_path.is_file():
    raise SystemExit(0)
with config_path.open() as config_file:
    config = json.load(config_file)
o_groups = config.get("o_groups")
if o_groups is not None and (o_groups < tp_size or o_groups % tp_size):
    raise SystemExit(
        f"checkpoint o_groups={o_groups} requires a positive integral local group count; "
        f"TP_SIZE={tp_size} is invalid"
    )
PY
}
validate_model_parallelism

# Environment copied from the prefill role in deepseek-pro-kvpool.yaml.
export MODEL_PATH="${A3_MODEL_PATH}"
export STARAGENT_DISABLED=true
export ALIYUN_LOG_ENV_TAGS='MODEL_INSTANCE_NAME|MODEL_SERVICE_NAME|MODEL_NAME'
export VLLM_USE_V1=1
export VLLM_VERSION=0.20.2
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_IMAGE_FETCH_TIMEOUT=30
export VLLM_VIDEO_FETCH_TIMEOUT=60
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=6000
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export PROMETHEUS_MULTIPROC_DIR=/tmp/
export ASCEND_PROCESS_LOG_PATH='./logs/ascend/'
export VLLM_LOGGING_CONFIG_PATH=/home/admin/vllm/production_logging_config.json
export PORT=8088
export ASCEND_BASE_PORT=22000
export VLLM_ASCEND_ENABLE_OMNIINFER_SAMPLER=0
export API_SERVER_COUNT=1
export LOCAL_DP_SIZE=1
export NET_CARD_NAME="${NETWORK_INTERFACE}"
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
export TP_SOCKET_IFNAME="${NETWORK_INTERFACE}"
export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_BASE_PORT=9100
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export MC_LOG_LEVEL=INFO
export MC_LOG_DIR="${ROLE_DIR}/logs/mooncake"
export MOONCAKE_CONFIG_PATH="${ROLE_DIR}/mooncake_single_node.json"
export MOONCAKE_MASTER="${LOCAL_IP}:50051"
export ASCEND_CONNECT_TIMEOUT=100000
export ASCEND_TRANSFER_TIMEOUT=100000
export PYTHONHASHSEED=0
export GLOG_alsologtostderr=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=1024
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_HOST_LOG_FILE_NUM=1000
export PYTHONPATH="/usr/local/python3.11.15/lib/python3.11/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export POD_IP="${LOCAL_IP}"

# This is a cold-prefill experiment: do not let an inherited KV-pool or prefix
# cache configuration alter the request path.
unset VLLM_ASCEND_PER_REQUEST_SAVE_WAIT MOONCAKE_BATCH_CAPTURE HCCL_EVENT_TIMEOUT
unset VLLM_PREFIX_CACHE_RETENTION_INTERVAL
unset VLLM_ASCEND_KVPOOL_RETENTION_LOOKUP_FACTOR VLLM_ASCEND_ASYNC_SAVE
unset VLLM_ASCEND_KVTRACE_LOG

MODEL_LOADER_CONFIG='{"enable_multithread_load":true,"num_threads":8}'
SPECULATIVE_CONFIG='{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}'
ADDITIONAL_CONFIG=$(jq -cn \
    --argjson prefill_overlap "${ENABLE_PREFILL_COMM_COMPUTE_OVERLAP}" \
    --argjson multistream_dsa_preprocess "${ENABLE_MULTISTREAM_DSA_PREPROCESS}" \
    --argjson multistream_overlap_shared_expert "${ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT}" \
    --argjson c128_owner_shard "${ENABLE_C128_OWNER_SHARD}" \
    --argjson c128_owner_compact_allocation "${ENABLE_C128_OWNER_COMPACT_ALLOCATION}" \
    --argjson c128_owner_debug "${ENABLE_C128_OWNER_DEBUG}" \
    --argjson c128_owner_selective_stage "${ENABLE_C128_OWNER_SELECTIVE_STAGE}" \
    --argjson c128_owner_local_compressor "${ENABLE_C128_OWNER_LOCAL_COMPRESSOR}" \
    --argjson dsa_cp_local_current_kv "${ENABLE_DSA_CP_LOCAL_CURRENT_KV}" \
    --argjson kv_cache_allocation_accounting "${ENABLE_KV_CACHE_ALLOCATION_ACCOUNTING}" \
    --argjson dsa_layer_sharding "${ENABLE_DSA_LAYER_SHARDING}" \
    --argjson fused_mc2 "${ENABLE_FUSED_MC2}" \
    '({
      enable_cpu_binding:true,
      enable_dsa_cp:true,
      enable_shared_expert_dp:true,
      prefill_comm_compute_overlap:$prefill_overlap,
      multistream_dsa_preprocess:$multistream_dsa_preprocess,
      multistream_overlap_shared_expert:$multistream_overlap_shared_expert,
      enable_c128_owner_shard:$c128_owner_shard,
      enable_c128_owner_compact_allocation:$c128_owner_compact_allocation,
      enable_c128_owner_debug:$c128_owner_debug,
      enable_c128_owner_selective_stage:$c128_owner_selective_stage,
      enable_c128_owner_local_compressor:$c128_owner_local_compressor,
      enable_dsa_cp_local_current_kv:$dsa_cp_local_current_kv,
      enable_kv_cache_allocation_accounting:$kv_cache_allocation_accounting,
      enable_fused_mc2:$fused_mc2
    } + if $dsa_layer_sharding == 1 then {layer_sharding:["q_b_proj", "o_proj"]} else {} end)')

if [[ "${ENABLE_MOONCAKE_KV_CONNECTOR}" == 1 ]]; then
    KV_TRANSFER_CONFIG=$(jq -cn --arg engine_id "${LOCAL_IP}" '
      {
        kv_connector:"MooncakeHybridConnector",
        kv_role:"kv_producer",
        engine_id:$engine_id,
        kv_port:"30100",
        kv_connector_extra_config:{
          prefill:{dp_size:1,tp_size:16},
          decode:{dp_size:1,tp_size:16}
        }
      }')
fi

if [[ "${ENABLE_TORCH_PROFILER}" == 1 ]]; then
    PROFILER_CONFIG=$(jq -cn --arg trace_dir "${TORCH_PROFILER_DIR}" '
      {
        profiler:"torch",
        torch_profiler_dir:$trace_dir,
        torch_profiler_with_stack:false,
        torch_profiler_with_memory:false,
        ignore_frontend:true,
        max_iterations:2
      }')
fi

if [[ "${SYNTHETIC_ROUTED_EXPERTS}" != 0 ]]; then
    [[ "${ALLOW_SYNTHETIC_WEIGHTS}" == 1 ]] || {
        echo "SYNTHETIC_ROUTED_EXPERTS requires ALLOW_SYNTHETIC_WEIGHTS=1; production weights are incompatible" >&2
        exit 2
    }
    [[ "${SYNTHETIC_ROUTED_EXPERTS}" =~ ^[0-9]+$ ]] \
        && (( SYNTHETIC_ROUTED_EXPERTS >= 6 )) \
        && (( SYNTHETIC_ROUTED_EXPERTS % TP_SIZE == 0 )) || {
        echo "SYNTHETIC_ROUTED_EXPERTS must be an integer >= 6 and divisible by TP_SIZE=${TP_SIZE}" >&2
        exit 2
    }
    # Disable hash-router layers because their tid2eid checkpoint tensor is
    # sized for 384 experts. Dummy loading makes this a path-performance test,
    # not an accuracy result.
    SYNTHETIC_HF_OVERRIDES=$(jq -cn --argjson n "${SYNTHETIC_ROUTED_EXPERTS}" \
        '{n_routed_experts:$n,num_hash_layers:0}')
    # DummyModelLoader rejects all model-loader extra config. Keep the
    # production safetensor loader configuration untouched outside this mode.
fi

ensure_mooncake_master() {
    local master_bin=/usr/local/python3.11.15/lib/python3.11/site-packages/mooncake/mooncake_master
    if pgrep -f '[m]ooncake_master.*--port 50051' >/dev/null 2>&1; then
        return
    fi
    [[ -x "${master_bin}" ]] || { echo "Missing ${master_bin}" >&2; exit 1; }
    jq -n --arg master "${MOONCAKE_MASTER}" \
        '{use_ascend_direct:true,metadata_server:"P2PHANDSHAKE",protocol:"ascend",device_name:"eth0",global_segment_size:"30GB",local_buffer_size:"6GB",master_server_address:$master}' \
        > "${MOONCAKE_CONFIG_PATH}"
    nohup env -u GLOG_alsologtostderr "${master_bin}" --max_threads 32 --metrics_port 9003 --port 50051 -v=1 \
        > "${ROLE_DIR}/log_mooncake_single_node.log" 2>&1 &
    sleep 1
    pgrep -f '[m]ooncake_master.*--port 50051' >/dev/null || {
        echo "Mooncake master failed to start" >&2
        exit 1
    }
}

# Complete command for this role. There is no Python launcher or secondary
# run_dp_template.sh between this array and the vLLM process.
VLLM_CMD=(
    vllm serve "${A3_MODEL_PATH}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --trust-remote-code
    --served-model-name auto
    --distributed-executor-backend mp
    --enable-log-requests
    --enable-prompt-tokens-details
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --block-size 128
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-disable-hybrid-kv-cache-manager
    --no-enable-prefix-caching
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"
    # Omit all explicit DP rendezvous flags: defaults are DP=1 without the
    # external load-balancer mode inherited by the original 4P2D launcher.
    --tensor-parallel-size "${TP_SIZE}"
    --seed 1024
    --enforce-eager
    --quantization ascend
    --enable-expert-parallel
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v4
    --tokenizer-mode deepseek_v4
    --reasoning-parser deepseek_v4
    --additional-config "${ADDITIONAL_CONFIG}"
)
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    VLLM_CMD+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi
if [[ "${ENABLE_MOONCAKE_KV_CONNECTOR}" == 1 ]]; then
    VLLM_CMD+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
fi
if [[ "${SYNTHETIC_ROUTED_EXPERTS}" == 0 ]]; then
    VLLM_CMD+=(--model-loader-extra-config "${MODEL_LOADER_CONFIG}")
else
    VLLM_CMD+=(--load-format dummy --hf-overrides "${SYNTHETIC_HF_OVERRIDES}")
fi
if [[ "${ENABLE_MTP}" == 1 ]]; then
    VLLM_CMD+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi
if [[ "${ENABLE_TORCH_PROFILER}" == 1 ]]; then
    VLLM_CMD+=(--profiler-config "${PROFILER_CONFIG}")
fi
if [[ -n "${OTLP_TRACES_ENDPOINT}" ]]; then
    VLLM_CMD+=(--otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}")
fi

ENV_KEYS=(
    MODEL_PATH ASCEND_PROCESS_LOG_PATH VLLM_USE_V1 VLLM_VERSION VLLM_RPC_TIMEOUT
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS HCCL_EXEC_TIMEOUT HCCL_CONNECT_TIMEOUT
    PORT ASCEND_BASE_PORT API_SERVER_COUNT LOCAL_DP_SIZE NET_CARD_NAME HCCL_IF_IP
    VLLM_BASE_PORT MOONCAKE_CONFIG_PATH MOONCAKE_MASTER
    ASCEND_RT_VISIBLE_DEVICES POD_IP
)

print_effective_config() {
    local key
    printf 'role=%s local_ip=%s prefill_comm_compute_overlap=%s multistream_dsa_preprocess=%s multistream_overlap_shared_expert=%s c128_owner_shard=%s c128_owner_compact_allocation=%s c128_owner_debug=%s c128_owner_selective_stage=%s c128_owner_local_compressor=%s dsa_cp_local_current_kv=%s kv_cache_allocation_accounting=%s dsa_layer_sharding=%s enable_fused_mc2=%s enable_mtp=%s mooncake_kv_connector=%s synthetic_routed_experts=%s torch_profiler=%s\n' \
        "${ROLE_NAME}" "${LOCAL_IP}" "${ENABLE_PREFILL_COMM_COMPUTE_OVERLAP}" "${ENABLE_MULTISTREAM_DSA_PREPROCESS}" "${ENABLE_MULTISTREAM_OVERLAP_SHARED_EXPERT}" "${ENABLE_C128_OWNER_SHARD}" "${ENABLE_C128_OWNER_COMPACT_ALLOCATION}" "${ENABLE_C128_OWNER_DEBUG}" "${ENABLE_C128_OWNER_SELECTIVE_STAGE}" "${ENABLE_C128_OWNER_LOCAL_COMPRESSOR}" "${ENABLE_DSA_CP_LOCAL_CURRENT_KV}" "${ENABLE_KV_CACHE_ALLOCATION_ACCOUNTING}" "${ENABLE_DSA_LAYER_SHARDING}" "${ENABLE_FUSED_MC2}" "${ENABLE_MTP}" "${ENABLE_MOONCAKE_KV_CONNECTOR}" "${SYNTHETIC_ROUTED_EXPERTS}" "${ENABLE_TORCH_PROFILER}"
    printf 'dp_size=%s dp_rank=%s tp_size=%s api_port=%s\n' \
        "${DP_SIZE}" "${DP_RANK}" "${TP_SIZE}" "${VLLM_PORT}"
    printf 'safetensors_load_strategy=%s gpu_memory_utilization=%s max_model_len=%s max_num_batched_tokens=%s num_gpu_blocks_override=%s\n' \
        "${SAFETENSORS_LOAD_STRATEGY}" "${GPU_MEMORY_UTILIZATION}" "${MAX_MODEL_LEN}" \
        "${MAX_NUM_BATCHED_TOKENS}" "${NUM_GPU_BLOCKS_OVERRIDE:-<unset>}"
    printf '\nEnvironment:\n'
    for key in "${ENV_KEYS[@]}"; do
        printf '%s=%q\n' "${key}" "${!key-}"
    done
    printf 'VLLM_ASCEND_PER_REQUEST_SAVE_WAIT=<unset>\n'
    if [[ "${ENABLE_MOONCAKE_KV_CONNECTOR}" == 1 ]]; then
        printf '\n--kv-transfer-config:\n'
        jq . <<<"${KV_TRANSFER_CONFIG}"
    else
        printf '\n--kv-transfer-config: disabled for isolated prefill\n'
    fi
    printf '\nCommand:\n'
    printf '%q ' "${VLLM_CMD[@]}"
    printf '\n'
}

if [[ "${PRINT_CONFIG_ONLY:-0}" == 1 ]]; then
    print_effective_config
    exit 0
fi

ensure_not_running
cd "${ROLE_DIR}"
mkdir -p /home/admin/logs/vllm "${ASCEND_PROCESS_LOG_PATH}" \
    "${ROLE_DIR}/logs/runtime" "${ROLE_DIR}/logs/vllm" "${MC_LOG_DIR}"
if [[ "${ENABLE_MOONCAKE_KV_CONNECTOR}" == 1 ]]; then
    ensure_mooncake_master
fi
ulimit -c unlimited
ulimit -n 1048576

{
    printf '\n===== %s %s launch =====\n' "$(date '+%F %T')" "${ROLE_NAME}"
    print_effective_config
} >> "${LOG_FILE}"

nohup setsid "${VLLM_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
VLLM_PID=$!
printf '%s\n' "${VLLM_PID}" > "${PID_FILE}"
ps -o pgid= -p "${VLLM_PID}" | tr -d ' ' > "${ROLE_DIR}/.vllm_pgids"
sleep 1
kill -0 "${VLLM_PID}" 2>/dev/null || {
    echo "${ROLE_NAME} exited during launch; inspect ${LOG_FILE}" >&2
    exit 1
}

echo "Started ${ROLE_NAME}: pid=${VLLM_PID} log=${LOG_FILE}"
