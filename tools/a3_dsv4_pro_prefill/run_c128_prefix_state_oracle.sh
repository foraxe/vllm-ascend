#!/usr/bin/env bash
set -eo pipefail

TEST_FILE=${1:?usage: run_c128_prefix_state_oracle.sh TEST_FILE RUN_DIR}
RUN_DIR=${2:?usage: run_c128_prefix_state_oracle.sh TEST_FILE RUN_DIR}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
CUSTOM_OP_VENDOR=${CUSTOM_OP_VENDOR:-/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}

source "${CANN_ENV}"
set -u

mkdir -p "${RUN_DIR}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_VERSION=${VLLM_VERSION:-0.20.2}
export VLLM_ASCEND_APPLY_DSV4_PATCH=${VLLM_ASCEND_APPLY_DSV4_PATCH:-1}
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OP_VENDOR}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${CUSTOM_OP_VENDOR}/op_api/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export DSA_CP_ORACLE_RESULT_JSON="${RUN_DIR}/oracle_result.json"

{
    echo "hypothesis=Each TP8-local 640-token C128 compressor call followed by the shared 3080-token tail preserves the global reference outputs, live state, and future continuation."
    echo "baseline=one global 5120-token compressor call, then 3080-token tail, then 24-token continuation"
    echo "candidate=eight isolated state histories, each with one rank-local 640-token call, then the same tail and continuation"
    echo "metric=all eight prefix outputs, tail outputs, logical live state [8072:8200], continuation outputs, and future live state"
    echo "pass=all output comparisons satisfy atol=0.05 rtol=0.01 and all state comparisons satisfy atol=0.0001 rtol=0.001"
    echo "fail=any numerical comparison exceeds its tolerance"
    echo "kill=timeout after ${TIMEOUT_SECONDS}s or any custom-op ABI/runtime error"
    echo "test_file=${TEST_FILE}"
    echo "result_json=${DSA_CP_ORACLE_RESULT_JSON}"
    echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE}"
    echo "PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF}"
    echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
    echo "custom_op_api_lib=${CUSTOM_OP_VENDOR}/op_api/lib"
    python --version
    python -c "import torch, torch_npu, vllm, vllm_ascend; print(f'torch={torch.__version__} torch_npu={torch_npu.__version__} vllm={vllm.__version__}')"
} > "${RUN_DIR}/experiment_manifest.txt" 2>&1

npu-smi info > "${RUN_DIR}/npu_before.txt" 2>&1
set +e
timeout "${TIMEOUT_SECONDS}" python -m pytest -sv "${TEST_FILE}" \
    > >(tee "${RUN_DIR}/pytest.log") \
    2>&1
pytest_status=$?
set -e
npu-smi info > "${RUN_DIR}/npu_after.txt" 2>&1
echo "${pytest_status}" > "${RUN_DIR}/pytest.exitcode"
exit "${pytest_status}"
