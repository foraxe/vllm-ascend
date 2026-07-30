#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <durable-artifact-directory> [probe options...]" >&2
  exit 2
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$1"
shift
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-60}"
TOTAL_TIMEOUT_SECONDS="${TOTAL_TIMEOUT_SECONDS:-180}"

mkdir -p "${ARTIFACT_DIR}/source"
cp "${SOURCE_DIR}/vmm_bridge.cpp" \
  "${SOURCE_DIR}/remote_tensor_probe.py" \
  "${SOURCE_DIR}/run_probe.sh" \
  "${ARTIFACT_DIR}/source/"
RUN_SOURCE_DIR="${ARTIFACT_DIR}/source"

source /usr/local/Ascend/cann/set_env.sh

g++ -std=c++17 -O2 -g -fPIC -shared \
  -Wall -Wextra -Werror \
  -I"${CANN_HOME}/include" \
  "${RUN_SOURCE_DIR}/vmm_bridge.cpp" \
  -L"${CANN_HOME}/lib64" \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -lascendcl \
  -o "${ARTIFACT_DIR}/libdsa_vmm_bridge.so" \
  2>&1 | tee "${ARTIFACT_DIR}/build.log"

nm -D "${ARTIFACT_DIR}/libdsa_vmm_bridge.so" \
  | grep ' dsa_vmm_' \
  | tee "${ARTIFACT_DIR}/symbols.log"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PROBE_GIT_SHA="${PROBE_GIT_SHA:-unknown}"

set +e
timeout --signal=TERM --kill-after=15s \
  "${TOTAL_TIMEOUT_SECONDS}s" \
  python3 -u "${RUN_SOURCE_DIR}/remote_tensor_probe.py" \
    --library "${ARTIFACT_DIR}/libdsa_vmm_bridge.so" \
    --output "${ARTIFACT_DIR}/result.json" \
    --stage-timeout "${STAGE_TIMEOUT_SECONDS}" \
    "$@" \
  > >(tee "${ARTIFACT_DIR}/run.log") 2>&1
probe_status=$?
set -e

printf '%s\n' "${probe_status}" >"${ARTIFACT_DIR}/exit_code"
exit "${probe_status}"
