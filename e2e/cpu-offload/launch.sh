#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=${ENGRAM_BASE_LAUNCH:-/a3_inference/itask/workdir/shared/ningyunxiao.nyx/dsv4_1/engram-delivery-20260916/launch_a3.sh}
export ENGRAM_NATIVE_CPU_REFERENCE=1
# Retain the CANN SDK paths established by the container environment.
export PYTHONPATH="$root:${PYTHONPATH:-}"
exec bash "$base" "${1:?DP rank}" "${2:?local IP}" "${3:?master IP}" hbm
