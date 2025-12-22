#!/bin/bash

set -euo pipefail

# Resolve repository root based on this script's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAQ_DIR="${REPO_ROOT}/heimdall_daq_fw/Firmware"
DOA_DIR="${REPO_ROOT}/krakensdr_doa"

# Stop DAQ
cd "${DAQ_DIR}"
./daq_stop.sh || true

# Stop Web UI
cd "${DOA_DIR}"

SYSTEM_OS="$(uname -s)"
if [[ "${SYSTEM_OS}" == "Darwin" ]]; then
    KILL_SIGNAL=9
else
    KILL_SIGNAL=64
fi

kill_procs() {
    local pattern="$1"
    local pids
    pids=$(ps ax | grep "${pattern}" | awk '{print $1}')
    if [[ -n "${pids}" ]]; then
        sudo kill -"${KILL_SIGNAL}" ${pids} 2>/dev/null || true
    fi
}

kill_procs ".*[p]ython3 .*_ui/_web_interface/app.py"
kill_procs "[p]hp"
