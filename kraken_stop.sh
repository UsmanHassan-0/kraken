#!/bin/bash

set -euo pipefail

# Resolve repository root based on this script's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAQ_DIR="${REPO_ROOT}/daq/Firmware"
DOA_DIR="${REPO_ROOT}/gui"
LOG_ROOT="${REPO_ROOT}/logs"
STOP_LOG="${LOG_ROOT}/kraken_stop.log"

mkdir -p "${LOG_ROOT}"
: > "${STOP_LOG}"
exec > >(tee -a "${STOP_LOG}") 2>&1

# Ensure sudo is available for port cleanup and process termination
sudo -v

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

is_kraken_cmd() {
    case "$1" in
        *ui/web_interface/app.py*|*daq/Firmware/daq_core/rtl_daq.out*|*daq/Firmware/daq_core/rebuffer.out*|*daq/Firmware/daq_core/decimate.out*|*daq/Firmware/daq_core/iq_server.out*|*daq_core/delay_sync.py*|*daq_core/hw_controller.py*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

kill_procs ".*[p]ython3 .*ui/web_interface/app.py"
kill_procs ".*[p]ython3 .*_ui/_web_interface/app.py"
kill_procs "[p]hp"

kill_ports() {
    local ports=("$@")
    if command -v lsof >/dev/null 2>&1; then
        while true; do
            local any_listeners=0
            for p in "${ports[@]}"; do
                local pids
                pids=$(sudo lsof -nP -t -sTCP:LISTEN -iTCP:"${p}" 2>/dev/null || true)
                if [[ -z "${pids}" ]]; then
                    continue
                fi
                any_listeners=1
                for pid in ${pids}; do
                    local cmd
                    cmd=$(ps -o args= -p "${pid}" 2>/dev/null || true)
                    if is_kraken_cmd "${cmd}"; then
                        echo "Kraken listener on port ${p}: pid=${pid} cmd=${cmd}"
                    fi
                    sudo kill -9 "${pid}" 2>/dev/null || true
                done
            done
            if [[ "${any_listeners}" -eq 0 ]]; then
                break
            fi
            sleep 0.2
        done
    fi
    if command -v fuser >/dev/null 2>&1; then
        for p in "${ports[@]}"; do
            sudo fuser -k "${p}/tcp" >/dev/null 2>&1 || true
        done
    fi
}

# Ensure the UI port is fully released
kill_ports 8080
