#!/bin/bash

set -euo pipefail

# Resolve repository root based on this script's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the Kraken environment
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate kraken
fi

# Optional: clear Python bytecode caches when -c is provided
while getopts c flag; do
    case "${flag}" in
        c) sudo py3clean "${REPO_ROOT}" ;;
    esac
done

# Stop any existing stack before starting fresh
if [[ -x "${REPO_ROOT}/kraken_stop.sh" ]]; then
    "${REPO_ROOT}/kraken_stop.sh" || true
fi

DAQ_DIR="${REPO_ROOT}/daq/Firmware"
DOA_DIR="${REPO_ROOT}/gui"

# Start DAQ
cd "${DAQ_DIR}"
sudo env "PATH=$PATH" ./daq_start_sm.sh
sleep 1

# Start Web UI
cd "${DOA_DIR}"

LOGS_GUI="logs/gui"
LOGS_DAQ="logs/daq"
SHARE_DIR="share"

mkdir -p "${SHARE_DIR}" "${LOGS_GUI}" "${LOGS_DAQ}"

# Sync DAQ logs into shared folder
./util/sync_daq_logs.sh >/dev/null 2>/dev/null &

echo "Web Interface running at 0.0.0.0:8080"
python3 ui/web_interface/app.py >"${LOGS_GUI}/ui.log" 2>&1 &
