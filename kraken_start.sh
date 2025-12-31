#!/bin/bash

set -euo pipefail

# Resolve repository root based on this script's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${REPO_ROOT}/logs"
LOGS_GUI="${LOG_ROOT}/gui"
LOGS_DAQ="${LOG_ROOT}/daq"
START_LOG="${LOG_ROOT}/kraken_start.log"
SHARE_DIR="share"

mkdir -p "${LOG_ROOT}" "${LOGS_GUI}" "${LOGS_DAQ}" "${SHARE_DIR}"
: > "${START_LOG}"

# Activate the Kraken environment from Miniforge only
. "$HOME/miniforge3/etc/profile.d/conda.sh" || { echo "Miniforge not found at ~/miniforge3; run ./install_kraken.sh"; exit 1; }
conda activate kraken || { echo "Failed to activate conda env 'kraken'"; exit 1; }

# Optional: clear Python bytecode caches when -c is provided
while getopts c flag; do
    case "${flag}" in
        c) sudo py3clean "${REPO_ROOT}" ;;
    esac
done

DAQ_DIR="${REPO_ROOT}/daq/Firmware"
DOA_DIR="${REPO_ROOT}/gui"

# Stop any existing stack before starting fresh
if [[ -x "${REPO_ROOT}/kraken_stop.sh" ]]; then
    "${REPO_ROOT}/kraken_stop.sh" || true
fi

# Start DAQ
cd "${DAQ_DIR}"
sudo -E env "PATH=$PATH" ./daq_start_sm.sh >"daq_start.log" 2>&1
sleep 1

# Start Web UI
cd "${DOA_DIR}"

# Sync DAQ logs into shared folder
./util/sync_daq_logs.sh >/dev/null 2>/dev/null &

echo "Web Interface running at 0.0.0.0:8080" | tee -a "${START_LOG}"
# Run the GUI with sudo so it can read root-owned FIFOs created by DAQ
sudo -E env "PATH=$PATH" \
  python3 ui/web_interface/app.py >"${LOGS_GUI}/ui.log" 2>&1 &
