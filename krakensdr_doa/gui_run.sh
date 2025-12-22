#!/bin/bash

IPADDR="0.0.0.0"

SHARED_FOLDER="_share"
SHARED_FOLDER_DOA_LOGS="${SHARED_FOLDER}/logs/krakensdr_doa"
SHARED_FOLDER_DAQ_LOGS="${SHARED_FOLDER}/logs/heimdall_daq_fw"

# Create folder, if it does not exists, that will contain data shared with clients
mkdir -p "${SHARED_FOLDER}"
# Create folder, if it does not exists, that will contain logs shared with clients
mkdir -p "${SHARED_FOLDER_DOA_LOGS}"
mkdir -p "${SHARED_FOLDER_DAQ_LOGS}"

# Start rsync to sync DAQ logs into shared folder
./util/sync_daq_logs.sh >/dev/null 2>/dev/null &

echo "Web Interface Running at $IPADDR:8080"
python3 _ui/_web_interface/app.py >"${SHARED_FOLDER_DOA_LOGS}/ui.log" 2>&1 &
