#!/usr/bin/env bash

GUI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAQ_LOGS_PATH="${GUI_ROOT}/../daq/Firmware/logs"
LOGS_DAQ="${GUI_ROOT}/../logs/daq"

if [ -d "$DAQ_LOGS_PATH" ] && [ -d "$LOGS_DAQ" ]; then
    while true; do
        cp -afu "${DAQ_LOGS_PATH}"/* "${LOGS_DAQ}/"
        sleep 5
    done
fi
