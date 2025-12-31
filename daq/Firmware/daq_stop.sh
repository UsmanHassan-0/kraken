#/bin/sh!
echo "Shut down DAQ chain .."
sudo -v
#sudo kill -64 $(ps aux | grep 'rtl' | awk '{print $2}')
#sudo killall -s 9 rtl*

sudo pkill -64 rtl_daq.out
sudo kill -64 $(ps ax | grep "[p]ython3 testing/test_data_synthesizer.py" | awk '{print $1}') 2> /dev/null
sudo pkill -64 sync.out
sudo pkill -64 decimate.out
sudo pkill -64 rebuffer.out
sudo kill -64 $(ps ax | grep "[p]ython3 daq_core/delay_sync.py" | awk '{print $1}') 2> /dev/null
sudo kill -64 $(ps ax | grep "[p]ython3 daq_core/hw_controller.py" | awk '{print $1}') 2> /dev/null
sudo kill -64 $(ps ax | grep "[p]ython3 daq_core/iq_eth_sink.py" | awk '{print $1}') 2> /dev/null
sudo pkill -64 iq_server.out

is_kraken_cmd() {
    case "$1" in
        *daq_core/rtl_daq.out*|*daq_core/rebuffer.out*|*daq_core/decimate.out*|*daq_core/iq_server.out*|*daq_core/delay_sync.py*|*daq_core/hw_controller.py*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

kill_ports() {
    ports="$*"
    if command -v lsof >/dev/null 2>&1; then
        while :; do
            any_listeners=0
            for p in $ports; do
                pids=$(sudo lsof -nP -t -sTCP:LISTEN -iTCP:"$p" 2>/dev/null | tr '\n' ' ')
                if [ -z "$pids" ]; then
                    continue
                fi
                any_listeners=1
                for pid in $pids; do
                    cmd=$(ps -o args= -p "$pid" 2>/dev/null)
                    if is_kraken_cmd "$cmd"; then
                        echo "Kraken listener on port ${p}: pid=${pid} cmd=${cmd}"
                    fi
                    sudo kill -9 "$pid" 2>/dev/null || true
                done
            done
            if [ "$any_listeners" -eq 0 ]; then
                break
            fi
            sleep 0.2
        done
    fi
    if command -v fuser >/dev/null 2>&1; then
        for p in $ports; do
            sudo fuser -k "${p}/tcp" >/dev/null 2>&1 || true
        done
    fi
}

# Ensure IQ server (5000) and hardware controller (5001) ports are released
kill_ports 5000 5001
