#!/bin/bash

echo -e "\e[33mConfig file check bypassed [ WARNING ]\e[39m"

sudo sysctl -w kernel.sched_rt_runtime_us=-1

# Read config ini file
out_data_iface_type=$(awk -F'=' '/out_data_iface_type/ {gsub (" ", "", $0); print $2}' daq_chain_config.ini)

# (re) create control FIFOs
rm data_control/fw_decimator_in 2> /dev/null
rm data_control/bw_decimator_in 2> /dev/null

rm data_control/fw_decimator_out 2> /dev/null
rm data_control/bw_decimator_out 2> /dev/null

rm data_control/fw_delay_sync_iq 2> /dev/null
rm data_control/bw_delay_sync_iq 2> /dev/null

rm data_control/fw_delay_sync_hwc 2> /dev/null
rm data_control/bw_delay_sync_hwc 2> /dev/null

mkfifo data_control/fw_decimator_in
mkfifo data_control/bw_decimator_in

mkfifo data_control/fw_decimator_out
mkfifo data_control/bw_decimator_out

mkfifo data_control/fw_delay_sync_iq
mkfifo data_control/bw_delay_sync_iq

mkfifo data_control/fw_delay_sync_hwc
mkfifo data_control/bw_delay_sync_hwc

# Remove old log files
rm logs/*.log 2> /dev/null

# The Kernel limits the maximum size of all buffers that libusb can allocate to 16MB by default.
# In order to disable the limit, you have to run the following command as root:
sudo sh -c "echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb"

# This command clear the caches
echo '3' | sudo tee /proc/sys/vm/drop_caches > /dev/null

# Stop any prior DAQ chain before checking ports
sudo env "PATH=$PATH" ./daq_stop.sh

# Generating FIR filter coefficients
python3 fir_filter_designer.py
out=$?
if test $out -ne 0
    then
        echo -e "\e[91mDAQ chain not started!\e[39m"
        exit
fi

# Start main program chain -Thread 0 Normal (non squelch mode)
echo "Starting DAQ Subsystem"
chrt -f 99 daq_core/rtl_daq.out 2> logs/rtl_daq.log | \
chrt -f 99 daq_core/rebuffer.out 0 2> logs/rebuffer.log &

# Decimator - Thread 1
chrt -f 99 daq_core/decimate.out 2> logs/decimator.log &

# Delay synchronizer - Thread 2
chrt -f 99 python3 daq_core/delay_sync.py 2> logs/delay_sync.log &

# Hardware Controller data path - Thread 3
chrt -f 99 sudo env "PATH=$PATH" python3 daq_core/hw_controller.py 2> logs/hwc.log &
# root priviliges are needed to drive the i2c master

if [ $out_data_iface_type = eth ]; then
    echo "Output data interface: IQ ethernet server"
    chrt -f 99 daq_core/iq_server.out 2>logs/iq_server.log &
elif [ $out_data_iface_type = shmem ]; then
    echo "Output data interface: Shared memory"
fi


echo -e "      )  (     "
echo -e "      (   ) )  "
echo -e "       ) ( (   "
echo -e "     _______)_ "
echo -e "  .-'---------|" 
echo -e " (  |/\/\/\/\/|"
echo -e "  '-./\/\/\/\/|"
echo -e "    '_________'"
echo -e "     '-------' "
echo -e "               "
echo -e "Have a coffee watch radar"
