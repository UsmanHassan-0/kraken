# KrakenSDR Signal Processor
#
# Copyright (C) 2018-2021  Carl Laufer, Tamás Pető
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# - coding: utf-8 -*-

import copy
import json
import logging
import os

# Import built-in modules
import threading
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Tuple

# Import optimization modules
import numba as nb

# Math support
import numpy as np
import numpy.linalg as lin

# Signal processing support
import scipy
from iq_header import IQHeader
from kraken_sdr_receiver import ReceiverRTLSDR
from numba import njit
from pyargus import directionEstimation as de
from scipy import fft, signal
from signal_utils import can_store_file, fm_demod, write_wav
from variables import (
    SOFTWARE_GIT_SHORT_HASH,
    SOFTWARE_VERSION,
    SYSTEM_UNAME,
    root_path,
    shared_path,
    status_file_path,
)

# os.environ['OPENBLAS_NUM_THREADS'] = '4'
# os.environ['NUMBA_CPU_NAME'] = 'cortex-a72'

DEFAULT_VFO_FIR_ORDER_FACTOR = int(2)

NEAR_ZERO = 1e-15


class SignalProcessor(threading.Thread):
    def __init__(self, data_que, module_receiver: ReceiverRTLSDR, logging_level=10):
        """
        Parameters:
        -----------
        :param: data_que: Que to communicate with the UI (web iface/Qt GUI)
        :param: module_receiver: Kraken SDR DoA DSP receiver modules
        """
        super(SignalProcessor, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        self.root_path = root_path

        self.module_receiver = module_receiver
        self.data_que = data_que
        self.en_spectrum = False
        self.en_record = False
        self.wav_record_path = f"{shared_path}/records/fm"
        self.en_iq_files = False
        self.iq_record_path = f"{shared_path}/records/iq"
        self.en_DOA_estimation = True
        self.doa_measure = "Linear"
        self.compass_offset = 0.0
        self.first_frame = 1  # Used to configure local variables from the header fields
        self.processed_signal = np.empty(0)

        Path(f"{self.wav_record_path}/").mkdir(parents=True, exist_ok=True)
        Path(f"{self.iq_record_path}").mkdir(parents=True, exist_ok=True)

        # Squelch feature
        self.data_ready = False
        self.dsp_decimation = 1

        # DOA processing options
        # self.en_DOA_Bartlett = False
        # self.en_DOA_Capon    = False
        # self.en_DOA_MEM      = False
        # self.en_DOA_MUSIC    = False
        self.DOA_algorithm = "MUSIC"
        self.DOA_offset = 0
        self.DOA_UCA_radius_m = np.Infinity
        self.DOA_inter_elem_space = 0.5
        self.DOA_ant_alignment = "UCA"
        self.DOA_theta = np.linspace(0, 359, 360)
        self.array_offset = 0.0
        self.DOA_expected_num_of_sources = 1
        self.DOA_decorrelation_method = "Off"

        # Processing parameters
        self.spectrum_window_size = fft.next_fast_len(4096)
        self.spectrum_plot_size = 1024
        self.spectrum_window = "hann"
        self.run_processing = True  # False
        self.is_running = False
        self.channel_number = 4  # Update from header
        self.spectrum_fig_type = "Single"  # 0 Single, 1 Full

        # Result vectors
        self.DOA = np.ones(181)

        # VFO settings
        self.max_vfos = 16
        self.vfo_bw = [12500] * self.max_vfos
        self.vfo_fir_order_factor = [DEFAULT_VFO_FIR_ORDER_FACTOR] * self.max_vfos
        self.vfo_freq = [self.module_receiver.daq_center_freq] * self.max_vfos
        self.vfo_default_squelch_mode = "Auto"
        self.vfo_squelch_mode = ["Auto"] * self.max_vfos
        self.vfo_squelch = [-120] * self.max_vfos
        self.vfo_default_demod = "None"
        self.vfo_demod = ["Default"] * self.max_vfos
        self.vfo_default_iq = "False"
        self.vfo_iq = ["Default"] * self.max_vfos
        self.vfo_demod_channel = [np.array([])] * self.max_vfos
        self.vfo_theta_channel = [[]] * self.max_vfos
        self.vfo_iq_channel = [np.array([])] * self.max_vfos
        self.vfo_blocked = [False] * self.max_vfos
        self.vfo_time = [0] * self.max_vfos
        self.max_demod_timeout = 60
        self.default_auto_db_offset = 5  # 5dB for Auto Squelch
        self.default_auto_channel_db_offset = 3  # 3dB for Auto Channel Squelch and Scan modes
        # Ratio of Auto Channel, mean that how big should be measurement of spectrum outside of vfo_bw
        self.ratio_auto_channel = 3

        self.en_fm_demod = False
        self.vfo_fm_demod = [False] * self.max_vfos
        self.fm_demod_channels = [None] * self.max_vfos
        self.fm_demod_channels_thetas = [[]] * self.max_vfos
        self.iq_channels = [None] * self.max_vfos

        self.active_vfos = 1
        self.output_vfo = 0
        self.vfo_mode = "Standard"
        self.optimize_short_bursts = False

        # self.DOA_theta =  np.linspace(0,359,360)
        self.spectrum = None  # np.ones((self.channel_number+2,N), dtype=np.float32)
        self.peak_hold_spectrum = np.ones(self.spectrum_window_size) * -200
        self.en_peak_hold = False

        self.latency = 100
        self.processing_time = 0
        self.timestamp = int(time.time() * 1000)

        self.doa_max_list = [-1] * self.max_vfos

        self.theta_0_list = []
        self.freq_list = []
        self.doa_result_log_list = []
        self.confidence_list = []
        self.max_power_level_list = []
        self.fm_demod_channel_list = []

        # TODO: NEED to have a funtion to update the file name if changed in the web ui
        self.data_recording_file_name = "mydata.csv"
        data_recording_file_path = os.path.join(os.path.join(self.root_path, self.data_recording_file_name))
        self.data_record_fd = open(data_recording_file_path, "a+")
        self.en_data_record = False
        self.write_interval = 1
        self.last_write_time = [time.time()] * self.max_vfos

        self.adc_overdrive = False
        self.number_of_correlated_sources = []
        self.snrs = []
        self.dropped_frames = 0

    @property
    def vfo_demod_modes(self):
        vfo_demod = [self.vfo_default_demod] * self.max_vfos
        for i in range(len(self.vfo_demod)):
            demod = self.vfo_demod[i]
            if demod != "Default":
                vfo_demod[i] = demod
        return vfo_demod

    @property
    def vfo_iq_enabled(self):
        vfo_iq = [True if self.vfo_default_iq == "True" else False] * self.max_vfos
        for i in range(len(self.vfo_iq)):
            demod = self.vfo_iq[i]
            if demod != "Default":
                vfo_iq[i] = True if demod == "True" else False
        return vfo_iq

    def resetPeakHold(self):
        if self.spectrum_fig_type == "Single":
            self.peak_hold_spectrum = np.ones(self.spectrum_window_size) * -200

    def mean_spectrum(self, measured_spec):
        def is_enabled_auto_squelch(v):
            return v == "Auto" or (v == "Default" and self.vfo_default_squelch_mode == "Auto")

        auto_squelch = any(is_enabled_auto_squelch(vfo_squelch_mode) for vfo_squelch_mode in self.vfo_squelch_mode)
        if auto_squelch:
            measured_spec_mean = np.mean(measured_spec)
            vfo_auto_squelch = measured_spec_mean + self.default_auto_db_offset

            for i in range(len(self.vfo_squelch)):
                auto_squelch = is_enabled_auto_squelch(self.vfo_squelch_mode[i])
                if auto_squelch:
                    self.vfo_squelch[i] = vfo_auto_squelch

    def calculate_squelch(self, sampling_freq, N, measured_spec, real_freqs):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

        self.mean_spectrum(measured_spec)

        for i, vfo_squelch_mode in enumerate(self.vfo_squelch_mode[: self.active_vfos]):
            if vfo_squelch_mode == "Auto Channel" or (
                vfo_squelch_mode == "Default" and self.vfo_default_squelch_mode == "Auto Channel"
            ):
                vfo_bw_freq_window = int(self.vfo_bw[i] / (sampling_freq / N))
                freq_idx, nearsest = find_nearest(real_freqs, self.vfo_freq[i])
                vfo_freq_window = int(vfo_bw_freq_window / 2 + self.ratio_auto_channel * vfo_bw_freq_window)
                vfo_start_measure_spec = freq_idx - min(abs(freq_idx), vfo_freq_window)
                vfo_end_measure_spec = freq_idx + min(abs(N - freq_idx), vfo_freq_window)
                measured_spec_mean = np.mean(measured_spec[vfo_start_measure_spec:vfo_end_measure_spec])
                self.vfo_squelch[i] = measured_spec_mean + self.default_auto_channel_db_offset

    def save_processing_status(self) -> None:
        """This method serializes system status to file."""

        status = {}
        daq_status = {}

        status["timestamp_ms"] = int(time.time() * 1e3)
        status["hardware_id"] = self.module_receiver.iq_header.hardware_id.rstrip("\x00")
        status["unit_id"] = self.module_receiver.iq_header.unit_id
        status["host_os_type"] = SYSTEM_UNAME.system
        status["host_os_version"] = SYSTEM_UNAME.release
        status["host_os_architecture"] = SYSTEM_UNAME.machine
        status["software_version"] = SOFTWARE_VERSION
        status["software_git_short_hash"] = SOFTWARE_GIT_SHORT_HASH
        status["uptime_ms"] = int(time.monotonic() * 1e3)

        iq_header_emtpy = self.module_receiver.iq_header.frame_type == IQHeader.FRAME_TYPE_EMPTY

        if not iq_header_emtpy:
            status["timestamp_ms"] = self.module_receiver.iq_header.time_stamp
            daq_status["data_frame_index"] = self.module_receiver.iq_header.cpi_index
            daq_status["frame_sync"] = not bool(self.module_receiver.iq_header.check_sync_word())
            daq_status["sample_delay_sync"] = bool(self.module_receiver.iq_header.delay_sync_flag)
            daq_status["iq_sync"] = bool(self.module_receiver.iq_header.iq_sync_flag)
            daq_status["noise_source_enabled"] = bool(self.module_receiver.iq_header.noise_source_state)
            daq_status["adc_overdrive"] = bool(self.module_receiver.iq_header.adc_overdrive_flags)
            daq_status["sampling_frequency_hz"] = self.module_receiver.iq_header.adc_sampling_freq
            daq_status["bandwidth_hz"] = self.module_receiver.iq_header.sampling_freq
            daq_status["decimated_bandwidth_hz"] = self.module_receiver.iq_header.sampling_freq // self.dsp_decimation
            daq_status["buffer_size_ms"] = (
                (self.module_receiver.iq_header.cpi_length / self.module_receiver.iq_header.sampling_freq) * 1e3
                if self.module_receiver.iq_header.sampling_freq > 0.0
                else 0.0
            )

        status["daq_status"] = daq_status
        status["daq_ok"] = (
            not iq_header_emtpy
            and daq_status.get("frame_sync", False)
            and daq_status.get("sample_delay_sync", False)
            and daq_status.get("iq_sync", False)
        )
        status["daq_num_dropped_frames"] = self.dropped_frames

        try:
            with open(status_file_path, "w", encoding="utf-8") as file:
                json.dump(status, file)
        except Exception:
            pass

    def run(self):
        """
        Main processing thread
        """
        # scipy.fft.set_workers(4)
        while True:
            self.is_running = False
            time.sleep(1)
            while self.run_processing:
                self.is_running = True
                que_data_packet = []

                # -----> ACQUIRE NEW DATA FRAME <-----
                get_iq_failed = self.module_receiver.get_iq_online()

                start_time = time.time()
                self.save_processing_status()

                que_data_packet.append(["iq_header", self.module_receiver.iq_header])
                self.logger.debug("IQ header has been put into the data que entity")

                # Check frame type for processing
                """
                    You can enable here to process other frame types (such as call type frames)
                """
                en_proc = (
                    self.module_receiver.iq_header.frame_type == self.module_receiver.iq_header.FRAME_TYPE_DATA
                )  # or \
                # (self.module_receiver.iq_header.frame_type == self.module_receiver.iq_header.FRAME_TYPE_CAL)# For debug purposes

                self.data_ready = False

                if not self.module_receiver.iq_samples.size and get_iq_failed:
                    if not self.dropped_frames:
                        logging.error(
                            """The data frame was lost while processing was active!
                            This might indicate issues with USB data cable or USB host,
                            inadequate power supply, overloaded CPU, wrong host OS settings, etc."""
                        )
                    self.dropped_frames += 1
                elif en_proc:
                    self.timestamp = self.module_receiver.iq_header.time_stamp
                    self.adc_overdrive = self.module_receiver.iq_header.adc_overdrive_flags

                    # Configure processing parameteres based on the settings of the DAQ chain
                    if self.first_frame:
                        self.channel_number = self.module_receiver.iq_header.active_ant_chs
                        self.spectrum = np.ones(
                            (self.channel_number + 4, self.spectrum_window_size),
                            dtype=np.float32,
                        )
                        self.first_frame = 0

                    self.processed_signal = np.ascontiguousarray(self.module_receiver.iq_samples)
                    sampling_freq = self.module_receiver.iq_header.sampling_freq

                    global_decimation_factor = max(
                        int(self.dsp_decimation), 1
                    )  # max(int(self.phasetest[0]), 1) #ps_len // 65536 #int(self.phasetest[0]) + 1

                    if global_decimation_factor > 1:
                        self.processed_signal = signal.decimate(
                            self.processed_signal,
                            global_decimation_factor,
                            n=global_decimation_factor * 5,
                            ftype="fir",
                        )
                        sampling_freq = sampling_freq // global_decimation_factor

                    self.data_ready = True

                    if self.spectrum_fig_type == "Single":
                        m = 0
                        N = self.spectrum_window_size
                        self.spectrum = (
                            np.ones(
                                (self.channel_number + (self.active_vfos * 2 + 1), N),
                                dtype=np.float32,
                            )
                            * -200
                        )  # Only 0.1 ms, not performance bottleneck

                        single_ch = self.processed_signal[1, :]

                        noverlap = int(N * 0)
                        window = "blackman"
                        if self.optimize_short_bursts:
                            noverlap = int(N * 0.5)
                            window = ("tukey", 0.15)

                        f, Pxx_den = signal.welch(
                            single_ch,
                            sampling_freq,
                            nperseg=N,
                            nfft=N,
                            noverlap=noverlap,  # int(N_perseg*0.0),
                            detrend=False,
                            return_onesided=False,
                            window=window,
                            # 'blackman', #('tukey', 0.25), #tukey window gives better time resolution for squelching
                            scaling="spectrum",
                        )
                        self.spectrum[1 + m, :] = fft.fftshift(10 * np.log10(Pxx_den))
                        if self.en_peak_hold:
                            self.spectrum[2 + m, :] = np.maximum(self.peak_hold_spectrum, self.spectrum[1 + m, :])
                            self.peak_hold_spectrum = self.spectrum[2 + m, :]

                        self.spectrum[0, :] = fft.fftshift(f)
                    else:
                        N = 32768
                        self.spectrum = np.ones(
                            (self.channel_number + (self.active_vfos * 2 + 1), N),
                            dtype=np.float32,
                        )
                        for m in range(self.channel_number):  # range(1): #range(self.channel_number):
                            f, Pxx_den = signal.periodogram(
                                self.processed_signal[m, :],
                                sampling_freq,
                                nfft=N,
                                detrend=False,
                                return_onesided=False,
                                window="blackman",
                                scaling="spectrum",
                            )
                            self.spectrum[1 + m, :] = fft.fftshift(10 * np.log10(Pxx_den))
                        self.spectrum[0, :] = fft.fftshift(f)

                    max_amplitude = np.max(self.spectrum[1, :])  # Max amplitude out of all 5 channels
                    que_data_packet.append(["max_amplitude", max_amplitude])

                    # -----> DoA PROCESSING <-----
                    try:
                        if self.data_ready:
                            spectrum_window_size = len(self.spectrum[0, :])
                            active_vfos = self.active_vfos if self.vfo_mode == "Standard" else 1
                            write_freq = 0
                            update_list = [False] * self.max_vfos
                            conf_val = 0
                            theta_0 = 0
                            DOA_str = ""
                            confidence_str = ""
                            max_power_level_str = ""
                            doa_result_log = np.empty(0)

                            self.theta_0_list.clear()
                            self.freq_list.clear()
                            self.doa_result_log_list.clear()
                            self.max_power_level_list.clear()
                            self.confidence_list.clear()
                            self.number_of_correlated_sources.clear()
                            self.snrs.clear()
                            self.fm_demod_channel_list.clear()

                            relative_freqs = self.spectrum[0, ::-1]
                            real_freqs = self.module_receiver.daq_center_freq - relative_freqs
                            measured_spec = self.spectrum[1, :]

                            self.calculate_squelch(sampling_freq, N, measured_spec, real_freqs)

                            for i in range(active_vfos):
                                # If chanenl freq is out of bounds for the current tuned bandwidth, reset to the middle freq
                                if abs(self.vfo_freq[i] - self.module_receiver.daq_center_freq) > sampling_freq / 2:
                                    self.vfo_freq[i] = self.module_receiver.daq_center_freq

                                freq = (
                                    self.vfo_freq[i] - self.module_receiver.daq_center_freq
                                )  # ch_freq is relative to -sample_freq/2 : sample_freq/2, so correct for that and get the actual freq

                                if self.vfo_mode == "Auto":  # Mode 1 is Auto Max Mode
                                    max_index = self.spectrum[1, :].argmax()
                                    freq = self.spectrum[0, max_index]
                                    self.vfo_freq[i] = freq + self.module_receiver.daq_center_freq

                                decimation_factor = max(
                                    (sampling_freq // self.vfo_bw[i]), 1
                                )  # How much decimation is required to get to the requested bandwidth

                                # Get max amplitude of the channel from the FFT for squelching
                                # From channel frequency determine array index of channel
                                vfo_width_idx = int(
                                    (spectrum_window_size * self.vfo_bw[i]) / (sampling_freq)
                                )  # Width of channel in array indexes based on FFT size
                                vfo_width_idx = max(vfo_width_idx, 2)

                                freqMin = -sampling_freq / 2

                                vfo_center_idx = int((((freq - freqMin) * spectrum_window_size) / sampling_freq))

                                vfo_upper_bound = vfo_center_idx + vfo_width_idx // 2
                                vfo_lower_bound = vfo_center_idx - vfo_width_idx // 2

                                if self.spectrum_fig_type == "Single":  # Do CH1 only (or make channel selectable)
                                    spectrum_channel = self.spectrum[
                                        1,
                                        max(vfo_lower_bound, 0) : min(vfo_upper_bound, spectrum_window_size),
                                    ]
                                    max_amplitude = np.max(spectrum_channel)
                                else:
                                    spectrum_channel = self.spectrum[
                                        :,
                                        max(vfo_lower_bound, 0) : min(vfo_upper_bound, spectrum_window_size),
                                    ]
                                    max_amplitude = np.max(
                                        spectrum_channel[
                                            1 : self.module_receiver.iq_header.active_ant_chs + 1,
                                            :,
                                        ]
                                    )

                                # *** HERE WE NEED TO PERFORM THE SPECTRUM UPDATE TOO ***
                                if self.en_spectrum:
                                    # Selected Channel Window
                                    signal_window = np.zeros(spectrum_window_size) - 120
                                    signal_window[
                                        max(vfo_lower_bound, 4) : min(vfo_upper_bound, spectrum_window_size - 4)
                                    ] = 0  # max_amplitude
                                    self.spectrum[self.channel_number + (2 * i + 1), :] = (
                                        signal_window  # np.ones(len(spectrum[1,:])) * self.module_receiver.daq_squelch_th_dB # Plot threshold line
                                    )

                                    # Squelch Window
                                    signal_window[
                                        max(vfo_lower_bound, 4) : min(vfo_upper_bound, spectrum_window_size - 4)
                                    ] = self.vfo_squelch[i]
                                    self.spectrum[self.channel_number + (2 * i + 2), :] = (
                                        signal_window  # np.ones(len(spectrum[1,:])) * self.module_receiver.daq_squelch_th_dB # Plot threshold line
                                    )

                                # -----> DoA ESIMATION <-----

                                # datetime object containing current date and time
                                now = datetime.now()
                                now_dt_str = now.strftime("%d-%b-%Y_%Hh%Mm%Ss")
                                if (
                                    self.en_DOA_estimation
                                    and self.channel_number > 1
                                    and max_amplitude > self.vfo_squelch[i]
                                    and (i == self.output_vfo or self.output_vfo < 0)
                                ):
                                    write_freq = int(self.vfo_freq[i])
                                    # Do channelization
                                    if self.vfo_demod_modes[i] == "FM":
                                        decimate_sampling_freq = 48_000
                                        decimation_factor = int(sampling_freq / decimate_sampling_freq)

                                    fir_order_factor = max(self.vfo_fir_order_factor[i], DEFAULT_VFO_FIR_ORDER_FACTOR)
                                    vfo_channel = channelize(
                                        self.processed_signal,
                                        freq,
                                        decimation_factor,
                                        fir_order_factor,
                                        sampling_freq,
                                    )
                                    iq_channel = vfo_channel[1]

                                    # Method to check IQ diffs when noise source forced ON
                                    # iq_diffs = calc_sync(self.processed_signal)
                                    # print("IQ DIFFS: " + str(iq_diffs))
                                    # print("IQ DIFFS ANGLE: " + str(np.rad2deg(np.angle(iq_diffs))))
                                    #
                                    theta_0 = self.estimate_DOA(vfo_channel, self.vfo_freq[i])

                                    if not numba_isfinite(self.DOA):
                                        logging.error("""Estimated DOA is not finite.""")
                                        continue

                                    doa_result_log = DOA_plot_util(self.DOA)
                                    conf_val = calculate_doa_papr(self.DOA)

                                    self.doa_max_list[i] = theta_0
                                    update_list[i] = True

                                    # DOA_str = str(int(theta_0))
                                    DOA_str = str(int(360 - theta_0))
                                    confidence_str = "{:.2f}".format(np.max(conf_val))
                                    max_power_level_str = "{:.1f}".format((np.maximum(-100, max_amplitude)))

                                    self.theta_0_list.append(theta_0)
                                    self.confidence_list.append(np.max(conf_val))
                                    self.max_power_level_list.append(np.maximum(-100, max_amplitude))
                                    self.freq_list.append(write_freq)
                                    self.doa_result_log_list.append(doa_result_log)

                                    if self.vfo_demod_modes[i] or self.vfo_iq_enabled[i]:
                                        if theta_0 not in self.vfo_theta_channel[i]:
                                            self.vfo_theta_channel[i].append(theta_0)

                                    self.vfo_time[i] += self.processed_signal[1].size / sampling_freq
                                    if 0 < self.max_demod_timeout < self.vfo_time[i] and (
                                        self.vfo_demod_modes[i] == "FM" or self.vfo_iq_enabled[i]
                                    ):
                                        self.vfo_demod_channel[i] = np.array([])
                                        self.vfo_theta_channel[i] = []
                                        self.vfo_iq_channel[i] = np.array([])
                                    elif self.vfo_demod_modes[i] == "FM":
                                        fm_demod_channel = fm_demod(iq_channel, decimate_sampling_freq, self.vfo_bw[i])
                                        self.vfo_demod_channel[i] = np.concatenate(
                                            (self.vfo_demod_channel[i], fm_demod_channel)
                                        )
                                    elif self.vfo_iq_enabled[i]:
                                        self.vfo_iq_channel[i] = np.concatenate((self.vfo_iq_channel[i], iq_channel))
                                else:
                                    self.vfo_time[i] = 0
                                    self.vfo_blocked[i] = False
                                    fm_demod_channel = self.vfo_demod_channel[i]
                                    iq_channel = self.vfo_iq_channel[i]
                                    thetas = self.vfo_theta_channel[i]
                                    vfo_freq = int(self.vfo_freq[i])
                                    self.fm_demod_channel_list.append(
                                        (now_dt_str, vfo_freq, fm_demod_channel, iq_channel, thetas)
                                    )
                                    self.vfo_demod_channel[i] = np.array([])
                                    self.vfo_theta_channel[i] = []
                                    self.vfo_iq_channel[i] = np.array([])

                            que_data_packet.append(["doa_thetas", self.DOA_theta])
                            que_data_packet.append(["DoA Result", doa_result_log])
                            que_data_packet.append(["DoA Max", theta_0])
                            que_data_packet.append(["DoA Confidence", conf_val])
                            que_data_packet.append(["DoA Squelch", update_list])
                            que_data_packet.append(["DoA Max List", self.doa_max_list])
                            if self.vfo_mode == "Auto":
                                que_data_packet.append(["VFO-0 Frequency", self.vfo_freq[0]])

                            def adjust_theta(theta):
                                if self.doa_measure == "Compass":
                                    return (360 - theta + self.compass_offset) % 360
                                else:
                                    return theta

                            def average_thetas(thetas):
                                avg_theta = sum(thetas) / len(thetas)
                                diff_thetas = copy.copy(thetas)
                                for i in range(len(diff_thetas)):
                                    diff_thetas[i] = abs(diff_thetas[i] - avg_theta)

                                return avg_theta, max(diff_thetas)

                            for (
                                now_dt_str,
                                vfo_freq,
                                fm_demod_channel,
                                iq_channel,
                                thetas,
                            ) in self.fm_demod_channel_list:
                                store_demod_channel = fm_demod_channel.size > 0
                                store_iq_channel = iq_channel.size > 0
                                if ((not store_demod_channel) and (not store_iq_channel)) or (not thetas):
                                    continue
                                avg_theta, max_diff_theta = average_thetas(thetas)
                                if max_diff_theta > 10:
                                    doa_max_str = []
                                    for theta in thetas:
                                        doa_max_str.append(f"{adjust_theta(theta):.1f}")
                                    doa_max_str = "_".join(doa_max_str)
                                else:
                                    doa_max_str = f"{adjust_theta(avg_theta):.1f}"

                                if store_demod_channel:
                                    record_file_name = f"{now_dt_str},FM_{vfo_freq / 1e6:.3f}MHz"
                                    filename = f"{self.wav_record_path}/{record_file_name},DOA_{doa_max_str}.wav"
                                    if can_store_file(self.wav_record_path):
                                        write_wav(
                                            filename,
                                            48_000,
                                            fm_demod_channel,
                                        )
                                    else:
                                        self.logger.error(
                                            "No disk space left for storing %s, demodulation and recording disabled.",
                                            filename,
                                        )
                                        self.vfo_demod[:] = ["None"] * len(self.vfo_demod)
                                if store_iq_channel:
                                    record_file_name = f"{now_dt_str},IQ_{vfo_freq / 1e6:.3f}MHz"
                                    filename = f"{self.iq_record_path}/{record_file_name},DOA_{doa_max_str}.iq"
                                    if can_store_file(self.iq_record_path):
                                        iq_channel.tofile(filename)
                                    else:
                                        self.logger.error(
                                            "No disk space left for storing %s, IQ recording disabled.", filename
                                        )
                                        self.vfo_iq[:] = ["False"] * len(self.vfo_iq)
                    except Exception:
                        self.logger.error(traceback.format_exc())
                        self.data_ready = False

                    # -----> SPECTRUM PROCESSING <-----
                    if self.en_spectrum and self.data_ready:
                        spectrum_plot_data = reduce_spectrum(
                            self.spectrum, self.spectrum_plot_size, self.channel_number
                        )
                        que_data_packet.append(["spectrum", spectrum_plot_data])

                    daq_cpi = int(
                        self.module_receiver.iq_header.cpi_length * 1000 / self.module_receiver.iq_header.sampling_freq
                    )
                    # We don't include processing latency here, because reported timestamp marks end of the data frame
                    # so latency is essentially an acquisition time.
                    self.latency = daq_cpi
                    self.processing_time = int(1000 * (time.time() - start_time))

                    if self.data_ready and self.theta_0_list:
                        message = ""
                        for j, freq in enumerate(self.freq_list):
                            sub_message = ""
                            sub_message += f"{self.timestamp}, {360 - self.theta_0_list[j]}, {self.confidence_list[j]}, {self.max_power_level_list[j]}, "
                            sub_message += f"{freq}, {self.DOA_ant_alignment}, {self.latency}"

                            doa_result_log = self.doa_result_log_list[j] + np.abs(np.min(self.doa_result_log_list[j]))
                            for i in range(len(doa_result_log)):
                                sub_message += ", " + "{:.2f}".format(doa_result_log[i])

                            sub_message += " \n"

                            if self.en_data_record:
                                time_elapsed = time.time() - self.last_write_time[j]
                                if time_elapsed > self.write_interval:
                                    self.last_write_time[j] = time.time()
                                    self.data_record_fd.write(sub_message)

                            message += sub_message

                stop_time = time.time()

                que_data_packet.append(["update_rate", stop_time - start_time])
                que_data_packet.append(
                    [
                        "latency",
                        (
                            (int(stop_time * 10**3) - self.module_receiver.iq_header.time_stamp)
                            if not get_iq_failed
                            else 0
                        ),
                    ]
                )

                # Put data into buffer, but if there is no data because its a cal/trig wait frame etc, then only write if the buffer is empty
                # Otherwise just discard the data so that we don't overwrite good DATA frames.
                try:
                    self.data_que.put(
                        que_data_packet, False
                    )  # Must be non-blocking so DOA can update when dash browser window is closed
                except Exception:
                    # Discard data, UI couldn't consume fast enough
                    pass

    def estimate_DOA(self, processed_signal, vfo_freq):
        """
        Estimates the direction of arrival of the received RF signal
        """

        antennas_alignment = "UCA"

        # Calculating spatial correlation matrix
        R = corr_matrix(processed_signal)
        M = R.shape[0]

        if self.DOA_decorrelation_method == "FBA":
            R = de.forward_backward_avg(R)
        elif self.DOA_decorrelation_method == "TOEP":
            R = toeplitzify(R)
        elif self.DOA_decorrelation_method == "FBSS":
            smoothing_degree = 1
            subarray_size = M - smoothing_degree
            if subarray_size > 1:
                R = de.spatial_smoothing(processed_signal.T, subarray_size, "forward-backward")
            else:
                # Too few channels for spatial smoothing, skipping it.
                pass

        M = R.shape[0]

        # If rank of the correlation matrix is not equal to its full one,
        # then we are likely dealing with correlated sources and (or) low SNR signals
        number_of_correlated_sources = M - np.linalg.matrix_rank(R)
        self.number_of_correlated_sources.append(number_of_correlated_sources)
        snr = SNR(R)
        self.snrs.append(snr)

        frq_ratio = vfo_freq / self.module_receiver.daq_center_freq
        inter_element_spacing = self.DOA_inter_elem_space * frq_ratio

        scanning_vectors = gen_scanning_vectors(M, inter_element_spacing, antennas_alignment, int(self.array_offset))

        # DOA estimation
        if self.DOA_algorithm == "Bartlett":  # self.en_DOA_Bartlett:
            DOA_Bartlett_res = de.DOA_Bartlett(R, scanning_vectors)
            self.DOA = DOA_Bartlett_res
        if self.DOA_algorithm == "Capon":  # self.en_DOA_Capon:
            DOA_Capon_res = de.DOA_Capon(R, scanning_vectors)
            self.DOA = DOA_Capon_res
        if self.DOA_algorithm == "MEM":  # self.en_DOA_MEM:
            DOA_MEM_res = de.DOA_MEM(R, scanning_vectors, column_select=0)
            self.DOA = DOA_MEM_res
        if self.DOA_algorithm == "TNA":
            self.DOA = DOA_TNA(R, scanning_vectors)
        if self.DOA_algorithm == "MUSIC":  # self.en_DOA_MUSIC:
            DOA_MUSIC_res = DOA_MUSIC(
                R, scanning_vectors, signal_dimension=self.DOA_expected_num_of_sources
            )  # de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
            self.DOA = DOA_MUSIC_res
        theta_0 = self.DOA_theta[np.argmax(self.DOA)]

        return theta_0

    def update_recording_filename(self, filename):
        self.data_record_fd.close()
        self.data_recording_file_name = filename
        data_recording_file_path = os.path.join(os.path.join(self.root_path, self.data_recording_file_name))
        self.data_record_fd = open(data_recording_file_path, "a+")
        self.en_data_record = False

    def get_recording_filesize(self):
        return round(
            os.path.getsize(os.path.join(os.path.join(self.root_path, self.data_recording_file_name))) / 1048576,
            2,
        )  # Convert to MB


def calc_sync(iq_samples):
    iq_diffs = np.ones(4, dtype=np.complex64)

    # Calculate Spatial correlation matrix to determine amplitude-phase missmatches
    Rxx = iq_samples.dot(np.conj(iq_samples.T))
    # Perform eigen-decomposition
    eigenvalues, eigenvectors = lin.eig(Rxx)
    # Get dominant eigenvector
    max_eig_index = np.argmax(np.abs(eigenvalues))
    vmax = eigenvectors[:, max_eig_index]
    iq_diffs = 1 / vmax
    iq_diffs /= iq_diffs[0]

    return iq_diffs


# Reduce spectrum size for plotting purposes by taking the MAX val every few values
# Significantly faster with numba once we added nb.prange
@njit(fastmath=True, cache=True)
def reduce_spectrum(spectrum, spectrum_size, channel_number):
    spectrum_elements = len(spectrum[:, 0])

    spectrum_plot_data = np.zeros((spectrum_elements, spectrum_size), dtype=np.float32)
    group = len(spectrum[0, :]) // spectrum_size
    for m in nb.prange(spectrum_elements):
        for i in nb.prange(spectrum_size):
            spectrum_plot_data[m, i] = np.max(spectrum[m, i * group : group * (i + 1)])
    return spectrum_plot_data


# Get the FIR filter
@lru_cache(maxsize=32)
def get_fir(n, q, padd):
    return signal.dlti(signal.firwin(n, 1.0 / (q * padd), window="hann"), 1.0)


# Get the frequency rotation exponential
@lru_cache(maxsize=32)
def get_exponential(freq, sample_freq, sig_len):
    # Auto shift peak frequency center of spectrum, this frequency will be decimated:
    # https://pysdr.org/content/filters.html
    f0 = -freq  # +10
    Ts = 1.0 / sample_freq
    t = np.arange(0.0, Ts * sig_len, Ts)
    exponential = np.exp(2j * np.pi * f0 * t)  # this is essentially a complex sine wave

    return np.ascontiguousarray(exponential)


@njit(fastmath=True, cache=True)
def numba_mult(a, b):
    return a * b


@njit(cache=True)
def numba_isfinite(a):
    return np.all(np.isfinite(a))


# Memoize the total shift filter
@lru_cache(maxsize=32)
def shift_filter(decimation_factor, fir_order_factor, freq, sampling_freq, padd):
    fir_order = decimation_factor * fir_order_factor
    fir_order = fir_order + (fir_order - 1) % 2
    system = get_fir(fir_order, decimation_factor, padd)
    b = system.num
    a = system.den
    exponential = get_exponential(-freq, sampling_freq, len(b))
    b = numba_mult(b, exponential)
    return signal.dlti(b, a)


# This function takes the full data, and efficiently returns only a filtered and decimated requested channel
# Efficient method: Create BANDPASS Filter for frequency of interest, decimate with that bandpass filter, then do the final shift
def channelize(processed_signal, freq, decimation_factor, fir_order_factor, sampling_freq):
    system = shift_filter(
        decimation_factor, fir_order_factor, freq, sampling_freq, 1.1
    )  # Decimate with a BANDPASS filter
    decimated = signal.decimate(processed_signal, decimation_factor, ftype=system)
    exponential = get_exponential(
        freq, sampling_freq / decimation_factor, len(decimated[0, :])
    )  # Shift the signal AFTER to get back to normal decimate behaviour
    return numba_mult(decimated, exponential)

    # Old Method
    # Auto shift peak frequency center of spectrum, this frequency will be decimated:
    # https://pysdr.org/content/filters.html
    # f0 = -freq #+10
    # Ts = 1.0/sample_freq
    # t = np.arange(0.0, Ts*len(processed_signal[0, :]), Ts)
    # exponential = np.exp(2j*np.pi*f0*t) # this is essentially a complex sine wave

    # Decimate down to BW
    # decimation_factor = max((sample_freq // bw), 1)
    # decimated_signal = signal.decimate(processed_signal, decimation_factor, n = decimation_factor * 2, ftype='fir')

    # return decimated_signal


# NUMBA optimized Thermal Noise Algorithm (TNA) function.
# Based on `pyargus` DOA_Capon
@njit(fastmath=True, cache=True)
def DOA_TNA(R, scanning_vectors):
    # --> Input check

    if R.shape[0] != scanning_vectors.shape[0]:
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return np.ones(1, dtype=np.complex64) * -2

    ADSINR = np.zeros(scanning_vectors.shape[1], dtype=np.complex64)

    # TODO: perhaps we can store scanning_vectors in column-major order from the very begining to
    # avoid such conversion?
    S_ = np.asfortranarray(scanning_vectors)

    # --- Calculation ---
    try:
        R_inv_2 = np.linalg.matrix_power(R, -2)
    except np.linalg.LinAlgError:
        print("ERROR: Singular or non-square matrix")
        return np.ones(1, dtype=np.complex64) * -3

    # TODO: it seems like rising correlation matrix to power benefits from added precision.
    # This might be artifact of the testing and if it is then we can switching the whole
    # processing chain from double to single precision for considerable performance uplift,
    # especially on low grade hardware.
    R_inv_2 = R_inv_2.astype(np.complex64)

    for i in range(scanning_vectors.shape[1]):
        S_theta_ = S_[:, i]
        ADSINR[i] = np.dot(np.conj(S_theta_), np.dot(R_inv_2, S_theta_))

    ADSINR = np.reciprocal(ADSINR)

    return ADSINR


# NUMBA optimized MUSIC function. About 100x faster on the Pi 4
# @njit(fastmath=True, cache=True, parallel=True)
@njit(fastmath=True, cache=True)
def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution=1):
    # --> Input check
    if R[:, 0].size != R[0, :].size:
        print("ERROR: Correlation matrix is not quadratic")
        return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

    if R[:, 0].size != scanning_vectors[:, 0].size:
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return np.ones(1, dtype=np.complex64) * -2

    ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
    M = R[:, 0].size  # np.size(R, 0)

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    sigmai = np.abs(sigmai)

    idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
    vi = vi[:, idx]

    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension

    E = np.empty((M, noise_dimension), dtype=np.complex64)
    for i in range(noise_dimension):
        E[:, i] = vi[:, i]

    E_ct = E @ E.conj().T
    theta_index = 0
    for i in range(scanning_vectors[0, :].size):
        S_theta_ = scanning_vectors[:, i]
        S_theta_ = np.ascontiguousarray(S_theta_.T)
        ADORT[theta_index] = 1 / np.abs(S_theta_.conj().T @ E_ct @ S_theta_)
        theta_index += 1

    return ADORT


# Rather naive way to estimate SNR (in dBs) based on the assumption that largest and smallest eigenvalues
# of the correlation matrix corresponds to the powers of the signal plus noise  and noise respectively.
# Even though it won't estimate SNR beyond dominant signal, if it is already quite small,
# then any additional signals have even lower SNR.
def SNR(R: np.ndarray) -> float:
    ev = np.abs(scipy.linalg.eigvals(R))
    ev.sort()
    noise_power = ev[0]
    signal_plus_noise_power = ev[-1]
    power_ratio = (signal_plus_noise_power - noise_power) / noise_power
    snr = 10.0 * np.log10(power_ratio)
    return snr


def xi(uca_radius_m: float, frequency_Hz: float) -> Tuple[float, int]:
    wavelength_m = scipy.constants.speed_of_light / frequency_Hz
    x = 2.0 * np.pi * uca_radius_m / wavelength_m
    L = int(np.floor(x))
    return x, L


# Numba optimized version of pyArgus corr_matrix_estimate with "fast". About 2x faster on Pi4
# @njit(fastmath=True, cache=True)
def corr_matrix(X: np.ndarray) -> np.ndarray:
    N = X[0, :].size
    R = np.dot(X, X.conj().T)
    R = np.divide(R, N)
    return R


# This is so-called "Rectification" or "Toeplizification" of correlation matrix method
# investigated by P. Vallet and P. Loubaton, "Toeplitz rectification and DOA estimation with MUSIC",
# 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
# Florence, Italy, 2014, pp. 2237-2241, doi: 10.1109/ICASSP.2014.6853997. and references therein.
def toeplitzify(R: np.ndarray) -> np.ndarray:
    M = R.shape[0]
    ms = np.arange(0, -M, -1, dtype=int)
    c = [1.0 / (float(M - abs(m))) * np.trace(R, m) for m in ms]
    return scipy.linalg.toeplitz(c)


# This is one of so-called "Toeplitz Reconstruction" of correlation matrix methods
# investigated A. M. McDonald and M. A. van Wyk,
# "A Condition for Unbiased Direction-of-Arrival Estimation with Toeplitz Decorrelation Techniques",
# 2019 IEEE Asia Pacific Conference on Postgraduate Research in Microelectronics and Electronics (PrimeAsia),
# Bangkok, Thailand, 2019, pp. 45-48, doi: 10.1109/PrimeAsia47521.2019.8950749. and references therein.
# with important addition of the F-B averaging suggested by R. M. Shubair, et al.,
# "A new technique for UCA-based DOA estimation of coherent signals,"
# 2016 16th Mediterranean Microwave Symposium (MMS),
# Abu Dhabi, United Arab Emirates, 2016, pp. 1-3, doi: 10.1109/MMS.2016.7803806. and references therein.
def fb_toeplitz_reconstruction(R: np.ndarray) -> np.ndarray:
    R_f = scipy.linalg.toeplitz(R[:, 0], R[0, :])
    R_b = scipy.linalg.toeplitz(np.flip(R[:, -1]), np.flip(R[-1, :]))
    return 0.5 * (R_f + R_b.conj())


# LRU cache memoize about 1000x faster.
@lru_cache(maxsize=32)
def gen_scanning_vectors(M, DOA_inter_elem_space, type, offset):
    thetas = np.linspace(
        0, 359, 360
    )  # Remember to change self.DOA_thetas too, we didn't include that in this function due to memoization cannot work with arrays
    if type != "UCA":
        type = "UCA"

    # convert UCA inter element spacing back to its radius
    to_r = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2.0 * np.pi / M)))
    r = DOA_inter_elem_space * to_r
    x = r * np.cos(2 * np.pi / M * np.arange(M))
    y = -r * np.sin(2 * np.pi / M * np.arange(M))  # For this specific array only

    scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
    for i in range(thetas.size):
        scanning_vectors[:, i] = np.exp(
            1j * 2 * np.pi * (x * np.cos(np.deg2rad(thetas[i] + offset)) + y * np.sin(np.deg2rad(thetas[i] + offset)))
        )

    return np.ascontiguousarray(scanning_vectors)


@njit(fastmath=True, cache=True)
def DOA_plot_util(DOA_data, log_scale_min=-100):
    """
    This function prepares the calulcated DoA estimation results for plotting.

    - Noramlize DoA estimation results
    - Changes to log scale
    """
    # Normalization
    max_doa_amplitude = np.max(np.abs(DOA_data))
    DOA_data = (np.abs(DOA_data) / max_doa_amplitude) if max_doa_amplitude > NEAR_ZERO else np.abs(DOA_data)

    # Change to logscale
    DOA_data = 10 * np.log10(DOA_data)

    for i in range(len(DOA_data)):  # Remove extremely low values
        if DOA_data[i] < log_scale_min:
            DOA_data[i] = log_scale_min

    return DOA_data


@njit(fastmath=True, cache=True)
def calculate_doa_papr(DOA_data):
    mean_doa_amplitude = np.mean(np.abs(DOA_data))
    return 10 * np.log10(np.max(np.abs(DOA_data)) / mean_doa_amplitude) if mean_doa_amplitude > NEAR_ZERO else 0.0


# Old time-domain squelch algorithm (Unused as freq domain FFT with overlaps gives significantly better sensitivity with acceptable time resolution expense
"""
    K = 10
    self.filtered_signal = self.raw_signal_amplitude #convolve(np.abs(self.raw_signal_amplitude),np.ones(K), mode = 'same')/K

    # Burst is always started at the begining of the processed block, ensured by the squelch module in the DAQ FW
    burst_stop_index  = len(self.filtered_signal) # CARL FIX: Initialize this to the length of the signal, incase the signal is active the entire time
    self.logger.info("Original burst stop index: {:d}".format(burst_stop_index))

    min_burst_size = K
    burst_stop_amp_val = 0
    for n in np.arange(K, len(self.filtered_signal), 1):
        if self.filtered_signal[n] < self.squelch_threshold:
            burst_stop_amp_val = self.filtered_signal[n]
            burst_stop_index = n
            burst_stop_index-=K # Correction with the length of filter
            break

        #burst_stop_index-=K # Correction with the length of filter


    self.logger.info("Burst stop index: {:d}".format(burst_stop_index))
    self.logger.info("Burst stop ampl val: {:f}".format(burst_stop_amp_val))
    self.logger.info("Processed signal length: {:d}".format(len(self.processed_signal[0,:])))

    # If sign
    if burst_stop_index < min_burst_size:
        self.logger.debug("The length of the captured burst size is under the minimum: {:d}".format(burst_stop_index))
        burst_stop_index = 0

    if burst_stop_index !=0:
        self.logger.info("INSIDE burst_stop_index != 0")

       self.logger.debug("Burst stop index: {:d}".format(burst_stop_index))
       self.logger.debug("Burst stop ampl val: {:f}".format(burst_stop_amp_val))
       self.squelch_mask = np.zeros(len(self.filtered_signal))
       self.squelch_mask[0 : burst_stop_index] = np.ones(burst_stop_index)*self.squelch_threshold
       # Next line removes the end parts of the samples after where the signal ended, truncating the array
       self.processed_signal = self.module_receiver.iq_samples[: burst_stop_index, self.squelch_mask == self.squelch_threshold]
       self.logger.info("Raw signal length when burst_stop_index!=0: {:d}".format(len(self.module_receiver.iq_samples[0,:])))
       self.logger.info("Processed signal length when burst_stop_index!=0: {:d}".format(len(self.processed_signal[0,:])))

       #self.logger.info(' '.join(map(str, self.processed_signal)))

       self.data_ready=True
   else:
       self.logger.info("Signal burst is not found, try to adjust the threshold levels")
       #self.data_ready=True
       self.squelch_mask = np.ones(len(self.filtered_signal))*self.squelch_threshold
       self.processed_signal = np.zeros([self.channel_number, len(self.filtered_signal)])
"""
