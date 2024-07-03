# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np
from copy import deepcopy
import pickle

TS_PARAMS_F_HZ              = "Frequency (Hz)"
TS_PARAMS_SAMPLE_B          = "Size per sample (bits)"
TS_PARAMS_LENGTH_S          = "Length (s)"
TS_PARAMS_START_S           = "Start time (s)"
TS_PARAMS_END_S             = "End time (s)"
TS_PARAMS_PHASE_DEG         = "Sampling phase (°)"
TS_PARAMS_OFFSET_B          = "Input signal offset (bits)"
TS_PARAMS_POWER_W           = "Sampling power (W)"
TS_PARAMS_EPC_J             = "Energy per conversion (J)"
TS_PARAMS_STEP_HISTORY      = "Step history"
TS_PARAMS_LATENCY_HISTORY   = "Latency history"
TS_PARAMS_AIDI              = "AIDI"
TS_PARAMS_DR_BPS            = "Datarate (bps)"

TS_PARAMS_INPUT_SERIES      = "Input series"
TS_PARAMS_OPERATION         = "Operation"



class Timeseries:
    """
    Class for representing time series data.

    Attributes:
        name (str): Name of the time series.
        data (list): List of data points.
        time (list): List of time points.
        f_Hz (float): Frequency of the time series.
        length_s (float): Length of the time series in seconds.
        sample_b (int): Size per sample in bits.
        params (dict): Dictionary containing additional parameters.
        scores (dict): Dictionary containing scores associated with the time series.
    """

    def __init__(self,
                name,
                data=None,
                time=None,
                f_Hz=0,
                length_s=0,
                ):
        """
        Initializes a TimeSeries object.

        Args:
            name (str): Name of the time series.
            data (list, optional): List of data points. Defaults to None.
            time (list, optional): List of time points. Defaults to None.
            f_Hz (float, optional): Frequency of the time series. Defaults to 0.
            length_s (float, optional): Length of the time series in seconds. Defaults to 0.
            sample_b (int, optional): Size per sample in bits. Defaults to 0.
        """
        self.name = name
        self.data = np.array(data, dtype=np.float32) if data is not None else np.array([], dtype=np.float32)
        self.time = np.array(time, dtype=np.float32) if time is not None else np.array([], dtype=np.float32)

        if time is None and length_s != 0:
            T_s = length_s / len(data)
            f_Hz = 1.0 / T_s
            time = np.arange(0, length_s, T_s, dtype=np.float32)
        elif time is not None and f_Hz == 0:
            f_Hz = 1.0 / (time[1] - time[0])
            length_s = len(time) / f_Hz
        elif time is not None:
            length_s = len(time) / f_Hz

        self.params = {
            TS_PARAMS_F_HZ: f_Hz,
            TS_PARAMS_LENGTH_S: length_s,
            TS_PARAMS_STEP_HISTORY: [],
            TS_PARAMS_LATENCY_HISTORY: [],
        }


    def __str__(self):
        """
        Returns a string representation of the time series object.
        """
        return self.name

    def export(self, path="../out/", name=""):
        """
        Exports the time series data to a text file.

        Args:
            path (str, optional): Path to export the file. Defaults to "../out/".
            name (str, optional): Name of the exported file. Defaults to "".
        """
        if name == "":
            name = self.name.replace(" ", "_")
        with open(path + name + ".timeseries", 'w+') as f:
            for t, d in zip(self.time, self.data):
                f.write(f"{t}, {d}\n")

    def export_bin(self, outfile="../out/", bytes=4, bigendian=False):
        """
        Exports the time series data to a binary file.

        Args:
            outfile (str, optional): Path to export the file. Defaults to "../out/".
            bytes (int, optional): Size of each data point in bytes. Defaults to 4.
            bigendian (bool, optional): Indicates whether to use big-endian byte order. Defaults to False.
        """
        if outfile == "../out/":
            outfile += self.name.replace(" ", "_")
        # Save the array to a binary file
        if bytes == 4:
            wordsize = np.int32
        elif bytes == 2:
            wordsize = np.int16
        elif bytes == 1:
            wordsize = np.int8
        else:
            raise ValueError("Invalid word size in bytes! Choose between 1, 2 and 4")
        data = np.array(self.data).astype(wordsize)
        with open(outfile + ".bin", 'wb') as f:
            if bigendian:
                data.byteswap(True).tofile(f)
            else:
                data.tofile(f)

    def dump(self, path="../out/", name=""):
        """
        Dumps the TimeSeries object to a pickle file.

        Args:
            path (str, optional): Path to dump the file. Defaults to "../out/".
            name (str, optional): Name of the dumped file. Defaults to "".
        """
        if name == "":
            name = self.name.replace(" ", "_")
        from copy import deepcopy as cp
        cop = cp(self)
        cop.data = np.asarray(cop.data)
        with open(path + name + ".pkl", 'wb') as f:
            pickle.dump(cop, f)

    @classmethod
    def load(cls, filename):
        """
        Loads a TimeSeries object from a pickle file.

        Args:
            filename (str): Path to the pickle file.

        Returns:
            Timeseries: Loaded TimeSeries object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def copy(self):
        """
        Creates a deep copy of the TimeSeries object.

        Returns:
            Timeseries: Deep copy of the TimeSeries object.
        """
        return deepcopy(self)

    def print_params(self):
        """
        Prints the parameters of the TimeSeries object.
        """
        for k, v in self.params.items():
            print(f"{k}\t{v}")

    def min_bits_required(self):
        """
        Calculates the minimum number of bits required for data and time points.

        Returns:
            tuple: A tuple containing the minimum number of bits required for data and time points, respectively.
        """
        try:
            max_d, max_t = int(max(self.data)), int(max(self.time))
        except:
            return 0, 0
        d_b, t_b = max_d.bit_length(), max_t.bit_length()
        return d_b, t_b


def copy_series(series):
    """
    Creates a deep copy of the input time series.

    Args:
        series (Timeseries): Input time series to be copied.

    Returns:
        Timeseries: Deep copy of the input time series.
    """
    return deepcopy(series)
