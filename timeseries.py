# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np
from copy import deepcopy
import pickle
import hashlib

from ts_params import *

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

    def accumulate_param(self,  k, v):
        self.params[k] = self.params.get(k, 0) + v

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

    def generate_unique_id(self):
        """
        Generates a unique ID for the timeseries by encoding its parameters keys and values.
        It excludes the ID from the parameters to be encoded.
        """
        filtered_params = {key: value for key, value in self.params.items() if key != 'TS_PARAMS_ID'}
        params_string = str(sorted(filtered_params.items())).encode()
        self.params[TS_PARAMS_ID] = hashlib.md5(params_string).hexdigest()
        self.params[TS_PARAMS_SHORT_ID] = self.params[TS_PARAMS_ID][-5:]

def copy_series(series):
    """
    Creates a deep copy of the input time series.

    Args:
        series (Timeseries): Input time series to be copied.

    Returns:
        Timeseries: Deep copy of the input time series.
    """
    return deepcopy(series)

catalog = {}

def reset_catalog():
    catalog = {}

def update_catalog( series_list ):
    # Create a dictionary to map IDs to TimeSeries objects for O(1) lookup time.
    catalog_additions = {series.params[TS_PARAMS_ID]: series for series in series_list}
    catalog.update(catalog_additions)

def get_series_by_id(id):
    """
    Given a catalog of TimeSeries objects and a target ID, find and return the TimeSeries with that ID.

    Args:
        catalog (list of TimeSeries): The list containing TimeSeries objects.
        id (str): The unique identifier to search for.

    Returns:
        TimeSeries: The TimeSeries object with the matching ID or None if not found.
    """
    # Return the TimeSeries object with the given ID, or None if the ID is not in the dictionary.
    return catalog.get(id, None)

def save_series(series, filename, input_series = []):
    import dill as pickle
    print("💾 Saving...")
    sizes_MB    = np.array([len(pickle.dumps(s.data))+len(pickle.dumps(s.time)) for s in series])/(1024**2)
    summ_MB     = len(pickle.dumps(series))/(1024**2)
    avg_size_MB = np.average(sizes_MB)
    std_size_MB = np.std(sizes_MB)
    print(f"Size: {summ_MB:0.1f} MB, {len(sizes_MB)} × avg: {avg_size_MB:0.1f} MB (std: {std_size_MB:0.1f} MB)")

    # plot_sizes_and_params(series)

    with open( filename, 'wb') as f:
        pickle.dump( series, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Steps saved in {filename}")


def load_series(filename):
    import dill as pickle
    print("📂 Loading...")
    with open(filename, "rb") as f:
        series = pickle.load(f)

    update_catalog( series )
    return series