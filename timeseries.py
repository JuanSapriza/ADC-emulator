# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np
from copy import deepcopy
import pickle

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


def copy_series(series):
    """
    Creates a deep copy of the input time series.

    Args:
        series (Timeseries): Input time series to be copied.

    Returns:
        Timeseries: Deep copy of the input time series.
    """
    return deepcopy(series)


def save_series(series, filename, input_series = []):
    import dill as pickle

    print("💾 Saving...")

    def calculate_sizes_and_params(series):
        sizes_MB = []
        all_attributes = set()
        params_sizes = {}  # This will store cumulative sizes of each key in `params`

        # Measure the size of each attribute and each param for each instance
        for s in series:
            attr_sizes = {}
            for attr_name, attr_value in s.__dict__.items():
                if attr_name == 'params' and isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        size_mb = len(pickle.dumps(value)) / (1024**2)
                        if key in params_sizes:
                            params_sizes[key].append(size_mb)
                        else:
                            params_sizes[key] = [size_mb]
                else:
                    attr_size_mb = len(pickle.dumps(attr_value)) / (1024**2)
                    attr_sizes[attr_name] = attr_size_mb
                    all_attributes.add(attr_name)
            # if sum(attr_sizes.values()) > 1: print(f"\n❗❗LARGE SERIES DETECTED! {sum(attr_sizes.values()):0.2f} MB -- {s.params[TS_PARAMS_STEP_HISTORY][-1]}")
            sizes_MB.append(attr_sizes)

        # Calculate average sizes for params
        avg_params_sizes = {key: np.mean(values) for key, values in params_sizes.items()}
        return sizes_MB, all_attributes, avg_params_sizes

    def plot_sizes_and_params(series):
        import matplotlib.pyplot as plt
        sizes_MB, all_attributes, avg_params_sizes = calculate_sizes_and_params(series)
        indices = range(len(series))

        # Bar chart for attributes
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)  # First subplot for the bar chart
        bottoms = np.zeros(len(series))

        for attr in sorted(all_attributes):
            attr_sizes = [item.get(attr, 0) for item in sizes_MB]
            plt.bar(indices, attr_sizes, bottom=bottoms, label=attr)
            bottoms += np.array(attr_sizes)

        plt.title('Memory Usage of Each Time Series Signal')
        plt.xlabel('Signal Index')
        plt.ylabel("Size (MB)")
        plt.legend(title="Attributes", loc="upper right")

        # Pie chart for average params sizes
        plt.subplot(1, 2, 2)  # Second subplot for the pie chart
        labels = avg_params_sizes.keys()
        sizes = avg_params_sizes.values()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Average Memory Usage Distribution in params')

        plt.tight_layout()
        plt.show()


    sizes_MB    = np.array([len(pickle.dumps(s.data))+len(pickle.dumps(s.time)) for s in series])/(1024**2)
    summ_MB     = len(pickle.dumps(series))/(1024**2)
    avg_size_MB = np.average(sizes_MB)
    std_size_MB = np.std(sizes_MB)
    print(f"Size: {summ_MB:0.1f} MB, {len(sizes_MB)} × avg: {avg_size_MB:0.1f} MB (std: {std_size_MB:0.1f} MB)")

    # plot_sizes_and_params(series)

    with open( filename, 'wb') as f:
        pickle.dump( (series, input_series), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Steps saved in {filename}")


def load_series(filename):
    import dill as pickle
    print("📂 Loading...")
    with open(filename, "rb") as f:
        series, input_series = pickle.load(f)
    return series, input_series