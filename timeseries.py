# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np
from copy import deepcopy
import pickle

TS_PARAMS_F_HZ      = "Frequency (Hz)"
TS_PARAMS_SAMPLE_B  = "Size per sample (bits)"
TS_PARAMS_LENGTH_S  = "Length (s)"
TS_PARAMS_PHASE_DEG = "Sampling phase (°)"
TS_PARAMS_OFFSET_B  = "Input signal offset (bits)"
TS_PARAMS_POWER_W   = "Sampling power (W)"
TS_PARAMS_EPC_J     = "Energy per conversion (J)"


class Timeseries:
    def __init__(self,
                 name,
                 data       = None,
                 time       = None,
                 f_Hz       = 0,
                 length_s   = 0,
                 sample_b   = 0,
                  ):
        self.name   = name
        self.data   = data if data is not None else []

        if time is None and length_s != 0:
            T_s = length_s/len(data)
            f_Hz = 1.0/T_s
            time = np.arange(0,length_s,T_s)
        elif time is not None and f_Hz == 0:
            f_Hz = 1.0/(time[1]-time[0])
            length_s = len(time)/f_Hz
        elif time is not None:
            length_s = len(time)/f_Hz

        self.params = {
            TS_PARAMS_F_HZ      : f_Hz,
            TS_PARAMS_LENGTH_S  : length_s,
            TS_PARAMS_SAMPLE_B  : sample_b,
        }

        self.time       = time if time is not None else []
        self.scores     = {}

    def __str__(self):
        return self.name

    def export(self, path="../out/", name="" ):
        if name == "": name = self.name.replace(" ", "_")
        with open( path+name+".timeseries", 'w+') as f:
            for t, d in zip(self.time, self.data): f.write(f"{t}, {d}\n")

    def export_bin(self, outfile="../out/", bytes=4, bigendian = False):
        '''
        This will onlywork with synchronous timeseries, as it will discard time information.
        '''
        if outfile == "../out/": outfile += self.name.replace(" ", "_")
        # Save the array to a binary file
        if      bytes == 4: wordsize = np.int32
        elif    bytes == 2: wordsize = np.int16
        elif    bytes == 1: wordsize = np.int8
        else:   raise ValueError("Invalid word size in bytes! Choose between 1, 2 and 4")
        data = np.array(self.data).astype(wordsize)
        with  open( outfile + ".bin", 'wb') as f:
            if bigendian: data.byteswap(True).tofile(f)
            else: data.tofile(f)

    def dump(self, path="../out/", name=""):
        if name == "": name = self.name.replace(" ", "_")
        from copy import deepcopy as cp
        cop = cp(self)
        cop.data = np.asarray(cop.data)
        with open( path + name + ".pkl", 'wb') as f:
            pickle.dump( cop, f )

    @classmethod
    def load(cls, filename ):
        with open( filename, 'rb' ) as f:
            return pickle.load(f)

    def copy(self):
        return deepcopy(self)

def copy_series( series ):
    return deepcopy( series)