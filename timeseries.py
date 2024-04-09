# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np
import pickle

class Timeseries:
    def __init__(self,
                 name,
                 data = None,
                 time = None,
                 f_Hz = 0,
                 length_s = 0,
                 dx = None ):
        self.name   = name
        self.data   = data if data is not None else []

        if time is None and length_s != 0:
            T_s = length_s/len(data)
            f_Hz = 1/T_s
            time = np.arange(0,length_s,T_s)
        self.time   = time if time is not None else []

        self.f_Hz   = f_Hz
        self.dx     = dx if dx is not None else []

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
