# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

import numpy as np

from timeseries import *
from processes import *


class ADC:
    def __init__( self,
                  name          : str                   = "myVirtualADC",
                  units         : str                   = "Arbitrary units",
                  f_Hz          : float                 = 0,
                  dynRange      : list[float, float]    = [0,0],
                  bandwidth     : list[float, float]    = [0,0],
                  noise_dev     : float                 = 0,
                  phase_deg     : float                 = 0,
                  SNR_dB        : float                 = 100,
                  THD_pc        : float                 = 0,
                  epc_J         : float                 = 0,
                  tpc_s         : float                 = 0,
                  res_b         : int                   = 0,
                  time_bits     : int                   = 0,
                  buf_smpl      : int                   = 1,
                  channels      : int                   = 1,
                  diff          : bool                  = False,
                  interrupt     : int                   = -1,
                  ac_coupling_n : int                   = 0,
                  signed        : bool                  = False,
                  map           : bool                  = False,
                  out_range_b     : int                   = 0,
                  series        : Timeseries            = None,
                #   levels        : callable[[None],list] = None,
                #   in_custom     : callable[[Timeseries], Timeseries] = None,
                #   out_custom    : callable[[Timeseries], Timeseries] = None,
                #   time_custom   : callable[[Timeseries], Timeseries] = None,
                  linear_range  : list[float, float] = [0,0],
                  ):
        """
        Create a virtual ADC with specified characteristics.

        Parameters:
        - name (str): Name of the virtual ADC to identify it. Default "myVirtualADC".
        - units (str): Units in which the data is represented. Default "Arbitrary units".
        - f_Hz (float): Sampling frequency of the ADC. If larger than the input frequency of the signal, this will be oversampled. Default 0 for no fixed sampling rate.
        - dynRange (List[float,float]): Dynamic Range of the ADC (in the input units), lower and upper bounds. Default [0,0] for no limits.
        - bandwidth (List[float,float]): Frequency bounds of a band-pass filter at the input of the ADC. Default [0,0] for no filter.
        - noise_dev (float): Deviation of a Gaussian noise at the input. Default 0 for no noise.
        - SNR_dB (float): Signal-to-Noise Ratio of the ADC in decibels. Default 100 dB.
        - THD_pc (float): Total Harmonic Distortion (per cent). Default is 0%.
        - epc_J (float): Energy per Conversion in Joules. Default 0 for an ideal conversion.
        - tpc_s (float): Time per Conversion in seconds. Default 0 for no delay in the acquisition.
        - res_b (int): Number of bits of amplitude resolution. Default 0 for full precision.
        - out_range_b (int): Amplify the output (without affecting resolution, just a scaling of the data), to match this output range (in bits)
        - time_bits (int): Number of bits of timing resolution. Default 0 for full precision.
        - buf_smpl (int): Number of samples to buffer before transmitting or interrupting. Defualt 1 for transmission upon acquisition.
        - channels (int): Number of channels of the ADC. Channels are time-division multiplexed (TDM). Default 1 for single channel.
        - diff (bool): Whether the ADC should output the difference between channels. Requires even number of channels. Default 0 for no differential output.
        - interrupt (bool): GPIO number to toggle to announce an available sample. Default -1 for no interrupts.
        - ac_coupling_n (int): Number of samples of the input signal to use as window of the moving mean over which to apply a moving mean
        - signed (bool): Whether results can be negative or not
        - map (bool): Only for DAC. Whether to map the input range (dynamic range) into the output full scale.
        - levels (callable[[None],list]): Function defining a list of levels for Level-Crossing (LC) ADC. None for no LC ADC.
        - in_custom (callable[[Timeseries], Timeseries]): A function that performs a custom operation over the input Timeseries, resulting in a modified Timeseries. Default None for no operation.
        - out_custom (callable[[Timeseries], Timeseries]): A function that performs a custom operation over the output Timeseries, resulting in a modified Timeseries. Default None for no operation.
        - time_custom (callable[[Timeseries], Timeseries]): A function that defines the sampling time. Default None to perform a fixed-rate sampling based on the f_Hz argument.
        """
        self.name           = name
        self.units          = units
        self.f_Hz           = f_Hz
        self.phase_deg      = phase_deg
        self.dynRange       = dynRange
        self.bandwidth      = bandwidth     # Not yet used
        self.noise_dev      = noise_dev     # Not yet used
        self.SNR_dB         = SNR_dB        # Not yet used
        self.THD_pc         = THD_pc        # Not yet used
        self.epc_J          = epc_J
        self.tpc_s          = tpc_s         # Not yet used
        self.res_b          = res_b
        self.out_range_b      = out_range_b
        self.time_bits      = time_bits     # Not yet used
        self.buf_smpl       = buf_smpl      # Not yet used
        self.channels       = channels
        self.diff           = diff          # Not yet used
        self.interrupt      = interrupt     # Not yet used
        self.ac_coupling_n  = ac_coupling_n
        self.signed         = signed
        self.map            = map
        # self.levels         = levels
        # self.in_custom      = in_custom
        # self.out_custom     = out_custom
        # self.time_custom    = time_custom
        self.linear_range   = linear_range  # Not yet used
        self.latency_s      = 0             # Not yet used
        self.energy_J       = 0
        self.power_avg_W    = 0
        self.conversion     = None
        if series != None: self.feed(series)


    def sample( self, value, time = None ):
        value = self.clip_val( value )
        value = self.quantize_val( value, np.round )
        value = self.amplify_val( value )
        self.conversion.data.append( value )
        self.conversion.time.append( time )
        return value


    def feed( self, series: Timeseries ):

        if self.ac_coupling_n != 0: series = mean_sub( series, self.ac_coupling_n )
        # @ToDo: Move the following methods to the processes package
        series = self.resample( series, timestamps = None, f_Hz = self.f_Hz, phase_deg=self.phase_deg)
        series = self.clip( series )
        series = self.quantize( series, np.ceil)
        series = self.amplify_output( series )
        # series = self.measEnergy( series )
        self.conversion = Timeseries("Conversion " + self.name,
                                    data        = series.data,
                                    time        = series.time
                                    )
        self.conversion.params.update(series.params)
        self.conversion.params[TS_PARAMS_EPC_J] = self.epc_J

    def measEnergy(self, series: Timeseries ):
        series.energy_J = series.epc_J * len(series.data)
        return series

    def resample(self, series: Timeseries, timestamps = None, f_Hz = 0, phase_deg = 0  ):
        if timestamps == None:
            if f_Hz == 0:
                raise   ValueError("Either timestamps or f_Hz should be provided.")
            dephase_s   = (phase_deg/180)*(1/f_Hz)
            timestamps  = np.arange(series.time[0]+ dephase_s, series.time[-1], 1 / f_Hz)

        resampled_data = np.interp(timestamps, series.time, series.data)
        o = Timeseries("resampled")
        o.data = resampled_data
        o.time = timestamps
        o.params.update(series.params)
        o.params[TS_PARAMS_F_HZ] = f_Hz
        return o


    def clip(self, series: Timeseries ):
        if self.dynRange[0] >= self.dynRange[1]:
            raise ValueError("The Dynamic Range should be defined as [Lower bound, Upper bound] and these should be different.")

        o = Timeseries(series.name + f" Clipped({self.dynRange[0]},{self.dynRange[1]})")
        o.time = series.time
        o.params.update(series.params)
        for s in series.data:
            d = s
            if s < self.dynRange[0]:
                d = self.dynRange[0]
            elif s > self.dynRange[1]:
                d = self.dynRange[1]
            o.data.append(d)
        return o

    def clip_val( self, value ):
        d = value
        if value < self.dynRange[0]:
            d = self.dynRange[0]
        elif value > self.dynRange[1]:
            d = self.dynRange[1]
        return d

    def quantize(self, series: Timeseries, approximation: callable ):
        if self.dynRange[0] >= self.dynRange[1]:
            raise ValueError("The Dynamic Range should be defined as [Lower bound, Upper bound] and these should be different.")
        o = Timeseries(series.name + f" Q({self.res_b})")
        o.params.update(series.params)
        o.time                          = series.time
        o.params[TS_PARAMS_SAMPLE_B]    = self.res_b
        o.params[TS_PARAMS_AMPL_RANGE]  = [0,2**self.res_b-1 ] if not self.signed else [-2**(self.res_b-1),2**(self.res_b-1)-1 ]
        if self.signed:
            for s  in series.data:
                d = int(approximation( (2**self.res_b-1) * s/(self.dynRange[1]-self.dynRange[0]) ))
                o.data.append(d)
        else:
            for s  in series.data:
                d = int(approximation( (2**self.res_b)* s/(self.dynRange[1]-self.dynRange[0]) ))
                o.data.append(d)
        return o

    def quantize_val( self, value, approximation ):
        if self.signed:
            if self.map:
                return int(approximation( (2**self.res_b)* value/(self.dynRange[1]-self.dynRange[0]) )) - 2**(self.res_b-1)
            else:
                return int(approximation( (2**self.res_b -1 )* value/(self.dynRange[1]-self.dynRange[0]) ))
        else:
            # @ToDo: This logic is wrong! only works with symmetric DR around 0!
            return int(approximation( (2**self.res_b)*( value + self.dynRange[1])/(self.dynRange[1]-self.dynRange[0]) ))

    def amplify_output( self, series ):
        if self.out_range_b == 0: return series
        series.data = np.array(series.data) << int( self.out_range_b - self.res_b )
        return series # @ToDo: make other steps be the same as this (not having N timeseries!)

    def amplify_val( self, value ):
        if self.out_range_b == 0: return value
        return value << int( self.out_range_b - self.res_b )


class mcADC:
    def __init__(self,
                 name: str = "MyMultiChannelADC",
                 channels: list[ADC] = [] ):
        self.name       = name
        self.channels   = channels
        self.conversion = None

    def TDM(self):
        s           = self.channels[0].conversion
        f_tdm_Hz    = s.params[TS_PARAMS_F_HZ] * len(self.channels)
        T_tdm_s     = 1/f_tdm_Hz
        length_s    = len(s.data)*(1/s.params[TS_PARAMS_F_HZ])
        time        = np.arange(0,length_s,T_tdm_s)
        data        = []
        for i in range(len(s.time)):
            for c, c_idx in zip(self.channels, range(len(self.channels))):
                # CODIFICATION = 0bCC...CCDD...DD where C represent bits for the channel index and D bits for data in the resolution of the ADC.
                encoded = int(c.conversion.data[i]) + c_idx*(2**(c.res_b))
                data.append(encoded)
                # print(f"{bin(encoded)[2:].zfill(c.res_b+2)}")

        self.conversion = Timeseries( name = f"TDM {len(self.channels)} channels",
                                     data = data,
                                     time = time,
                                     f_Hz = f_tdm_Hz,
                                    )

