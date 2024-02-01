
import numpy as np
from timeseries import *
from processes import *


class ADC:
    def __init__( self,
                  name       : str                   = "myVirtualADC",
                  units      : str                   = "Arbitrary units",
                  f_Hz       : float                 = 0,
                  dynRange   : list[float, float]    = [0,0],
                  bandwidth  : list[float, float]    = [0,0],
                  noise_dev  : float                 = 0,
                  SNR_dB     : float                 = 100,
                  THD_pc     : float                 = 0,
                  epc_J      : float                 = 0,
                  tpc_s      : float                 = 0,
                  ampl_bits  : int                   = 0,
                  time_bits  : int                   = 0,
                  buf_smpl   : int                   = 1,
                  channels   : int                   = 1,
                  diff       : bool                  = False,
                  interrupt  : int                   = -1,
                #   levels     : callable[[None],list] = None,
                #   in_custom  : callable[[Timeseries], Timeseries] = None,
                #   out_custom : callable[[Timeseries], Timeseries] = None,
                #   time_custom: callable[[Timeseries], Timeseries] = None,
                #   linear_range : list[float, float] = [0,0],
                  series     : Timeseries           = None,
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
        - ampl_bits (int): Number of bits of amplitude resolution. Default 0 for full precision.
        - time_bits (int): Number of bits of timing resolution. Default 0 for full precision.
        - buf_smpl (int): Number of samples to buffer before transmitting or interrupting. Defualt 1 for transmission upon acquisition.
        - channels (int): Number of channels of the ADC. Channels are time-division multiplexed (TDM). Default 1 for single channel.
        - diff (bool): Whether the ADC should output the difference between channels. Requires even number of channels. Default 0 for no differential output.
        - interrupt (bool): GPIO number to toggle to announce an available sample. Default -1 for no interrupts.
        - levels (callable[[None],list]): Function defining a list of levels for Level-Crossing (LC) ADC. None for no LC ADC.
        - in_custom (callable[[Timeseries], Timeseries]): A function that performs a custom operation over the input Timeseries, resulting in a modified Timeseries. Default None for no operation.
        - out_custom (callable[[Timeseries], Timeseries]): A function that performs a custom operation over the output Timeseries, resulting in a modified Timeseries. Default None for no operation.
        - time_custom (callable[[Timeseries], Timeseries]): A function that defines the sampling time. Default None to perform a fixed-rate sampling based on the f_Hz argument.
        """
        self.name           = name
        self.units          = units
        self.f_Hz           = f_Hz
        self.dynRange       = dynRange
        self.bandwidth      = bandwidth
        self.noise_dev      = noise_dev
        self.SNR_dB         = SNR_dB
        self.THD_pc         = THD_pc
        self.epc_J          = epc_J
        self.tpc_s          = tpc_s
        self.ampl_bits      = ampl_bits
        self.time_bits      = time_bits
        self.buf_smpl       = buf_smpl
        self.channels       = channels
        self.diff           = diff
        self.interrupt      = interrupt
        # self.levels         = levels
        # self.in_custom      = in_custom
        # self.out_custom     = out_custom
        # self.time_custom    = time_custom
        self.latency_s   = 0
        self.energy_J    = 0
        self.conversion  = None
        if series != None: self.feed(series)



    def feed( self, series: Timeseries ):

        series = resample( series, timestamps = None, f_Hz = self.f_Hz)
        series = clip( series, self.dynRange )
        series = quantize( series, self.ampl_bits, np.ceil)

        convert = series
        self.conversion = Timeseries("Conversion " + self.name,
                                    convert.data,
                                    convert.time,
                                    convert.f_Hz )


def resample( series: Timeseries, timestamps = None, f_Hz = 0  ):
    if timestamps == None:
        if f_Hz == 0:
            raise   ValueError("Either timestamps or f_Hz should be provided.")
        timestamps = np.arange(series.time[0], series.time[-1], 1 / f_Hz)

    resampled_data = np.interp(timestamps, series.time, series.data)
    o = Timeseries("resampled")
    o.data = resampled_data
    o.time = timestamps
    return o


def clip( series: Timeseries, dynRange: list[float, float] = [0,0] ):
    if dynRange[0] >= dynRange[1]:
        raise ValueError("The Dynamic Range should be defined as [Lower bound, Upper bound] and these should be different.")

    o = Timeseries(series.name + f" Clipped({dynRange[0]},{dynRange[1]})")
    o.time = series.time
    o.f_Hz = series.f_Hz
    for s in series.data:
        d = s
        if s < dynRange[0]:
            d = dynRange[0]
        elif s > dynRange[1]:
            d = dynRange[1]
        o.data.append(d)
    return o



def quantize( series: Timeseries, bits: int, approximation: callable ):
    o = Timeseries(series.name + f" Q({bits})")
    o.time = series.time
    o.f_Hz = series.f_Hz
    max_amplitude = max(abs(np.asarray(series.data)))
    min_amplitude_step = max_amplitude/ ( (2**bits)/2 -1)
    for s, i  in zip(series.data, range(len(series.data))):
        d = approximation(s/min_amplitude_step)*min_amplitude_step
        o.data.append(d)
    return o



class mcADC:
    def __init__(self,
                 name: str = "MyMultiChannelADC",
                 channels: list[ADC] = None):
        self.name       = name
        self.channels   = channels
        self.conversion = None

    def TDM(self):
        f_tdm_Hz    = self.channels[0].f_Hz * len(self.channels)
        T_tdm_s     = 1/f_tdm_Hz
        length      = len(self.channels[0])
        time        = np.arange(0,length,T_tdm_s)
        data        = []
        for i in range(length):
            for c in self.channels:
                # ADD CODIFICATION
                data.append(data[i])
        self.conversion = Timeseries( name = f"TDM {len(self.channels)} channels"
