# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

from scipy.signal import butter,filtfilt
from scipy import interpolate
import numpy as np

from timeseries import Timeseries


class Process:
    def __init__(self, process, *args):
        self.process = process
        self.args = args

class Signal:
    def __init__(self, series: Timeseries):
        self.series     = series
        self.processes  = []
        self.steps      = [series]

    def add_process(self, process, *args ):
        self.processes.append(Process(process, *args))

    def apply_process(self, process, *args):
        self.steps.append(process(self.series,*args))


def pas(series, e):
    '''
    Implement a PAS
    '''
    data = series.data
    time = series.time
    dx  = 1 # Time differential
    i   = 1 # Sample index
    f   = 0 # Cost function
    x   = 0 # Time sample
    y   = 0 # Value sample
    t_  = 0 # Time of the previous fiducial point
    p   = 0 # Time of a peak
    l   = 0 # Length
    o = Timeseries(series.name + " PAS") # Output array
    for i in range(1,len(time)):
        dy = data[i] - data[i-1]
        x += dx
        y += dy
        f = f + x*dx + y*dy
        displ = abs(y) + x
        if displ < l and p == 0:
            p += i - 1
        l = displ
        if abs(f) > e: # ToDo: Maybe adjust e to reach a certain level of compression?
            if p == 0:
                t = i -1
            else:
                t= p
            o.data.append( data[t] )
            o.dx = t - t_# This is the value that would be saved
            o.time.append( time[t] )
            f = 0
            p = 0
            x = (i-t)*dx
            y = data[i] - data[t]
            t_ = t
            l = abs(y) + x
    return o

def neo(series, win):
    o = Timeseries(series.name + " NEO")
    o.f_Hz = series.f_Hz
    t_diff = int(o.f_Hz*win)
    for i in range(t_diff,len(series.data)):
        dx = series.time[i] - series.time[i-t_diff]
        dy = series.data[i] - series.data[i-t_diff]
        dydx = dy/dx
        o.data.append( dydx**2 - series.data[i]*dydx )
        o.time.append( series.time[i] )
    return o

def aso(series, win):
    o = Timeseries(series.name + " ASO")
    o.f_Hz = series.f_Hz
    t_diff = int(o.f_Hz*win)
    for i in range(1,len(series.data)):
        dx = series.time[i] - series.time[i-t_diff]
        dy = series.data[i] - series.data[i-t_diff]
        dydx = dy/dx
        o.data.append( series.data[i]*dydx )
        o.time.append( series.time[i] )
    return o

def as2o(series, win):
    o = Timeseries(series.name + " AS2O")
    o.f_Hz = series.f_Hz
    t_diff = int(o.f_Hz*win) if o.f_Hz != 0 else win
    for i in range(t_diff,len(series.data)):
        dx = series.time[i] - series.time[i-t_diff]
        dy = series.data[i] - series.data[i-t_diff]
        dydx = dy/dx
        o.data.append( series.data[i]*dydx**2 )
        o.time.append( series.time[i] )
    return o

def needle(series, win):
    o = Timeseries(series.name + " needle'd")
    o.f_Hz = series.f_Hz
    t_diff = int(o.f_Hz*win) if o.f_Hz != 0 else win
    k = Timeseries("inflections")
    d = []
    d.append(0)
    o.data.append(1)
    o.time.append(0)
    for j in range(1,len(series.data)):
        d.append( 1 if series.data[j] - series.data[j-t_diff] > 0 else -1 )
        if d[j]*d[j-1] < 0:
            k.data.append( j-1 )
    for i in range(1,len(k.data)):
        dx = series.time[k.data[i]] - series.time[k.data[i-1]]
        dy = series.data[ k.data[i] ] - series.data[ k.data[i-1] ]
        o.data.append( (dy**2)/dx )
        o.time.append( series.time[k.data[i]] )
    return o



def mean_sub(series, win):
    o = Timeseries(series.name + " Mean")
    o.f_Hz = series.f_Hz
    for i in range(win,len(series.time)):
        ### print('',end='\r')
        m = np.average(series.data[i-win:i])
        o.data.append( series.data[i] - m )
        o.time.append( series.time[i] )
        ### print(i, end='')
    ### print('',end='\r')
    return o

def pseudo_mean(series, bits):
    o = Timeseries(series.name + " pMean")
    o.f_Hz = series.f_Hz
    m = int(series.data[0])
    mb = int(series.data[0]) << bits
    for i in range(len(series.time)):
        mb = int(mb - m + series.data[i]) # m[i]xb = m[i-1]xb - m[i-1] + s[i]]
        m = mb >> bits  # m[i] = m[i]xb /b
        o.data.append( m )
        o.time.append( series.time[i] )
        ### print('\r',i, end='')
    ### print('',end='\r')
    return o

def lpf_butter(series, cutoff, order):
    o = Timeseries(series.name + " LPF")
    o.f_Hz = series.f_Hz
    normal_cutoff = 2/cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    o.data = filtfilt(b, a, series.data)
    o.time = series.time
    return o


# Function to create a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply a Butterworth filter to a signal
def bpf_butter(series, lowcut, highcut, order=4):
    o = Timeseries(series.name + " BPF")
    o.f_Hz = series.f_Hz
    b, a = butter_bandpass(lowcut, highcut, o.f_Hz, order=order)
    o.data = filtfilt(b, a, series.data)
    o.time = series.time
    return o



def norm(series, bits):
    o = Timeseries(series.name + " Norm")
    o.time = series.time
    o.f_Hz = series.f_Hz
    sorted = np.abs(series.data)
    sorted.sort()
    maxs = sorted[-10:]
    maxd = np.average(maxs)
    max_val = int(2**bits/2 -1)
    for s in series.data:
        d = max_val*s/maxd
        if d > max_val:
            d = max_val
        elif d < -max_val:
            d = -max_val
        o.data.append(d)
    return o

def offset_to_pos_and_map(series, bits):
    o = Timeseries(series.name + " map abs", time = series.time, f_Hz = series.f_Hz)

    # Push everything above 0
    minv = min(series.data)
    if minv < 0:
        data = []
        for s in series.data:
            data.append(s-minv)

    maxd = max(data)
    max_val = int(2**bits/2 -1)
    for s in data:
        d = max_val*s/maxd
        if d > max_val:
            d = max_val
        elif d < -max_val:
            d = -max_val
        o.data.append(d)
    return o

def quant(series, bits):
    o = Timeseries(series.name + f" Q({bits})")
    o.time = series.time
    o.f_Hz = series.f_Hz
    sorted = np.abs(series.data)
    sorted.sort()
    maxs = sorted[-10:]
    maxd = np.average(maxs)
    max_val = int(2**bits/2 -1)
    ratio = 1 #maxd/max_val  # Just to scale it next to the other signal
    for s in series.data:
        d = int(max_val*s/maxd)*ratio
        if d > max_val:
            d = max_val
        elif d < -max_val:
            d = -max_val

        o.data.append(d)
    return o

def get_density(series, win):
    o = Timeseries("LCdens")
    abst = 0
    lstt = 0
    buf = []
    for t in series.time:
        abst += t
        if abst - lstt > win:
            o.time.append(abst)
            o.data.append(len(buf))
            lstt = abst
            buf = []
        else:
            buf.append(t)
    return o


def spike_det_dt(series, threshold):
    o = Timeseries("sDet")
    for s,t in zip(series.data, series.time):
        if s > threshold or s < -threshold:
            o.time.append(t)
    return o


def spike_det_lc(series, dt, count):
    o = Timeseries("sDETlc")
    i = count

    # print(">>", series.data[:10])

    data = [d for d in series.data if d != 0]
    time = [t for t,d in zip(series.time, series.data) if d != 0]
    for i in range(count, len(data)):
        if all(d == data[i] for d in data[i-count+1:i+1]): # A burst
            if time[i] - time[i-count] < dt: # fast
                o.time.append(time[i])
    return o


def lc_aso(series, lvls):
    o = Timeseries("LCASO")
    lvl = first_level(lvls)
    t =  series.time[0]
    for i in range(1, len(series.data) ):
        dt = series.time[i]
        t += dt
        dir = series.data[i]
        lvl += dir
        y = lvls[lvl]
        dy = y - lvls[ lvl- dir ]
        slope = dy/dt
        aso = y*slope
        o.time.append(t)
        o.data.append(aso)
    return o


def oversample(series, order):
    o = Timeseries(f"Sx{order}")
    o.f_Hz = series.f_Hz*order
    f = interpolate.interp1d(series.time, series.data)
    num_points = int((series.time[-1] - series.time[0]) * o.f_Hz) + 1
    o.time = np.linspace( series.time[0], series.time[-1], num_points )
    o.data = f(o.time)
    return o



def compute_sdr(original_signal, new_signal, interpolate=False):

    # If interpolation is enabled, interpolate the new signal
    if interpolate:
        # Interpolate the new signal to match the length of the original signal
        interpolated_new_signal = np.interp(
            np.arange(len(original_signal)),
            np.arange(0, len(original_signal), len(original_signal) / len(new_signal)),
            new_signal
        )
    else:
        # Otherwise, use sample-and-hold
        interpolated_new_signal = np.repeat(new_signal, len(original_signal) // len(new_signal))
        interpolated_new_signal = np.resize(interpolated_new_signal, len(original_signal))

    # Compute the power of the original signal and the distortion
    power_original_signal = np.sum(np.square(original_signal))
    power_distortion = np.sum(np.square(original_signal - interpolated_new_signal))

    # Ensure non-zero power_distortion to avoid division by zero
    if power_distortion == 0:
        raise ValueError("Power of distortion should be non-zero for SDR computation.")

    # Compute the SDR in dB
    sdr = 10 * np.log10(power_original_signal / power_distortion)

    return sdr