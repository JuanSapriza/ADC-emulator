#!/usr/bin/env python3

import io, os, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp

from processes import *

CHANNELS = 10
processes = []
# figure, axs = plt.subplots(CHANNELS+1, 1, figsize=(10, 8))

def add_to_plot(series, row, col, alpha=1):
    ax = plt.subplot2grid((CHANNELS, 2), (row, col), rowspan = 1 if col == 0 else CHANNELS )
    ax.set_title(series.name)
    print(f'Samples \t {len(series.time)} \t {series}')
    ax.scatter(series.time, series.data, alpha = alpha, s=2  )
    processes.append(series)
    series.dump()



'''
Time,FP1,FP2,C3,C4,O1,O2,SPR1,SPR2,ECG1,ECG2
0.005,86.512,-61.877,73.345,-196.646,120.742,-10.405,-121.088,-6.189,1.378,-24.799
0.010,54.116,-194.703,32.428,-161.863,129.175,65.907,-121.581,-27.004,22.375,-29.840
'''

with open('../in/multimodal/S1_ADAS1.txt') as f: y = f.readlines()
titles = y[0].strip().split(',')
y = y[1:]
data = np.asarray([[float(f) for f in l.split(',')] for l in y])

channels = []
for i in range(len(titles)):
    channels.append([row[i] for row in data])

from vADCs import *

time = channels[0]
channels = channels[1:]
titles = titles[1:]
T_s = time[1] - time[0]
f_Hz = 1/T_s
length_s = 15

sample_max = int(length_s/T_s)
for c in range(len(channels)): channels[c] = channels[c][:sample_max]



adc_channels = []
for ch, ch_idx in zip(channels, range(CHANNELS)):

    adc_channels.append( ADC( name      = f"{titles[ch_idx]}",
                            units       = "uV",
                            f_sample_Hz = f_Hz,
                            ampl_bits   = 14,
                            dynRange    = [-1500, 1500],
                            series      =  Timeseries('ch1',
                                                      channels[ch_idx],
                                                      length_s = length_s)
                            )
                        )

    add_to_plot(adc_channels[ch_idx].conversion, ch_idx, 0)



mcadc = mcADC("MultiChannel ADC", adc_channels[:CHANNELS] )

mcadc.TDM()

add_to_plot(mcadc.conversion, 0, 1)

plt.subplots_adjust(hspace=2)
plt.tight_layout()
plt.show()

