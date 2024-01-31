#!/usr/bin/env python3

import io, os, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp

from processes import *

processes = []

def add_to_plot(series, alpha=1):
    print(f'Samples \t {len(series.time)} \t {series}')
    ax.plot(series.time, series.data, alpha = alpha )
    processes.append(series)
    series.dump()

figure, ax = plt.subplots(figsize=(10, 8))

with open('../in/Biopac.txt') as f:
    y = f.readlines()
data = np.asarray([int(np.round(float(f))) for f in y])

time_s = 60
T_s = time_s/len(data)
f_Hz = 1/T_s
time = np.arange(0,time_s,T_s)

''' The raw signal '''
s = Signal( Timeseries('Raw', data, time, f_Hz) )

s.apply_process( pas, 500e3 )
s.apply_process( lpf_butter, 100, 2 )
other = Signal( cp(s.steps[-1]) )
other.steps[0].name = "Other"

s.apply_process(pas, 100e-3)
other.apply_process(pas, 2000e-3)

for step in s.steps: add_to_plot(step)
for step in other.steps: add_to_plot(step)

plt.legend(processes)
plt.show()