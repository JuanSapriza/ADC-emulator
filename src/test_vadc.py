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

with open('../in/Biopac.txt') as f: y = f.readlines()
data = np.asarray([int(np.round(float(f))) for f in y])
max_ch1 = max(abs(data))
ch1 = Timeseries('ch1', data, length_s = 60)
add_to_plot(ch1)

with open('../in/Epiphone.txt') as f: y = f.readlines()
data = np.asarray([int(np.round(float(f))) for f in y])
max_ch2 = max( abs(data))
norm_data = []
for d in data: norm_data.append(d*max_ch1/max_ch2)

# PERFORM THIS OPERATIONS WITH SIGNALS!!
ch2 = Timeseries('ch1', norm_data, length_s = 60)

add_to_plot(ch2)

print("Original frequency (Hz) ch1", ch1.f_Hz)
print("Original frequency (Hz) ch2", ch2.f_Hz)


from vADCs import *

adc_ch1 = ADC( name     = "CH1 (Biopac)",
            units       = "uV",
            f_Hz        = 200,
            ampl_bits   = 5,
            dynRange    = [-300, 300]
            )

adc_ch2 = cp(adc_ch1)
adc_ch2.name = "CH2 (Epiphone)"
adc_ch2.f_Hz = 1000


adc_ch1.feed(ch1)
add_to_plot(adc_ch1.conversion)

adc_ch2.feed(ch2)
add_to_plot(adc_ch2.conversion)

mcadc = mcADC( name = "Multichannel ADC",
              channels = [adc_ch1, adc_ch2]
              )


# ax.set_xlim(56, 59)
plt.legend(processes)
plt.show()