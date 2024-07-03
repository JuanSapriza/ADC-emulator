# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

from timeseries import *
from processes import *

TS_PARAMS_LC_LVLS           = "LC levels"
TS_PARAMS_LC_LVL_W_FRACT    = "LC level width by fraction"
TS_PARAMS_LC_LVL_W_B        = "LC level width by bits"
TS_PARAMS_LC_STRAT          = "LC strategy"
TS_PARAMS_LC_FREQ_LIM       = "LC ADC datarate limit"
TS_PARAMS_LC_ACQ_AMP_B      = "LC Acquisition word size of Amplitude"
TS_PARAMS_LC_ACQ_DIR_B      = "LC Acquisition word size of Direction"
TS_PARAMS_LC_ACQ_TIME_B     = "LC Acquisition word size of Time"
TS_PARAMS_LC_ACQ_AMP_STRAT  = "LC Acquisition strategy amplitude"
TS_PARAMS_LC_ACQ_DIR_STRAT  = "LC Acquisition strategy direction"
TS_PARAMS_LC_ACQ_TIME_STRAT = "LC Acquisition strategy time"
TS_PARAMS_LC_ACQ_F_HZ       = "LC Acquisition ~ frequency"

def lcadc_simple(series, lvls):
    '''
    Analog LC ADC simple implementation.

    Compares against level width and returns:
    - data: A signed int with the number of levels crossed.
    - time: The absolute time of the crossing.

    Args:
        series (Timeseries): Input time series.
        lvls (list): Levels list for compatibility.

    Returns:
        Timeseries: Level crossing time series.
    '''
    lvl_width = lvls[1] - lvls[0]
    o = Timeseries("LC simple")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS] = lvls
    current_level = np.trunc(series.data[0] / lvl_width)  # Level number in the list.
    o_data = []
    o_time = []
    for i in range(1, len(series.data)):
        diff = np.trunc(((series.data[i] - current_level * lvl_width) / lvl_width)).astype(int)
        if diff != 0:
            o_data.append(np.sign(diff))
            o_time.append(series.time[i])
            current_level = current_level + np.sign(diff)
    o.data = np.array(o_data, dtype=np.float32)
    o.time = np.array(o_time, dtype=np.float32)
    return o.copy()

def lcadc_fraction(series, lvl_w_fraction):
    '''
    Analog LC ADC with level width fractions.

    Takes levels list for compatibility with other methods.
    Returns:
    - data: A signed int with the number of levels crossed.
    - time: The number of samples skipped.

    Args:
        series (Timeseries): Input time series.
        lvl_w_fraction (float): Level width fraction.

    Returns:
        Timeseries: Level crossing time series.
    '''
    lvl_w = (np.abs(series.params[TS_PARAMS_AMPL_RANGE][1])+np.abs(series.params[TS_PARAMS_AMPL_RANGE][0])) / lvl_w_fraction
    lvls = list(np.arange(series.params[TS_PARAMS_AMPL_RANGE][0], series.params[TS_PARAMS_AMPL_RANGE][1], lvl_w))
    o = Timeseries("LC fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS] = lvls
    current_level = np.trunc(series.data[0] / lvl_w)  # Level number in the list.
    last_sample = 0
    o_data = [0]
    o_time = [0]
    for i in range(1, len(series.data)):
        diff = np.trunc(((series.data[i] - current_level * lvl_w) / lvl_w)).astype(int)
        if diff != 0:
            o_data.append(np.sign(diff))
            current_level = current_level + np.sign(diff)
            o_time.append(i - last_sample - 1)
            last_sample = i
    o.data = np.array(o_data, dtype=np.float32)
    o.time = np.array(o_time, dtype=np.float32)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

def lc_subsampler_fraction(series, lvl_w_fraction):
    '''
    LC subsampler implementation compatible with C code.

    Takes level width as a fraction of the range of the input samples.
    Returns:
    - data: A signed int of up to 16 bits with the number of levels crossed.
    - time: The number of samples skipped.

    Args:
        series (Timeseries): Input time series.
        lvl_w_fraction (float): Level width fraction.

    Returns:
        Timeseries: Level crossing time series.
    '''
    sample_b = int(series.params[TS_PARAMS_SAMPLE_B])
    fraction_b = np.log2(lvl_w_fraction)
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS] = list(range(0, 2**sample_b, lvl_w))

    current_lvl = 0  # The last level to be crossed.
    lvl_up, lvl_down = 0, 0  # The value to be crossed to consider that a level was crossed.
    x_up, x_down = False, False  # Whether the sample crossed the upper or lower level
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xing = False  # Whether the sample crossed any level. THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_VAL = 2**16
    MIN_VAL = 0
    MAX_SKIP = MAX_VAL
    MAX_XING = MAX_VAL

    o_data = []
    o_time = []

    for i in range(len(series.data)):
        dir = 0  # Reset the direction signal
        while True:
            xings = 0
            while True:
                lvl_up = current_lvl if current_lvl >= MAX_VAL - lvl_w else current_lvl + lvl_w
                lvl_down = current_lvl if current_lvl <= MIN_VAL + lvl_w else current_lvl - lvl_w
                x_up = (current_lvl != lvl_up) and (series.data[i] >= lvl_up)
                x_down = (current_lvl != lvl_down) and (series.data[i] <= lvl_down)

                dir |= x_up
                xing = x_up or x_down
                xings += xing

                if xing and dir:
                    current_lvl = lvl_up
                elif xing and not dir:
                    current_lvl = lvl_down

                if not (xing and xings != MAX_XING):
                    break

            if xings or skipped == MAX_SKIP:
                o_data.append(xings if dir else -xings)
                o_time.append(skipped)
                skipped = 0
            else:
                skipped += 1

            if xings == MAX_XING:
                continue
            break

    o.data = np.array(o_data, dtype=np.int16)
    o.time = np.array(o_time, dtype=np.int16)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

def lc_subsampler(series, lvl_w_b, time_in_skips=True):
    '''
    Simple LC subsampler.

    Takes level width in bits.
    Returns:
    - data: A signed int with the number of levels crossed.
    - time: The number of samples skipped.

    Args:
        series (Timeseries): Input time series.
        lvl_w_b (int): Level width in bits.
        time_in_skips (bool): Whether to return time in skips or absolute time.

    Returns:
        Timeseries: Level crossing time series.
    '''
    o = Timeseries(series.name + f" LCsubs({lvl_w_b})")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS] = list(range(0, 2**series.params[TS_PARAMS_SAMPLE_B], 2**lvl_w_b))

    lvl_width = 2**lvl_w_b
    current_lvl = ((series.data[0]) // lvl_width) * lvl_width
    last_crossing = 0
    o_data = [current_lvl]
    o_time = [0]

    for i in range(1, len(series.data)):
        diff = (series.data[i] - current_lvl) // lvl_width
        if diff != 0:
            o_data.append(diff)
            if time_in_skips:
                o_time.append(i - last_crossing - 1)
                last_crossing = i
            else:
                o_time.append(series.time[i] - sum(o_time[:len(o_time)]))
            current_lvl = max(0, current_lvl + diff * lvl_width)

    o.data = np.array(o_data, dtype=np.int16)
    o.time = np.array(o_time, dtype=np.int16)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

def lc_task_detect_spike(series, length=10, dt_s=0.025):
    '''
    Detect spikes in a level crossing series.

    Args:
        series (Timeseries): Input time series.
        length (int): Length of the spike detection window.
        dt_s (float): Time threshold for spike detection.

    Returns:
        list: Indices of detected spikes.
    '''
    data = series.data[1:]
    time = np.array(series.time[1:]) / series.params[TS_PARAMS_F_HZ]

    switch_indexes = []
    current_value = data[0]
    count = length
    accum_time = sum(time[:length])
    one_way = 0

    for i in range(length, len(data)):
        if data[i] == current_value:
            count = count + 1 if count < length else count
            accum_time = accum_time + time[i] if count < length else accum_time + time[i] - time[i-length]
        else:
            one_way = 1 if (count == length and accum_time <= dt_s) else 0
            current_value, accum_time, count = data[i], 0, 0

        if count == length and accum_time <= dt_s and one_way == 1:
            current_value, accum_time, count = data[i], 0, 0
            switch_indexes.append(i - length + 2)

    return switch_indexes

def lc_task_detect_spike_online(series, start_time_s=0, length=10, dt_n=0, Block=True):
    '''
    Online spike detection in a level crossing series.

    Args:
        series (Timeseries): Input time series.
        start_time_s (float): Start time for detection.
        length (int): Length of the spike detection window.
        dt_s (float): Time threshold for spike detection.

    Returns:
        Timeseries: Detected spikes time series.
    '''
    o = Timeseries(series.name + " LC R-peak detection")
    o.params.update(series.params)
    data = series.data[1:]
    series_time = series.time[1:]
    time = (np.array(series.time[1:]) + 1) / series.params[TS_PARAMS_F_HZ]  # +1 because skipped=0 is still one more sample
    count = 0
    blocked = False

    o_time = []
    o_data = []
    for i in range(length, len(data)):
        if np.sign(data[i]) != np.sign(data[i-1]) or series_time[i] > dt_n:
            count += abs(data[i-1])
            if count >= length:
                if not blocked:
                    o_time.append(start_time_s + sum(np.array(time[:i])))
                    o_data.append(count)
                    count, blocked = 0, Block
                else:
                    count, blocked = 0, False
            else:
                count = 0
        elif series_time[i] <= dt_n:
            count += abs(data[i-1])

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()

def lc_aso(series, lvls):
    '''
    Level Crossing Amplitude Slope Operator (ASO).

    Args:
        series (Timeseries): Input time series.
        lvls (list): Levels list.

    Returns:
        Timeseries: ASO time series.
    '''
    o = Timeseries("LCASO")
    o.params.update(series.params)
    lvl = int(np.floor(len(lvls) / 2))
    t = series.time[0]
    o_time = []
    o_data = []

    for i in range(1, len(series.data)):
        dt = series.time[i]
        t += dt
        dir = series.data[i]
        lvl += dir
        y = lvls[lvl]
        dy = y - lvls[lvl - dir]
        slope = dy / dt
        aso = y * slope
        o_time.append(t)
        o_data.append(aso)

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()

def lcadc_reconstruct(series, lvls, start_lvl, start_time_s, end_time_s):
    '''
    Reconstructs a LC'd signal.

    Args:
        series (Timeseries): Input time series.
        lvls (list): Levels list.
        start_lvl (int): Starting level.
        start_time_s (float): Start time for reconstruction.
        end_time_s (float): End time for reconstruction.

    Returns:
        Timeseries: Reconstructed time series.
    '''
    o = Timeseries(series.name + " LCrec")
    o.params.update(series.params)
    lvl = start_lvl
    lvl = int(min(max(0, lvl + series.data[0]), len(lvls) - 1))
    o_data = [lvls[lvl]]
    o_time = [start_time_s]

    for i in range(1, len(series.data)):
        o_time.append(o_time[-1] + ((series.time[i] + 1) / series.params[TS_PARAMS_F_HZ]))
        lvl = int(min(max(0, lvl + series.data[i]), len(lvls) - 1))
        o_data.append(lvls[lvl])

    o_data.append(lvls[lvl])
    o_time.append(end_time_s)
    o.data = np.array(o_data, dtype=np.float32)
    o.time = np.array(o_time, dtype=np.float32)
    return o.copy()

def lcadc_reconstruct_arrows(series, start_time_s):
    '''
    Generate arrows for a LC'd signal.

    Args:
        series (Timeseries): Input time series.
        start_time_s (float): Start time for reconstruction.

    Returns:
        Timeseries: Reconstructed arrows time series. Transforms time to absolute values.
    '''
    o = Timeseries(series.name + " LCrecTime")
    o.params.update(series.params)
    o_time = [start_time_s]

    for i in range(1, len(series.data)):
        o_time.append(o_time[-1] + ((series.time[i] + 1) / series.params[TS_PARAMS_F_HZ]))

    o.time = np.array(o_time, dtype=np.float32)
    o.data = series.data
    return o.copy()
