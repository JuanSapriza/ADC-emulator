# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch

from timeseries import *
from processes import *


'''```````````````````````````````
 LC ADC
```````````````````````````````'''
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
    o.params[TS_PARAMS_START_S] = series.time[0]
    o.params[TS_PARAMS_END_S]   = series.time[-1]
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

def lcadc_fraction(series, params ):
    '''
    Analog LC ADC with level width fractions.

    Takes levels list for compatibility with other methods.
    Returns:
    - data: A signed int with the direction of the crossing.
    - time: The number of timer samples skipped.

    Args:
        series (Timeseries): Input time series.
        lvl_w_fraction (float): Level width fraction.

    Returns:
        Timeseries: Level crossing time series.
    '''
    lvl_w = (np.abs(series.params[TS_PARAMS_AMPL_RANGE][1])+np.abs(series.params[TS_PARAMS_AMPL_RANGE][0])) / params[TS_PARAMS_LC_LVL_W_FRACT]
    lvls = list(np.arange(series.params[TS_PARAMS_AMPL_RANGE][0], series.params[TS_PARAMS_AMPL_RANGE][1], lvl_w))
    o = Timeseries("LC fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS]     = lvls
    o.params[TS_PARAMS_START_S]     = series.time[0]
    o.params[TS_PARAMS_END_S]       = series.time[-1]

    MAX_TIME = 2**params[TS_PARAMS_LC_ACQ_TIME_B]

    current_level = np.trunc(series.data[0] / lvl_w)  # Level number in the list.
    last_sample = 0
    o_data = [current_level]
    o_time = [0]

    for i in range(0, len(series.data)):
        diff = np.trunc(((series.data[i] - current_level * lvl_w) / lvl_w)).astype(int)
        dir = np.sign(diff)
        diff = abs(diff)
        analog_samples = i - last_sample - 1
        timer_samples = max(1, np.round( analog_samples * params[TS_PARAMS_LC_TIMER_F_HZ] / series.params[TS_PARAMS_F_HZ] ) )

        consecutives = 0
        while diff > 0 or timer_samples >= MAX_TIME:
            o_data.append(dir)
            current_level = current_level + dir
            analog_samples = i - last_sample - 1
            timer_samples = timer_samples = max(1,np.round( analog_samples * params[TS_PARAMS_LC_TIMER_F_HZ] / series.params[TS_PARAMS_F_HZ] )) + consecutives
            o_time.append(timer_samples)
            last_sample = i
            diff -= 1
            timer_samples = 0
            consecutives += 1
    o.data = np.array(o_data, dtype=np.int16)
    o.time = np.array(o_time, dtype=np.int16)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

def lc_subsampler_fraction_old( series, params ):
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
    fraction_b = np.log2(params[TS_PARAMS_LC_LVL_W_FRACT])
    sample_b = int(series.params[TS_PARAMS_SAMPLE_B])
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS]         = list(range(0, 2**sample_b, lvl_w))
    o.params[TS_PARAMS_START_S]         = series.time[0]
    o.params[TS_PARAMS_END_S]           = series.time[-1]

    current_lvl = 0  # The last level to be crossed.
    lvl_up, lvl_down = 0, 0  # The value to be crossed to consider that a level was crossed.
    x_up, x_down = False, False  # Whether the sample crossed the upper or lower level
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xing = False  # Whether the sample crossed any level. THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_VAL = 2**sample_b
    MIN_VAL = 0
    MAX_SKIP = 2**params[TS_PARAMS_LC_ACQ_TIME_B] -1
    MAX_XING = 2**params[TS_PARAMS_LC_ACQ_AMP_B] -1

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
                while( xings or skipped == MAX_SKIP ):
                    data    = min( xings, MAX_XING )
                    xings   = max( 0, xings - MAX_XING)
                    o_data.append(data if dir else -data)
                    o_time.append(skipped)
                    skipped = 0
                skipped = 1

            else:
                skipped += 1

            if xings == MAX_XING:
                continue
            break

    o.data = np.array(o_data, dtype=np.int16)
    o.time = np.array(o_time, dtype=np.int16)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

'''```````````````````````````````
 LC Subsampler
```````````````````````````````'''

def lc_subsampler_fraction( series, params ):
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
    fraction_b = np.log2(params[TS_PARAMS_LC_LVL_W_FRACT])
    sample_b = int(series.params[TS_PARAMS_SAMPLE_B])
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS]         = list(range(0, 2**sample_b, lvl_w))
    o.params[TS_PARAMS_START_S]         = series.time[0]
    o.params[TS_PARAMS_END_S]           = series.time[-1]

    current_lvl = 0  # The last level to be crossed.
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_SKIP = 2**params[TS_PARAMS_LC_ACQ_TIME_B] -1
    MAX_XING = 2**params[TS_PARAMS_LC_ACQ_AMP_B] -1

    o_data = []
    o_time = []

    for i in range(len(series.data)):
        diff    = (series.data[i] - current_lvl)
        dir     = np.sign(diff)
        xings   = np.floor(np.abs(diff / lvl_w))
        current_lvl += xings*lvl_w*dir
        while( xings or skipped == MAX_SKIP ):
            data    = min( xings, MAX_XING )
            xings   = max( 0, xings - MAX_XING)
            o_data.append(data*dir)
            o_time.append(skipped)
            skipped = 0

        skipped += 1


    o.data = np.array(o_data, dtype=np.int16)
    o.time = np.array(o_time, dtype=np.int16)
    o.params[TS_PARAMS_LC_ACQ_F_HZ] = len(o.data) / series.params[TS_PARAMS_LENGTH_S]
    return o.copy()

def lc_subsampler_fraction_half_lsb( series, params ):
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
    fraction_b = np.log2(params[TS_PARAMS_LC_LVL_W_FRACT])
    sample_b = int(series.params[TS_PARAMS_SAMPLE_B])
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TS_PARAMS_LC_LVLS]         = list(range(0, 2**sample_b, lvl_w))
    o.params[TS_PARAMS_START_S]         = series.time[0]
    o.params[TS_PARAMS_END_S]           = series.time[-1]

    current_lvl = 0  # The last level to be crossed.
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_SKIP = 2**params[TS_PARAMS_LC_ACQ_TIME_B] -1
    MAX_XING = 2**params[TS_PARAMS_LC_ACQ_AMP_B] -1

    o_data = []
    o_time = []

    for i in range(len(series.data)):
        diff    = (series.data[i] - current_lvl)
        dir     = np.sign(diff)
        xings   = np.abs(diff // lvl_w)
        current_lvl += xings*lvl_w*dir

        while( xings or skipped == MAX_SKIP ):
            data    = min( xings, MAX_XING )
            xings   = max( 0, xings - MAX_XING)
            o_data.append(data*dir)
            o_time.append(skipped)
            skipped = 0

        skipped += 1


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
    o.params[TS_PARAMS_START_S] = series.time[0]
    o.params[TS_PARAMS_END_S]   = series.time[-1]

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

'''```````````````````````````````
 LC Feature extraction
```````````````````````````````'''

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

def lc_task_detect_spike_online(series, length=10, dt_n=0, Block=True):
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
                    o_time.append(series.params[TS_PARAMS_START_S] + sum(np.array(time[:i])))
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


'''```````````````````````````````
 Reconstruct LC signal
```````````````````````````````'''

def lc_reconstruct(series):
    '''
    Reconstructs a LC'd signal.

    Args:
        series (Timeseries): Input time series.
        lvls (list): Levels list.
        start_lvl (int): Starting level.

    Returns:
        Timeseries: Reconstructed time series.
    '''
    o = Timeseries(series.name + " LCrecTime")
    o.params.update(series.params)
    o_time = [ series.params[TS_PARAMS_START_S]]
    lvl_w = series.params[TS_PARAMS_LC_LVLS][1]
    o_data = [series.data[0]*lvl_w]
    try:
        f_Hz = series.params[TS_PARAMS_LC_TIMER_F_HZ]
    except:
        f_Hz = series.params[TS_PARAMS_F_HZ]

    for i in range(1, len(series.data)):
        if series.time[i] == 0:
            o_data[-1] += series.data[i]
        else:
            o_time.append(o_time[-1] + ((series.time[i] ) / f_Hz))
            o_data.append( o_data[-1] + series.data[i]*lvl_w )

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()

def lc_reconstruct_arrows(series):
    '''
    Generate arrows for a LC'd signal.

    Args:
        series (Timeseries): Input time series.

    Returns:
        Timeseries: Reconstructed arrows time series. Transforms time to absolute values.
    '''
    o = Timeseries(series.name + " LCrecTime")
    o.params.update(series.params)
    o_time = [ series.params[TS_PARAMS_START_S]]
    o_data = [0]

    try:
        f_Hz = series.params[TS_PARAMS_LC_TIMER_F_HZ]
    except:
        f_Hz = series.params[TS_PARAMS_F_HZ]

    for i in range(0, len(series.data)):
        if series.time[i] == 0:
            o_data[-1] += series.data[i]
        else:
            o_time.append(o_time[-1] + ((series.time[i] ) / f_Hz ))
            o_data.append( series.data[i] )

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()

def lc_rec_zoh_fmin(series):
    '''
    Reconstruct the LC signal through ZOH to the minimal sampling frequency possible.
    This approach is compatible with both LC-ADC and LC-subsampling

    Args:
        series (Timeseries): Input time series, a LC signal having in data the level differences, and in time
        number of skipped samples (at the original

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate)
    '''
    o = Timeseries(series.name + " rec. ZOH fmin")
    o.params.update(series.params)

    t_min_s = (min(series.time)+1)/series.params[TS_PARAMS_F_HZ]
    start_s = series.params[TS_PARAMS_START_S]
    end_s   = series.params[TS_PARAMS_END_S]

    in_time = start_s + np.cumsum( series.time / series.params[TS_PARAMS_F_HZ] )
    o_time = np.arange(start_s, end_s, t_min_s)

    lvl = 0  # <<<<<<<<<<<<<<< FIX THIS
    lvls = series.params[TS_PARAMS_LC_LVLS]
    o_data = []

    # Apply zero-order hold
    current_index = 0
    o_data = np.zeros_like(o_time, dtype=int)
    for i, time in enumerate(in_time):
        # Find the index in the new timeline where the current time falls
        while current_index < len(o_time) and o_time[current_index] <= time-t_min_s:
            value = lvls[lvl]
            o_data[current_index] = value
            current_index += 1
        lvl += series.data[i]

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    return o.copy()

def lc_rec_linear_interp(series):
    '''
    Reconstruct the LC signal through linear interpolation to the minimal sampling frequency possible.
    This approach is compatible with both LC-ADC and LC-subsampling.

    Args:
        series (Timeseries): Input time series, a LC signal having in data the level differences, and in time
                             number of skipped samples (at the original frequency).

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate).
    '''
    o = Timeseries(series.name + " rec. Linear Interpolation")
    o.params.update(series.params)

    t_min_s = (min(series.time)+1)/series.params[TS_PARAMS_F_HZ]
    start_s = series.params[TS_PARAMS_START_S]
    end_s   = series.params[TS_PARAMS_END_S]

    in_time = start_s + np.cumsum(series.time / series.params[TS_PARAMS_F_HZ])
    o_time = np.arange(start_s, end_s, t_min_s)

    lvls = series.params[TS_PARAMS_LC_LVLS]
    o_data = np.zeros_like(o_time, dtype=int)

    # Initialize current level based on initial conditions
    current_lvl = 0
    next_index = 0

    # Iterate over each output time and interpolate
    for j in range(len(o_time)):
        # Advance to the right interval in the input time
        while next_index < len(in_time) and in_time[next_index] < o_time[j]:
            current_lvl += series.data[next_index]
            next_index += 1

        # Set the output data by interpolating linearly
        if next_index == 0:
            # Before the first known point
            o_data[j] = lvls[current_lvl]
        elif next_index < len(in_time):
            # Interpolate between current_lvl and current_lvl + series.data[next_index]
            if o_time[j] == in_time[next_index-1]:
                # Exact match with the known data point
                o_data[j] = lvls[current_lvl]
            else:
                # Linear interpolation
                t_prev = in_time[next_index - 1]
                t_next = in_time[next_index]
                lvl_prev = current_lvl
                lvl_next = current_lvl + series.data[next_index]
                # Interpolate levels
                o_data[j] = lvls[int(np.interp(o_time[j], [t_prev, t_next], [lvl_prev, lvl_next]))]
        else:
            # After the last known point
            o_data[j] = lvls[current_lvl]

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    return o.copy()

def lc_rec_poly_interp(series, order):
    '''
    Reconstruct the LC signal through polynomial interpolation of a specified order.
    This approach is compatible with both LC-ADC and LC-subsampling.

    Args:
        series (Timeseries): Input time series, a LC signal having in data the level differences, and in time
                             number of skipped samples (at the original frequency).
        order (int): The order of the polynomial used for interpolation.

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate).
    '''
    o = Timeseries(series.name + f" rec. Poly Interp order {order}")
    o.params.update(series.params)

    t_min_s = (min(series.time)+1)/series.params[TS_PARAMS_F_HZ]
    start_s = series.params[TS_PARAMS_START_S]
    end_s   = series.params[TS_PARAMS_END_S]

    in_time = start_s + np.cumsum(series.time / series.params[TS_PARAMS_F_HZ])
    o_time = np.arange(start_s, end_s, t_min_s)

    lvls = series.params[TS_PARAMS_LC_LVLS]
    o_data = np.zeros_like(o_time, dtype=int)

    # Initialize current level based on initial conditions
    current_lvl = 0
    points = [(in_time[0], current_lvl)]

    # Collect points for polynomial interpolation
    for i in range(len(series.data)):
        current_lvl += series.data[i]
        points.append((in_time[i], current_lvl))

    # Convert points to useable format
    x_points, y_points = zip(*points)
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    # Fit a polynomial of the given order
    coeffs = np.polyfit(x_points, y_points, order)

    # Evaluate the polynomial at each point in o_time
    for j in range(len(o_time)):
        # Polynomial evaluation
        interp_value = np.polyval(coeffs, o_time[j])
        # Ensure interpolated value does not exceed bounds
        interp_level_index = int(np.clip(round(interp_value), 0, len(lvls)-1))
        o_data[j] = lvls[interp_level_index]

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    return o.copy()


'''```````````````````````````````
 Reconstruct sparse signal (absolute values)
```````````````````````````````'''

def rec_zoh_fmin(series):
    '''
    Reconstruct the LC signal through ZOH to the minimal sampling frequency possible.
    This approach is compatible with both LC-ADC and LC-subsampling

    Args:
        series (Timeseries): Input time series, a sparse signal having in data the absolute amplitudes, and in time
        the absolute times

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate)
    '''
    o = Timeseries(series.name + " rec. ZOH fmin")
    o.params.update(series.params)

    if len(series.time) < 2 : return None
    t_min_s = min(np.diff(series.time))
    start_s = series.params[TS_PARAMS_START_S]
    end_s   = series.params[TS_PARAMS_END_S]

    o_time = np.arange(start_s, end_s, t_min_s)
    o_data = []

    # Apply zero-order hold
    current_index = 0
    o_data = np.zeros_like(o_time)
    value = series.data[0]
    for i, time in enumerate(series.time):
        # Find the index in the new timeline where the current time falls
        while current_index < len(o_time) and o_time[current_index] <= time-t_min_s:
            o_data[current_index] = value
            current_index += 1
        value = series.data[i]
    o.data = np.array(o_data)
    o.time = np.array(o_time)
    o.params[TS_PARAMS_F_HZ] = 1/t_min_s
    return o.copy()


def rec_linear_fmin(series):
    '''
    Reconstruct the LC signal through linear interpolation to the minimal sampling frequency possible.
    This approach is compatible with both LC-ADC and LC-subsampling.

    Args:
        series (Timeseries): Input time series, a sparse signal having in data the absolute amplitudes, and in time
                             the absolute times.

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate).
    '''
    o = Timeseries(series.name + " rec. Linear Interpolation fmin")
    o.params.update(series.params)

    if len(series.time) < 2 : return None
    t_min_s = min(np.diff(series.time))
    start_s = series.params[TS_PARAMS_START_S]
    end_s   = series.params[TS_PARAMS_END_S]

    o_time = np.arange(start_s, end_s, t_min_s)
    o_data = np.zeros_like(o_time, dtype=int)

    # Perform linear interpolation using numpy.interp
    # numpy.interp(x, xp, fp) where:
    # x is the array of points at which to interpolate
    # xp is the array of the data points' x-values
    # fp is the array of the data points' y-values
    o_data = np.interp(o_time, series.time, series.data)

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    o.params[TS_PARAMS_F_HZ] = 1/t_min_s
    return o.copy()