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
    lvl_w = (np.abs(series.params[TSP_AMPL_RANGE][1])+np.abs(series.params[TSP_AMPL_RANGE][0])) / params[TSP_LC_LVL_W_FRACT]
    lvls = list(np.arange(series.params[TSP_AMPL_RANGE][0], series.params[TSP_AMPL_RANGE][1], lvl_w))
    o = Timeseries("LC fraction")
    o.params.update(series.params)
    o.params[TSP_LC_LVLS]     = lvls
    o.params[TSP_LC_LVL_W_B]      = np.log2(lvl_w)
    o.params[TSP_LC_LVL_W_FRACT]  = params[TSP_LC_LVL_W_FRACT]
    o.params[TSP_START_S]     = series.time[0]
    o.params[TSP_END_S]       = series.time[-1]
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_DIFF_TM_N

    MAX_TIME = 2**params[TSP_LC_ACQ_TIME_B]

    current_level = np.trunc(series.data[0] / lvl_w)  # Level number in the list.
    last_sample = 0
    o_data = [current_level]
    o_time = [0]

    for i in range(0, len(series.data)):
        diff = np.trunc(((series.data[i] - current_level * lvl_w) / lvl_w)).astype(int)
        dir = np.sign(diff)
        diff = abs(diff)
        analog_samples = i - last_sample # - 1
        timer_samples = max(1, np.round( analog_samples * params[TSP_TIMER_F_HZ] / series.params[TSP_F_HZ] ) )

        consecutives = 0
        while diff > 0 or timer_samples >= MAX_TIME:
            o_data.append(dir)
            current_level = current_level + dir
            analog_samples = i - last_sample # - 1
            timer_samples = timer_samples = max(1,np.round( analog_samples * params[TSP_TIMER_F_HZ] / series.params[TSP_F_HZ] )) + consecutives
            o_time.append(timer_samples)
            last_sample = i
            diff -= 1
            timer_samples = 0
            consecutives += 1
    o.data = np.array(o_data )
    o.time = np.array(o_time )
    o.params[TSP_LC_ACQ_F_HZ] = len(o.data) / series.params[TSP_LENGTH_S]
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
    fraction_b = np.log2(params[TSP_LC_LVL_W_FRACT])
    sample_b = int(series.params[TSP_SAMPLE_B])
    if sample_b <= fraction_b: return None
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TSP_LC_LVLS]         = list(range(0, 2**sample_b, lvl_w))
    o.params[TSP_LC_LVL_W_B]      = np.log2(lvl_w)
    o.params[TSP_LC_LVL_W_FRACT]  = params[TSP_LC_LVL_W_FRACT]
    o.params[TSP_START_S]         = series.time[0]
    o.params[TSP_END_S]           = series.time[-1]
    o.params[TSP_TIME_FORMAT]     = TIME_FORMAT_DIFF_FS_N

    current_lvl = 0  # The last level to be crossed.
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_SKIP = 2**params[TSP_LC_ACQ_TIME_B] -1
    MAX_XING = 2**params[TSP_LC_ACQ_AMP_B] -1

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


    o.data = np.array( o_data )
    o.time = np.array( o_time )
    o.params[TSP_LC_ACQ_F_HZ] = len(o.data) / series.params[TSP_LENGTH_S]
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
    fraction_b = np.log2(params[TSP_LC_LVL_W_FRACT])
    sample_b = int(series.params[TSP_SAMPLE_B])
    lvl_w_b = sample_b - fraction_b
    lvl_w = int(2**lvl_w_b)
    o = Timeseries("LC in C from fraction")
    o.params.update(series.params)
    o.params[TSP_LC_LVLS]         = list(range(0, 2**sample_b, lvl_w))
    o.params[TSP_START_S]         = series.time[0]
    o.params[TSP_END_S]           = series.time[-1]
    o.params[TSP_TIME_FORMAT]     = TIME_FORMAT_DIFF_FS_N

    current_lvl = 0  # The last level to be crossed.
    dir = 0  # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0  # The count of crossings between two consecutive samples.
    skipped = 0  # The count of samples that did not cross any level. Reset on every crossing.

    MAX_SKIP = 2**params[TSP_LC_ACQ_TIME_B] -1
    MAX_XING = 2**params[TSP_LC_ACQ_AMP_B] -1

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


    o.data = np.array(o_data )
    o.time = np.array(o_time )
    o.params[TSP_LC_ACQ_F_HZ] = len(o.data) / series.params[TSP_LENGTH_S]
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
    time = np.array(series.time[1:]) / series.params[TSP_F_HZ]

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
        Timeseries: Detected spikes time series. On time the time of the spike. On data the signed count of the level crossings that led to that detection.
    '''
    o = Timeseries(series.name + " LC peak detection")
    o.params.update(series.params)
    data = series.data[1:]

    series_time = series.time[1:]
    try:
        f_Hz = series.params[TSP_TIMER_F_HZ]
    except:
        f_Hz = series.params[TSP_F_HZ]
    time = (np.array(series.time)) / f_Hz

    count = 0
    blocked = False

    o_time = []
    o_data = []
    for i in range(length, len(data)):
        if np.sign(data[i]) != np.sign(data[i-1]) or series_time[i] > dt_n or data[i] == 0:
            count += abs(data[i-1])
            if count >= length:
                if not blocked:
                    o_time.append(series.params[TSP_START_S] + sum(np.array(time[:i+1])))
                    o_data.append(count*np.sign(data[i-1]))
                    count, blocked = 0, Block
                else:
                    count, blocked = 0, False
            else:
                count = 0
        elif series_time[i] <= dt_n:
            count += abs(data[i-1])

    PLOT = 0
    if PLOT:
        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(10,3))
        arr = lc_reconstruct_time(series)
        [plt.axvline(l, color='gray', linestyle='-', alpha=0.2 ) for l in o_time ];
        [ plt.axhline(l, color='gray', linestyle='-', alpha=0.2 ) for l in series.params[TSP_LC_LVLS] ];
        [plt.arrow(t,0, dx=0,dy=d, color='r') for t, d in zip(arr.time, arr.data) ];
        plt.xlim(10,10.8)
        plt.show()

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()


def lc_task_detect_spike_online_2(series, length=10, dt_n=0, Block=True):
    '''
    Online spike detection in a level crossing series.

    Args:
        series (Timeseries): Input time series.
        start_time_s (float): Start time for detection.
        length (int): Length of the spike detection window.
        dt_s (float): Time threshold for spike detection.

    Returns:
        Timeseries: Detected spikes time series. On time the time of the spike. On data the signed count of the level crossings that led to that detection.
    '''
    o = Timeseries(series.name + " LC peak detection")
    o.params.update(series.params)
    data = series.data[1:]

    series_time = series.time[1:]
    f_Hz = series.params[TSP_F_HZ]

    time = (np.array(series.time)) / f_Hz

    count = 0
    blocked     = False
    possible    = False
    short_used  = 0

    o_time = []
    o_data = []
    for i in range(length, len(data)):
        if np.sign(data[i]) != np.sign(data[i-1]) or series_time[i] > dt_n or data[i] == 0:
            count += abs(data[i-1])
            if count >= length - (1-short_used):
                if count == length - (1-short_used): short_used = 1
                if not blocked:
                    if possible:
                        o_time.append(series.params[TSP_START_S] + sum(np.array(time[:i+1])))
                        o_data.append(count*np.sign(data[i-1]))
                        count, blocked = 0, Block
                        possible = False
                        short_used = 0
                    else:
                        possible = True
                        count = 0
                else:
                    count, blocked = 0, False
                    short_used = 0
                    possible = False

            else:
                count = 0
                possible = False
                short_used = 0
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
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S
    o_time = [ series.params[TSP_START_S]]
    lvl_w = series.params[TSP_LC_LVLS][1]
    o_data = [0] #[series.data[0]*lvl_w]

    try:
        f_Hz = series.params[TSP_TIMER_F_HZ]
    except:
        f_Hz = series.params[TSP_F_HZ]

    for i in range(0, len(series.data)):
        if series.time[i] == 0:
            o_data[-1] += series.data[i]
        else:
            o_time.append(o_time[-1] + ((series.time[i] ) / f_Hz))
            o_data.append( o_data[-1] + series.data[i]*lvl_w )

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()


def lc_reconstruct_w_inflections(series):
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
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S
    o_time = [ series.params[TSP_START_S]]
    lvl_w = series.params[TSP_LC_LVLS][1]
    o_data = [0] #[series.data[0]*lvl_w]

    try:
        f_Hz = series.params[TSP_TIMER_F_HZ]
    except:
        f_Hz = series.params[TSP_F_HZ]

    for i in range(0, len(series.data)):
        if series.time[i] == 0:
            o_data[-1] += series.data[i]
        else:
            if np.sign(series.data[i]) != np.sign(series.data[i-1]):
                time = o_time[-1] + ((series.time[i] ) / f_Hz)/2
                data = o_data[-1] + np.sign(series.data[i-1])*lvl_w/2
                o_time.append( time )
                o_data.append( data )
                time = o_time[-1] + ((series.time[i] ) / f_Hz)/2
                data = o_data[-2] + series.data[i]*lvl_w
                o_time.append( time )
                o_data.append( data )

            else:
                time = o_time[-1] + ((series.time[i] ) / f_Hz)
                data = o_data[-1] + series.data[i]*lvl_w
                o_time.append( time )
                o_data.append( data )

    o.time = np.array(o_time, dtype=np.float32)
    o.data = np.array(o_data, dtype=np.float32)
    return o.copy()

def lc_reconstruct_time(series):
    '''
    Generate arrows for a LC'd signal.

    Args:
        series (Timeseries): Input time series.

    Returns:
        Timeseries: Reconstructed arrows time series. Transforms time to absolute values.
    '''
    o = Timeseries(series.name + " LCrecTime")
    o.params.update(series.params)
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S
    o_time = [ series.params[TSP_START_S]]
    o_data = [0]

    try: o.params[TSP_F_HZ] = series.params[TSP_TIMER_F_HZ]
    except: pass
    f_Hz = o.params[TSP_F_HZ]

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
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S

    t_min_s = (min(series.time)+1)/series.params[TSP_F_HZ]
    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

    in_time = start_s + np.cumsum( series.time / series.params[TSP_F_HZ] )
    o_time = np.arange(start_s, end_s, t_min_s)

    lvl = 0
    lvls = series.params[TSP_LC_LVLS]
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

    missing = len(o_time) - current_index
    o_data[-missing:] = o_data[current_index-1]

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
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S

    t_min_s = (min(series.time)+1)/series.params[TSP_F_HZ]
    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

    in_time = start_s + np.cumsum(series.time / series.params[TSP_F_HZ])
    o_time = np.arange(start_s, end_s, t_min_s)

    lvls = series.params[TSP_LC_LVLS]
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
                # print(t_prev, t_next, lvl_prev, lvl_next)
                # Interpolate levels
                o_data[j] = np.interp(o_time[j], [t_prev, t_next], [lvl_prev, lvl_next])
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
    o.params[TSP_TIME_FORMAT] = TIME_FORMAT_ABS_S

    t_min_s = (min(series.time)+1)/series.params[TSP_F_HZ]
    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

    in_time = start_s + np.cumsum(series.time / series.params[TSP_F_HZ])
    o_time = np.arange(start_s, end_s, t_min_s)

    lvls = series.params[TSP_LC_LVLS]
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
    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

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

    missing = len(o_time) - current_index
    o_data[-missing:] = o_data[current_index-1]

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    o.params[TSP_F_HZ] = 1/t_min_s
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
    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

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
    o.params[TSP_F_HZ] = 1/t_min_s
    return o.copy()


from scipy.interpolate import interp1d

def rec_piecewise_poly_fmin(series, order=2):
    '''
    Reconstruct the LC signal through piecewise polynomial interpolation to the minimal sampling frequency possible.
    This approach is compatible with both LC-ADC and LC-subsampling.

    Args:
        series (Timeseries): Input time series, a sparse signal having in data the absolute amplitudes, and in time
                             the absolute times.
        order (int): The order of the polynomial used for interpolation between each pair of points.

    Returns:
        Timeseries: Reconstructed timeseries (at a fixed rate).
    '''
    o = Timeseries(series.name + " rec. Piecewise Polynomial Interpolation fmin")
    o.params.update(series.params)

    if len(series.time) < 2 : return None

    try: t_min_s = 1/series.params[TSP_F_HZ]
    except: t_min_s = min(np.diff(series.time))

    start_s = series.params[TSP_START_S]
    end_s   = series.params[TSP_END_S]

    o_time = np.arange(start_s, end_s, t_min_s)

    # Setup piecewise polynomial interpolation using interp1d
    # 'slinear', 'quadratic', 'cubic' or integer specifying the order of the spline interpolation
    if len(series.time) > order:
        interpolator = interp1d(series.time, series.data, kind=order, fill_value="extrapolate")
    else:
        # If not enough points for the desired order, fall back to linear
        interpolator = interp1d(series.time, series.data, kind='linear', fill_value="extrapolate")

    o_data = interpolator(o_time)

    o.data = np.array(o_data)
    o.time = np.array(o_time)
    o.params[TSP_F_HZ] = 1/t_min_s
    return o.copy()




def calculate_ser(rec, og):
    """
    Calculate the Signal to Error Ratio (SER) between two time series.

    Returns:
    float: The Signal to Error Ratio.
    """
    # Ensure the time series for lpfd is interpolated over the time points of rec
    interpolator = interp1d(og.time, og.data, kind='linear', bounds_error=False, fill_value='extrapolate')
    og_interpolated = interpolator(rec.time)

    # Calculate the error signal
    error_signal = rec.data - og_interpolated

    # Calculate the power of the reference signal and the error signal
    signal_power = np.mean(np.square(rec.data))
    error_power = np.mean(np.square(error_signal))

    # Compute SER as the ratio of signal power to error power
    if error_power == 0:
        return np.inf  # To handle the case of zero error power
    ser = signal_power / error_power
    # Convert SER to decibels
    ser_db = 10 * np.log10(ser)

    return ser_db