# Copyright 2024 EPFL
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Author: Juan Sapriza - juan.sapriza@epfl.ch


from timeseries import *
from processes import *

def first_level(lvls):
    return int( np.floor( len(lvls)/2 ) )

UP = 1
NO = 0
DN = -1
CHANGE = -1

def lcadc(series, lvls, save_last=True):
    o = Timeseries(series.name + " LCADC")
    first = first_level(lvls)
    o.params[TS_PARAMS_F_HZ] = 0
    now = first
    last_l = first
    nxt = 0
    dir = 0
    last_t = 0

    def save(i, last_t, dir, last_l):
        if save_last and last_l >= 0: #also save last time\
            last_t, last_l = save( last_l, last_t, NO, CHANGE )
        dt = series.time[i] - last_t
        o.time.append(dt)
        o.data.append(dir)
        return series.time[i], last_l

    def switch( i, nxt, dir, last_l):
        last_l = i
        dir *= CHANGE
        poss =  nxt + 2*dir
        if poss > 0 and poss < len(lvls):
            nxt = poss
        return dir, nxt, last_l

    def next(nxt, dir, last_l):
        last_l = CHANGE
        now = nxt
        nxt += dir
        return now, nxt, last_l


    start = 0
    while True:
        if series.data[start+1] > series.data[start]:
            # We started going up
            dir = UP
        elif series.data[start+1] < series.data[start]:
            dir = DN
        else:
            start += 1
            continue
        nxt = now + dir
        break

    o.time.append(0)
    o.data.append(dir)
    last_t = series.time[start]

    for i in range(start+1,len(series.data)):
        s = series.data[i]
        if dir == UP:
            if s > lvls[nxt]:
                last_t, last_l = save(i, last_t, dir, last_l)
                now, nxt, last_l = next(nxt, dir, last_l)
            elif s < lvls[now]:
                dir, nxt, last_l = switch(i, nxt,dir, last_l)
        elif dir == DN:
            if s < lvls[nxt]:
                last_t, last_l = save(i, last_t, dir, last_l)
                now, nxt, last_l = next(nxt, dir, last_l)
            elif s > lvls[now]:
                dir, nxt, last_l = switch(i,nxt,dir, last_l)
    return o


CMP_H = DIR_U = NEXT = 1
CMP_L = DIR_D = SWTC = -1
CMP_N = 0

class Comparator:
    def __init__( self ):
        self.high   = 0
        self.low    = 0
        self.ptr    = 0
        self.dir    = DIR_U
        self.cmp    = CMP_N
        self.lvls_n = 0
        self.buf    = Timeseries("All level crossings")
        self.prund  = Timeseries("debounced")
        self.diffd  = Timeseries("Differences between points")


    def set_comparators(self, lvls):
        self.lvls_n = len(lvls)
        if self.dir == DIR_U:
            self.high   = min(self.lvls_n-1,self.ptr + 1)
            self.low    = self.ptr
        elif self.dir == DIR_D:
            self.high   = self.ptr
            self.low    = max(0,self.ptr - 1)

    def compare(self, s, lvls ):
        try:
            if      s >= lvls[self.high]:   self.cmp = CMP_H
            elif    s <= lvls[self.low]:    self.cmp = CMP_L
            else:                           self.cmp = CMP_N
        except:
            print(lvls)
            print(self.high, self.low)


    def check_cross(self):
        case = self.cmp * self.dir
        if case == NEXT: self.next()
        if case == SWTC: self.switch()
        return case

    def next(self):
        self.ptr = max(0, min( self.ptr + self.dir, self.lvls_n ))

    def switch(self):
        self.dir *= -1

    def save(self, t):
        self.buf.data.append(self.ptr)
        self.buf.time.append(t)

    def debounce(self):
        for i in range(2, len(self.buf.data)):
            if      self.buf.data[i] == self.buf.data[i-1]: continue # It's the same level! keep the first one only
            elif    self.buf.data[i] == self.buf.data[i-2]: continue # We are going back and forth, ignore this one
            else:
                self.prund.data.append( self.buf.data[i] )
                self.prund.time.append( self.buf.time[i] )

    def differentiate(self, series ):
        # ### print(series.time[0], series.data[0])
        self.diffd.data.append( 0 )
        self.diffd.time.append( series.time[0] )
        for i in range(1,len(series.data) ):
            dl = series.data[i] - series.data[i-1] # This should be either -1, 0 or 1
            dt = series.time[i] - series.time[i-1]
            # print(f"{series.time[i]}/{series.data[i]:.1f} = {dt}/{dl}")
            self.diffd.data.append( dl )
            self.diffd.time.append( dt )



def lcadc_fil(series, lvls):
    o = Timeseries(series.name + f" LCfil({lvls[1]-lvls[0]})")
    first =  first_level(lvls)
    c = Comparator()
    c.ptr = first
    c.buf.data.append(first)
    c.buf.time.append(series.time[0])

    for i in range( 1, len(series.data) ):
        s = series.data[i]
        t = series.time[i]
        c.set_comparators(lvls)
        c.compare(s, lvls)
        case = c.check_cross()
        if case == NEXT:
            c.save(t)
    c.debounce()
    c.differentiate(c.prund)
    o.data = c.diffd.data
    o.time = c.diffd.time
    return o

def lcadc_naive(series, lvls):
    o = Timeseries(series.name + f" LCnaive({lvls[1]-lvls[0]})")
    first =  first_level(lvls)
    c = Comparator()
    c.ptr = first
    c.buf.data.append(first)
    c.buf.time.append(series.time[0])

    for i in range( 1, len(series.data) ):
        s = series.data[i]
        t = series.time[i]
        c.set_comparators(lvls)
        c.compare(s, lvls)
        case = c.check_cross()
        if case == NEXT:
            c.save(t)
    # c.debounce()
    c.differentiate(c.buf)
    o.data = c.diffd.data
    o.time = c.diffd.time
    return o

def lcadc_simple( series, lvl_width ):
    o = Timeseries( "LC simple" )
    current_level   = np.floor(series.data[0]/lvl_width)
    for i in range(1, len(series.data)):
        diff =  np.floor((series.data[i] - current_level*lvl_width)/lvl_width)
        if diff != 0:
            o.data.append(np.sign(diff))
            o.time.append( series.time[i] )
            current_level = current_level + np.sign(diff)
    return o

def lcadc_simple_debounce( series, lvl_width ):
    o = Timeseries( "LC simple" )
    current_level   = np.floor(series.data[0]/lvl_width)
    for i in range(1, len(series.data)):
        diff =  np.trunc(((series.data[i] - current_level*lvl_width)/lvl_width)).astype(int)
        if diff != 0:
            o.data.append(np.sign(diff))
            o.time.append( series.time[i] )
            current_level = current_level + np.sign(diff)
    return o


def lcadc_reconstruct(series, lvls, offset=0):
    o = Timeseries(series.name + " LCrec")
    first = first_level(lvls)
    lvl = first + offset
    o.time.append(series.time[0])
    o.data.append(0)
    for i in range(1, len(series.data)):
        o.time.append( o.time[i-1] + series.time[i] )
        lvl = int(min( max(0, lvl + series.data[i] ), len(lvls) -1 ))
        o.data.append( lvls[lvl] )
    return o

def lcadc_reconstruct_simple(series, lvls, start_lvl, start_time, end_time ):
    o = Timeseries(series.name + " LCrec")
    lvl = start_lvl
    o.time.append(start_time)
    o.data.append(lvls[lvl])
    for i in range(0, len(series.data)):
        o.time.append( series.time[i] )
        lvl = int(min( max(0, lvl + series.data[i] ), len(lvls) -1 ))
        o.data.append( lvls[lvl] )
    o.data.append(lvls[lvl])
    o.time.append(end_time)
    return o

def lcadc_reconstruct_time(series, height=16):
    o = Timeseries(series.name + " LCrecTime")
    o.time.append(series.time[0])
    o.time.append(series.time[0])
    o.time.append(series.time[0])
    o.data.append(0)
    o.data.append(0)
    o.data.append(0)
    dt = 0.00001
    i = 3
    for x in range(1, len(series.data)):
        t = o.time[i-2] + series.time[x]
        o.time.append( t -dt )
        o.time.append( t )
        o.time.append( t +dt )
        o.data.append( 0 )
        o.data.append( height*series.data[x] )
        o.data.append( 0 )
        i += 3
    return o

def lcadc_reconstruct_arrows( series ):
    o = Timeseries(series.name + " LCrecTime")
    for i in range(1, len(series.data)):
        t = np.sum( series.time[:i] )
        d = series.data[i]
        o.time.append(t)
        o.data.append(d)
    return o


def lc_task_detect_spike( series, length = 10, dt = 0.025 ):
    data = series.data[1:]
    time = series.time[1:]

    switch_indexes  = []
    current_value   = data[0]
    count           = length
    accum_time      = sum(time[:length])
    one_way         = 0

    for i in range(length, len(data)):
        if data[i] == current_value:
            count       = count + 1 if count < length else count
            accum_time  = accum_time + time[i] if count < length else accum_time + time[i] - time[ i-length ]
        else:
            one_way = 1 if (count == length and accum_time <= dt) else 0
            current_value, accum_time, count = data[i], 0, 0

        if count == length and accum_time <= dt and one_way == 1:
            current_value, accum_time, count = data[i], 0, 0
            switch_indexes.append(i - length + 2)

    return switch_indexes

def lc_task_detect_spike_online( series, length = 10, dt = 0.0025 ):
    o = Timeseries(series.name + " LC R-peak detection")
    data = series.data[1:]
    time = series.time[1:]
    count = 0
    blocked = 0

    for i in range(length, len(data)):
        # print(np.sign(data[i]) != np.sign(data[i-1]) or time[i] > dt, time[i] <= dt, count >= length)
        if np.sign(data[i]) != np.sign(data[i-1]) or time[i] > dt:
            if count >= length:
                if not blocked:
                    o.time.append( sum(series.time[:i+1]) )
                    count, blocked = 0, 1
                else:
                    count, blocked = 0, 0
            else:
                count = 0
        if time[i] <= dt:
            count += abs(data[i])
    return o



def lc_subsampler_C( series, lvl_w_b ):
    o = Timeseries("LC in C")
    lvl_w = 2**lvl_w_b

    current_lvl = 0                     # The last level to be crossed.
    lvl_up, lvl_down = 0, 0            # The value to be crossed to consider that a level was crossed.
    x_up, x_down = False, False         # Whether the sample crossed the upper or lower level
    dir = 0                             # The direction of the crossing (1=up, 0=down). THIS SIGNAL SHOULD BE EXPOSED.
    xing = False                        # Whether the sample crossed any level. THIS SIGNAL SHOULD BE EXPOSED.
    xings = 0                           # The count of crossings between two consecutive samples.
    skipped = 0                         # The count of samples that did not cross any level. Reset on every crossing.

    MAX_VAL = 2**16
    MIN_VAL = 0
    MAX_SKIP = 127
    MAX_XING = 255

    for i in range( len(series.data) ):
        dir = 0  # Reset the direction signal
        while True:
            xings = 0
            while True:
                # The upper level is computed.
                # If the difference between the current level and the upper bound of the range is less than a level,
                # the next level is set to the top of the range.
                # The crossing up is only computed if the current level is not already the top.
                # e.g. If the signal is stuck in 255, 255, 255 (in saturation, for instance)
                # then no crossings will be detected, although the sample is always coinciding with the upper level.
                # The same logic is applied to the bottom end of the range.
                lvl_up      = current_lvl if current_lvl >= MAX_VAL - lvl_w else current_lvl + lvl_w
                lvl_down    = current_lvl if current_lvl <= MIN_VAL + lvl_w else current_lvl - lvl_w
                x_up        = (current_lvl != lvl_up)   and (series.data[i] >= lvl_up)
                x_down      = (current_lvl != lvl_down) and (series.data[i] <= lvl_down)

                # The direction of the crossing is 1 if the crossing was with the upper level. Therefore, it is equal
                # to the signal that reports exactly that.
                # It is bit-wise-or'd to keep the value of direction between iterations (when the sample crossed
                # multiple levels - then, the last time, the x_up would be 0 and we don't want that to propagate to dir).
                dir |= x_up
                # A crossing is decteted if either level was crossed.
                xing = x_up or x_down
                # The count of crossings is kept by increasing the count on every crossing.
                xings += xing

                # Once the signal crosses one of the levels, the current level is set to it.
                if xing and dir:
                    current_lvl = lvl_up
                elif xing and not dir:
                    current_lvl = lvl_down

                # If there was a crossing, there could be more. The iteration is repeated until all level crossings
                # have been detected.
                # Additionally, it could happen that the crossings counter reaches its limit.
                # In that case, we also exit to generate a sample, but continue on the same sample until we have reached
                # the appropriate level.
                # Note that this scenario is very unlikely if the level width was properly selected.
                # If this protections wants to be by-passed, then the crossing counter could be forced to saturate and the
                # resulting LC signal will have an offset from then onwards (due to the missed crossings). If the DC level
                # of the signal is irrelevant for the processing task, this approximation could be acceptable.
                if not(xing and xings != MAX_XING):
                    break

            # If the sample crossed at least one level, or if we have skipped too many samples (the samples counter saturates),
            # then an acquisition needs to be performed.
            if xings or skipped == MAX_SKIP:
                # The data is formatted as:
                # MSBs    - Skipped samples count
                # middle  - Direction
                # LSBs    - Levels crossed.
                o.data.append( xings if dir else -xings )
                o.time.append( skipped )

                # If a sample is acquired, then the skipped samples is reset, otherwise it's increased.
                skipped = 0
            else:
                skipped += 1

            # If an acquisition was performed because the crossing counter reached it's limit, then we should re-iterate
            # over this sample until no more crossings are detected.
            if xings == MAX_XING:
                continue
            break

    return o





def lc_subsampler( series, lvl_w_b, time_in_skips = False ):
    o = Timeseries(series.name + f" LCsubs({lvl_w_b})")

    lvl_width     = 2**lvl_w_b
    current_lvl   = ((series.data[0]) // lvl_width)*lvl_width
    last_crossing = 0
    o.data.append(0)
    if time_in_skips:
        o.time.append(series.time[0])
    else:
        o.time.append(0)

    for i in range(1, len(series.data)):
        diff = (series.data[i] - current_lvl) // lvl_width
        if diff != 0:
            o.data.append(diff)
            if time_in_skips:
                o.time.append( i - last_crossing )
                last_crossing = i
            else:
                o.time.append( (series.time[i] - sum(o.time[ : len(o.time) ])) )
            current_lvl = max( 0, current_lvl + diff*lvl_width)

    # Average acquisition rate over the sampled period
    o.params[TS_PARAMS_F_HZ] = len(o.data)/series.params[TS_PARAMS_LENGTH_S]

    return o

## lcadc_naive is the LC algorithm
## This one instead should define the sampling format
def lcadc(analog_signal: Timeseries, lvl_width_fraction = 0.1):
    lvls = lvls_uniform_u32b_by_fraction(lvl_width_fraction)
    scaled_signal  = offset_to_pos_and_map( analog_signal, 32)
    return lcadc_naive(scaled_signal, lvls), lvls



def lc_analog(analog_signal: Timeseries, lvl_width_fraction = 0.1):
    lvls = lvls_uniform_u32b_by_fraction(lvl_width_fraction)
    scaled_signal  = offset_to_pos_and_map( analog_signal, 32)
    return lcadc_naive(scaled_signal, lvls), lvls

def lc_subsample(analog_signal, adc_fs_Hz = 100, adc_res_b = 8, lvl_w_b = 1 ):
    fradc = ADC(    name        = "ADC for LC subsampling",
                    units       = "Normalized",
                    f_Hz        = adc_fs_Hz,
                    res_b       = adc_res_b,
                    dynRange    = [-1, 1],
                    series      = analog_signal )
    return lc_subsampler(fradc.conversion, lvl_w_b ), fradc.conversion


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

def lvls_shifted():
    lvls_a = [ 2, 4, 8, 16, 32, 64, 128 ]
    lvls_b = [3 + l for l in lvls_a ]
    lvls_c = lvls_a + lvls_b
    lvls_c.sort()
    lvls_d = [0]+lvls_c
    lvls_c.reverse()
    lvls_c = [-l for l in lvls_c]
    lvls_d = lvls_c+lvls_d
    return lvls_d

def lvls_pwr2():
    return [ -128, -96, -64, -48, -32, -24, -16, -12, -8, -6, -4, 0,
            4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

def lvls_unif(width, bits):
    return list(range(-2**(bits-1)-1,2**(bits-1),width))

def lvls_centered():
    return [-128, -95, -63, -47, -39, -32, -27, -23, -15, 0, 15, 23, 27, 32, 39, 47, 63, 95, 128]

def lvls_pwr2_high():
    return [ -128, -96, -64, -48, -32, -24, -16, -12, -8, 8, 12, 16, 24, 32, 48, 64, 96, 128]

def lvls_uniform():
    return list(range(-128,129,16))

def lvls_uniform_dense():
    return list(range(-128,129,8))

def lvls_uniform_u32b(width):
    return list(range(0,np.iinfo(np.uint32).max, width))

def lvls_uniform_u32b_by_fraction(fraction):
    l = list(np.arange(0,np.iinfo(np.uint32).max, np.iinfo(np.uint32).max/fraction))
    return l