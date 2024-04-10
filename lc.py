from timeseries import *


def first_level(lvls):
    return int( np.floor( len(lvls)/2 ) )

UP = 1
NO = 0
DN = -1
CHANGE = -1

def lcadc(series, lvls, save_last=True):
    o = Timeseries(series.name + " LCADC")
    first = first_level(lvls)
    o.f_Hz = 0
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


def lcadc_reconstruct(series, lvls, offset=0):
    o = Timeseries(series.name + " LCrec")
    first = first_level(lvls)
    lvl = first + offset
    o.time.append(series.time[0])
    o.data.append(lvls[lvl])
    for i in range(1, len(series.data)):
        o.time.append( o.time[i-1] + series.time[i] )
        lvl = min( max(0, lvl + series.data[i] ), len(lvls) -1 )
        o.data.append( lvls[lvl] )
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
        o.data.append( height if series.data[x] >0 else -height )
        o.data.append( 0 )
        i += 3
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
            switch_indexes.append(i - length + 2)
            current_value, accum_time, count = data[i], 0, 0

    return switch_indexes





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