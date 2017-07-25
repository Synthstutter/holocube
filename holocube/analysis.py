# make some tools for experiments and analysis
 
from pylab import *
import numpy 
import os
from scipy.stats import *

def plot_trace(data, dt=.001, stderr=True, zerostart=False, orientation='horizontal', **kwargs):
    '''Plot the mean data trace, and the standard deviation in light color.'''
    mn = data.mean(0)
    err = data.std(0)
    if stderr==True: err /= sqrt(len(data))
    if zerostart: mn-=mn[0]
    times = arange(len(mn))*dt

    xs = concatenate([times, times[::-1]])
    ys = concatenate([mn + err, mn[::-1]-err[::-1]])

    if orientation=='horizontal':
        lines = plot(times, mn, **kwargs)
        c = lines[-1].get_color()
        fill(xs, ys, color=c, alpha=.1)
    else: # vertical
        lines = plot(mn, times, **kwargs)
        c = lines[-1].get_color()
        fill(ys, xs, color=c, alpha=.1)
        ylim(times[-1], 0) #time goes downward

def plot_corr_bars(r_data, labels=None, nsshow=False, start=0, **kwargs):
    data = r_data
    data_mean = data.mean(1)
    data_sem = data.std(1)/sqrt(r_data.shape[1])
    bar(arange(r_data.shape[0]) + start -.4, data_mean, yerr=data_sem, **kwargs)
    if labels: xticks(arange(r_data.shape[0]) + start, labels)
    else: xticks(arange(r_data.shape[0]) + start, [str(num) for num in arange(r_data.shape[0]) + start])
    bot, top = gca().get_ylim()
    inc = abs(top-bot)/10.
    # t-tests
    for i in arange(r_data.shape[0]):
        for j in arange(i+1, r_data.shape[0]):
            test = ttest_rel(r_data[i], r_data[j])
            ch='ns'
            if test[-1]/2.<=.05: ch='*t = %2.2f p = %2.4f'%test
            if test[-1]/2.<=.01: ch='**t = %2.2f p = %2.4f'%test
            if test[-1]/2.<=.001: ch='***t = %2.2f p = %2.4f'%test
            if nsshow or ch != 'ns':
                plot(array([i,i,j,j])+start, [top-inc/2., top, top, top-inc/2])
                text(mean([i, j])+start, top, ch, va='center', ha='center', backgroundcolor='w',\
                     fontsize=8)
                top += inc

def plot_test_bars(data, labels=None, nsshow=False, start=0, **kwargs):
    numbars, numsamps = data.shape
    data_mean = data.mean(1)
    data_sem = data.std(1)/sqrt(data.shape[1])
    bar(arange(numbars) + start -.4, data_mean, yerr=data_sem, **kwargs)
    if labels: xticks(arange(numbars) + start, labels)
    else: xticks(arange(numbars) + start, [str(num) for num in arange(numbars) + start])
    bot, top = gca().get_ylim()
    inc = abs(top-bot)/10.
    # test
    aov_f, aov_p = stats.f_oneway(*data)
    groups = hstack(labels)


def plot_pcorr_bars(p_data, labels=None, nsshow=False, start=0, **kwargs):
    ath_data = arctanh(p_data)
    ath_data_mean = ath_data.mean(1)
    ath_data_sem = ath_data.std(1)/sqrt(p_data.shape[1])
    data_mean = tanh(ath_data_mean)
    data_sem = array([data_mean - tanh(ath_data_mean - ath_data_sem),\
                      tanh(ath_data_mean + ath_data_sem) - data_mean])
    bar(arange(p_data.shape[0]) + start -.4, data_mean, yerr=data_sem, **kwargs)
    if labels != None: xticks(arange(p_data.shape[0]) + start, labels)
    else: xticks(arange(p_data.shape[0]) + start, [str(num) for num in arange(p_data.shape[0]) + start])
    bot, top = gca().get_ylim()
    inc = abs(top-bot)/10.
    # t-tests
    for i in arange(p_data.shape[0]):
        for j in arange(i+1, ath_data.shape[0]):
            test = ttest_rel(ath_data[i], ath_data[j])
            ch='ns'
            if test[-1]/2.<=.05: ch='*t = %2.2f p = %2.4f'%test
            if test[-1]/2.<=.01: ch='**t = %2.2f p = %2.4f'%test
            if test[-1]/2.<=.001: ch='***t = %2.2f p = %2.4f'%test
            if nsshow or ch != 'ns':
                plot(array([i,i,j,j])+start, [top-inc/2., top, top, top-inc/2])
                text(mean([i, j])+start, top, ch, va='center', ha='center', backgroundcolor='w',\
                     fontsize=8)
                top += inc

def edge_inds(data, duration=9, thresh=6):
    '''Returns the indexes of rising square wave edges that last at
    least duration samples and rise at least thresh standard
    deviations above baseline of the correlated signal.'''
    # first cross correlate with a rising edge kernel
    kernel = ones((duration))
    kernel[0] = -1.
    c = correlate(data-data.mean(), kernel, 'valid')
    # isolate the regions above thresh to work with a small list of candidates
    csorted = sort(c)
    cbots, ctops = csorted[:(len(c)/2)].mean(), median(csorted[-100:])
    cmid = cbots + (cbots + ctops)/2.
    c_hi_inds = where(c>cmid)[0]
    # find the boundaries of above thresh regions (skip more than one index)
    dc_hi_inds = ediff1d(c_hi_inds, to_begin=3, to_end=3)
    strts = where(dc_hi_inds>2)[0]
    inds = zip(take(c_hi_inds, strts[:-1]), take(c_hi_inds, strts[1:], mode='clip'))
    # one point is higher than its neighbors, return one index for each region
    return array([i+c[i:j].argmax() for i,j in inds])

def get_motion(d1, d2, d3, duration=9, thresh=6):
    out = zeros(len(d1))
    out[edge_inds(d1, duration, thresh)] = 1
    out[edge_inds(d2, duration, thresh)] = 2
    out[edge_inds(d3, duration, thresh)] = 3
    nz = nonzero(out)
    dnout = ediff1d(out[nz], to_end=0)
    dnout[dnout==-2] = 1
    dnout[dnout==2] = -1
    out[nz] = dnout
    return out

def start_inds(id_data, thresh=200):
    '''Returns the sorted list of starting indexes for each
    experiment, where sequential pulses are less than thresh samples
    apart, but experiments are more than thresh samples apart.'''
    # find the edges, and how many samples separate them
    einds = edge_inds(id_data)
    einddiffs = ediff1d(einds, to_begin=0, to_end=thresh+1)
    # where are there breaks longer than thresh?
    skipinds = where(einddiffs>thresh)[0]
    # now count the pulses in between thresh-sized breaks, subtract one for indexing
    order = ediff1d(skipinds, to_begin=skipinds[0])
    startindsinds = concatenate([[0], skipinds[:-1]])
    return einds[startindsinds][order.argsort()]
    
def get_exp_data(cdir, id_chan=5, rest_time=4000):

    fns = os.listdir(cdir)
    fns = [fn for fn in fns if fn.endswith('.npy')]
    fns.sort()
    num_trials = len(fns)

    # check the first trial just to get the number of exps
    trial_data = numpy.load(fns[0])
    num_chans = trial_data.shape[0]
    starts = start_inds(trial_data[id_chan])
    starts.sort()
    num_exps = len(starts)
    nsamps = min(diff(starts))-rest_time #get the smallest len between starts (and 1000 (1 sec) less than this)

    out = zeros((num_chans, num_exps, num_trials, nsamps))

    for trial_num in range(num_trials):
        fn = fns[trial_num]
        print (trial_num, fn)
        trial_data = numpy.load(fn)
        starts = start_inds(trial_data[id_chan])

        for chan_num in range(num_chans):
            for exp_num in range(num_exps):
                out[chan_num, exp_num, trial_num] = trial_data[chan_num, starts[exp_num]:(starts[exp_num] + nsamps)]
    return out

def edge_ind_vals(data, nvals=2, duration=4, thresh=.2):
    '''Returns the indexes of rising square wave edges that last at
    least duration samples and rise at least thresh standard
    deviations above baseline of the correlated signal.'''
    out = zeros(len(data), dtype='int')
    # first cross correlate with a rising edge kernel
    kernel = ones((duration))
    kernel[0] = -1.
    c = correlate(data-data.mean(), kernel, 'valid')
    # isolate the regions above thresh stds to work with a small list of candidates
    cptp = c.ptp()
    c_hi_inds = where(c>cptp*thresh)[0]
    # find the boundaries of above thresh regions (skip more than one index)
    dc_hi_inds = ediff1d(c_hi_inds, to_begin=3, to_end=3)
    strts = where(dc_hi_inds>2)[0]
    inds = zip(take(c_hi_inds, strts[:-1]), take(c_hi_inds, strts[1:], mode='clip'))
    # one point is higher than its neighbors, return one index for each region
    inds = [ind for ind in inds if ind[0] != ind[1]]
    xi = array([i+c[i:j].argmax() for i,j in inds])
    # the value of the highest point in each region of the correlation
    xv = c[xi]
    sxv = concatenate(([-inf], sort(xv)))
    # find the indexes of the largest gaps in peak height
    if hasattr(nvals, '__iter__'): nvals = concatenate([[0],nvals])
    else: nvals = arange(nvals+1, dtype='int')
    argbreaks = sort(argsort(diff(sxv))[::-1][:(len(nvals)-1)])
    for i in range(len(nvals)-1):
        out[xi[xv>sxv[argbreaks[i]]]] += 1 #add one each level
    return choose(out, nvals)

    
def test_inds(id_data, thresh=.2, duration=4, min_interval=35):
    '''New function to grab exp durations and sort them'''
    inds = edge_ind_vals(id_data, 2, thresh=thresh, duration=duration)
    # get rid of any pips that occur less than min_interval after another
    for pip_ind in where(inds==2)[0]: inds[(pip_ind+1):(pip_ind + min_interval)] = 0
    tinds = where(inds==2)[0].reshape(-1,2)
    order = []
    for i in range(len(tinds)):
        order.append(sum(inds[range(*tinds[i])]))
    return tinds[argsort(order)]
    
def exp_data(cdir='./', id_chan=2, thresh=.2, duration=4):
    fns = os.listdir(cdir)
    fns = [cdir+'/'+fn for fn in fns if fn.endswith('.npy')]
    fns.sort()
    num_trials = len(fns)

    # check the first trial just to get the number of exps
    trial_data = numpy.load(fns[0])
    num_chans = trial_data.shape[0]
    tinds = test_inds(trial_data[id_chan], thresh=thresh, duration=duration)
    num_tests = len(tinds)
    num_samps = max(diff(tinds))[0]

    out = zeros((num_chans, num_tests, num_trials, num_samps))

    for trial_num in range(num_trials):
        fn = fns[trial_num]
        print (trial_num, fn)
        trial_data = numpy.load(fn)
        tinds = test_inds(trial_data[id_chan], thresh=thresh, duration=duration)

        for chan_num in range(num_chans):
            for test_num in range(num_tests):
                put(out[chan_num, test_num, trial_num], range(num_samps), trial_data[chan_num, range(*tinds[test_num])], mode='clip')
    return out


def spikelist(num, llen, dist=10):
    sl = zeros((llen))
    sl[arange(0,num)*dist] = 1
    return sl

def triangle_wave(numframes, gain=1., cycles=1.):
    return gain*arcsin(sin(linspace(0,cycles*2*pi,numframes)))


def numflash(num, llen, dist=10, col_1=255, col_2=96):
    '''take a test length and a number and produce a sequence to send to flash4'''
    sl = zeros((llen), dtype='int')
    sl[(arange(0,num)+2)*dist] = col_2
    sl[0] = col_1
    sl[-2] = col_1
    sl[-1] = 0
    return sl

def seqflash(seq, valmap=[[-1,96],[1,255]]):
    out = zeros((len(seq)), dtype='int')
    for i in arange(valmap):
        out[seq==valmap[i][0]] = valmap[i][1]
    return out

def seqflash3(seq, valmap=[[-1,96],[1,255]]):
    out = zeros((len(seq)), dtype='int')
    for i in arange(valmap):
        out[seq==valmap[i][0]] = valmap[i][1]
    return out.reshape((-1,3))

# def exp_flash(num, num_pts, ref=0, dist=10, col_1=255, col_2=96):
#     '''take a test length and a number and produce a sequence to send to ref.set_ref_color.'''
#     sl = zeros((num_pts, 4), dtype='int')
#     sl[:,0] = ref
#     sl[(arange(0,num)+2)*dist,1] = col_2
#     sl[0,1] = col_1
#     sl[-2,1] = col_1
#     sl[-1,1] = 0
#     return sl

# def exp_flash(num, num_pts, dist=10, col_1=255, col_2=96):
#     '''take a test length and a number and produce a sequence to send to ref.set_ref_color.'''
#     sl = zeros((num_pts, 3), dtype='int')
#     # sl[:,0] = ref
#     sl[(arange(0,num)+2)*dist,1] = col_2
#     sl[0,1] = col_1
#     sl[-2,1] = col_1
#     sl[-1,1] = 0
#     return sl #.tolist()

def exp_flash(num, num_pts, dist=10, col_1=255, col_2=96):
    '''take a test length and a number and produce a sequence to send to ref.set_ref_color.'''
    sl = zeros((num_pts), dtype='O')
    for i in range(num_pts):
        sl[i] = (0,0,0)
    for i in range(num):
        sl[(i + 2)*dist] = (0,col_2,0)
    sl[0] = (0,col_1,0)
    sl[-2] = (0,col_1,0)
    return sl



# some tools to extract pos ori data
def extract_pos(pos_ori):
    return pos_ori[0]

def extract_yaw(pos_ori):
    return arctan2(pos_ori[...,3,0], pos_ori[...,3,2])

def extract_pitch(pos_ori):
    return arctan2(pos_ori[...,3,1], pos_ori[...,3,2])
    

# msequence from a matlab script

def mseq(baseVal, powerVal, shift=1, whichSeq=1):
    bitNum=baseVal**powerVal-1;
    register=ones([powerVal]);
    if baseVal==2:
        if powerVal==2:
            tap = [[1,2]]
        elif powerVal==3:
            tap = [[1,3],
                   [2,3]]
        elif powerVal==4:
            tap = [[1,4],
                   [3,4]]
        elif powerVal==5:
            tap = [[2,5],
                   [3,5],
                   [1,2,3,5],
                   [2,3,4,5],
                   [1,2,4,5],
                   [1,3,4,5]]
        elif powerVal==6:
            tap = [[1,6],
                   [5,6],
                   [1,2,5,6],
                   [1,4,5,6],
                   [1,3,4,6],
                   [2,3,5,6]]
        elif powerVal==7:
            tap = [[1,7],
                   [6,7],
                   [3,7],
                   [4,7],
                   [1,2,3,7],
                   [4,5,6,7],
                   [1,2,5,7],
                   [2,5,6,7],
                   [2,3,4,7],
                   [3,4,5,7],
                   [1,3,5,7],
                   [2,4,6,7],
                   [1,3,6,7],
                   [1,4,6,7],
                   [2,3,4,5,6,7],
                   [1,2,3,4,5,7],
                   [1,2,4,5,6,7],
                   [1,2,3,5,6,7]]
        elif powerVal==8:
            tap = [[1,2,7,8],
                   [1,6,7,8],
                   [1,3,5,8],
                   [3,5,7,8],
                   [2,3,4,8],
                   [4,5,6,8],
                   [2,3,5,8],
                   [3,5,6,8],
                   [2,3,6,8],
                   [2,5,6,8],
                   [2,3,7,8],
                   [1,5,6,8],
                   [1,2,3,4,6,8],
                   [2,4,5,6,7,8],
                   [1,2,3,6,7,8],
                   [1,2,5,6,7,8]]
        elif powerVal==9:
            tap = [[4,9],
                   [5,9],
                   [3,4,6,9],
                   [3,5,6,9],
                   [4,5,8,9],
                   [1,4,5,9],
                   [1,4,8,9],
                   [1,5,8,9],
                   [2,3,5,9],
                   [4,6,7,9],
                   [5,6,8,9],
                   [1,3,4,9],
                   [2,7,8,9],
                   [1,2,7,9],
                   [2,4,7,9],
                   [2,5,7,9],
                   [2,4,8,9],
                   [1,5,7,9],
                   [1,2,4,5,6,9],
                   [3,4,5,7,8,9],
                   [1,3,4,6,7,9],
                   [2,3,5,6,8,9],
                   [3,5,6,7,8,9],
                   [1,2,3,4,6,9],
                   [1,5,6,7,8,9],
                   [1,2,3,4,8,9],
                   [1,2,3,7,8,9],
                   [1,2,6,7,8,9],
                   [1,3,5,6,8,9],
                   [1,3,4,6,8,9],
                   [1,2,3,5,6,9],
                   [3,4,6,7,8,9],
                   [2,3,6,7,8,9],
                   [1,2,3,6,7,9],
                   [1,4,5,6,8,9],
                   [1,3,4,5,8,9],
                   [1,3,6,7,8,9],
                   [1,2,3,6,8,9],
                   [2,3,4,5,6,9],
                   [3,4,5,6,7,9],
                   [2,4,6,7,8,9],
                   [1,2,3,5,7,9],
                   [2,3,4,5,7,9],
                   [2,4,5,6,7,9],
                   [1,2,4,5,7,9],
                   [2,4,5,6,7,9],
                   [1,3,4,5,6,7,8,9],
                   [1,2,3,4,5,6,8,9]]
        elif powerVal==10:
            tap = [[3,10],
                   [7,10],
                   [2,3,8,10],
                   [2,7,8,10],
                   [1,3,4,10],
                   [6,7,9,10],
                   [1,5,8,10],
                   [2,5,9,10],
                   [4,5,8,10],
                   [2,5,6,10],
                   [1,4,9,10],
                   [1,6,9,10],
                   [3,4,8,10],
                   [2,6,7,10],
                   [2,3,5,10],
                   [5,7,8,10],
                   [1,2,5,10],
                   [5,8,9,10],
                   [2,4,9,10],
                   [1,6,8,10],
                   [3,7,9,10],
                   [1,3,7,10],
                   [1,2,3,5,6,10],
                   [4,5,7,8,9,10],
                   [2,3,6,8,9,10],
                   [1,2,4,7,8,10],
                   [1,5,6,8,9,10],
                   [1,2,4,5,9,10],
                   [2,5,6,7,8,10],
                   [2,3,4,5,8,10],
                   [2,4,6,8,9,10],
                   [1,2,4,6,8,10],
                   [1,2,3,7,8,10],
                   [2,3,7,8,9,10],
                   [3,4,5,8,9,10],
                   [1,2,5,6,7,10],
                   [1,4,6,7,9,10],
                   [1,3,4,6,9,10],
                   [1,2,6,8,9,10],
                   [1,2,4,8,9,10],
                   [1,4,7,8,9,10],
                   [1,2,3,6,9,10],
                   [1,2,6,7,8,10],
                   [2,3,4,8,9,10],
                   [1,2,4,6,7,10],
                   [3,4,6,8,9,10],
                   [2,4,5,7,9,10],
                   [1,3,5,6,8,10],
                   [3,4,5,6,9,10],
                   [1,4,5,6,7,10],
                   [1,3,4,5,6,7,8,10],
                   [2,3,4,5,6,7,9,10],
                   [3,4,5,6,7,8,9,10],
                   [1,2,3,4,5,6,7,10],
                   [1,2,3,4,5,6,9,10],
                   [1,4,5,6,7,8,9,10],
                   [2,3,4,5,6,8,9,10],
                   [1,2,4,5,6,7,8,10],
                   [1,2,3,4,6,7,9,10],
                   [1,3,4,6,7,8,9,10]]
        elif powerVal==11:
            tap=[[9,11]]
        elif powerVal==12:
            tap=[[6,8,11,12]]
        elif powerVal==13:
            tap=[[9,10,12,13]]
        elif powerVal==14:
            tap=[[4,8,13,14]]
        elif powerVal==15:
            tap=[[14,15]]
        elif powerVal==16:
            tap=[[4,13,15,16]]
        elif powerVal==17:
            tap=[[14,17]]
        elif powerVal==18:
            tap=[[11,18]]
        elif powerVal==19:
            tap=[[14,17,18,19]]
        elif powerVal==20:
            tap=[[17,20]]
        elif powerVal==21:
            tap=[[19,21]]
        elif powerVal==22:
            tap=[[21,22]]
        elif powerVal==23:
            tap=[[18,23]]
        elif powerVal==24:
            tap=[[17,22,23,24]]
        elif powerVal==25:
            tap=[[22,25]]
        elif powerVal==26:
            tap=[[20,24,25,26]]
        elif powerVal==27:
            tap=[[22,25,26,27]]
        elif powerVal==28:
            tap=[[25,28]]
        elif powerVal==29:
            tap=[[27,29]]
        elif powerVal==30:
            tap=[[7,28,29,30]]
        else:
            print ('M-sequence {}**{} is not defined'.format(baseVal,powerVal))
    elif baseVal==3:
        if powerVal==2:
            tap = [[2,1],
                   [1,1]]
        elif powerVal==3:
            tap=[[0,1,2],
                 [1,0,2],
                 [1,2,2],
                 [2,1,2]]
        elif powerVal==4:
            tap=[[0,0,2,1],
                 [0,0,1,1],
                 [2,0,0,1],
                 [2,2,1,1],
                 [2,1,1,1],
                 [1,0,0,1],
                 [1,2,2,1],
                 [1,1,2,1]]
        elif powerVal==5:
            tap=[[0,0,0,1,2], 
                 [0,0,0,1,2],
                 [0,0,1,2,2],
                 [0,2,1,0,2],
                 [0,2,1,1,2],
                 [0,1,2,0,2],
                 [0,1,1,2,2],
                 [2,0,0,1,2],
                 [2,0,2,0,2],
                 [2,0,2,2,2],
                 [2,2,0,2,2],
                 [2,2,2,1,2],
                 [2,2,1,2,2],
                 [2,1,2,2,2],
                 [2,1,1,0,2],
                 [1,0,0,0,2],
                 [1,0,0,2,2],
                 [1,0,1,1,2],
                 [1,2,2,2,2],
                 [1,1,0,1,2],
                 [1,1,2,0,2]]
        elif powerVal==6:
            tap=[[0,0,0,0,2,1],
                 [0,0,0,0,1,1],
                 [0,0,2,0,2,1],
                 [0,0,1,0,1,1],
                 [0,2,0,1,2,1],
                 [0,2,0,1,1,1],
                 [0,2,2,0,1,1],
                 [0,2,2,2,1,1],
                 [2,1,1,1,0,1],
                 [1,0,0,0,0,1],
                 [1,0,2,1,0,1],
                 [1,0,1,0,0,1],
                 [1,0,1,2,1,1],
                 [1,0,1,1,1,1],
                 [1,2,0,2,2,1],
                 [1,2,0,1,0,1],
                 [1,2,2,1,2,1],
                 [1,2,1,0,1,1],
                 [1,2,1,2,1,1],
                 [1,2,1,1,2,1],
                 [1,1,2,1,0,1],
                 [1,1,1,0,1,1],
                 [1,1,1,2,0,1],
                 [1,1,1,1,1,1]]
        elif powerVal==7:
            tap=[[0,0,0,0,2,1,2],
                 [0,0,0,0,1,0,2],
                 [0,0,0,2,0,2,2],
                 [0,0,0,2,2,2,2],
                 [0,0,0,2,1,0,2],
                 [0,0,0,1,1,2,2],
                 [0,0,0,1,1,1,2],
                 [0,0,2,2,2,0,2],
                 [0,0,2,2,1,2,2],
                 [0,0,2,1,0,0,2],
                 [0,0,2,1,2,2,2],
                 [0,0,1,0,2,1,2],
                 [0,0,1,0,1,1,2],
                 [0,0,1,1,0,1,2],
                 [0,0,1,1,2,0,2],
                 [0,2,0,0,0,2,2],
                 [0,2,0,0,1,0,2],
                 [0,2,0,0,1,1,2],
                 [0,2,0,2,2,0,2],
                 [0,2,0,2,1,2,2],
                 [0,2,0,1,1,0,2],
                 [0,2,2,0,2,0,2],
                 [0,2,2,0,1,2,2],
                 [0,2,2,2,2,1,2],
                 [0,2,2,2,1,0,2],
                 [0,2,2,1,0,1,2],
                 [0,2,2,1,2,2,2]]
        else:
            print ('M-sequence {}**{} is not defined'.format(baseVal,powerVal))
    elif baseVal==5:
        if powerVal==2:
            tap=[[4,3],
                 [3,2],
                 [2,2],
                 [1,3]]
        if powerVal==3:
            tap=[[0,2,3],
                 [4,1,2],
                 [3,0,2],
                 [3,4,2],
                 [3,3,3],
                 [3,3,2],
                 [3,1,3],
                 [2,0,3],
                 [2,4,3],
                 [2,3,3],
                 [2,3,2],
                 [2,1,2],
                 [1,0,2],
                 [1,4,3],
                 [1,1,3]]
        if powerVal==4:
            tap=[[0,4,3,3],
                 [0,4,3,2],
                 [0,4,2,3],
                 [0,4,2,2],
                 [0,1,4,3],
                 [0,1,4,2],
                 [0,1,1,3],
                 [0,1,1,2],
                 [4,0,4,2],
                 [4,0,3,2],
                 [4,0,2,3],
                 [4,0,1,3],
                 [4,4,4,2],
                 [4,3,0,3],
                 [4,3,4,3],
                 [4,2,0,2],
                 [4,2,1,3],
                 [4,1,1,2],
                 [3,0,4,2],
                 [3,0,3,3],
                 [3,0,2,2],
                 [3,0,1,3],
                 [3,4,3,2],
                 [3,3,0,2],
                 [3,3,3,3],
                 [3,2,0,3],
                 [3,2,2,3],
                 [3,1,2,2],
                 [2,0,4,3],
                 [2,0,3,2],
                 [2,0,2,3],
                 [2,0,1,2],
                 [2,4,2,2],
                 [2,3,0,2],
                 [2,3,2,3],
                 [2,2,0,3],
                 [2,2,3,3],
                 [2,1,3,2],
                 [1,0,4,3],
                 [1,0,3,3],
                 [1,0,2,2],
                 [1,0,1,2],
                 [1,4,1,2],
                 [1,3,0,3],
                 [1,3,1,3],
                 [1,2,0,2],
                 [1,2,4,3],
                 [1,1,4,2]]
        else:
            print ('M-sequence {}**{} is not defined'.format(baseVal,powerVal))
    elif baseVal==5:
        if powerVal==2:
            tap=[[1,1],
                 [1,2]]
        else:
            print ('M-sequence {}**{} is not defined'.format(baseVal,powerVal))

    ms=zeros([bitNum])

    if whichSeq == None:
        whichSeq = ceil(rand(1)*len(tap))
    else:
        if (whichSeq > len(tap)) or (whichSeq < 1):
            print (' wrapping arround!')
            whichSeq = (whichSeq%len(tap)) + 1

    weights=zeros([powerVal])

    if baseVal==2:
        weights[array(tap[whichSeq-1])-1] = 1
    elif baseVal>2:
        weights = tap[whichSeq-1]

    for i in arange(bitNum):
        ms[i] = (dot(weights,register)+baseVal) % baseVal
        register[1:] = register[:-1]
        register[0] = ms[i]
        
    if not shift==None:
        shift = shift%len(ms)
        ms = concatenate([ms[shift:], ms[:shift]])

    if baseVal==2:
        ms = ms*2-1
    elif baseVal==3:
        ms[ms==2]=-1
    elif baseVal==5:
        ms[ms==4]=-1
        ms[ms==2]=-2
    elif baseVal==9:
        ms[ms==5]=-1
        ms[ms==6]=-2
        ms[ms==7]=-3
        ms[ms==8]=-4
    else:
        print ('Wrong baseVal!')

    return ms

