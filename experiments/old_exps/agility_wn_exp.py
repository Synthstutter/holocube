# horizon test with rotation

import holocube.hc5 as hc5
from numpy import *
from holocube.tools import mseq

# horizon test
expi = 1

# how long for the exp?
numframes = 1023

# a set of points
pts = hc5.stim.Points(hc5.window, 20000, dims=[[-5,5],[-5,5],[-5,5]], color=.5, pt_size=3)


# the motions
wn1 = cumsum(mseq(2,10,0,1))
wn1_dir = array(sign(ediff1d(wn1,None, 0)), dtype='int')

lights1 = mod(cumsum(wn1_dir),3)
lights1 = array([[0,e*127,0] for e in lights1])
lights1[-1] = 0

tampl = .04
rampl = 1
hc5.scheduler.add_exp()
# slip
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_px,        wn1*tampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_pos, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)

# lift
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_py,        wn1*tampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_pos, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)

# thrust
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_pz,        wn1*tampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_pos, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)

# pitch
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_rx,        wn1*rampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_ori, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)

# yaw
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_ry,        wn1*rampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_ori, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)

# roll
expnumspikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.set_rz,        wn1*rampl],
           [hc5.window.set_ref, 0, expnumspikes],
           [hc5.window.set_ref, 1, lights1]]

ends =    [[pts.on,            0],
           [hc5.window.reset_ori, 1],
           [pts.shuffle,       1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)


# add the rest
num_frames = 300
rbar = hc5.stim.cbarr_class(hc5.window, dist=1)

starts =  [[hc5.window.set_far,         2],
           [hc5.window.set_bg,          [0.0,0.0,0.0,1.0]],
           [hc5.arduino.set_lmr_scale,  -.1],
           [rbar.set_ry,               0],
           [rbar.switch,               True] ]
middles = [[rbar.inc_ry,               hc5.arduino.lmr]]
ends =    [[rbar.switch,               False],
           [hc5.window.set_far,         2]]
 
hc5.scheduler.add_rest(num_frames, starts, middles, ends)

