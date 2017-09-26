# sf test, try a series of spatial frequencies to get a curve, left right
import holocube.hc5 as hc
from numpy import *
import numpy as np

# count exps
test_num = 1

num_frames = 160

# a grating stimulus
sp = hc.stim.grating_class(hc.window, fast=True)

# a series of contrasts
sf = 20.*pi/180
sf = 1./sf
cs = 1
tf = np.round(logspace(log10(4), log10(20), 8))
os = array([0, pi])


for t in tf:
    for o in os:
        sp.add_grating(sf=sf, tf=t, o=o, c=cs, sd=.35, maxframes=num_frames)
anim_seq = arange(num_frames)

 # add the experiment
hc.scheduler.add_exp()

test_num_flash = hc.tools.test_num_flash(test_num, num_frames)

for i in arange(len(sp.gratings)):

    
    starts =  [[hc.window.set_viewport_projection, 0, 0],
               [hc.window.set_bg,                  [0,0,0,1.0]],
               [sp.choose_grating,                 i],
               [sp.on,                             True]]

    middles = [[hc.window.set_ref,                  0, test_num_flash],
               [sp.animate,                        anim_seq]]
    
    ends =    [[sp.on,                             False],
               [hc.window.set_viewport_projection, 0, 1]]

    # add each test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

    test_num += 1
    
    
# add the rest (shorter than default)
num_frames = 360
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts =  [[hc.window.set_far,         2],
           [hc.window.set_bg,          [0,0,0,1.0]],
           [hc.arduino.set_lmr_scale,  -.1],
           [rbar.set_ry,               0],
           [rbar.switch,               True] ]
middles = [[rbar.inc_ry,               hc.arduino.lmr]]
ends =    [[rbar.switch,               False],
           [hc.window.set_far,         2]]
 
hc.scheduler.add_rest(num_frames, starts, middles, ends)

