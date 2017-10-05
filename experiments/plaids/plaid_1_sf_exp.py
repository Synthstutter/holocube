# splaid rotate to 
import holocube.hc5 as hc
from numpy import *

# count exps
test_num = 1

num_frames = 240

# a grating stimulus
sp = hc.stim.grating_class(hc.window, fast=True)

# a series of rotations
# tfs = logspace(0, 4, 16, base=2) #cy/sec
tf = 5
sfs = logspace(-5, -3, 6, base=2) #cy/deg
sfs = sfs *180./pi #cy/deg -> cy/rad
# sf = .03 *180/pi
os = array([0, pi])
cs = linspace(0., 1., 7)

for sf in sfs:
    for c in cs:
        sp.add_plaid(sf1=sf, tf1=tf, o1=11*pi/12, c1=c,
                     sf2=sf, tf2=tf, o2=1*pi/12, c2=1-c, 
                     sdb=.35, maxframes=num_frames)

anim_seq = arange(num_frames)

# add the experiment
hc.scheduler.add_exp()

for i in arange(len(sp.gratings)):
    sf = i/7
    sf += 1
    c = i%7
    c += 2
    
    sf_flash = hc.tools.test_num_flash(sf, num_frames)
    contrast_flash = hc.tools.test_num_flash(c, num_frames)
    
    starts =  [[hc.window.set_viewport_projection, 0, 0],
               [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
               [sp.choose_grating,         i],
               [sp.on,                     True]]

    middles = [[hc.window.set_ref,                0, sf_flash],
               [hc.window.set_ref,                1, contrast_flash],
               [sp.animate,                anim_seq]]
    
    ends =    [[sp.on,                     False],
               [hc.window.set_viewport_projection, 0, 1]]

    # add each test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

    test_num += 1
    
    
# add the rest
num_frames = 240
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts =  [[hc.window.set_far,         2],
           [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
           [hc.arduino.set_lmr_scale,  -.1],
           [rbar.set_ry,               0],
           [rbar.switch,               True] ]
middles = [[rbar.inc_ry,               hc.arduino.lmr]]
ends =    [[rbar.switch,               False],
           [hc.window.set_far,         2]]
 
hc.scheduler.add_rest(num_frames, starts, middles, ends)
