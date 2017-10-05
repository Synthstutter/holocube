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
sfs = logspace(-5, -3, 5, base=2) #cy/deg
sfs = sfs *180./pi #cy/deg -> cy/rad
# sf = .03 *180/pi
os = linspace(0, 2*pi, 13)[:-1]
c = .5

for sf in sfs:
    for o in os:
        sp.add_plaid(sf1=sf, tf1=tf, o1=o,           c1=c,
                     sf2=sf, tf2=tf, o2=o + 5*pi/6., c2=1-c, 
                     sdb=.35, maxframes=num_frames)

anim_seq = arange(num_frames)

# add the experiment
hc.scheduler.add_exp()

for i in arange(len(sp.gratings)):
    sf = i/12
    sf += 1
    o = i%12
    quad = o/3
    quad += 2
    ang = o%3
    ang += 3
    
    sf_flash = hc.tools.test_num_flash(sf, num_frames)
    quad_flash = hc.tools.test_num_flash(quad, num_frames)
    ang_flash = hc.tools.test_num_flash(ang, num_frames)
    
    starts =  [[hc.window.set_viewport_projection, 0, 0],
               [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
               [sp.choose_grating,         i],
               [sp.on,                     True]]

    middles = [[hc.window.set_ref,                0, sf_flash],
               [hc.window.set_ref,                1, quad_flash],
               [hc.window.set_ref,                2, ang_flash],
               [sp.animate,                anim_seq]]
    
    ends =    [[sp.on,                     False],
               [hc.window.set_viewport_projection, 0, 1]]

    # add each test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

    test_num += 1
    
    
# add the rest
num_frames = 200
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
