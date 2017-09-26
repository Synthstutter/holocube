# splaid rotate to 
import holocube.hc5 as hc
from numpy import *

# count exps
test_num = 1

num_frames = 240

# a grating stimulus
sp = hc.stim.grating_class(hc.window, fast=True)

# a series of rotations
tf = 5 #cy/sec
sf = .03 *180/pi #cy/deg -> cy/rad
os = array([0, pi])

sf1 = .035
tfs = array([-2, 0.0001, 2, 4, 4*sqrt(2), 8, 20], dtype=float)
tf1 = tfs[4]

for tf in tfs:
    for (o1, o2) in array([(0, pi/4.), (pi, 3.*pi/4.)]):
        sp.add_plaid(sf1=sf1, tf1 = tf,   o1 = o1, c1=.5,
                     sf2=sf1, tf1 = tf1, o2 = o2, c2=.5,
                     sdb=.35, maxframes=num_frames)

speeds = tfs/sf1
tf2 = 5.
sfs = tf2/speeds
sf2 = sfs[4]

for sf in sfs:
    for (o1, o2) in array([(0, pi/4.), (pi, 3.*pi/4.)]):
        sp.add_plaid(sf1=sf,  tf1 = tf2, o1 = o1, c1=.5,
                     sf2=sf2, tf1 = tf2, o2 = o2, c2=.5,
                     sdb=.35, maxframes=num_frames)

for o in array([0, pi/4., pi, 3./4.*pi]):
    sp.add_grating(sf = sf1, tf = tf1, o=o, c = 1)

anim_seq = arange(num_frames)

# add the experiment
hc.scheduler.add_exp()

for i in arange(len(sp.gratings)):

    test_num_flash = hc.tools.test_num_flash(test_num, num_frames)
    
    starts =  [[hc.window.set_viewport_projection, 0, 0],
               [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
               [sp.choose_grating,         i],
               [sp.on,                     True]]

    middles = [[hc.window.set_ref,                0, test_num_flash],
               [sp.animate,                anim_seq]]
    
    ends =    [[sp.on,                     False],
               [hc.window.set_viewport_projection, 0, 1]]

    # add each test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

    test_num += 1
    
    
# add the rest
num_frames = 300
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

