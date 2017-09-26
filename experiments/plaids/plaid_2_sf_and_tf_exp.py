# splaid rotate to 
import holocube.hc5 as hc
from numpy import *

# count exps
test_num = 1

num_frames = 240

# a grating stimulus
sp = hc.stim.grating_class(hc.window, fast=True)

# a set of temporal freqs for one component, with a constant sf, while the other
# component has a constant tf
tfs = array([-2, 0.0001, 2, 4, 4*sqrt(2), 8, 20], dtype=float) #cy/sec
tf_const = tfs[4] #cy/sec
sf_const = .035*180./pi #cy/deg -> cy/rad


for tf in tfs:
    for (or1, or2) in array([(0, pi/4.), (pi, 3.*pi/4.)]):
        sp.add_plaid(sf1=sf_const, tf1=tf,       o1 = or1, c1=.5,
                     sf2=sf_const, tf2=tf_const, o2 = or2, c2=.5,
                     sdb=.35, maxframes=num_frames)

# a set of sfs for one component, with a constant tf, while the other component 
# has a constant sf, calculated such that the speeds are equal to those above
speeds = tfs/sf_const
sfs = tf_const/speeds #cy/deg -> cy/rad
sf_const = sfs[4] #cy/deg -> cy/rad

for sf in sfs:
    for (or1, or2) in array([(0, pi/4.), (pi, 3.*pi/4.)]):
        sp.add_plaid(sf1=sf,       tf1 = tf_const, o1 = or1, c1=.5,
                     sf2=sf_const, tf2 = tf_const, o2 = or2, c2=.5,
                     sdb=.35, maxframes=num_frames)

for o in array([0, pi/4., pi, 3./4.*pi]):
    sp.add_grating(sf = sf_const, tf = tf_const, o=o, c = 1, 
                   sd=.35, maxframes=num_frames)

anim_seq = arange(num_frames)

# add the experiment
hc.scheduler.add_exp()

for i in arange(len(sp.gratings)):

    right_flash = hc.tools.test_num_flash(i % 2, num_frames)
    sf_a_flash = hc.tools.test_num_flash(i/16, num_frames) #the 16's
    sf_b_flash = hc.tools.test_num_flash((i%16)/4, num_frames) #the 4's
    sf_c_flash = hc.tools.test_num_flash((i%4)/2, num_frames) #the 2's
    
    starts =  [[hc.window.set_viewport_projection, 0, 0],
               [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
               [sp.choose_grating,         i],
               [sp.on,                     True]]

    middles = [[hc.window.set_ref,                0, right_flash],
               [hc.window.set_ref,                1, sf_a_flash],
               [hc.window.set_ref,                2, sf_b_flash],
               [hc.window.set_ref,                3, sf_c_flash], 
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

