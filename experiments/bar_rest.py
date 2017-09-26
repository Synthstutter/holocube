# rest with a rotating bar

import holocube.hc5 as hc
from numpy import *

bar = hc.stim.cbarr_class(hc.window, dist=1)

num_frames = inf

starts =  [[hc.window.set_far,        2],
           [hc.arduino.set_lmr_scale, -0.1],
           [bar.set_ry,               0],
           [bar.switch,               True] ]

middles = [[bar.inc_ry,               hc.arduino.lmr]]

ends =    [[bar.switch,               False],
           [hc.window.set_far,        2]]

hc.scheduler.add_idle(num_frames, starts, middles, ends)

