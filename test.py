#! /usr/bin/env python
# run a series of exps with pyg and ard

from pylab import *
import pyglet
from pyglet.window import key
import holocube.hc5 as hc

# ardname = '/dev/ttyACM0'
ardname = 'dummy'
tracker_name = 'dummy'

project = 0
bg = [1., 1., 1., 1.]
bg = [.1, .1, .1, 1.]
# bg = [0., 0., 0., 1.]
near, far = .01, 1.
randomize = False

# start the components
hc.window.start(project=project, bg_color=bg, near=near, far=far, load_coords=True, config='test_viewport_config.txt', projection_screen=0)
hc.window.ref_bg_color = [.3,.3,.3,1]
hc.arduino.start(ardname)
# hc.filereader.start(tracker_name)
hc.scheduler.start(hc.window, randomize=randomize, default_rest_time=1)

hc.window.add_keypress_action(key.UP, hc.window.inc_pitch, -.05)
hc.window.add_keypress_action(key.DOWN, hc.window.inc_pitch, .05)
hc.window.add_keypress_action(key.LEFT, hc.window.inc_yaw, -.05)
hc.window.add_keypress_action(key.RIGHT, hc.window.inc_yaw, .05)

hc.window.add_keypress_action((key.UP, key.MOD_SHIFT), hc.window.inc_pitch, -.50)
hc.window.add_keypress_action((key.DOWN, key.MOD_SHIFT), hc.window.inc_pitch, .50)
hc.window.add_keypress_action((key.LEFT, key.MOD_SHIFT), hc.window.inc_yaw, -.50)
hc.window.add_keypress_action((key.RIGHT, key.MOD_SHIFT), hc.window.inc_yaw, .50)

from experiments import *
print ('ready')

# for debugging
s = hc.scheduler
e = s.exps
w = hc.window

# run pyglet
pyglet.app.run()

