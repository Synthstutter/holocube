#! /usr/bin/env python
# run a series of exps with pyg and ard

from pylab import *
import pyglet
import holocube.hc5 as hc

ardname = 'dummy'
# ardname = '/dev/ttyACM0'
randomize = 0

# start the components
# hc.window.start(config_file='test_viewport.config')
hc.window.start(config_file='viewport.config')
hc.arduino.start(ardname)
hc.scheduler.start(hc.window, randomize=randomize, default_rest_time=.1, beep_file=None)

hc.scheduler.load_dir('experiments')
print('ready')

# for debugging
s = hc.scheduler
e = s.exps
w = hc.window
a = hc.arduino

# run pyglet
pyglet.app.run()
