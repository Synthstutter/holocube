# plaid 4, just to see the fly follows rigid directions

import holocube.hc5 as hc5
from numpy import *

# count exps
expi = 2

numframes = 600 #5 secs
sf = 10/pi

# a grating stimulus
sp = hc5.stim.grating_class(hc5.window, fast=True)

# EXPS #
# two stims that reverse the grating, second grating is slower
tfs1 = zeros(numframes)
tfs1[120:360] = 2
tfs1[360:] = -2

tfs2 = zeros(numframes)
tfs2[120:360] = 1
tfs2[360:] = -1

# left
o2 = pi/2-pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)

# and right
o2 = pi/2+pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)


# two stims that reverse the grating, second grating is faster
tfs2[120:360] = 4
tfs2[360:] = -4

# left
o2 = pi/2-pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)

# and right
o2 = pi/2+pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)


# two stims that increase speed of second grating
# to reverse the apparent direction
tfs1[120:] = 2

tfs2 = zeros(numframes)
tfs2[120:360] = 1
tfs2[360:] = 4

# left
o2 = pi/2-pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)

# and right
o2 = pi/2+pi/6
sp.add_plaid(sf1=sf, tf1=tfs1, o1=pi/2,
             sf2=sf, tf2=tfs2, o2=o2, sd=.35, maxframes=numframes)





for i in arange(6):
    expflashes = array(hc5.schedulers.spikelist(expi, numframes)*255, dtype='int')
    r1 = zeros(numframes, dtype='int')
    r1[:120] = 255
    r2 = zeros(numframes, dtype='int')
    r2[120:360] = 255
    r3 = zeros(numframes, dtype='int')
    r3[360:-1] = 255
    
    hc5.scheduler.add_test(hc5.window.set_perspective, False,                   0,
                           sp.choose_grating,          i,                       0,
                           sp.on,                      1,                       0,
                           
                           hc5.window.ref_color_1,     r1,                      1,
                           hc5.window.ref_color_2,     r2,                      1,
                           hc5.window.ref_color_3,     r3,                      1,
                           hc5.window.ref_color_4,     expflashes,              1,
                           sp.animate,                 arange(numframes),       1,
                           
                           sp.on,                      0,                      -1,
                           hc5.window.set_perspective, True,                   -1)

    expi += 1

# tracking bar
bar = hc5.stim.cbarr_class(hc5.window)
tri = arcsin(sin(linspace(0,4*pi,numframes)))*40
tri_dir = array(sign(ediff1d(tri,None, 0)), dtype='int')
lights = mod(cumsum(tri_dir),3)
expnumlights = array(hc5.schedulers.spikelist(expi, numframes)*255, dtype='int')

hc5.scheduler.add_test(bar.on,                     1,                        0,
                       hc5.window.ref_light,      -1,                        0,
                       
                       bar.set_ry,                 tri,                      1,
                       hc5.window.ref_light,       lights,                   1,
                       hc5.window.ref_color_4,     expnumlights,             1,
                       
                       hc5.window.ref_light,      -1,                       -1,
                       bar.on, 0,                                           -1)    


hc5.scheduler.save_exp()
