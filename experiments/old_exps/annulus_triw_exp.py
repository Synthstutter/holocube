# horizon test with rotation
import holocube.hc5 as hc5
from numpy import *
from holocube.tools import mseq

# horizon test
expi = 1

# how long for the exp?
numframes = 500

# how fast is forward velocity and horizontal translational velocities, triangle wav velocity?
trng_wav_v = 0.01
fwd_v = [0.00, 0.01, 0.02]

# where should our theta boundaries be?
theta_ranges = [[0.0, 0.90374825082675658],
 [0.90374825082675658, 1.331170660136958],
 [1.331170660136958, 1.7152900947440821],
 [1.7152900947440821, 2.123889803846899]]

def calc_theta(x,y,z):
    '''from cartesian coords return spherical coords declination theta (pi - theta)'''
    r = sqrt(x**2 + y**2 + z**2)
    theta = pi - arccos(z/r)
    return theta

def inds_btw_thetas(coords_array, theta_min, theta_max):
    ''' check if coords in range of thetas. return frame and point inds for which wn should be active '''
    for frame in coords_array:
        for point in arange(len(coords_array[0][0])):
            if theta_min <= calc_theta(frame[0][point], frame[1][point], frame[2][point]) <= theta_max:
                frame[3][point] = 1
            else:
                frame[3][point] = 0
    return array([frame[3] for frame in coords_array], dtype='bool')

def hrz_trngl_wv(number_frames, wav_freq = 0.5, cpu_freq = 120):
    ''' triangular wave function '''
    frames_p_cycle = cpu_freq/wav_freq
    frame_shift = int(frames_p_cycle/4)
    curr_frame = frame_shift
    trans = zeros(number_frames)
    while curr_frame < (number_frames + frame_shift):
        if curr_frame%frames_p_cycle <= frames_p_cycle/2:
            trans[curr_frame- frame_shift]= 1
        if curr_frame%frames_p_cycle > frames_p_cycle/2:
            trans[curr_frame-frame_shift]= -1
        curr_frame += 1
    return trans
    
# motions
tr_wav = hrz_trngl_wv(numframes, 0.5, 120)


# lights
tr_lights = array([(0, 175 + step*80, 0) for step in tr_wav], dtype='int') 

# a set of points
pts = hc5.stim.Points(hc5.window, 5000, dims=[[-2,2],[-2,2],[-30,5]], color=.5, pt_size=3)

# simulation to calculate points in frames that are between thetas
coords_over_t = array([zeros([numframes, 4, pts.num]) for v in fwd_v])

# add fwd_v position to each z coordinate
coords_over_t[:,0] = array([[pts.coords[0] , pts.coords[1], pts.coords[2], zeros(pts.num)] for item in fwd_v])
for ind, val in enumerate(fwd_v):
    for frame in arange(1, numframes):
        coords_over_t[ind][frame] = array([pts.coords[0], pts.coords[1], coords_over_t[ind][frame-1][2] + fwd_v[ind], zeros(pts.num)])

act_inds = array([[inds_btw_thetas(coords_over_t[ind], t_range[0], t_range[1]) for t_range in theta_ranges] for ind, val in enumerate(fwd_v)])

orig_x = pts.coords[0, :].copy()
select_all = array([True]*pts.num)

hc5.scheduler.add_exp()

## experiments


####################### No fwd_v #########################################
test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[0]],
           [pts.subset_inc_px,  act_inds[0][0], tr_wav*trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[0]*numframes],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[0]],
           [pts.subset_inc_px,  act_inds[0][1], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[0]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[0]],
           [pts.subset_inc_px,  act_inds[0][2], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[0]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[0]],
           [pts.subset_inc_px,  act_inds[0][3], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[0]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

################### fwd V = 0.01 ##################################

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[1]],
           [pts.subset_inc_px,  act_inds[1][0], tr_wav*trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[1]*numframes],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[1]],
           [pts.subset_inc_px,  act_inds[1][1], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[1]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[1]],
           [pts.subset_inc_px,  act_inds[1][2], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[1]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[1]],
           [pts.subset_inc_px,  act_inds[1][3], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[1]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

################### fwd V = 0.02 ##################################

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[2]],
           [pts.subset_inc_px,  act_inds[2][0], tr_wav*trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[2]*numframes],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[2]],
           [pts.subset_inc_px,  act_inds[2][1], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[2]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[2]],
           [pts.subset_inc_px,  act_inds[2][2], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[2]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [[pts.on,            1],
           # [hc5.window.set_far,  100],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]]]

middles = [[pts.inc_pz,        fwd_v[2]],
           [pts.subset_inc_px,  act_inds[2][3], tr_wav * trng_wav_v],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights]]

ends =    [[pts.on,            0],
           [pts.inc_pz, -fwd_v[2]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [pts.subset_set_px, select_all, orig_x ],
           [hc5.window.reset_pos, 1]]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


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

