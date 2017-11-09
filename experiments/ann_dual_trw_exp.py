# horizon test with rotation
import holocube.hc5 as hc5
from numpy import *
from holocube.tools import mseq

# horizon test
expi = 1

# how long for the exp?
numframes = 500
num_points = 5000

# how fast is forward velocity and horizontal translational velocities, triangle wav velocity?
trng_wav_v = 0.01
fwd_v = [ 0.01, 0.02]

# where should our theta boundaries be?
theta_ranges = [[0.0, 0.72273424781341566],
 [0.72273424781341566, 1.0471975511965979],
 [1.0471975511965979, 1.318116071652818],
 [1.318116071652818, 1.5707963267948966]]

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
    

# a set of points
pts_fwd = hc5.stim.Points(hc5.window, num_points, dims=[[-2,2],[-2,2],[-30,5]], color=.5, pt_size=3)
pts_trw = hc5.stim.Points(hc5.window, num_points, dims=[[-2,2],[-2,2],[-30,5]], color=.5, pt_size=3)
pts_dual = hc5.stim.Points(hc5.window, num_points, dims=[[-2,2],[-2,2],[-30,5]], color=.5, pt_size=3)

# motions
tr_wav = hrz_trngl_wv(numframes, 0.5, 120)
trw_motion = tr_wav * trng_wav_v

# lights
tr_lights = array([(0, 175 + step*80, 0) for step in tr_wav], dtype='int') 

# simulation to calculate points in frames that are between thetas
coords_over_t_fwd = array([zeros([numframes, 4, pts_fwd.num]) for v in fwd_v])
coords_over_t_trw = zeros([numframes, 4, pts_trw.num])
coords_over_t_dual = array([zeros([numframes, 4, pts_dual.num]) for v in fwd_v])

# add fwd_v position to each z coordinate
coords_over_t_fwd[:,0] = array([[pts_fwd.coords[0] , pts_fwd.coords[1], pts_fwd.coords[2], zeros(pts_fwd.num)] for item in fwd_v])
for ind, val in enumerate(fwd_v):
    for frame in arange(1, numframes):
        coords_over_t_fwd[ind][frame] = array([pts_fwd.coords[0], pts_fwd.coords[1], coords_over_t_fwd[ind][frame-1][2] + fwd_v[ind], zeros(pts_fwd.num)])

act_inds_fwd = array([[inds_btw_thetas(coords_over_t_fwd[ind], t_range[0], t_range[1]) for t_range in theta_ranges] for ind, val in enumerate(fwd_v)])

coords_over_t_trw[0] = array([pts_trw.coords[0] , pts_trw.coords[1], pts_trw.coords[2], zeros(pts_trw.num)])
for frame in arange(1, numframes):
    coords_over_t_trw[frame] = array([coords_over_t_trw[frame-1][0] + tr_wav[frame]*trng_wav_v, pts_trw.coords[1], pts_trw.coords[2], zeros(pts_fwd.num)])


act_inds_trw = array([inds_btw_thetas(coords_over_t_trw, t_range[0], t_range[1]) for t_range in theta_ranges])

coords_over_t_dual[:,0] = array([[pts_dual.coords[0] , pts_dual.coords[1], pts_dual.coords[2], zeros(pts_dual.num)] for item in fwd_v])
for ind, val in enumerate(fwd_v):
    for frame in arange(1, numframes):
        coords_over_t_dual[ind][frame] = array([coords_over_t_dual[ind][frame-1][0] + trw_motion[frame], pts_dual.coords[1], coords_over_t_dual[ind][frame-1][2] + fwd_v[ind], zeros(pts_dual.num)])

act_inds_dual = array([[inds_btw_thetas(coords_over_t_dual[ind], t_range[0], t_range[1]) for t_range in theta_ranges] for ind, val in enumerate(fwd_v)])


## keep original y values for coords so that we can move things in and out of "space"
orig_y_fwd = array([pts_fwd.coords[1, :].copy()]*numframes)
orig_y_trw = array([pts_trw.coords[1, :].copy()]*numframes)
orig_y_dual =  array([pts_dual.coords[1, :].copy()]*numframes)

orig_x_trw = pts_trw.pos[0].copy()
orig_x_dual = pts_dual.pos[0].copy()



far_y_fwd = array([[10] * pts_fwd.num] * numframes)
far_y_trw = array([[10] * pts_trw.num] * numframes)
far_y_dual = array([[10] * pts_dual.num] * numframes)


select_all = array([[1]*num_points] * numframes,  dtype='bool' )

hc5.scheduler.add_exp()

## experiments



######################################### 0.01 speed ####################################################

vel = 0
theta_fwd = 0
theta_trw = 1


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 1
theta_trw = 0

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


theta_fwd = 0
theta_trw = 2

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_fwd = 2
theta_trw = 0

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_fwd = 0
theta_trw = 3

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


theta_fwd = 3
theta_trw = 0

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 1
theta_trw = 2

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_fwd = 2
theta_trw = 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 1
theta_trw = 3

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 3
theta_trw = 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 2
theta_trw = 3

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1



theta_fwd = 3
theta_trw = 2

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_fwd.on,            1],
           [pts_trw.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_fwd.inc_pz,        fwd_v[vel]],
           [pts_fwd.subset_set_py, select_all, far_y_fwd],
           [pts_fwd.subset_set_py, act_inds_fwd[vel][theta_fwd], orig_y_fwd], 
           [pts_trw.subset_set_py,  select_all, far_y_trw],
           [pts_trw.subset_set_py,  act_inds_trw[theta_trw], orig_y_trw ],
           [pts_trw.inc_px, trw_motion],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_trw.on,            0],
           [pts_fwd.on,            0],
           [ pts_trw.set_px, orig_x_trw],
           [pts_fwd.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1


#########################combined #############################
theta_trw = 0


test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_dual.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_dual.subset_set_py,  select_all, far_y_dual],
           [pts_dual.inc_px, trw_motion],
           [pts_dual.inc_pz, fwd_v[vel]], 
           [pts_dual.subset_set_py,  act_inds_dual[vel][theta_trw], orig_y_dual ],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_dual.on,            0],
           [ pts_dual.set_px, orig_x_dual],
           [pts_dual.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_trw += 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_dual.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_dual.subset_set_py,  select_all, far_y_dual],
           [pts_dual.inc_px, trw_motion],
           [pts_dual.inc_pz, fwd_v[vel]], 
           [pts_dual.subset_set_py,  act_inds_dual[vel][theta_trw], orig_y_dual ],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_dual.on,            0],
           [ pts_dual.set_px, orig_x_dual],
           [pts_dual.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_trw += 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_dual.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_dual.subset_set_py,  select_all, far_y_dual],
           [pts_dual.inc_px, trw_motion],
           [pts_dual.inc_pz, fwd_v[vel]], 
           [pts_dual.subset_set_py,  act_inds_dual[vel][theta_trw], orig_y_dual ],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_dual.on,            0],
           [ pts_dual.set_px, orig_x_dual],
           [pts_dual.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
hc5.scheduler.add_test(numframes, starts, middles, ends)
expi += 1

theta_trw += 1

test_num_spikes = hc5.tools.test_num_flash(expi, numframes)
starts =  [
           [pts_dual.on,             1],
           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
         
            ]

middles = [
           [pts_dual.subset_set_py,  select_all, far_y_dual],
           [pts_dual.inc_px, trw_motion],
           [pts_dual.inc_pz, fwd_v[vel]], 
           [pts_dual.subset_set_py,  act_inds_dual[vel][theta_trw], orig_y_dual ],
           [hc5.window.set_ref, 0, test_num_spikes],
           [hc5.window.set_ref, 1, tr_lights],
           ]

ends =    [
           [pts_dual.on,            0],
           [ pts_dual.set_px, orig_x_dual],
           [pts_dual.inc_pz, -fwd_v[vel]*numframes],
           [hc5.window.set_ref, 1, [0,0,0]],
           [hc5.window.reset_pos, 1]
           ]
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
