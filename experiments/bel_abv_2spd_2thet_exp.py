# the same as 3 spd 4 theta lmr but with the theta angles corrected so that they run only up to 90 degrees
import holocube.hc5 as hc5
from numpy import *
from tools.annulus_tools import *

# horizon test
expi = 1

# how long for the exp?
numframes = 180
num_points = 5000

# how fast is forward velocity and horizontal translational velocities, triangle wav velocity?

# where should our theta boundaries be?
def pair_ranges(seg_ends):
    return array([seg_ends[:-1],seg_ends[1:]]).T

slip_v = array([-0.01, 0.01])
slip_dir = [[1,0,0], [1,0,0]]
slip_theta_ranges =  array([
       [0.40145062, 1.05443582],
       [1.35575444, 1.78583821],
       ])
slip_phis = [-pi, pi]
slip_phi_ranges = pair_ranges(slip_phis)

fwd_v = [0.00, 0.01]
fwd_dir = [[0,0,1], [0,0,1]]
fwd_theta_ranges = array([
       [0, 0.40145062],
       [0.40145062, 1.05443582],
       [1.05443582, 1.35575444],
       [1.35575444, 1.78583821],
       [1.78583821, pi],
       ])
fwd_phis = [-pi, 0 , pi]
fwd_phi_ranges = pair_ranges(fwd_phis)

no_mot_v = [0.00]
no_mot_dir = [[0,0,1]]
no_mot_theta_ranges = array([
       [0, 0.40145062],
       [0.40145062, 1.05443582],
       [1.05443582, 1.35575444],
       [1.35575444, 1.78583821],
       [1.78583821, pi],
       ])
no_mot_phis = [-pi, 0 , pi]
no_mot_phi_ranges = pair_ranges(fwd_phis)


slip_left_seq = hc5.tools.test_num_flash(3, numframes)
slip_right_seq = hc5.tools.test_num_flash(4, numframes)

hc5.scheduler.add_exp()

## experiments
pts_fwd = Moving_points(fwd_theta_ranges, fwd_phi_ranges, numframes = 180, num_points = 5000)
pts_slip = Moving_points(slip_theta_ranges, slip_phi_ranges, numframes = 180, num_points = 5000)
pts_no_mot = Moving_points(no_mot_theta_ranges, no_mot_phi_ranges, numframes = 180, num_points = 5000)

test = Ann_test_creator()

for i_fwd_v, v_fwd_v in enumerate(fwd_v):
    v_ind_seq = hc5.tools.test_num_flash(i_fwd_v + 2, numframes)
    fwd_direction = fwd_dir[i_fwd_v]
    for i_phi, v_phi in enumerate(fwd_phi_ranges):
        fwd_phi = v_phi
        no_mot_phi = no_mot_phi_ranges[no_mot_phi_ranges!=fwd_phi].reshape(-1,2)[0]      
        phi_ind_seq = hc5.tools.test_num_flash(i_phi + 4, numframes)
        for i_theta, v_theta in enumerate(slip_theta_ranges):
            slip_theta = v_theta
            fwd_thetas= fwd_theta_ranges[fwd_theta_ranges!=v_theta].reshape(-1,2)
            no_mot_thetas= no_mot_theta_ranges[no_mot_theta_ranges!=v_theta].reshape(-1,2)
            theta_ind_seq = hc5.tools.test_num_flash(i_theta + 1, numframes)
            for i_slip_v, v_slip_v in enumerate(slip_v):
                slip_direction = slip_dir[i_slip_v] 
                if i_slip_v ==1:
                    slip_ind_seq = slip_right_seq                
                else: 
                    slip_ind_seq = slip_left_seq

                pts_fwd.vel, pts_fwd.direction = v_fwd_v, fwd_direction
                pts_fwd.calc_act_inds()
                pts_slip.vel, pts_slip.direction = v_slip_v, slip_direction
                pts_slip.calc_act_inds()
                pts_no_mot.vel, pts_no_mot.direction = no_mot_v[0], no_mot_dir[0]
                pts_no_mot.calc_act_inds()

                test.add_pts_region(pts_fwd, fwd_thetas, [fwd_phi]* fwd_thetas.shape[0])
                test.add_pts_region(pts_no_mot, no_mot_thetas, [no_mot_phi]* no_mot_thetas.shape[0])
                test.add_pts_region(pts_slip, [slip_theta], slip_phi_ranges)
                test.add_lights(0, v_ind_seq)
                test.add_lights(1, phi_ind_seq)
                test.add_lights(2, theta_ind_seq)
                test.add_lights(3, slip_ind_seq)
                hc5.scheduler.add_test(numframes, test.starts, test.middles, test.ends)
                test.reset()
# add the rest
numframes = 300
rbar = hc5.stim.cbarr_class(hc5.window, dist=1)

starts =  [[hc5.window.set_far,         2],
           [hc5.window.set_bg,          [0.0,0.0,0.0,1.0]],
           [hc5.arduino.set_lmr_scale,  -.1],
           [rbar.set_ry,               0],
           [rbar.switch,               True] ]
middles = [[rbar.inc_ry,               hc5.arduino.lmr]]
ends =    [[rbar.switch,               False],
           [hc5.window.set_far,         2]]

hc5.scheduler.add_rest(numframes, starts, middles, ends)

