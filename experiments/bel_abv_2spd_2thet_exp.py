# the same as 3 spd 4 theta lmr but with the theta angles corrected so that they run only up to 90 degrees
import holocube.hc5 as hc5
from numpy import *

# horizon test
expi = 1

# how long for the exp?
numframes = 180
num_points = 5000

# how fast is forward velocity and horizontal translational velocities, triangle wav velocity?
slip_v = array([-0.01, 0.01])

# where should our theta boundaries be?
def pair_ranges(seg_ends):
    return array([seg_ends[:-1],seg_ends[1:]]).T

theta_ranges = array([
       [0.40145062, 1.05443582],
       [1.35575444, 1.78583821],
       ])

## phis should range from -pi to pi. Can't do things like -5pi/4 just yet
phi_pairs = [-pi, 0 , pi]
phi_ranges = pair_ranges(phi_pairs)

slip_phis = [-pi, pi]
slip_phi_ranges = pair_ranges(slip_phis)

def calc_theta_phi(x,y,z, phi_rot = 0):
    '''from cartesian coords return spherical coords,  declination theta (pi - theta), and phi'''
    r = sqrt(x**2 + y**2 + z**2)
    theta = pi - arccos(z/r) # declination
    phi = arctan2(y,x) - phi_rot
    phi[phi > pi] -= 2*pi
    phi[phi < -pi] += 2*pi
    return theta, phi

def inds_btw_sph_range(coords_array, theta_min, theta_max, phi_min, phi_max):
    ''' check if coords in range of thetas. return frame and point inds for which wn should be active '''
    theta, phi = calc_theta_phi(coords_array[:,0], coords_array[:,1], coords_array[:,2], phi_rot = 0)
    bool_array = (theta >= theta_min) * (theta <= theta_max) * (phi >= phi_min) * (phi <= phi_max)
    return bool_array

x_dims = [-abs(slip_v).max()*numframes-2, abs(slip_v).max()*numframes+2]

class Moving_points():
    '''returns active indexes as dictionary. Key (vel, theta range, phi range)'''
    def __init__(self, vels, direction, theta_ranges, phi_ranges):
        self.pts = hc5.stim.Points(hc5.window, num_points, dims=[x_dims,[-2,2],[-30,5]], color=.5, pt_size=3)
        self.theta_ranges = theta_ranges
        self.phi_ranges = phi_ranges
        self.vels = vels
        self.direction = direction
        coords_over_t = array([zeros([numframes, 3, self.pts.num]) for v in self.vels])
        coords_over_t[:,0] = array([[self.pts.coords[0] , self.pts.coords[1], self.pts.coords[2]] for v in self.vels])
        self.act_inds = {}
        self.not_act_inds = {}
        for i_vel, v_vel in enumerate(self.vels):
            direc = self.direction[i_vel] 
            dist = linalg.norm(direc)
            mag  = v_vel/dist 
            x_disp = direc[0] *mag
            y_disp = direc[1] * mag
            z_disp = direc[2] * mag
            for frame in arange(1, numframes):
                coords_over_t[i_vel][frame] = array([coords_over_t[i_vel][frame-1][0] + x_disp,
                                                   coords_over_t[i_vel][frame-1][1] + y_disp,
                                                   coords_over_t[i_vel][frame-1][2] + z_disp,
                                                   ])
            for t_range in self.theta_ranges:
                for p_range in self.phi_ranges:
                    self.act_inds[( str(v_vel), str(direc), str(t_range), str(p_range))] = array(inds_btw_sph_range(coords_over_t[i_vel], t_range[0], t_range[1], p_range[0], p_range[1]))
                    self.not_act_inds[( str(v_vel), str(direc), str(t_range), str(p_range))] = True - self.act_inds[( str(v_vel), str(direc), str(t_range), str(p_range))]             
        self.orig_y = array([self.pts.coords[1, :].copy()]*numframes)
        self.orig_x = self.pts.pos[0].copy()
        self.far_y = array([[10] * self.pts.num] * numframes)
        self.select_all = array([[1]*num_points] * numframes,  dtype='bool')


fwd_v = [0.00, 0.01]

pts_fwd = Moving_points(fwd_v, [[0,0,1], [0,0,1]], theta_ranges, phi_ranges)
pts_no_mot = Moving_points([0], [[0,0,1]], theta_ranges, phi_ranges)
pts_slip = Moving_points(slip_v, [[1,0,0], [1,0,0]], theta_ranges, slip_phi_ranges) 

slip_left_seq = hc5.tools.test_num_flash(3, numframes)
slip_right_seq = hc5.tools.test_num_flash(4, numframes)

hc5.scheduler.add_exp()
import pdb; pdb.set_trace()

## experiments
for i_fwd_v, v_fwd_v in enumerate(pts_fwd.vels):
    fwd_dir = pts_fwd.direction[i_fwd_v]
    v_ind_seq = hc5.tools.test_num_flash(i_fwd_v + 2, numframes)
    for i_phi, v_phi in enumerate(phi_ranges):
        phi_ind_seq = hc5.tools.test_num_flash(i_phi + 4, numframes)
        for i_theta, v_theta in enumerate([theta_ranges]):
            theta_ind_seq = hc5.tools.test_num_flash(i_theta + 1, numframes)
            for i_slip_v, v_slip_v in enumerate(slip_v):
                if i_slip_v ==1:
                    vslip = pts_slip.vels[1]
                    slip_dir = pts_slip.direction[1]
                    slip_ind_seq = slip_right_seq                
                else: 
                    vslip = pts_slip.vels[0]
                    slip_dir = pts_slip.direction[0]
                    slip_ind_seq = slip_left_seq

                starts =  [
                           [pts_slip.pts.on,         1],
                           [pts_fwd.pts.on,           1],
                           [pts_no_mot.pts.on,           1],                    
                           [hc5.window.set_bg,  [0.0,0.0,0.0,1.0]],
                            ]

                middles = [

                           [pts_fwd.pts.subset_set_py, pts_fwd.select_all, pts_fwd.orig_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(v_theta) , str(pts_fwd.phi_ranges[0]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(v_theta) , str(pts_fwd.phi_ranges[1]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(pts_fwd.theta_ranges[0]) , str(pts_fwd.phi_ranges[i_phi]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(pts_fwd.theta_ranges[1]) , str(pts_fwd.phi_ranges[i_phi]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(pts_fwd.theta_ranges[2]) , str(pts_fwd.phi_ranges[i_phi]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(pts_fwd.theta_ranges[3]) , str(pts_fwd.phi_ranges[i_phi]))], pts_fwd.far_y],
                           [pts_fwd.pts.subset_set_py, pts_fwd.act_inds[
                               (str(v_fwd_v), str(fwd_dir), str(pts_fwd.theta_ranges[4]) , str(pts_fwd.phi_ranges[i_phi]))], pts_fwd.far_y],
             
                           [pts_fwd.pts.inc_px,    fwd_dir[0] * v_fwd_v/linalg.norm(fwd_dir)],
                           [pts_fwd.pts.inc_py,    fwd_dir[1] * v_fwd_v/linalg.norm(fwd_dir)],
                           [pts_fwd.pts.inc_pz,    fwd_dir[2] * v_fwd_v/linalg.norm(fwd_dir)],

                           [pts_no_mot.pts.subset_set_py, pts_no_mot.select_all, pts_no_mot.orig_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(v_theta) , str(pts_no_mot.phi_ranges[0]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(v_theta) , str(pts_no_mot.phi_ranges[1]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(pts_no_mot.theta_ranges[0]) , str(pts_no_mot.phi_ranges[1- i_phi]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(pts_no_mot.theta_ranges[1]) , str(pts_no_mot.phi_ranges[1- i_phi]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(pts_no_mot.theta_ranges[2]) , str(pts_no_mot.phi_ranges[1- i_phi]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(pts_no_mot.theta_ranges[3]) , str(pts_no_mot.phi_ranges[1- i_phi]))], pts_no_mot.far_y],
                           [pts_no_mot.pts.subset_set_py, pts_no_mot.act_inds[
                               (str(pts_no_mot.vels[0]), str(pts_no_mot.direction[0]), str(pts_no_mot.theta_ranges[4]) , str(pts_no_mot.phi_ranges[1- i_phi]))], pts_no_mot.far_y],
             
                           [pts_no_mot.pts.inc_px,    0],
                           [pts_no_mot.pts.inc_py,    0],
                           [pts_no_mot.pts.inc_pz,    0],

                           [pts_slip.pts.subset_set_py, pts_slip.select_all, pts_slip.far_y],
                           [pts_slip.pts.subset_set_py, pts_slip.act_inds[
                               (str(vslip),str(slip_dir), str(v_theta) , str(pts_slip.phi_ranges[0]))], pts_slip.orig_y],
                           [pts_slip.pts.inc_px,    slip_dir[0] * vslip/linalg.norm(slip_dir)],
                           [pts_slip.pts.inc_py,    slip_dir[1] * vslip/linalg.norm(slip_dir)],
                           [pts_slip.pts.inc_pz,    slip_dir[2] * vslip/linalg.norm(slip_dir)],

                           [hc5.window.set_ref, 0, v_ind_seq],
                           [hc5.window.set_ref, 1, theta_ind_seq],
                           [hc5.window.set_ref, 2, slip_ind_seq],
                           [hc5.window.set_ref, 3, phi_ind_seq],
                           ]

                ends =    [
                           [pts_fwd.pts.on,            0],
                           [pts_fwd.pts.inc_px,    -fwd_dir[0] * v_fwd_v/linalg.norm(fwd_dir)*numframes],
                           [pts_fwd.pts.inc_py,    -fwd_dir[1]* v_fwd_v/linalg.norm(fwd_dir)*numframes],
                           [pts_fwd.pts.inc_pz,    -fwd_dir[2]* v_fwd_v/linalg.norm(fwd_dir)*numframes],

                           [pts_no_mot.pts.on,            0],
                           [pts_no_mot.pts.inc_px,    0],
                           [pts_no_mot.pts.inc_py,    0],
                           [pts_no_mot.pts.inc_pz,    0],

                           [pts_slip.pts.on,            0],
                           [pts_slip.pts.inc_px,    -slip_dir[0]* vslip/linalg.norm(slip_dir)*numframes],
                           [pts_slip.pts.inc_py,    -slip_dir[1]* vslip/linalg.norm(slip_dir)*numframes],
                           [pts_slip.pts.inc_pz,    -slip_dir[2]* vslip/linalg.norm(slip_dir)*numframes],

                           [hc5.window.set_ref, 0, [0,0,0]],
                           [hc5.window.set_ref, 1, [0,0,0]],
                           [hc5.window.set_ref, 2, [0,0,0]],
                           [hc5.window.set_ref, 3, [0,0,0]],
                           [hc5.window.reset_pos],
                           ]
                hc5.scheduler.add_test(numframes, starts, middles, ends)

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

