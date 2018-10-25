# the same as 3 spd 4 theta lmr but with the theta angles corrected so that they run only up to 90 degrees
import holocube.hc5 as hc5
from numpy import *

# horizon test
expi = 1

# how long for the exp?
numframes = 180
num_points = 5000

# how fast is forward velocity and horizontal translational velocities, triangle wav velocity?

# where should our theta boundaries be?
def pair_ranges(seg_ends):
    return array([seg_ends[:-1],seg_ends[1:]]).T

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

x_dims = [-abs(0.01)*numframes-2, abs(0.01)*numframes+2]

class Moving_points():
    '''returns active indexes as dictionary. Key (vel, theta range, phi range)'''
    def __init__(self,theta_ranges, phi_ranges):
        self.pts = hc5.stim.Points(hc5.window, num_points, dims=[x_dims,[-2,2],[-30,5]], color=.5, pt_size=3)
        self.theta_ranges = theta_ranges
        self.phi_ranges = phi_ranges
        self.vel = 0
        self.dir = [1,0,0]
        self.act_inds = {}        
        
    def calc_act_inds(self):    
        coords_over_t = zeros([numframes, 3, self.pts.num])
        coords_over_t[0] = array([self.pts.coords[0] , self.pts.coords[1], self.pts.coords[2]])
        dist = linalg.norm(self.direction)
        mag  = self.vel/dist 
        x_disp = self.direction[0] *mag
        y_disp = self.direction[1] * mag
        z_disp = self.direction[2] * mag
        for frame in arange(1, numframes):
            coords_over_t[frame] = array([coords_over_t[frame-1][0] + x_disp,
                                            coords_over_t[frame-1][1] + y_disp,
                                            coords_over_t[frame-1][2] + z_disp,
                                           ])
        for t_range in self.theta_ranges:
            for p_range in self.phi_ranges:
                self.act_inds[(str(t_range), str(p_range))] = array(inds_btw_sph_range(coords_over_t, t_range[0], t_range[1], p_range[0], p_range[1]))
                    
        self.orig_y = array([self.pts.coords[1, :].copy()]*numframes)
        self.orig_x = self.pts.pos[0].copy()
        self.far_y = array([[10] * self.pts.num] * numframes)
        self.select_all = array([[1]*num_points] * numframes,  dtype='bool')

class Ann_test_creator():
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
   
    def reset(self):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
   
    def add_to_starts(self, arr):
        self.starts.append(arr)

    def add_to_middles(self, arr):
        self.middles.append(arr)

    def add_to_ends(self, arr):
        self.ends.append(arr)
        
    def add_inits(self):
        self.add_to_starts([hc5.window.set_bg,  [0.0,0.0,0.0,1.0]])
        ends = [[hc5.window.set_ref, 0, [0,0,0]],
               [hc5.window.set_ref, 1, [0,0,0]],
               [hc5.window.set_ref, 2, [0,0,0]],
               [hc5.window.set_ref, 3, [0,0,0]],
               [hc5.window.reset_pos]]
        for end in ends:                   
            self.add_to_ends(end)

    def add_pts_region(self, points, theta_ranges_to_show, phi_ranges_to_show):
        self.add_to_starts([points.pts.on, 1])
        self.add_to_middles([points.pts.subset_set_py, points.select_all, points.far_y])
        act_inds = []
        for i_region in arange(len(theta_ranges_to_show)):
            act_inds.append(points.act_inds[(str(theta_ranges_to_show[i_region]) , str(phi_ranges_to_show[i_region]))])
        act_inds = array(act_inds).sum(axis = 0)
        zero_array = zeros(act_inds.shape, dtype = bool)
        zero_array[act_inds>0] = True
        act_inds = zero_array
        self.add_to_middles([points.pts.subset_set_py, act_inds, points.orig_y])
        self.add_to_middles([points.pts.inc_px,    points.direction[0] * points.vel/linalg.norm(points.direction)])
        self.add_to_middles([points.pts.inc_py,    points.direction[1] * points.vel/linalg.norm(points.direction)])
        self.add_to_middles([points.pts.inc_pz,    points.direction[2] * points.vel/linalg.norm(points.direction)])
        self.add_to_ends([points.pts.on, 0])

    def add_lights(self, ref_light, seq):
        '''adds light sequence to middles. turns off light at end'''
        self.add_to_middles([hc5.window.set_ref, ref_light, seq])
        self.add_to_ends([hc5.window.set_ref, 0, (0,0,0)])

        
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

slip_left_seq = hc5.tools.test_num_flash(3, numframes)
slip_right_seq = hc5.tools.test_num_flash(4, numframes)

hc5.scheduler.add_exp()

## experiments
pts_fwd = Moving_points(fwd_theta_ranges, fwd_phi_ranges)
pts_slip = Moving_points(v_slip_v, slip_direction, slip_theta_ranges, slip_phi_ranges)

test = Ann_test_creator()

for i_fwd_v, v_fwd_v in enumerate(fwd_v):
    v_ind_seq = hc5.tools.test_num_flash(i_fwd_v + 2, numframes)
    fwd_direction = fwd_dir[i_fwd_v]
    for i_phi, v_phi in enumerate(fwd_phi_ranges):
        fwd_phi = v_phi
        phi_ind_seq = hc5.tools.test_num_flash(i_phi + 4, numframes)
        for i_theta, v_theta in enumerate(slip_theta_ranges):
            slip_theta = v_theta
            fwd_thetas= fwd_theta_ranges[fwd_theta_ranges!=v_theta].reshape(-1,2)
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
                test.add_pts_region(pts_fwd, fwd_thetas, [fwd_phi]* fwd_thetas.shape[0])
                test.add_pts_region(pts_fwd, fwd_thetas, [fwd_phi]* fwd_thetas.shape[0])
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

