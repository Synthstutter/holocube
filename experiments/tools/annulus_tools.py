from numpy import *
import holocube.hc5 as hc5

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

class Moving_points():
    '''returns active indexes as dictionary. Key (vel, theta range, phi range)'''
    def __init__(self, numframes,  num_points = 5000,dimensions= [[-4,4],[-2,2],[-30,5]]):
        self.pts = hc5.stim.Points(hc5.window, num_points, dims=dimensions, color=.5, pt_size=3)
        self.vel = 0
        self.dir = [1,0,0]
        self.act_inds = []        
        self.numframes = numframes
        
    def calc_act_inds(self, theta_range, phi_range):    
        coords_over_t = zeros([self.numframes, 3, self.pts.coords.shape[1]])
        coords_over_t[0] = array([self.pts.coords[0] , self.pts.coords[1], self.pts.coords[2]])
        dist = linalg.norm(self.direction)
        mag  = self.vel/dist 
        x_disp = self.direction[0] *mag
        y_disp = self.direction[1] * mag
        z_disp = self.direction[2] * mag
        for frame in arange(1, self.numframes):
            coords_over_t[frame] = array([coords_over_t[frame-1][0] + x_disp,
                                            coords_over_t[frame-1][1] + y_disp,
                                            coords_over_t[frame-1][2] + z_disp,
                                           ])
        self.act_inds = array(inds_btw_sph_range(coords_over_t, theta_range[0], theta_range[1], phi_range[0], phi_range[1]))
        
    def get_selector_funcs(self):
        self.orig_y = array([self.pts.coords[1, :].copy()]*self.numframes)
        self.orig_x = array([self.pts.coords[0, :].copy()]*self.numframes)
        self.far_y = array([[10] * self.pts.num] * self.numframes)
        self.select_all = array([[1]*self.pts.num] * self.numframes,  dtype='bool')

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
        act_inds = []
        for i_theta, v_theta in enumerate(theta_ranges_to_show):
            v_phi = phi_ranges_to_show[i_theta]
            points.calc_act_inds(v_theta, v_phi)
            act_inds.append(points.act_inds)
        act_inds = array(act_inds).sum(axis = 0)
        pts_ever_visible = act_inds.sum(axis = 0, dtype = 'bool')
        act_inds = act_inds[:, pts_ever_visible]
        to_remove = array([not i for i in  pts_ever_visible])
        points.pts.remove_subset(to_remove)
        zero_array = zeros(act_inds.shape, dtype = bool)
        zero_array[act_inds>0] = True
        act_inds = zero_array
        points.get_selector_funcs()
        self.add_to_starts([points.pts.on, 1])
        self.add_to_middles([points.pts.subset_set_py, points.select_all, points.far_y])
        self.add_to_middles([points.pts.subset_set_py, act_inds, points.orig_y])
        self.add_to_middles([points.pts.inc_px,    points.direction[0] * points.vel/linalg.norm(points.direction)])
        self.add_to_middles([points.pts.inc_py,    points.direction[1] * points.vel/linalg.norm(points.direction)])
        self.add_to_middles([points.pts.inc_pz,    points.direction[2] * points.vel/linalg.norm(points.direction)])
        self.add_to_ends([points.pts.inc_px, -points.direction[0] * points.vel/linalg.norm(points.direction)*act_inds.shape[0]])
        self.add_to_ends([points.pts.inc_py, -points.direction[1] * points.vel/linalg.norm(points.direction)*act_inds.shape[0]])
        self.add_to_ends([points.pts.inc_pz, -points.direction[2] * points.vel/linalg.norm(points.direction)*act_inds.shape[0]])
        self.add_to_ends([points.pts.on, 0])

    def add_lights(self, ref_light, seq):
        '''adds light sequence to middles. turns off light at end'''
        self.add_to_middles([hc5.window.set_ref, ref_light, seq])
        self.add_to_ends([hc5.window.set_ref, 0, (0,0,0)])
