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
    def __init__(self, numframes,  numpoints = 5000, dimensions= [[-4,4],[-2,2],[-30,5]], vel = 0, direction = [1,0,0], theta_ranges  = [[0, pi]], phi_ranges = [[-pi, pi]], rx = 0, ry = 0, rz = 0):
        self.pts = hc5.stim.Points(hc5.window, numpoints, dims=dimensions, color=.5, pt_size=3)
        self.numframes = numframes
        self.vel = vel
        self.direction = direction
        self.theta_ranges = theta_ranges
        self.phi_ranges = phi_ranges
        
        self.act_inds = []        
        self.calc_act_inds()
        self.remove_unvisible_points()
        self.get_selector_funcs()
        
    def calc_act_inds(self):
        for i_theta_range, theta_range in enumerate(self.theta_ranges):
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
            self.act_inds.append(array(inds_btw_sph_range(coords_over_t, theta_range[0], theta_range[1], self.phi_ranges[i_theta_range][0], self.phi_ranges[i_theta_range][1])))
        
    def remove_unvisible_points(self):
        act_inds = array(self.act_inds).sum(axis = 0)
        pts_ever_visible = act_inds.sum(axis = 0, dtype = 'bool')
        act_inds = act_inds[:, pts_ever_visible]
        to_remove = array([not i for i in  pts_ever_visible])
        self.pts.remove_subset(to_remove)
        zero_array = zeros(act_inds.shape, dtype = bool)
        zero_array[act_inds>0] = True
        self.act_inds = zero_array

    def get_selector_funcs(self):
        self.orig_y = array([self.pts.coords[1, :].copy()]*self.numframes)
        self.orig_x = array([self.pts.coords[0, :].copy()]*self.numframes)
        self.far_y = array([[10] * self.pts.num] * self.numframes)
        self.select_all = array([[1]*self.pts.num] * self.numframes,  dtype='bool')

class Test_creator():
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self, numframes):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
        self.numframes = numframes
        
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
        ends = [[hc5.window.set_bg,  [0.0,0.0,0.0,1.0]], 
               [hc5.window.set_ref, 0, [0,0,0]],
               [hc5.window.set_ref, 1, [0,0,0]],
               [hc5.window.set_ref, 2, [0,0,0]],
               [hc5.window.set_ref, 3, [0,0,0]],
               [hc5.window.reset_pos]]
        for end in ends:                   
            self.add_to_ends(end)

    def add_index_lights(self, ref_light, index):
        '''adds index light sequence to middles. turns off light at end'''
        seq = hc5.tools.test_num_flash(index, self.numframes)
        self.add_to_middles([hc5.window.set_ref, ref_light, seq])
        self.add_to_ends([hc5.window.set_ref, 0, (0,0,0)])


    def add_to_scheduler(self):
        hc5.scheduler.add_test(self.numframes, self.starts, self.middles, self.ends)

    def add_rest_bar(self, numframes=300):
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

class Ann_test_creator(Test_creator):
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)
    
    def add_pts(self, annulus_points):
        for points in annulus_points:
            self.add_to_starts([points.pts.on, 1])
            self.add_to_middles([points.pts.subset_set_py, points.select_all, points.far_y])
            self.add_to_middles([points.pts.subset_set_py, points.act_inds, points.orig_y])
            self.add_to_middles([points.pts.inc_px,    points.direction[0] * points.vel/linalg.norm(points.direction)])
            self.add_to_middles([points.pts.inc_py,    points.direction[1] * points.vel/linalg.norm(points.direction)])
            self.add_to_middles([points.pts.inc_pz,    points.direction[2] * points.vel/linalg.norm(points.direction)])
            self.add_to_ends([points.pts.inc_px, -points.direction[0] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_py, -points.direction[1] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_pz, -points.direction[2] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.on, 0])

class Motion_ill_test_creator(Test_creator):
    '''create static image motion illusions. pick image location, onset time, end time'''
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)

    def add_image(self, img, left_edge=pi/4, right_edge=-pi/4, bottom_edge=-pi/4, top_edge=pi/4, rx = 0, ry = 0, rz = 0, start_t = 0.0, end_t = 1.0):
        image = hc5.stim.Quad_image(hc5.window, left= left_edge, right= right_edge, bottom= bottom_edge, top= top_edge, xdivs=24, ydivs=1, dist=1.5)
        image.load_image(img)
        image_state = array([False] * self.numframes)
        image_state[int(self.numframes*start_t): int(self.numframes*end_t)] = True
        self.add_to_starts([hc5.window.set_far,          3])
        self.add_to_ends([hc5.window.set_far,          2])
        self.add_to_starts([image.set_rx, rx])
        self.add_to_starts([image.set_ry, ry])
        self.add_to_starts([image.set_rz, rz])
        self.add_to_middles([image.on, image_state])
        self.add_to_ends([image.on, False])

    def add_bg_static(self, intensity= 1.0 , start_t = 0.0, end_t = 1.0):
        ''' add a static bg color from a start to end time'''
        bg_color = (255*intensity, 255*intensity, 255*intensity, 1.0)
        bg_state = array([(0.0,0.0,0.0,1.0)] * self.numframes)
        bg_state[int(self.numframes*start_t): int(self.numframes *end_t)] = bg_color
        self.add_to_middles([hc5.window.set_bg, bg_state])
