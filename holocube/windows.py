# windowing classes

import pyglet
from pyglet.gl import *
from pyglet.window import key
from numpy import *
from os.path import expanduser
import time

# for rotating the camera view angles
def rotmat(u=[0.,0.,1.], theta=0.0):
    '''Returns a matrix for rotating an arbitrary amount (theta)
    around an arbitrary axis (u, a unit vector).  '''
    ux, uy, uz = u
    cost, sint = cos(theta), sin(theta)
    uxu = array([[ux*ux, ux*uy, ux*uz],
                 [ux*uy, uy*uy, uz*uy],
                 [ux*uz ,uy*uz , uz*uz]])
    ux = array([[0, -uz, uy],
                [uz, 0, -ux],
                [-uy, ux, 0]])
    return cost*identity(3) + sint*ux + (1 - cost)*uxu

class Viewport_config_class():
    '''A class for a single window configuration information.'''
    def __init__(self, batch):
        self.coords = array([0,0,150,150], dtype='int')    # window left, bottom, width, height
        self.orth_coords = self.coords
        self.scale_factors = [1., 1., 1.]                  # x y and z flips for mirror reflections
        self.for_ax_ind = 2                 # camera faces along which axis
        self.for_ax_sign = -1               # forward or backwards on that axis
        self.up_ax_ind = 1                  # which axis is the top of the camera along
        self.up_ax_sign = 1                 # forward or backwards on that axis
        self.projection = 1                 # orthographic or perspective projection
        self.project = glFrustum
        self.draw = self.draw_perspective
        self.frustum = array([-0.1, 0.1, -0.1, 0.1, 0.1, 100.0]) #left right bottom top near far
        self.ortho = array([0.0, 150.0, 0.0, 150.0, -1.0, 1.0])
        self.sprite_pos = 0
        self.batch = batch
        self.num_ref_pts = 0
        self.ref_coords = array([])
        self.ref_colors = array([])
        self.ref_vl = None
        self.clear_color = [0.,0.,0.,1.]
        
    def draw_perspective(self, pos, ori):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(*self.frustum)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        glScalef(*self.scale_factors)
        # point the camera
        gluLookAt(pos[0],
                  pos[1],
                  pos[2],
                  pos[0] + self.for_ax_sign*ori[self.for_ax_ind, 0],
                  pos[1] + self.for_ax_sign*ori[self.for_ax_ind, 1],
                  pos[2] + self.for_ax_sign*ori[self.for_ax_ind, 2],
                  self.up_ax_sign*ori[self.up_ax_ind, 0],
                  self.up_ax_sign*ori[self.up_ax_ind, 1],
                  self.up_ax_sign*ori[self.up_ax_ind, 2])
        # draw everything
        self.batch.draw()

    def draw_ortho(self, pos, ori, reflect=False):
        # I have to set reflect to false for sprites since sprites don't scale to
        # negative values right now, and the main use of ortho is to display sprites
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluOrtho2D(self.sprite_pos, self.sprite_pos + self.coords[2], 0, self.coords[3])
        # if not self.half: gluOrtho2D(0, self.coords[2], 0, self.coords[3])
        # else: gluOrtho2D(0, self.coords[2]/2, 0, self.coords[3]/2)
        gluOrtho2D(0, self.orth_coords[2], 0, self.orth_coords[3])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        if not reflect:
            glScalef(*[abs(sf) for sf in self.scale_factors])
        else: #
            glScalef(*self.scale_factors)
        # point the camera --- ignore pos and ori for ortho projection
        gluLookAt (self.sprite_pos, 0.0, 0.0, self.sprite_pos, 0.0, -100.0, 0.0, 1.0, 0.0)
        # draw everything
        self.batch.draw()

    def draw_ref(self, pos, ori):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0,100,0,100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        # glClearColor(0.0, 0.0, 0.0, 1.0) #always use black for ref window bg color
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        glScalef(*self.scale_factors)
        # point the camera
        glPointSize(10)
        gluLookAt(0,0,0,0,0,-1,0,1,0)
        # gluLookAt(0,0,0,50,50,-1,0,1,0)
        # draw everything
        self.batch.draw()

    def set_ref_color(self, ind, color):
        # print (ind, color, self.ref_vl.colors[(ind*3):(ind*3 + 3)])
        self.ref_vl.colors[(ind*3):(ind*3 + 3)] = color


class Viewports_config_class():
    '''A class for holding multiple window configs, adding and removing and reading files'''
    def __init__(self, batch):
        self.vp = []
        self.curr_ind = 0
        self.default_batch = batch
        
    def read_file(self, fn='viewport_config.txt'):
        print('reading {}'.format(fn))
        f = open(fn, 'r')
        lines = [line.split() for line in f.readlines() if not line.strip().startswith('#') and line!='\n']
        f.close()
        self.vp = []
        for line in lines:
            vp = Viewport_config_class(self.default_batch)
            # vp.coords        = map(int, line[:4])
            # vp.scale_factors = map(float, line[4:7])
            vp.coords        = [int(num) for num in line[:4]]
            vp.scale_factors = [float(num) for num in line[4:7]]
            vp.for_ax_ind    = int(line[7])
            vp.for_ax_sign   = int(line[8])
            vp.up_ax_ind     = int(line[9])
            vp.up_ax_sign    = int(line[10])
            vp.projection    = int(line[11])
            # vp.frustum       = map(float, line[12:18])
            vp.frustum       = [float(num) for num in line[12:18]]
            if vp.projection == 0:      #orthographic projection
                vp.project = glOrtho
                vp.draw = vp.draw_perspective
                vp.orth_coords =  vp.coords
            elif vp.projection == 1:    #perspective projection
                vp.project = glFrustum
                vp.draw = vp.draw_perspective
            elif vp.projection == 2:    #ref window
                vp.ref_pt_size = int(line[18])
                vp.ref_coords = array([float(num) for num in line[19:]])
                vp.num_refs = len(vp.ref_coords)//3
                vp.ref_colors = tile((0,0,0),(vp.num_refs,1)).flatten()
                vp.draw = vp.draw_ref
                vp.project = glOrtho
                vp.batch = pyglet.graphics.Batch()
                # print (vp.num_refs, vp.ref_coords.flatten(), vp.ref_colors.flatten())
                vp.ref_vl = vp.batch.add(vp.num_refs, GL_POINTS, None,
                                         ('v3f/static', vp.ref_coords.flatten()),
                                         ('c3B/stream', vp.ref_colors.flatten()))
                vp.color = []
                for pt in range(vp.num_refs):
                    vp.color.append(vp.ref_vl.colors[(pt*3):(pt*3+3)])
                self.ref = vp
            self.vp.append(vp)

    def save_file(self, fn='viewport_config.txt'):
        f = open(fn, 'w')
        for vp in self.vp:
            f.write('{0.coords[0]} {0.coords[1]} {0.coords[2]} {0.coords[3]} {0.scale_factors[0]} {0.scale_factors[1]}  {0.scale_factors[2]} {0.for_ax_ind} {0.for_ax_sign} {0.up_ax_ind} {0.up_ax_sign} {0.projection} {0.frustum[0]} {0.frustum[1]} {0.frustum[2]} {0.frustum[3]} {0.frustum[4]} {0.frustum[5]}'.format(vp))
            if vp.projection == 2:  #ortho or perspective projection --- no ref dots
                # (' {}'*(len(a[19:]) + 1)).format(a[18], *a[19:])
                f.write(' {}'.format(vp.ref_pt_size))
                f.write((' {}'*len(vp.ref_vl.vertices)).format(*vp.ref_vl.vertices[:]))
            f.write('\n')
        f.close()
        print('saved {}'.format(fn))

    def add(self, batch):
        self.vp.append(Viewport_config_class(batch))
        self.curr_ind = len(self.vp) - 1

    def remove(self, add=False):
        del(self.vp[self.curr_ind])

    def choose_vp(self, vp, printit=False):
        if vp > len(self.vp):
            vp = len(self.vp)
        self.curr_ind = vp - 1
        if printit: print('Selected viewport {}'.format(vp))
        self.curr_ref_pt_ind = -1

    def choose_ref_pt(self, pt):
        print('choose_ref_pt')
        if self.vp[self.curr_ind].projection == 2:
            self.curr_ref_pt_ind = pt - 1
            print('Selected ref pt {}'.format(pt))
        else: print('Not a ref window')

    def print_info(self):
        for i in range(len(self.vp)):
            print('{}- for_ax={} for_sign={} up_ax={} up_sign={}'.format(i, self.vp[i].for_ax_ind, self.vp[i].for_ax_sign, self.vp[i].up_ax_ind, self.vp[i].up_ax_sign))

    def move(self, left=0, bottom=0, width=0, height=0):
        if self.vp[self.curr_ind].projection == 2 and self.curr_ref_pt_ind != -1:
            self.move_ref_pt(left, bottom, width, height)
        else:
            self.move_vp(left, bottom, width, height)

    def move_vp(self, left=0, bottom=0, width=0, height=0):
        '''Just move the viewport or change size'''
        self.vp[self.curr_ind].coords += array([left, bottom, width, height])

    def move_ref_pt(self, left=0, bottom=0, width=0, height=0):
        '''move the selected ref pt'''
        self.vp[self.curr_ind].ref_vl.vertices[3*self.curr_ref_pt_ind] += left
        self.vp[self.curr_ind].ref_vl.vertices[3*self.curr_ref_pt_ind + 1] += bottom

    def change_axis(self, axis=None):
        '''switch the axis of the viewport faces'''
        axes = [-3, -2, -1, 1, 2, 3] # these are the valid axes, curr_ax is the index
        curr_ax = axes.index(self.vp[self.curr_ind].for_ax_sign * (self.vp[self.curr_ind].for_ax_ind + 1))
        if axis is None: # if not specified, advance the axis index by one
            new_ax = mod(curr_ax + 1, 6)
        else:            # otherwise set the axis to whatever is specified
            new_ax = axes.index(axis)
        self.vp[self.curr_ind].for_ax_ind = abs(axes[new_ax]) - 1 #reassign index, sign, and up
        self.vp[self.curr_ind].for_ax_sign = sign(axes[new_ax])
        self.vp[self.curr_ind].up_ax_ind, self.vp[self.curr_ind].up_ax_sign = [(1,1),(2,-1),(1,1)][self.vp[self.curr_ind].for_ax_ind]

    def change_up(self, up=None):
        '''switch the top of the camera'''
        ups = [item for item in [(0, 1), (1, 1), (2, 1), (0,-1), (1,-1), (2,-1)] if item[0]!=self.vp[self.curr_ind].for_ax_ind]
        curr_ind = ups.index((self.vp[self.curr_ind].up_ax_ind, self.vp[self.curr_ind].up_ax_sign))
        self.vp[self.curr_ind].up_ax_ind, self.vp[self.curr_ind].up_ax_sign = ups[mod(curr_ind + 1, 4)]

    def change_scale(self, scale):
        scales = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]
        curr_ind = scales.index(self.vp[self.curr_ind].scale_factors)
        self.vp[self.curr_ind].scale_factors = scales[mod(curr_ind + 1, 4)]

    def change_projection(self, vp_ind, projection, half=False):
        '''change the projection of this viewport:
        0 - orthographic
        1 - frustum (perspective)
        2 - orthographic for ref window, not the scene as a whole'''
        if projection == 0:
            self.vp[vp_ind].draw = self.vp[vp_ind].draw_ortho
            self.vp[vp_ind].sprite_pos = int((vp_ind + 1)*1000)
            self.vp[vp_ind].orth_coords = self.vp[vp_ind].coords[:]
            if half:
                self.vp[vp_ind].orth_coords[2]/=2
                self.vp[vp_ind].orth_coords[3]/=2
        elif projection == 1:
            self.vp[vp_ind].draw = self.vp[vp_ind].draw_perspective
        elif projection == 2:
            self.vp[vp_ind].draw = self.vp[vp_ind].draw_ref

    def change_frustum(self, left, right, bottom, top, near, far):
        self.vp[self.curr_ind].frustum = array([left, right, bottom, top, near, far])

    def change_near(self, near):
        # change the near field in vp
        for vp in self.vp:
            vp.frustum[-2] = near

    def change_far(self, far):
        # change the far field in vp
        for vp in self.vp:
            vp.frustum[-1] = far

    def change_batch(self, batch):
        self.vp[self.curr_ind].batch = batch

    def make_ref_window(self):
        print('viewport {}, ref window with 4 points'.format(self.curr_ind))
        vp = self.vp[self.curr_ind]
        self.change_projection(2)
        self.change_frustum(0, 100, 0, 100, -1, 1)
        self.change_batch = pyglet.graphics.Batch()
        vp.num_refs = 4
        vp.ref_pt_size = 5
        vp.ref_coords = tarray([[10, 90, 10, 90.],[90, 90, 10, 10.],[0,0,0,0.]])
        vp.ref_colors = tile((255,255,255),(vp.num_refs,1))
        vp.ref_vl = vp.batch.add(vp.num_refs, GL_POINTS, None,
                                 ('v3f/static', vp.ref_coords.T.flatten()),
                                 ('c3B/stream', vp.ref_colors.flatten()))


    def change_ref_pt_size(self, size):
        self.vp[self.curr_ind].pt_size += size
        if self.vp[self.curr_ind].pt_size < 1:
            self.vp[self.curr_ind].pt_size = 1

        
def __DecimalToAnyBaseArrRecur__(arr, decimal, base):
    arr.append(decimal % base)
    div = decimal / base
    if(div == 0):
        return;
    __DecimalToAnyBaseArrRecur__(arr, div, base)

def DecimalToAnyBaseArr(decimal, base):
    arr = []
    __DecimalToAnyBaseArrRecur__(arr, decimal, base)
    return arr[::-1]
        
class Holocube_window(pyglet.window.Window):

    def __init__(self, ):
        # init the window class
        super(Holocube_window, self).__init__(style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)

        self.platform = pyglet.window.get_platform()
        self.disp = self.platform.get_default_display()
        self.screens = self.disp.get_screens()
        self.w, self.h = self.get_size()

        self.moves = []
        # self.keys = []
        self.key_actions = {}
        self.keypress_actions = {}
        self.keypressed_actions = {}
        self.activate() # give keyboard focus

        # make the batches for display
        self.world = pyglet.graphics.Batch()

        # where's the camera?
        self.pos = zeros((3))
        self.ori = identity(3)

        # history we can save ---obsolete-left in for compatability
        # self.n_history_frames = 120*30
        # self.history_ind = 0
        # self.history = zeros((self.n_history_frames, 3, 4))
        # new history scheme, saves whole trials with the spacebar
        # self.num_hist_frames = 120*30 #maximal length, 30 seconds
        # self.hist_frame_ind = 0       #current frame
        # self.hist_max_trial_ind = 0       #highest trial number
        # self.hist_last_trial_ind = -1       #highest trial number
        # self.hist_fn_ind = 0          #current file number appended
        # self.hist_data = zeros((50, self.n_history_frames, 3, 4)) #the history data structure
        # new new hisory structure, to be declared at the beginning of an experiment
        self.hist_num_frames = 0
        self.hist_num_tests = 0
        self.hist_test_ind = 0
        self.record_fn_ind = 0          #current file number appended
        self.record_data = array([0])

        # how far can we see?
        self.near = .01
        self.far = 10

        # motion
        self.slip, self.lift, self.thrust = 0., 0., 0.
        self.pitch, self.yaw, self.roll = 0., 0., 0.
        self.ind = 0 #for subclass tasks

        self.bg_color = [0.,0.,0.,1.]

        glEnable(GL_DEPTH_TEST)	# Enables Depth Testing
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_SCISSOR_TEST)
        # glPointSize(3)

    def start(self, project=False, projection_screen=0,
              config='viewport_config.txt',
              near=.01, far=10, pspeed=0.1, rspeed=0.1,
              wrap=[1,1,1], load_coords=False,
              bg_color=[0.,0.,0.,1.], forward_only=False,
              loop=True):

        if project:
            self.set_fullscreen(True, self.screens[projection_screen])
        self.bg_color = bg_color

        self.w, self.h = self.get_size()

        self.vp_axis_ind = 0
        self.vp_ind = 0
        self.vps = Viewports_config_class(self.world)
        self.vps.read_file(config)

    def set_ref(self, ind=0, color=[255,0,0]):
        '''change a ref light'''
        self.vps.ref.set_ref_color(ind, color)
    # def set_ref_color(self, ind, color):
    
    def set_far(self, far):
        # self.far = far
        self.vps.change_far(far)

    def set_near(self, near):
        # self.near = near
        self.vps.change_near(near)

    def set_bg(self, color, ref=False):
        for vp in self.vps.vp:
            if ref==False and vp.projection < 2:
                vp.clear_color = color
            elif ref==True and vp.projection==2:
                vp.clear_color = color

    def set_viewport_projection(self, viewport_ind=0, projection=0, half=False):
        '''switch the perspective back and forth for sprites.'''
        self.vps.change_projection(viewport_ind, projection, half=half)

    def save_state(self):
        self.bg_color_saved = self.bg_color
        self.near_saved =    self.near
        self.far_saved =     self.far
        self.on_draw_saved = self.on_draw
        self.pos_saved =     self.pos
        self.ori_saved =     self.ori

    def restore_state(self):
        self.bg_color = self.bg_color_saved
        self.near =    self.near_saved
        self.far =     self.far_saved
        self.on_draw = self.on_draw_saved
        self.pos =     self.pos_saved
        self.ori =     self.ori_saved

    # alter positions, set and inc
    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos
    
    def set_ori(self, ori):
        self.ori = ori

    def get_ori(self):
        return self.ori

    def set_pos_ori(self, pos_ori):
        self.pos = pos_ori[0]
        self.ori = pos_ori[1:]

    def get_pos_ori(self):
        return vstack((self.pos, self.ori))
    
    def set_xpos(self, pos):
        self.pos[0] = pos

    def set_px(self, pos):
        self.pos[0] = pos

    def set_ypos(self, pos):
        self.pos[1] = pos

    def set_py(self, pos):
        self.pos[1] = pos

    def set_zpos(self, pos):
        self.pos[2] = pos

    def set_pz(self, pos):
        self.pos[2] = pos

    def set_rx(self, rot):
        self.ori[0,0] = rot

    def set_ry(self, rot):
        curr_ang = arctan2(self.ori[2,2], self.ori[2,0]) #the x and z component of z (forward/backward direction)
        self.ori = dot(rotmat(array([0.,1.,0.]), curr_ang-rot), self.ori.T).T

    def set_rz(self, rot):
        self.ori[2,2] = rot

    def inc_slip(self, dis):
        self.pos += dis*self.ori[0]

    def inc_lift(self, dis):
        self.pos += dis*self.ori[1]

    def inc_thrust(self, dis):
        self.pos += dis*self.ori[2]

    def reset_pos(self, reset=True):
        if reset: self.pos = zeros((3))

    # alter orientations, only inc
    def inc_pitch(self, ang=0, rand=0):
        '''Increase the pitch angle by ang.
        If there is an imaginary part, it is added randomly.'''
        if ang.imag !=0:
            rand = ang.imag
            ang = ang.real
        angle = ang + random.uniform(-rand/2., rand/2.)
        self.ori = dot(rotmat(self.ori[0], angle), self.ori.T).T

    def inc_yaw(self, ang=0, rand=0):
        '''Increase the yaw angle by ang.
        If there is an imaginary part, it is added randomly.'''
        if ang.imag !=0:
            rand = ang.imag
            ang = ang.real
        angle = ang + random.uniform(-rand/2., rand/2.)
        self.ori = dot(rotmat(self.ori[1], angle), self.ori.T).T

    def inc_roll(self, ang=0, rand=0):
        '''Increase the roll angle by ang.
        If there is an imaginary part, it is added randomly.'''
        if ang.imag !=0:
            rand = ang.imag
            ang = ang.real
        angle = ang + random.uniform(-rand/2., rand/2.)
        self.ori = dot(rotmat(self.ori[2], angle), self.ori.T).T

    def reset_ori(self, reset=True):
        '''Reset the orientation to starting value.'''
        if reset: self.ori = identity(3)


    # to save a whole trial just like on the data computen
    # old
    # def record_frame(self, trial_num=-1):
    #     '''save the position and orientation at this instant'''
    #     if trial_num > -1:
    #         trial_ind = trial_num - 1
    #         if trial_ind != self.hist_last_trial_ind:
    #             self.hist_frame_ind = 0
    #             self.hist_last_trial_ind = trial_ind
    #             if trial_ind > self.hist_max_trial_ind:
    #                 self.hist_max_trial_ind = trial_ind+1
    #         self.hist_data[trial_ind, self.hist_frame_ind,:,0] = self.pos
    #         self.hist_data[trial_ind, self.hist_frame_ind,:,1:] = self.ori
    #         self.hist_frame_ind = mod(self.hist_frame_ind + 1, self.num_hist_frames)
    # #old
    # def save_trial(self, modifiers):
    #     if modifiers & key.MOD_SHIFT:
    #         print ('history erased') #shift space erases history without saving
    #     else:
    #         fn = 'data/' + time.strftime('d%Y-%m-%d') + '-%02d'%(self.hist_fn_ind)
    #         save(fn, self.hist_data[:self.hist_max_trial_ind+1, :self.hist_frame_ind])
    #         self.hist_fn_ind += 1
    #         print('saved {} - {}'.format(fn, self.hist_data[:self.hist_max_trial_ind+1, :self.hist_frame_ind].shape))
    #         # print 'saved ' + fn , fn.shape
    #     self.hist_max_trial_ind = 0
    #     self.hist_frame_ind = 0

    # new --- added start to define number of tests and frames at the beginning
    # also now generic to record any data
    def start_record(self, num_tests, num_frames, data_dims):
        record_dims = [num_tests, num_frames]
        record_dims.extend(data_dims)
        self.record_data = zeros(record_dims)

    def record(self, test_ind, frame_ind, data):
        self.record_data[test_ind, frame_ind] = data

    def save_record(self, fn=None):
        if fn == None:
            fn = 'data/hc-{}-{:02d}.npy'.format(time.strftime('%Y-%m-%d'), self.record_fn_ind)
        save(fn, self.record_data)
        self.record_fn_ind += 1
        print('saved {} - {}'.format(fn, self.record_data.shape))
    
    def save_png(self, num=0, vp_ind=0, prefix='frame'):
        self.vps.vp[vp_ind].draw_perspective(self.pos, self.ori)
        pyglet.image.get_buffer_manager().get_color_buffer().save('{}_{:06d}.png'.format(prefix, num))

    def add_keypress_action(self, key, action, *args, **kwargs):
        # did we provide a (key modifier) pair?
        if hasattr(key, '__iter__'):
            self.keypress_actions[key] = (action, args, kwargs)
        else: # just the key with no modifiers
            self.keypress_actions[(key, 0)] = (action, args, kwargs)

    def remove_keypress_action(self, key):
        del(self.keypress_actions[key])

    def print_keypress_actions(self):
        items = sorted(self.keypress_actions.items())
        for keypress, action in items:
            keysymbol = key.symbol_string(keypress[0]).lstrip(' _')
            modifiers = key.modifiers_string(keypress[1]).replace('MOD_', '').replace('|', ' ').lstrip(' ')
            func, args, kwargs = action[0].__name__, action[1], action[2]
            # print('{:<10} {:<6} --- {:<30}({}, {})'.format(modifiers, keysymbol, func, args, kwargs))

    def on_key_press(self, symbol, modifiers):
        # print symbol, modifiers
        # save a set of history positions and orientations
        # if symbol == key.SPACE:
        #     self.save_trial(modifiers)

        # close the window (for when it has no visible close box)
        if symbol == key.PAUSE or symbol == key.BREAK:
            print('quitting now...')
            self.close()
            
        # print iwnformation about everything
        elif symbol == key.I:
            print( 'pos:\n{}'.format(self.pos))
            print('ori\n{}'.format(self.ori))
            print('fps = {}\n'.format(pyglet.clock.get_fps()))
            self.vps.print_info()

        # reset to original position and orientation
        elif symbol == key.O:
            self.pos = zeros((3))
            self.ori = identity(3)

        # # other keys added by other modules
        elif (symbol, modifiers) in self.key_actions:
            fun, arg = self.key_actions[(symbol, modifiers)]
            fun(arg)

        elif (symbol, modifiers) in self.keypress_actions:
            fun, args, kwargs  = self.keypress_actions[(symbol, modifiers)]
            # print (fun, args, kwargs)
            fun(*args, **kwargs)

        # if there were no hits, report which keys were pressed
        else:
            if symbol not in [65507, 65508, 65513, 65514, 65505, 65506]: #if it's not a common modifier pressed on its own
                print('No action for {} {}'.format(key.modifiers_string(modifiers), key.symbol_string(symbol))) #print whatever it was
            
    def on_draw(self):
        # first clear the whole screen with black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0,0,self.w,self.h)
        # then set the bg_color to clear each viewport
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bg_color)
        # now let each viewport draw itself
        for vp in self.vps.vp:
            vp.draw(self.pos, self.ori)

if __name__=='__main__':
    project = 0
    bg = [1., 1., 1., 1.]
    near, far = .01, 1.

    import holocube.hc5 as hc5


    hc5.window.start(project=project, bg_color=bg, near=near, far=far)
    
    hc5.window.add_key_action(key._0, self.abort, True)

    # run pyglet
    pyglet.app.run()
