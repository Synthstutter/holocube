# windowing classes

import pyglet
from pyglet.gl import *
from pyglet.window import key
import arduino
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
        self.activate() # give keyboard focus

        # import alignment of the viewport windows
        self.vpcoords = array([[self.w//3, self.h//3,   self.h//3-5, self.h//3-5],
                               [self.w*2//3, self.h//3, self.h//3-5, self.h//3-5],
                               [self.w//3, self.h*2//3, self.h//3-5, self.h//3-5],
                               [0, self.h//3,      self.h//3-5, self.h//3-5],
                               [self.w//3, 0,      self.h//3-5, self.h//3-5],
                               [self.w*2//3, 0,      self.h//3-5, self.h//3-5],

                               [10,10 ,1,0],
                               [90,10 ,1,0],
                               [10,90 ,1,0],
                               [90,90 ,1,0]
                               ])
        # make the batches for display
        self.world = pyglet.graphics.Batch()

        self.refs = pyglet.graphics.Batch()
        self.num_refs = 4
        self.ref_coords = self.vpcoords[6:,:3].T
        self.ref_ptsize = 10
        # self.ref_coords = array([[-.44, -.30, -.10, .04],repeat(-.17,4),repeat(-.5,4)])
        self.ref_colors = tile((0,0,0),(self.num_refs,1))
        self.ref_vl = self.refs.add(self.num_refs, GL_POINTS, None,
                                    ('v3f/static', self.ref_coords.T.flatten()),
                                    ('c3B/stream', self.ref_colors.flatten()))
        
        # where's the camera?
        self.pos = zeros((3))
        self.ori = identity(3)

        # history we can save ---obsolete-left in for compatability
        self.n_history_frames = 120*30
        self.history_ind = 0
        self.history = zeros((self.n_history_frames, 3, 4))
        # new history scheme, saves whole trials with the spacebar
        self.num_hist_frames = 120*30 #maximal length, 30 seconds
        self.hist_frame_ind = 0       #current frame
        self.hist_max_trial_ind = 0       #highest trial number
        self.hist_last_trial_ind = -1       #highest trial number
        self.hist_fn_ind = 0          #current file number appended
        self.hist_data = zeros((50, self.n_history_frames, 3, 4)) #the history data structure

        # how far can we see?
        self.near = .01
        self.far = 10

        # motion
        self.slip, self.lift, self.thrust = 0., 0., 0.
        self.pitch, self.yaw, self.roll = 0., 0., 0.
        self.ind = 0 #for subclass tasks

        self.bgcolor = [0.,0.,0.,1.]
        self.refbgcolor = [0.,0.,0.,1.]

        self.viewport_ind = 0

        glEnable(GL_DEPTH_TEST)	# Enables Depth Testing
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_SCISSOR_TEST)
        # glPointSize(3)

    def start(self, project=False, projection_screen=0,
              vpcalibration=None,
              near=.01, far=10, pspeed=0.1, rspeed=0.1,
              wrap=[1,1,1], load_coords=False,
              bgcolor=[0.,0.,0.,1.], forward_only=False,
              projector_tilt=0):

        if project:
            self.set_fullscreen(True, self.screens[projection_screen])
            self.lr_scale = array([-1.,1.,1.])
            self.ud_scale = array([1.,-1.,1.])
        else:
            self.lr_scale = array([1.,1.,1.])
            self.ud_scale = array([1.,1.,1.])

        self.bgcolor = bgcolor

        # projecting or viewing on the monitor?
        if project or load_coords:
            if not vpcalibration: vpcalibration = '~/.viewport_calibration.data'
            try:
                self.vpcoords = loadtxt(expanduser(vpcalibration))
            except:
                print 'no coords'
                print vpcalibration
            self.vpcoords = array(self.vpcoords, dtype='int')

            self.ref_coords = self.vpcoords[6:,:3].T
            self.ref_ptsize = self.vpcoords[6,3]
            self.ref_colors = tile((0,0,0),(self.num_refs,1))
            self.ref_vl.delete()
            self.ref_vl = self.refs.add(self.num_refs, GL_POINTS, None,
                                        ('v3f/static', self.ref_coords.T.flatten()),
                                        ('c3B/stream', self.ref_colors.flatten()))
            
        # how far can we see?
        self.near = near
        self.far = far

        self.w, self.h = self.get_size()

        # motion
        self.pspeed, self.rspeed = pspeed, rspeed

        self.wrap = array(wrap)

        if forward_only: self.on_draw = self.on_draw_forward
        elif projector_tilt == 90: self.on_draw = self.on_draw_bot
        else: self.on_draw = self.on_draw_5


    def ref_color_1(self, color):
        self.ref_vl.colors[0:3] = [color,0,0]

    def ref_color_2(self, color):
        self.ref_vl.colors[3:6] = [color,0,0]

    def ref_color_3(self, color):
        self.ref_vl.colors[6:9] = [color,0,0]

    def ref_color_4(self, color):
        self.ref_vl.colors[9:12] = [color,0,0]

    def ref_light_4(self, color3):
        self.ref_vl.colors[9:12] = color3

    def flash1(self, color):
        self.ref_vl.colors[0] = color

    def flash11(self, color):
        self.ref_vl.colors[0:3] = [color, color, color]

    def flash31(self, color):
        self.ref_vl.colors[0:3] = [color[2],color[0],color[1]]
        
    def flash2(self, color):
        self.ref_vl.colors[3] = color
        
    def flash12(self, color):
        self.ref_vl.colors[3:6] = [color,color,color]
        
    def flash32(self, color):
        self.ref_vl.colors[3:6] = [color[2],color[0],color[1]]
        
    def flash3(self, color):
        self.ref_vl.colors[6] = color

    def flash13(self, color):
        self.ref_vl.colors[6:9] = [color,color,color]
        
    def flash33(self, color):
        self.ref_vl.colors[6:9] = [color[2],color[0],color[1]]
        
    def flash4(self, color):
        self.ref_vl.colors[9] = color

    def flash14(self, color):
        self.ref_vl.colors[9:12] = [color,color,color]
        
    def flash34(self, color):
        self.ref_vl.colors[9:12] = [color[2],color[0],color[1]]
        
    def ref_light(self, ref):
        '''put all lights to 0 except one to light up to 255.'''
        a = zeros(12, dtype='int')
        if ref !=-1: a[(ref*3):((ref+1)*3)] = 255
        self.ref_vl.colors = a

    def ref123(self, num, min=-1., max=1., levels=4):
        '''Try to represent a number with the first 3 ref lights'''
        frac = (num - min)/(max-min)
        n = int(frac*levels**3)
        ndigits = DecimalToAnyBaseArr(n, levels)
        color_1 = ndigits[0]*255/(levels-1)
        color_2 = ndigits[1]*255/(levels-1)
        color_3 = ndigits[2]*255/(levels-1)
        self.ref_color_1(color_1)
        self.ref_color_2(color_2)
        self.ref_color_3(color_3)


    def update_ref_coords(self):
        self.ref_coords = self.vpcoords[6:,:3].T
        # self.ref_coords = array([[-.44, -.30, -.10, .04],repeat(-.17,4),repeat(-.5,4)])
        self.ref_colors = tile((0,0,0),(self.num_refs,1))
        self.ref_vl.delete()
        self.ref_vl = self.refs.add(self.num_refs, GL_POINTS, None,
                                    ('v3f/static', self.ref_coords.T.flatten()),
                                    ('c3B/stream', self.ref_colors.flatten()))


    def set_far(self, far):
        self.far = far

    def set_near(self, near):
        self.near = near

    def set_bg(self, color):
        self.bgcolor = [color, color, color, 1.]

    def set_perspective(self, perspective=True):
        if perspective:
            self.on_draw = self.on_draw_5
        else:
            self.on_draw = self.on_draw_forward

    def save_state(self):
        self.bgcolor_saved = self.bgcolor
        self.near_saved =    self.near
        self.far_saved =     self.far
        self.on_draw_saved = self.on_draw
        self.pos_saved =     self.pos
        self.ori_saved =     self.ori

    def restore_state(self):
        self.bgcolor = self.bgcolor_saved
        self.near =    self.near_saved
        self.far =     self.far_saved
        self.on_draw = self.on_draw_saved
        self.pos =     self.pos_saved
        self.ori =     self.ori_saved

    # alter positions, set and inc
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
    def inc_pitch(self, ang):
        '''Increase the pitch angle by ang.
        If there is an imaginary part, it is added randomly.'''
        self.ori = dot(rotmat(self.ori[0], ang.real + random.uniform(0, ang.imag)), self.ori.T).T

    def inc_yaw(self, ang):
        '''Increase the yaw angle by ang.
        If there is an imaginary part, it is added randomly.'''
        self.ori = dot(rotmat(self.ori[1], ang.real + random.uniform(0, ang.imag)), self.ori.T).T

    def inc_roll(self, ang):
        '''Increase the roll angle by ang.
        If there is an imaginary part, it is added randomly.'''
        self.ori = dot(rotmat(self.ori[2], ang.real + random.uniform(0, ang.imag)), self.ori.T).T

    def reset_ori(self, reset=True):
        '''Reset the orientation to starting value.'''
        if reset: self.ori = identity(3)

    # to save a whole trial just like on the data computen
    def record_frame(self, trial_num=-1):
        '''save the position and orientation at this instant'''
        if trial_num>-1:
            trial_ind = trial_num-1
            if trial_ind != self.hist_last_trial_ind:
                self.hist_frame_ind = 0
                self.hist_last_trial_ind = trial_ind
                if trial_ind > self.hist_max_trial_ind:
                    self.hist_max_trial_ind = trial_ind+1
            self.hist_data[trial_ind, self.hist_frame_ind,:,0] = self.pos
            self.hist_data[trial_ind, self.hist_frame_ind,:,1:] = self.ori
            self.hist_frame_ind = mod(self.hist_frame_ind + 1, self.num_hist_frames)

    def save_trial(self, modifiers):
        if modifiers & key.MOD_SHIFT:
            print 'history erased' #shift space erases history without saving
        else:
            fn = 'data/' + time.strftime('d%Y-%m-%d') + '-%02d'%(self.hist_fn_ind)
            save(fn, self.hist_data[:self.hist_max_trial_ind+1, :self.hist_frame_ind])
            self.hist_fn_ind += 1
            print('saved {} - {}'.format(fn, self.hist_data[:self.hist_max_trial_ind+1, :self.hist_frame_ind].shape))
            # print 'saved ' + fn , fn.shape
        self.hist_max_trial_ind = 0
        self.hist_frame_ind = 0
            
    def save_png(self, num=0, prefix='frame'):
        pyglet.image.get_buffer_manager().get_color_buffer().save('{}_{:06d}.png'.format(prefix, num))
        
    def select_viewport(self, viewport_ind):
        self.viewport_ind = viewport_ind

    def move_viewport(self, parameters):
        xdist, ydist, scale = parameters
        self.vpcoords[self.viewport_ind, 0] += xdist
        self.vpcoords[self.viewport_ind, 1] += ydist
        self.vpcoords[self.viewport_ind, 2:] += scale
        self.update_ref_coords()

    def select_ref_dot(self, ref_dot_ind):
        self.ref_dot_ind = ref_dot_ind

    def move_ref_dot(self, parameters):
        pass
                    
    def add_key_action(self, symbol, action, argument, modifiers=0):
        self.key_actions[(symbol, modifiers)] = (action, argument)

    def on_key_press(self, symbol, modifiers):
        # if   symbol == key.G: self.slip -= self.pspeed
        # elif symbol == key.R: self.slip += self.pspeed

        # elif symbol == key.L: self.lift += self.pspeed
        # elif symbol == key.Z: self.lift -= self.pspeed

        # elif symbol == key.T: self.thrust -= self.pspeed
        # elif symbol == key.S: self.thrust += self.pspeed

        # elif symbol == key.H: self.yaw += self.rspeed
        # elif symbol == key.N: self.yaw -= self.rspeed

        # elif symbol == key.C: self.pitch += self.rspeed
        # elif symbol == key.W: self.pitch -= self.rspeed

        # elif symbol == key.M: self.roll -= self.rspeed
        # elif symbol == key.V: self.roll += self.rspeed

        # save a set of history positions and orientations
        if symbol == key.SPACE:
            self.save_trial(modifiers)

        # close the window (for when it has no visible close box)
        elif symbol == key.PAUSE or symbol == key.BREAK:
            self.close()
            
        # print information about everything
        elif symbol == key.I:
            print 'pos\n', self.pos
            print 'ori\n', self.ori
            print 'coords\n', self.vpcoords
            print 'fps = ', pyglet.clock.get_fps()
            print '\n'

        # reset to original position and orientation
        elif symbol == key.O:
            self.pos = zeros((3))
            self.ori = identity(3)

        elif symbol == key.P:
            self.save_png(prefix='screenshot')

        # # other keys added by other modules
        elif (symbol, modifiers) in self.key_actions:
            fun, arg = self.key_actions[(symbol, modifiers)]
            fun(arg)
            
    def on_key_release(self, symbol, modifiers):
        self.slip, self.lift, self.thrust = 0., 0., 0.
        self.pitch, self.yaw, self.roll = 0., 0., 0.

    def on_draw_bot(self):
        # if self.slip: self.pos += self.slip*self.ori[0]
        # if self.lift: self.pos += self.lift*self.ori[1]
        # if self.thrust: self.pos += self.thrust*self.ori[2]
        # # self.pos[self.pos>self.wrap] -= 2*self.wrap[self.pos>self.wrap]
        # # self.pos[self.pos<-self.wrap] += 2*self.wrap[self.pos<-self.wrap]

        # if self.pitch: self.ori = dot(rotmat(self.ori[0], self.pitch), self.ori.T).T
        # if self.yaw: self.ori = dot(rotmat(self.ori[1], self.yaw), self.ori.T).T
        # if self.roll: self.ori = dot(rotmat(self.ori[2], self.roll), self.ori.T).T

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0,0,self.w,self.h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bgcolor)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90.0, 1.0, self.near, self.far)

        glMatrixMode(GL_MODELVIEW)
        
        # forward
        glLoadIdentity()
        glScissor(*self.vpcoords[0])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[0])
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[2,0], self.pos[1]-self.ori[2,1], self.pos[2]-self.ori[2,2],
                  -self.ori[0,0], -self.ori[0,1], -self.ori[0,2])
        self.world.draw()
        # left
        glLoadIdentity()
        glScissor(*self.vpcoords[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[1])
        glScalef(-1.,1.,1.)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]+self.ori[0,0], self.pos[1]+self.ori[0,1], self.pos[2]+self.ori[0,2],
                  self.ori[2,0], self.ori[2,1], self.ori[2,2])
        self.world.draw()
        # back
        glLoadIdentity()
        glScissor(*self.vpcoords[2])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[2])
        glScalef(1.,1.,1.)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]+self.ori[2,0], self.pos[1]+self.ori[2,1], self.pos[2]+self.ori[2,2],
                  -self.ori[0,0], -self.ori[0,1], -self.ori[0,2])
        self.world.draw()
        # right
        glLoadIdentity()
        glScissor(*self.vpcoords[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[3])
        glScalef(-1.,1.,1.)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[0,0], self.pos[1]-self.ori[0,1], self.pos[2]-self.ori[0,2],
                  self.ori[2,0], self.ori[2,1], self.ori[2,2])
        self.world.draw()
        # down
        glLoadIdentity()
        glScissor(*self.vpcoords[4])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[4])
        glScalef(1.,-1.,1.)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[1,0], self.pos[1]-self.ori[1,1], self.pos[2]-self.ori[1,2],
                  -self.ori[2,0], -self.ori[2,1], -self.ori[2,2])
        self.world.draw()
        #ref window
        glLoadIdentity()
        glScissor(*self.vpcoords[5])
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[5])
        glPointSize(self.ref_ptsize)
        gluLookAt(0,0,0,0,0,-1,0,1,0)
        self.refs.draw()

    def on_draw_5(self):
        # if self.slip: self.pos += self.slip*self.ori[0]
        # if self.lift: self.pos += self.lift*self.ori[1]
        # if self.thrust: self.pos += self.thrust*self.ori[2]
        # # self.pos[self.pos>self.wrap] -= 2*self.wrap[self.pos>self.wrap]
        # # self.pos[self.pos<-self.wrap] += 2*self.wrap[self.pos<-self.wrap]

        # if self.pitch: self.ori = dot(rotmat(self.ori[0], self.pitch), self.ori.T).T
        # if self.yaw: self.ori = dot(rotmat(self.ori[1], self.yaw), self.ori.T).T
        # if self.roll: self.ori = dot(rotmat(self.ori[2], self.roll), self.ori.T).T


        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0,0,self.w,self.h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bgcolor)

        #ref window
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glScissor(*self.vpcoords[5])
        glClearColor(*self.refbgcolor)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[5])
        glPointSize(self.ref_ptsize)
        gluOrtho2D(0,100,0,100)
        self.refs.draw()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90.0, 1.0, self.near, self.far)

        glMatrixMode(GL_MODELVIEW)
        
        # left
        glLoadIdentity()
        glScissor(*self.vpcoords[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[1])
        glScalef(*self.lr_scale)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]+self.ori[0,0], self.pos[1]+self.ori[0,1], self.pos[2]+self.ori[0,2],
                  self.ori[1,0], self.ori[1,1], self.ori[1,2])
        self.world.draw()
        # up
        glLoadIdentity()
        glScissor(*self.vpcoords[2])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[2])
        # glScalef(1.,-1.,1.)
        glScalef(*self.ud_scale)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]+self.ori[1,0], self.pos[1]+self.ori[1,1], self.pos[2]+self.ori[1,2],
                  self.ori[2,0], self.ori[2,1], self.ori[2,2])
        self.world.draw()
        # right
        glLoadIdentity()
        glScissor(*self.vpcoords[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[3])
        # glScalef(-1.,1.,1.)
        glScalef(*self.lr_scale)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[0,0], self.pos[1]-self.ori[0,1], self.pos[2]-self.ori[0,2],
                  self.ori[1,0], self.ori[1,1], self.ori[1,2])
        self.world.draw()
        # down
        glLoadIdentity()
        glScissor(*self.vpcoords[4])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[4])
        # glScalef(1.,-1.,1.)
        glScalef(*self.ud_scale)
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[1,0], self.pos[1]-self.ori[1,1], self.pos[2]-self.ori[1,2],
                  -self.ori[2,0], -self.ori[2,1], -self.ori[2,2])
        self.world.draw()
        # forward
        glLoadIdentity()
        glScissor(*self.vpcoords[0])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[0])
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  self.pos[0]-self.ori[2,0], self.pos[1]-self.ori[2,1], self.pos[2]-self.ori[2,2],
                  self.ori[1,0], self.ori[1,1], self.ori[1,2])
        self.world.draw()


    def on_draw_forward(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0,0,self.w,self.h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bgcolor)

        # # forward
        glScissor(*self.vpcoords[0])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[0])
        
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, self.vpcoords[0,2], 0, self.vpcoords[0,3], -1, 1)
        self.world.draw()

        # ref window
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glScissor(*self.vpcoords[5])
        glClearColor(*self.refbgcolor)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(*self.vpcoords[5])
        glPointSize(self.ref_ptsize)
        gluOrtho2D(0,100,0,100)
        self.refs.draw()


if __name__=='__main__':
    project = 0
    bg = [1., 1., 1., 1.]
    near, far = .01, 1.

    import holocube.hc5 as hc5


    hc5.window.start(project=project, bgcolor=bg, near=near, far=far)
    
    hc5.window.add_key_action(key._0, self.abort, True)

    # run pyglet
    pyglet.app.run()
