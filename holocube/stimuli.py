# class for a moving bar
# lightspeed design 360 projector displays in the order b r g frames

from numpy import *
from numpy import resize as rsize
import pyglet
from pyglet.gl import *
import scipy.stats
from numpy.linalg import norm
from math import acos

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

class Sprite(pyglet.graphics.Group):
    def __init__(self, window, vp=0, half=False):
        self.window = window
        # self.pos = array([vp*1000,0.]) #changed to match change projection in windows.py
        self.pos = array([(vp + 1)*1000, 0.])
        # self.pos = array([0,0.])
        self.rot = array([0.])
        self.visible = False
        self.wd, self.ht = self.window.vps.vp[vp].coords[2], self.window.vps.vp[vp].coords[3]
        if half:
            self.wd = int(self.wd/2)
            self.ht = int(self.ht/2)
            
    def add(self, x=0, y=0):
        self.sprite = pyglet.sprite.Sprite(self.image, x=self.pos[0], y=self.pos[1], batch=self.window.world)
        self.visible = True

    def remove(self):
        self.sprite.delete()
        self.visible = False

    def on(self, onstate):
        if onstate and not self.visible: self.add()
        elif not onstate and self.visible: self.remove()

    def switch(self, onoff):
        '''A synonym for method on, adding string control'''
        if onoff=='on' or onoff==True or onoff==1:
            onstate = True
        elif onoff == 'off' or onoff==False or onoff==0:
            onstate = False
        else:
            onstate = False
            
        if onstate and not self.visible: self.add()
        elif not onstate and self.visible: self.remove()

    def change_vp(self, vp_ind=0):
        '''change the vp_ind by x shifting the entire sprite by 1000*vp_ind'''
        self.sprite.x = int(1000*vp_ind)

    def animate(self, frames):
        # self.sprite.draw
        pass

class Movable(pyglet.graphics.Group):
    '''Any opengl object that we can move and rotate outside of observer motion'''
    def __init__(self, window):
        self.window = window
        self.parent = None # required to use set_state
        self.pos = array([0,0,0.])
        self.rot = array([0,0,0.])
        self.ind = 0
        self.poss = array([[0,0,0.]])
        self.visible = False
        self.txtcoords = None
        self.colors = None

    def add(self):
        # are colors specified?
        if self.colors is None: self.colors = zeros((3, self.num), dtype='byte') + 255
        if self.txtcoords is None: self.txtcoords = zeros((2, self.num), dtype='float')
        self.vl = self.window.world.add(self.num, self.gl_type, self,
                                        ('v3f', self.coords.T.flatten()),
                                        ('c3B/stream', self.colors.T.flatten()),
                                        ('t2f', self.txtcoords.T.flatten()))
        self.visible = True

    def remove(self):
        self.vl.delete()
        self.visible = False

    def on(self, onstate):
        if onstate and not self.visible: self.add()
        elif not onstate and self.visible: self.remove()

    def switch(self, onoff):
        '''A synonym for method on, adding string control'''
        if onoff=='on' or onoff==True or onoff==1:
            onstate = True
        elif onoff == 'off' or onoff==False or onoff==0:
            onstate = False
        else:
            onstate = False
            
        if onstate and not self.visible: self.add()
        elif not onstate and self.visible: self.remove()

    def set_color(self, color=array([1,1,1])):
        self.vl.colors = take(atleast_1d(array(color, dtype='int8')), mod(arange(self.num*3),3), mode='wrap')

    def set_pos(self, pos):
        self.pos[:] = pos

    def set_rot(self, rot):
        self.rot[:] = rot

    def set_px(self, x):
        self.pos[0] = x

    def set_py(self, y):
        self.pos[1] = y

    def set_pz(self, z):
        self.pos[2] = z

    def set_rx(self, x):
        self.rot[0] = x

    def set_ry(self, y):
        self.rot[1] = y

    def set_rz(self, z):
        self.rot[2] = z

    def inc_px(self, x=.01):
        self.pos[0] += x

    def inc_py(self, y=.01):
        self.pos[1] += y

    def inc_pz(self, z=.01):
        self.pos[2] += z

    def inc_rx(self, x=pi/180):
        self.rot[0] += x

    def inc_ry(self, y=pi/180):
        self.rot[1] += y

    def inc_rz(self, z=pi/180):
        self.rot[2] += z

    def update_pos_ind(self, dt=0.0):
        take(self.poss, [self.ind], out=self.pos, mode='wrap')
        self.ind += 1

    def update_ry_func(self, dt=0.0, func=None):
        self.rot[1] = func()
    
    def subset_inc_px(self, bool_a, x):
        '''move in x direction just some of the vertices'''
        self.coords[0, bool_a] += x
        self.vl.vertices[::3] = self.coords[0]

    def subset_inc_py(self, bool_a, y):
        '''move in z direction just some of the vertices'''
        self.coords[1, bool_a] += y
        self.vl.vertices[1::3] = self.coords[1]

    def subset_inc_pz(self, bool_a, z):
        '''move in z direction just some of the vertices'''
        self.coords[2, bool_a] += z
        self.vl.vertices[2::3] = self.coords[2]

    def subset_set_px(self, bool_a, x_a):
        '''set x coordinate for some vertices '''
        self.coords[0, bool_a] = x_a[bool_a]
        self.vl.vertices[::3] = self.coords[0]

    def subset_set_py(self, bool_a, y_a):
        '''set y coordinate for some vertices '''
        self.coords[1, bool_a] = y_a[bool_a]
        self.vl.vertices[1::3] = self.coords[1]
                
    def subset_set_pz(self, bool_a, z_a):
        '''set z coordinate for some vertices '''
        self.coords[2, bool_a] = y_a[bool_a]
        self.vl.vertices[2::3] = self.coords[2]

    def set_state(self):
        glRotatef(self.rot[0], 1.0, 0.0, 0.0)
        glRotatef(self.rot[1], 0.0, 1.0, 0.0)
        glRotatef(self.rot[2], 0.0, 0.0, 1.0)
        glTranslatef(*self.pos)

    def unset_state(self):
        glTranslatef(*-self.pos)
        glRotatef(-self.rot[2], 0.0, 0.0, 1.0)
        glRotatef(-self.rot[1], 0.0, 1.0, 0.0)
        glRotatef(-self.rot[0], 1.0, 0.0, 0.0)


class Shape(Movable):
    '''An arbitrary polygon.'''
    def __init__(self, window, coords, color=1., add=False):
        super(Shape, self).__init__(window)
        self.gl_type = GL_POLYGON
        self.coords = array(coords)
        self.num = self.coords.shape[1]
        self.txtcoords = None
        self.color = color
        self.colors = array(repeat(color*255, self.num*3), dtype='byte')
        if add: self.add()

class Horizon(Shape):
    '''A horizon rendered out to some large distance off.'''
    def __init__(self, window, depth, dist, color=1., add=False):
        c = array([[-dist, -dist, dist, dist],[depth, depth, depth, depth], [-dist, dist, dist, -dist]])
        super(Horizon, self).__init__(window, coords=c, color=color)


# class pts_class(movable_fast_class):
class Points(Movable):

    def __init__(self, window, num=1000, dims=[(-1,1),(-1,1),(-1,1)],
                 color=1., pt_size=1, add=False):

        super(Points, self).__init__(window)
        self.gl_type = GL_POINTS
        self.pt_size = pt_size
        self.num = num
        self.color = color
        self.colors = array(repeat(color*255, self.num*3), dtype='byte')
        self.dims = array(dims)
        if len(self.dims.shape) == 1:
            self.dims = array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])
        self.txtcoords = None

        self.init_coords()
        if add: self.add()


    def set_num(self, num):
        self.num = num
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def shuffle(self, shuf=True):
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def init_coords(self):
        self.coords = array([random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                             random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                             random.uniform(self.dims[2][0], self.dims[2][1], self.num)])

    def set_pt_size(self, pt_size):
        self.pt_size = pt_size
        
    def set_state(self):
        super(Points, self).set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super(Points, self).unset_state()
        glPointSize(1)

class pts_trail_class(Movable):
    def __init__(self, window, num=1000, dims=[(-1,1),(-1,1),(-1,1)],
                 color=1., trail_size=10, add=False):

        self.gl_type = GL_POINTS
        self.num = num*trail_size
        self.trail_size = trail_size
        self.color = color
        self.colors = array(repeat(color*255, self.num*3), dtype='byte')
        self.dims = array(dims)
        if len(self.dims.shape) == 1:
            self.dims = array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])

        self.init_coords()
        if add: self.add()

        super(pts_class2, self).__init__(window)

    def set_num(self, num):
        self.num = num
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def shuffle(self):
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def init_coords(self):
        self.coords = array([random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                             random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                             random.uniform(self.dims[2][0], self.dims[2][1], self.num)])



class lines_class(Movable):
    def __init__(self, window, num=1000, dims=[(-1,1),(-1,1),(-1,1)],
                 color=1., ln_width=1, add=False):

        self.gl_type = GL_LINES
        self.ln_width = ln_width
        self.num = num*2
        self.color = color
        self.colors = array(repeat(color*255, self.num*3), dtype='byte')
        self.dims = array(dims)
        if len(self.dims.shape) == 1:
            self.dims = array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])

        self.txtcoords = None
        super(lines_class, self).__init__(window)

        self.init_coords()
        if add: self.add()

    def init_coords(self):
        self.coords = array([random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                             random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                             random.uniform(self.dims[2][0], self.dims[2][1], self.num)])

    def set_ln_width(self, ln_width):
        self.ln_width = ln_width

    def set_state(self):
        super(lines_class, self).set_state()
        glLineWidth(self.ln_width)

    def unset_state(self):
        super(lines_class, self).unset_state()
        glLineWidth(1)
    

class bar_class(Movable):
    '''A simple rotating bar.'''
    def __init__(self, window, width=.15, height=2, dist=.8, color=(1,1,1),
                 add=False):
        Movable.__init__(self, window)

        self.width=width
        self.height=height
        self.num = 4
        self.gl_type=GL_QUADS

        xl, xr = -width/2., width/2.
        yb, yt = -height/2., height/2.
        z = -dist

        self.coords = array([[xl, xl, xr, xr],[yb, yt, yt, yb],[z, z, z, z]])
        self.colors = tile(color,4)*255

        if add: self.add()

    def set_width_height(self, wh):
        width, height = wh
        xl, xr = -width/2., width/2.
        yb, yt = -height/2., height/2.
        self.coords[:2] = array([[xl, xl, xr, xr],[yb, yt, yt, yb]])

    def set_top(self, yt):
        self.coords[1,1:3] = yt

    def set_bot(self, yb):
        self.coords[1,0:4:3] = yb

    def set_top_ang(self, ang):
        x = self.coords[2,0]
        y = tan(ang*pi/180)*abs(x)
        self.set_top(y)

    def set_bot_ang(self, ang):
        x = self.coords[2,0]
        y = tan(ang*pi/180)*abs(x)
        self.set_bot(y)


class cbar_class(Movable):
    '''A multicolored rotating bar.'''
    def __init__(self, window, width=.15, height=2, dist=.8, color=(1,1,1),
                 add=False):
        Movable.__init__(self, window)

        self.width=width
        self.height=height
        self.num = 12
        self.gl_type=GL_QUADS

        xl, xr = -width/2., width/2.
        yb, yt = -height/2., height/2.
        z = -dist


        self.coords = array([[xl/4, xl/4, xr/4, xr/4, 2*xl/3, 2*xl/3, 2*xr/3, 2*xr/3, xl, xl, xr, xr],
                             [yb, yt, yt, yb, yb, yt, yt, yb, yb, yt, yt, yb],
                             [z+.02, z+.02, z+.02, z+.02, z+.01, z+.01, z+.01, z+.01, z, z, z, z]])
        self.colors = array([[255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255],
                             [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255],
                             [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255]])

        if add: self.add()


class cbarr_class(Movable):
    '''A multicolored rotating bar---reverse colors.'''
    def __init__(self, window, width=.15, height=2, dist=.8, color=(1,1,1),
                 add=False):
        Movable.__init__(self, window)

        self.width=width
        self.height=height
        self.num = 12
        self.gl_type=GL_QUADS

        xl, xr = -width/2., width/2.
        yb, yt = -height/2., height/2.
        z = -dist


        self.coords = array([[xl/4, xl/4, xr/4, xr/4, 2*xl/3, 2*xl/3, 2*xr/3, 2*xr/3, xl, xl, xr, xr],
                             [yb, yt, yt, yb, yb, yt, yt, yb, yb, yt, yt, yb],
                             [z+.02, z+.02, z+.02, z+.02, z+.01, z+.01, z+.01, z+.01, z, z, z, z]])
        self.colors = array([[0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                             [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                             [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0]])

        if add: self.add()

class bars_class(Movable):
    '''Oriented bar array in the frontal visual field.'''
    def __init__(self, window, dist=.25, w=.02, h=.15, o=pi/4, z=-0.8,
                 xlim=[-1., 1.], ylim=[-1., 1.], color=1, add=False):
        Movable.__init__(self, window)

        self.dist = dist
        self.width = w
        self.height = h
        self.orientation = o
        self.z = z
        self.xlim = xlim
        self.ylim = ylim

        self.add_bar_field()
        self.make_coord_list()

        self.num = len(self.bars)*4
        self.gl_type=GL_QUADS

        self.colors = array(repeat(color*255, self.num*3), dtype='byte')

        if add: self.add()

    def bar_pts(self, x, y, z, w, h, o):
        # get the 4 vertices of a bar given x and y of its center
        xl, xr, yb, yt = -h/2., h/2., -w/2., w/2. 
        
        pts = array([[xl, xl, xr, xr], [yb, yt, yt, yb], [z, z, z, z]])
        pts = dot(rotmat([0,0,-1.], -o), pts)
        pts[0] += x
        pts[1] += y
        return pts

    def add_bar_field(self):
        self.bars = []
        for i in arange(self.xlim[0], self.xlim[1], self.dist):
            for j in arange(self.ylim[0], self.ylim[1], self.dist):
                self.bars.append([i, j, self.z, self.width, self.height, self.orientation])

    def orient(self, o):
        self.orientation = o
        self.add_bar_field()
        self.make_coord_list()

    def d_height(self, h):
        self.height = h
        self.add_bar_field()
        self.make_coord_list()

    def make_coord_list(self):
        num_bars = len(self.bars)
        self.coords = zeros([3,4*num_bars])
        for i in arange(num_bars):
            b = self.bars[i]
            pts = self.bar_pts(b[0], b[1], b[2], b[3], b[4], b[5])
            self.coords[0,i*4:(i+1)*4] = pts[0]
            self.coords[1,i*4:(i+1)*4] = pts[1]
            self.coords[2,i*4:(i+1)*4] = pts[2]


class disk_class(Movable):
    '''discs'''
    def __init__(self, window, radius, color=(1,1,1), num_pts=60, add=False):
        Movable.__init__(self, window)
        a = linspace(0,2*pi,num_pts, endpoint=False)
        self.coords = array([radius*cos(a), radius*sin(a), -ones(num_pts)])
        self.num = num_pts
        self.colors = tile(color,num_pts)*255
        self.gl_type = GL_POLYGON
        self.txtcoords = None
        if add: self.add

class sphere_lines_class(Movable):
    '''A class of lines circling a sphere for calibration.'''
    def __init__(self, window, num_lines=12, num_segs=50, add=False):

        Movable.__init__(self, window)

        self.colors = None

        self.line_angs = linspace(-pi, pi, num_lines + 2)[1:-1]
        self.seg_angs = linspace(0, 2*pi, num_segs + 1)
        
        self.gl_type = GL_LINES

        self.num = num_lines*num_segs*2*3
        
        self.coords = zeros([3, self.num])
        self.init_coords()

        if add: self.add()

    def init_coords(self):
        ind = 0

        for line_ang in self.line_angs:
            for s in arange(len(self.seg_angs)-1):
                cseg_ang = self.seg_angs[s]
                nseg_ang = self.seg_angs[s+1]
                
                r = cos(line_ang)

                # around z
                # first coord of a segment
                self.coords[0, ind] = r*cos(cseg_ang)
                self.coords[1, ind] = r*sin(cseg_ang)
                self.coords[2, ind] = sin(line_ang)
                ind += 1

                # second part
                self.coords[0, ind] = r*cos(nseg_ang)
                self.coords[1, ind] = r*sin(nseg_ang)
                self.coords[2, ind] = sin(line_ang)
                ind += 1

                # around y
                # first coord of a segment
                self.coords[0, ind] = r*cos(cseg_ang)
                self.coords[1, ind] = sin(line_ang)
                self.coords[2, ind] = r*sin(cseg_ang)
                ind += 1

                # second part
                self.coords[0, ind] = r*cos(nseg_ang)
                self.coords[1, ind] = sin(line_ang)
                self.coords[2, ind] = r*sin(nseg_ang)
                ind += 1
                
                # around x
                # first coord of a segment
                self.coords[0, ind] = sin(line_ang)
                self.coords[1, ind] = r*sin(cseg_ang)
                self.coords[2, ind] = r*cos(cseg_ang)
                ind += 1

                # second part
                self.coords[0, ind] = sin(line_ang)
                self.coords[1, ind] = r*sin(nseg_ang)
                self.coords[2, ind] = r*cos(nseg_ang)
                ind += 1

class horizon_class(Movable):
    '''a horizon of any color that can be raised or tilted.'''
    def __init__(self, window, elevation=0, radius=2., flipped=False, color=1., add=False):
        self.gl_type=GL_QUADS
        self.color = color
        self.flipped = flipped
        self.elevation = elevation
        self.init_coords(elevation, radius)
        if add: self.add()
        super(horizon_class, self).__init__(window)

    def set_elevation(self, elevation):
        self.elevation = elevation
        self.init_coords(elevation)

    def set_color(self, color):
        self.color = color
        self.init_coords(self.elevation)

    def change_color(self, color):
        '''you have to have added the instance
        already so it has a vertex list. color is 0-255'''
        self.colors[:] = color
        self.vl.colors[:] = self.colors
        
    def init_coords(self, elevation=0, radius=2.):
        num_azis = 50
        num_elevs = 180

        xlist, ylist, zlist = [], [], []
        elevs = linspace(-pi/2, pi/2, num_elevs+1)
        if self.flipped: elevs = elevs[::-1] #reverse the elevations to make the horizon above instead of below
        azis = linspace(0, 2*pi, num_azis+1)
        for elev_ind in range(int(elevation)+90):
            lower_elev = elevs[elev_ind]
            upper_elev = elevs[elev_ind + 1]
            lower_y = radius*sin(lower_elev)
            upper_y = radius*sin(upper_elev)
            lower_rad = radius*cos(lower_elev)
            upper_rad = radius*cos(upper_elev)

            y1, y2, y3, y4 = lower_y, upper_y, upper_y, lower_y

            for azi_ind in range(num_azis):
                cur_azi = azis[azi_ind]
                next_azi = azis[azi_ind + 1]
                x1 = lower_rad * cos(cur_azi)
                x2 = upper_rad * cos(cur_azi)
                x3 = upper_rad * cos(next_azi)
                x4 = lower_rad * cos(next_azi)

                z1 = lower_rad * sin(cur_azi)
                z2 = upper_rad * sin(cur_azi)
                z3 = upper_rad * sin(next_azi)
                z4 = lower_rad * sin(next_azi)

                xlist.extend([x1, x2, x3, x4])
                ylist.extend([y1, y2, y3, y4])
                zlist.extend([z1, z2, z3, z4])

        self.coords = array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.txtcoords = None
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')


class Tree(Movable):
    def __init__(self, window, distance=3, num_sides=12, num_levels=3, color=0.0, add=False):
        self.gl_type=GL_QUADS
        self.color = color
        self.num_sides = num_sides
        self.num_levels = num_levels
        angs = linspace(0, 2*pi, self.num_sides+1)
        self.cos_angs = cos(angs)
        self.sin_angs = sin(angs)
        super(Tree, self).__init__(window)
        self.init_coords(distance=distance)
        if add: self.add()

    def add_branch(self, x1, y1, z1, x2, y2, z2, rad):
        '''Add all the coords for a slanted cylinder section of a branch.'''
        xstart, ystart, zstart = x1 + rad * self.cos_angs, repeat(y1, self.num_sides), z1 + rad * self.sin_angs
        xend, yend, zend       = x2 + rad * self.cos_angs, repeat(y2, self.num_sides), z2 + rad * self.sin_angs
        for side in range(self.num_sides):
            ind1, ind2 = side, mod(side + 1, self.num_sides)
            self.xlist.extend([xstart[ind1], xstart[ind2], xend[ind2], xend[ind1]])
            self.ylist.extend([ystart[ind1], ystart[ind2], yend[ind2], yend[ind1]])
            self.zlist.extend([zstart[ind1], zstart[ind2], zend[ind2], zend[ind1]])

    def add_tree(self, level=0, pstart_pt=array([0,-1,0.]), pend_pt=array([0,0,0.]), lens=(1,.2,.7), angs=(30., 15.), nbrs=3):
        # an array of vectors representing start and stop pts of branches
        start_pt = pend_pt #new start_pt
        # make a random branch length
        #lens[0] is mean length, lens[1] is std, lens[2] is decay with branch level
        ln = lens[2]**level*lens[0] + lens[2]**level*random.randn()*lens[1]

        # make a random angle
        ang = (angs[0] + 90 + angs[1]*random.randn())*pi/180 #random angle
        if level==0: ang = pi/2.
        nx, ny, nz = ln*cos(ang), ln*sin(ang), 0.
        raz = 2*pi*random.rand()  #swing around by a random azimuth
        nx, nz = nx*cos(raz), nx*sin(raz)

        # angle it to the existing branch
        stem = pend_pt - pstart_pt
        xyang, xzang = arctan2(stem[1], stem[0])-pi/2, arctan2(stem[2], stem[0])
        nx, ny, nz = nx*cos(xyang) - ny*sin(xyang), nx*sin(xyang) + ny*cos(xyang), nz #xy rot (along z) 
        nx, ny, nz = nx*cos(xyang) + nz*sin(xyang), ny, -nx*sin(xyang) + nz*cos(xyang) #xz rot (along y) 
        end_pt = array([nx, ny, nz])
        # and translate
        end_pt += start_pt

        #if poisson(nlevs)+1>lev:
        if level < self.num_levels:
            for i in arange(nbrs):
                self.add_branch(start_pt[0], start_pt[1], start_pt[2], end_pt[0], end_pt[1], end_pt[2], 0.05/(level+1))
                self.add_tree(level + 1, start_pt, end_pt, lens, angs, nbrs)


    def init_coords(self, distance=0):
        self.xlist, self.ylist, self.zlist = [], [], []
        self.add_tree(level=0)

        self.coords = array([self.xlist, self.ylist, self.zlist])
        self.coords[2] -= distance
        self.num = len(self.xlist)
        self.txtcoords = None
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        

class Forest(Movable):
    def __init__(self, window, numtrees=50, add=False):
        self.gl_type=GL_QUADS
        self.color = .5
        
        self.init_coords(numtrees)
        
        if add: self.add()
        super(Forest, self).__init__(window)

    def init_coords(self, numtrees):

        numsides = 12
        angs = linspace(0, 2*pi, numsides+1)
        cosangs = cos(angs)
        sinangs = sin(angs)
        xlist, ylist, zlist = [], [], []

        # first the ground
        grnd = -.2
        xlist.extend([-100, 100, 100, -100])
        zlist.extend([-100, -100, 100, 100])
        ylist.extend([grnd, grnd, grnd, grnd])

        # now the trunks
        for i in range(numtrees):
            diameter = random.uniform(.02, .2)
            height = random.uniform(.4, 10)
            x, z = random.uniform(-10, 10, 2)
            for j in range(numsides):
                xlist.extend([x+diameter*cosangs[j],
                              x+diameter*cosangs[j],
                              x+diameter*cosangs[j+1],
                              x+diameter*cosangs[j+1]])
                zlist.extend([z+diameter*sinangs[j],
                              z+diameter*sinangs[j],
                              z+diameter*sinangs[j+1],
                              z+diameter*sinangs[j+1]])
                ylist.extend([grnd, grnd+height, grnd+height, grnd])

        self.coords = array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')
        self.colors[:12] = 255

class grating_class(Sprite):
    '''Moving gratings and plaids in a single window.  Fast means put
    different values in the blue, red and green sequential channels.'''
    def __init__(self, window, vp=0, rate=120., add=False, fast=True, half=False):

        Sprite.__init__(self, window, vp=vp, half=half)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        h, v = meshgrid(linspace(-pi/4, pi/4, self.wd), linspace(-pi/4, pi/4, self.ht))
        self.center_dists = sqrt(h**2 + v**2)
        self.atans = arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.gratings = []

        if add: self.add()

    def add_grating(self, sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        # spatial
        phi_ss = 2*pi*sf*cos(self.atans + (o-pi/2))*self.center_dists

        # temporal
        if hasattr(tf, '__iter__'): tf_array = array(tf) #is it changing?
        else:
            tf_array = repeat(tf, min(abs(self.rate/tf), maxframes)) #or constant?

        nframes = len(tf_array)
        phi_ts = cumsum(-2*pi*tf_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t = 0 #first frame?
                else: prev_phi_t = phi_ts[f-1]
                phi_ts_b, phi_ts_r, phi_ts_g = linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else: #or grays
                phi_ts_b, phi_ts_r, phi_ts_g = repeat(phi_ts[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_b + phi_i))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_r + phi_i))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_g + phi_i))
            data[:,:,1] = lum[:,:]
	    # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add_grating_fast(self, sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, sdb=None, num_frames=120):
        num_frames *= 3
        frames = []
            
        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])
            
        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        sf = rsize(sf, (num_frames))
        tf = rsize(tf, (num_frames))
        c = rsize(c, (num_frames))
        o = rsize(o, (num_frames))
        phi_ss_array = 2*pi*sf[newaxis,newaxis,:]*cos(self.atans[:,:,newaxis] + (o[newaxis,newaxis,:]-pi/2))*self.center_dists[:,:,newaxis]
        phi_ts_array = cumsum(-2*pi*tf/float(self.rate*3))

        lum = array(127*(1 + mask_ss[:,:,newaxis]*c[newaxis,newaxis,:]*sin(phi_ss_array + phi_ts_array[newaxis,newaxis,:] + phi_i)), dtype='ubyte')
        frames = [pyglet.image.ImageData(self.wd, self.ht, 'RGB', lum[:,:,(f+2,f,f+1)].tostring()) for f in arange(0, num_frames, 3)]

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add_radial(self, sf=1., tf=1., c=1., phi_i=0, sd=None, sdb=None, maxframes=500):
        '''Add a circular radiating grating'''
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        # spatial
        # phi_ss = 2*pi*sf*cos(self.atans + (o-pi/2))*self.center_dists
        phi_ss = 2*pi*sf*self.center_dists

        # temporal
        if hasattr(tf, '__iter__'): tf_array = array(tf) #is it changing?
        else: tf_array = repeat(tf, min(abs(self.rate/tf), maxframes)) #or constant?

        nframes = len(tf_array)
        phi_ts = cumsum(-2*pi*tf_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t = 0 #first frame?
                else: prev_phi_t = phi_ts[f-1]
                phi_ts_b, phi_ts_r, phi_ts_g = linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else: #or grays
                phi_ts_b, phi_ts_r, phi_ts_g = repeat(phi_ts[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_b + phi_i))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_r + phi_i))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_g + phi_i))
            data[:,:,1] = lum[:,:]
	    # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add_plaid(self, sf1=.1, tf1=1, c1=1, o1=0, phi_i1=0.,
                  sf2=.1, tf2=1, c2=1, o2=0, phi_i2=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='byte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        # spatial
        phi_ss1 = 2*pi*sf1*cos(self.atans + (o1-pi/2))*self.center_dists
        phi_ss2 = 2*pi*sf2*cos(self.atans + (o2-pi/2))*self.center_dists
                    
        # temporal
        if hasattr(tf1, '__iter__'): lentf1=len(tf1)
        else: lentf1 = self.rate/tf1
        if hasattr(tf2, '__iter__'): lentf2=len(tf2)
        else: lentf2 = self.rate/tf2
        if hasattr(tf1, '__iter__'): tf1_array = array(tf1) #is it changing?
        else: tf1_array = repeat(tf1, min(max(lentf1, lentf2), maxframes)) #or constant?
        if hasattr(tf2, '__iter__'): tf2_array = array(tf2) #is it changing?
        else: tf2_array = repeat(tf2, min(max(lentf1, lentf2), maxframes)) #or constant?

        nframes = max(len(tf1_array), len(tf2_array))
        phi_t1s = cumsum(-2*pi*tf1_array/float(self.rate))
        phi_t2s = cumsum(-2*pi*tf2_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t1 = prev_phi_t2 = 0 #first frame?
                else:
                    prev_phi_t1 = phi_t1s[f-1]
                    prev_phi_t2 = phi_t2s[f-1]
                phi_t1s_b, phi_t1s_r, phi_t1s_g = linspace(prev_phi_t1, phi_t1s[f], 4)[1:]
                phi_t2s_b, phi_t2s_r, phi_t2s_g = linspace(prev_phi_t2, phi_t2s[f], 4)[1:]
            else: #or grays
                phi_t1s_b, phi_t1s_r, phi_t1s_g = repeat(phi_t1s[f],3)
                phi_t2s_b, phi_t2s_r, phi_t2s_g = repeat(phi_t2s[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_b + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_b + phi_i2)))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_r + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_r + phi_i2)))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_g + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_g + phi_i2)))
            data[:,:,1] = lum[:,:]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def choose_grating(self, num):
        '''Assign one of the already generated gratings as the current
        display grating'''
        self.image = self.gratings[num]


class Sphere_segment(Movable):
    '''a y-axis-centered spherical segment defined by an inner and outer angles_(in degrees measured from the y-axis). Outer angle should be larger than inner , both smaller than 90 degrees.'''
    def __init__(self, window, sphere_rad = 0.5, outer_ang = 0.5, inner_ang = 0, color=1., add=False):
        self.gl_type=GL_QUADS
        self.color = color
        self.sphere_rad = sphere_rad
        self.outer_ang = float(outer_ang)
        self.inner_ang = float(inner_ang)
        self.init_coords()
        if add: self.add()
        super(Sphere_segment, self).__init__(window)

    def set_angs(self, outer_ang, inner_ang):
        self.inner_ang = inner_ang
        self.outer_ang = outer_ang
        self.init_coords()
        
    def set_color(self, color):
        self.color = color
        self.init_coords()
        
    def change_color(self, color):
        '''you have to have added the instance
        already so it has a vertex list. color is 0-255'''
        self.colors[:] = color
        self.vl.colors[:] = self.colors
        
    def init_coords(self):
        num_azis = 180
        num_elevs = 180
        xlist, ylist, zlist = [], [], []
        elevs = linspace(-pi/2, pi/2, num_elevs+1)
        azis = linspace(0, 2*pi, num_azis+1)
        rad_a = sin(self.outer_ang/180*pi)*self.sphere_rad
        rad_b = sin(self.inner_ang/180*pi)*self.sphere_rad
        lower_elevation = cos(self.outer_ang/180*pi)*self.sphere_rad
        upper_elevation = cos(self.inner_ang/180*pi)*self.sphere_rad
        elevs = linspace(lower_elevation, upper_elevation, num_elevs)
        for elev_ind, elev_val in enumerate(elevs[:-1]):
            lower_y = elev_val
            upper_y = elevs[elev_ind + 1]
            lower_rad = sin(acos(lower_y/self.sphere_rad))*self.sphere_rad
            upper_rad = sin(acos(upper_y/self.sphere_rad))*self.sphere_rad
            y1, y2, y3, y4 = lower_y, upper_y, upper_y, lower_y
            for azi_ind in range(num_azis):
                cur_azi = azis[azi_ind]
                next_azi = azis[azi_ind + 1]
                x1 = lower_rad * cos(cur_azi)*self.sphere_rad
                x2 = upper_rad * cos(cur_azi)*self.sphere_rad
                x3 = upper_rad * cos(next_azi)*self.sphere_rad
                x4 = lower_rad * cos(next_azi)*self.sphere_rad

                z1 = lower_rad * sin(cur_azi)*self.sphere_rad
                z2 = upper_rad * sin(cur_azi)*self.sphere_rad
                z3 = upper_rad * sin(next_azi)*self.sphere_rad
                z4 = lower_rad * sin(next_azi)*self.sphere_rad

                xlist.extend([x1, x2, x3, x4])
                ylist.extend([y1, y2, y3, y4])
                zlist.extend([z1, z2, z3, z4])

        self.coords = array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.txtcoords = None
        self.colors = array(repeat(self.color*255, self.num*3), dtype='byte')

        
class Grating_cylinder(Sprite):
    '''Moving gratings and plaids in a single window.  Fast means put
    different values in the blue, red and green sequential channels.'''
    def __init__(self, window, vp=0, rate=120., center=[0,0],
                 add=False, fast=False, half=False):

        Sprite.__init__(self, window, vp=vp, half=half)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        # h, v = meshgrid(linspace(-pi/4, pi/4, self.wd), linspace(-pi/4, pi/4, self.ht))
        h, v = meshgrid(arctan2(linspace(-1, 1, self.wd),1), linspace(-pi/4, pi/4, self.ht))
        self.center_dists = sqrt((h+center[0])**2 + (v+center[1])**2)
        self.atans = arctan2(h+center[0], v+center[1])

        self.rate = rate
        self.fast = fast
        self.gratings = []
        self.vp = vp

        if add: self.add()

    def add_grating(self, sf=.1, tf=1, c=1, o=0, phi_i=0.,
                    sd=None, sdb=None, mask_reflect=False,
                    dots=False, dot_speed=[0,0,0.], maxframes=500):
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])


        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        if mask_reflect:
            half = int(self.wd/2)
            mask_ss[:,-half:] = mask_ss[:,:half][:,::-1]
            mask_ssb[:,-half:] = mask_ssb[:,:half][:,::-1]

        # spatial
        phi_ss = 2*pi*sf*cos(self.atans + (o-pi/2))*self.center_dists

        # temporal
        if hasattr(tf, '__iter__'): tf_array = array(tf) #is it changing?
        else: tf_array = repeat(tf, min(abs(self.rate/tf), maxframes)) #or constant?

        nframes = len(tf_array)
        phi_ts = cumsum(-2*pi*tf_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t = 0 #first frame?
                else: prev_phi_t = phi_ts[f-1]
                phi_ts_b, phi_ts_r, phi_ts_g = linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else: #or grays
                phi_ts_b, phi_ts_r, phi_ts_g = repeat(phi_ts[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_b + phi_i))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_r + phi_i))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_g + phi_i))
            data[:,:,1] = lum[:,:]
            # add the dots
            if dots: # if we are rendering dots
                self.relposs = dots.coords - self.window.pos[:,newaxis] - (f * array(dot_speed))[:,newaxis]
                self.dists = sqrt((self.relposs**2).sum(0))
                self.azs = arctan2(self.relposs[2], self.relposs[0])
                self.els = arctan2(self.relposs[1], self.dists)
                for i in range(len(self.dists)): # grab them one by one
                    # if self.dists[i] < self.window.far: #if this one is in range
                    if self.dists[i] < 2: #if this one is in range
                        # if (-pi/4 < self.els[i] <= pi/4) and not (-3*pi/4 < self.azs[i] <= pi/4):
                        if (-pi/4 < self.els[i] <= pi/4):
                            y = int((tan(self.els[i]) + 1)/2*self.ht)
                            if self.vp==3 and -pi/4 < self.azs[i] <= pi/4:
                                x = int((tan(self.azs[i]) + 1)/2*self.wd)
                                if mask_ssb[y,x]<.1:
                                    try:
                                        data[y-1:y+1,x-1:x+1,:] = 255
                                    except:
                                        data[y,x,:] = 255
                            elif self.vp==0 and -3*pi/4 < self.azs[i] <= -pi/4:
                                x = int((tan(self.azs[i]-pi/2) + 1)/2*self.wd)
                                if mask_ssb[y,x]<.1:
                                    try:
                                        data[y-1:y+1,x-1:x+1,:] = 255
                                    except:
                                        data[y,x,:] = 255
                            elif self.vp==1 and (3*pi/4 < self.azs[i] <= pi or -pi < self.azs[i] <= -3*pi/4):
                                x = int((tan(self.azs[i]-pi) + 1)/2*self.wd)
                                if mask_ssb[y,x]<.1:
                                    try:
                                        data[y-1:y+1,x-1:x+1,:] = 255
                                    except:
                                        data[y,x,:] = 255
                        
	    # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )


    def add_radial(self, sf=1., tf=1., c=1., phi_i=0, sd=None, sdb=None, maxframes=500):
        '''Add a circular radiating grating'''
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        # spatial
        # phi_ss = 2*pi*sf*cos(self.atans + (o-pi/2))*self.center_dists
        phi_ss = 2*pi*sf*self.center_dists

        # temporal
        if hasattr(tf, '__iter__'): tf_array = array(tf) #is it changing?
        else: tf_array = repeat(tf, min(abs(self.rate/tf), maxframes)) #or constant?

        nframes = len(tf_array)
        phi_ts = cumsum(-2*pi*tf_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t = 0 #first frame?
                else: prev_phi_t = phi_ts[f-1]
                phi_ts_b, phi_ts_r, phi_ts_g = linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else: #or grays
                phi_ts_b, phi_ts_r, phi_ts_g = repeat(phi_ts[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_b + phi_i))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_r + phi_i))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*127*(1 + mask_ss*c*sin(phi_ss + phi_ts_g + phi_i))
            data[:,:,1] = lum[:,:]
	    # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add_plaid(self, sf1=.1, tf1=1, c1=1, o1=0, phi_i1=0.,
                  sf2=.1, tf2=1, c2=1, o2=0, phi_i2=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = zeros((self.ht,self.wd,3), dtype='byte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max() #normalize
        else: mask_ssb = ones([self.ht, self.wd])

        # spatial
        phi_ss1 = 2*pi*sf1*cos(self.atans + (o1-pi/2))*self.center_dists
        phi_ss2 = 2*pi*sf2*cos(self.atans + (o2-pi/2))*self.center_dists
                    
        # temporal
        if hasattr(tf1, '__iter__'): lentf1=len(tf1)
        else: lentf1 = self.rate/tf1
        if hasattr(tf2, '__iter__'): lentf2=len(tf2)
        else: lentf2 = self.rate/tf2
        if hasattr(tf1, '__iter__'): tf1_array = array(tf1) #is it changing?
        else: tf1_array = repeat(tf1, min(max(lentf1, lentf2), maxframes)) #or constant?
        if hasattr(tf2, '__iter__'): tf2_array = array(tf2) #is it changing?
        else: tf2_array = repeat(tf2, min(max(lentf1, lentf2), maxframes)) #or constant?

        nframes = max(len(tf1_array), len(tf2_array))
        phi_t1s = cumsum(-2*pi*tf1_array/float(self.rate))
        phi_t2s = cumsum(-2*pi*tf2_array/float(self.rate))

        for f in arange(nframes):
            if self.fast: #different frames in each color (projected without color wheel)
                if f==0: prev_phi_t1 = prev_phi_t2 = 0 #first frame?
                else:
                    prev_phi_t1 = phi_t1s[f-1]
                    prev_phi_t2 = phi_t2s[f-1]
                phi_t1s_b, phi_t1s_r, phi_t1s_g = linspace(prev_phi_t1, phi_t1s[f], 4)[1:]
                phi_t2s_b, phi_t2s_r, phi_t2s_g = linspace(prev_phi_t2, phi_t2s[f], 4)[1:]
            else: #or grays
                phi_t1s_b, phi_t1s_r, phi_t1s_g = repeat(phi_t1s[f],3)
                phi_t2s_b, phi_t2s_r, phi_t2s_g = repeat(phi_t2s[f],3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_b + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_b + phi_i2)))
            data[:,:,2] = lum[:,:]
            # red channel
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_r + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_r + phi_i2)))
            data[:,:,0] = lum[:,:]
            # green channel
            lum = mask_ssb*63*(2 + mask_ss*(c1*sin(phi_ss1 + phi_t1s_g + phi_i1) +\
                                   c2*sin(phi_ss2 + phi_t2s_g + phi_i2)))
            data[:,:,1] = lum[:,:]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def choose_grating(self, num):
        '''Assign one of the already generated gratings as the current
        display grating'''
        self.image = self.gratings[num]





class Bar_array(Sprite):
    '''Moving array of bars in a single window.  Fast means put different
    values in the blue, red and green sequential channels.
    '''
    def __init__(self, window, vp=0, rate=120., add=False, fast=False):
        Sprite.__init__(self, window, vp=0)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.ht, self.wd))
        # self.sr_indices = self.indices.copy()
        # self.sr_indices[0] *= pi/2./self.ht
        # self.sr_indices[1] *= pi/2./self.wd
        h, v = meshgrid(linspace(-pi/4, pi/4, self.wd), linspace(-pi/4, pi/4, self.ht))
        self.center_dists = sqrt(h**2 + v**2)
        self.atans = arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.animations = []

        if add: self.add()

    def in_bar(self, testpts, x0, y0, wd, ht, ang):
        # points for the lines, crossing the middle of the bar, at zero, no rot
        zpts = array([[-wd, wd,   0,  0],
                      [  0,  0, -ht, ht],
                      [  1,  1,   1, 1.]])
        # rotate and translate
        cosang, sinang = cos(ang), sin(ang)
        tmat = array([[ cosang, sinang, x0],
                      [-sinang, cosang, y0],
                      [0,       0,      1.]])
        pts = dot(tmat, zpts)
        # assign vars
        x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = pts[:2].flatten()
        xc, yc = testpts
        # distances to each line, a and b are line endpoints, c is the pt
        dln1 = abs((x1b - x1a)*(y1a - yc) - (x1a - xc)*(y1b - y1a)) / sqrt((x1b - x1a)**2 + (y1b - y1a)**2)
        dln2 = abs((x2b - x2a)*(y2a - yc) - (x2a - xc)*(y2b - y2a)) / sqrt((x2b - x2a)**2 + (y2b - y2a)**2)
        # a test point that is less than width to the vertical line
        # and less than height to the horizontal line (before rotation)
        # is in the bar
        return logical_and(dln1<wd, dln2<ht)
            
    def add_bars(self, wd, ht, dist, ori=0, vel=[0,.1], color=255,
                 staggered=False, sd=None, num_frames=None):
        frames = []
        pnt_wd = wd/(pi/2)*self.wd
        pnt_ht = ht/(pi/2)*self.ht

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([self.wd, self.ht])

        # temporal
        if hasattr(vel[0], '__iter__'):  #is it changing?
            vel_array = array([vel[0,:], vel[1,:]])
            if num_frames==None: num_frames = len(vel[0])
        else:                             #or constant?
            vel_array = array([repeat(vel[0], num_frames),repeat(vel[1], num_frames)])

        dx, dy = 0., 0.
        for f in arange(num_frames):
            data = zeros((self.ht,self.wd,3), dtype='ubyte')

            if self.fast: #different frames in each color (projected without color wheel)
                for cframe in [2,0,1]:
                    dx += vel_array[0,f]/3.
                    dy += vel_array[1,f]/3.
                    for xa in arange(-pi/4 + mod(dx, dist*2), pi/4, dist*2): # place each bar
                        pnt_xa = (xa + pi/4)/(pi/2)*self.wd
                        for ya in arange(-pi/4 + mod(dy, dist), pi/4, dist):
                            pnt_ya = (ya + pi/4)/(pi/2)*self.ht
                            data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori),cframe] = color
                    for xa in arange(-pi/4 + mod(dx, dist*2) + dist, pi/4, dist*2): # place each bar
                        pnt_xa = (xa + pi/4)/(pi/2)*self.wd
                        if staggered: offset = dist/2
                        else: offset = 0
                        for ya in arange(-pi/4 + mod(dy, dist) + offset, pi/4, dist):
                            pnt_ya = (ya + pi/4)/(pi/2)*self.ht
                            data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori),cframe] = color
                        


            else: #or grays
                dx += vel_array[0,f]
                dy += vel_array[1,f]
                for xa in arange(-pi/4 + mod(dx, dist), pi/4, dist): # place each bar
                    for ya in arange(-pi/4 + mod(dy, dist), pi/4, dist):
                        # data[:,:,:] = self.in_bar(self.sr_indices, xa, ya, wd, ht, ori)[:,:,newaxis]*color*mask_ss[:,:,newaxis]
                        data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori),:] = color
                        # data *= mask_ss*color
            lum = data[:,:,:] * mask_ss[:,:,newaxis]
            data[:,:,:] = lum[:,:,:]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def choose_animation(self, num):
        '''Assign one of the already generated bar stims as the current
        display animation.'''
        self.image = self.animations[num]


class kinetogram_class(Sprite):
    '''Moving dots with some degree of coherence.  Fast means put different
    values in the blue, red and green sequential channels.
    '''
    def __init__(self, window, vp=0, rate=120., add=False, fast=False):
        Sprite.__init__(self, window, vp)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        # self.sr_indices = self.indices.copy()
        # self.sr_indices[0] *= pi/2./self.wd
        # self.sr_indices[1] *= pi/2./self.ht
        h, v = meshgrid(linspace(-pi/4, pi/4, self.wd), linspace(-pi/4, pi/4, self.ht))
        self.center_dists = sqrt(h**2 + v**2)
        self.atans = arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.animations = []

        if add: self.add()

    def add_kinetogram(self, radius, x, y, density=10, coherence=.5, sd=None,
                       duration=3, dotsize=2, velocity=[1.,0.],
                       dot_color=1, bg_color=0, num_frames=100):
        velocity = array(velocity)
        speed = norm(velocity)
        wd, ht = self.wd, self.ht
        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max() #normalize
        else: mask_ss = ones([wd, ht])

        # density dots/sr * 4*pi sr/sphere * 1 sphere/6 surface = density*2*pi/3 dots/surface
        numtot = 2*pi*density/3.
        numcoh, numran = int(round(coherence*numtot)), int(round((1 - coherence)*numtot))
        cohpts, ci = random.rand(numcoh, 2)*array([wd, ht]), random.randint(0, duration, (numcoh))
        ranpts, ri = random.rand(numran, 2)*array([wd, ht]), random.randint(0, duration, (numran))
        ranspd, ranang = random.normal(speed, speed/10., (numran)), random.uniform(0,2*pi, (numran))
        ranvel = ranspd.repeat(2).reshape((numran,2))
        ranvel[:,0]*= cos(ranang)
        ranvel[:,1]*= sin(ranang)

        lum = zeros((wd, ht, num_frames*3), dtype='ubyte') + bg_color
        for i in arange(num_frames*3):
            # move all the coherently moving pts by a given velocity
            cohpts += velocity
            # a fraction have expired, and are moved to a random location
            ci = mod(ci + 1, duration)
            cohpts[ci==0] = random.rand(len(cohpts[ci==0]), 2)*array([wd, ht])
            # wrap points over the edges
            mod(cohpts, [wd, ht], cohpts)
            # fill in the points of the frame
            lum[array(cohpts[:,0], dtype='ubyte'),array(cohpts[:,1], dtype='ubyte'), i] = dot_color

            ranpts += ranvel
            ri = mod(ri + 1, duration)
            ranpts[ri==0] = random.rand(len(ranpts[ri==0]), 2)*array([wd, ht])
            ranspd, ranang = random.normal(speed, speed/10., (len(ri)-count_nonzero(ri))), random.uniform(0,2*pi, (len(ri)-count_nonzero(ri)))
            ranvel[ri==0,0] = ranspd*cos(ranang)
            ranvel[ri==0,1] = ranspd*sin(ranang)
            mod(ranpts, [wd, ht], ranpts)
            # fill in the points of the frame
            lum[array(ranpts[:,0], dtype='ubyte'),array(ranpts[:,1], dtype='ubyte'), i] = dot_color

        # make a list of images
        if self.fast:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:,:,(f+2,f,f+1)].transpose([1,0,2]).tostring()) for f in arange(0, num_frames*3, 3)]
        else:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:,:,(f,f,f)].transpose([1,0,2]).tostring()) for f in arange(0, num_frames*3, 3)]
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add_kinetogram2(self, radius, x, y, density=10, coherence=.5, sd=None,
                        duration=3, dotsize=2, velocity=[1.,0.],
                        dot_color=1, bg_color=0, num_frames=100):
        velocity = array(velocity)
        speed = norm(velocity)
        wd, ht = self.wd, self.ht

        # gaussian mask
        if sd: mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
        else: mask_ss = ones([wd, ht])
        mask_ss /= mask_ss.max() #normalize

        # from the density argument, dots/sr, get dots to display on surface, dots/surface 
        # density dots/sr * 4*pi sr/sphere * 1 sphere/6 surface = density*2*pi/3 dots/surface
        numtot = int(2*pi*density/3.)
        numcoh = int(round(coherence*numtot))
        numran = numtot - numcoh

        # make the points, and put them in random states of duration
        pts = random.rand(numtot, 2)*array([wd, ht])
        durs = random.randint(0, duration, (numtot))
        spds = random.normal(speed, speed/10., (numtot))
        lum = zeros((wd, ht, num_frames*3), dtype='ubyte') + bg_color
        
        cohpts, ci = random.rand(numcoh, 2)*array([wd, ht]), 
        ranpts, ri = random.rand(numran, 2)*array([wd, ht]), random.randint(0, duration, (numran))
        ranspd, ranang = random.normal(speed, speed/10., (numran)), random.uniform(0,2*pi, (numran))
        ranvel = ranspd.repeat(2).reshape((numran,2))
        ranvel[:,0]*= cos(ranang)
        ranvel[:,1]*= sin(ranang)

        for i in arange(num_frames*3):
            # move all the coherently moving pts by a given velocity
            cohpts += velocity
            # a fraction have expired, and are moved to a random location
            ci = mod(ci + 1, duration)
            cohpts[ci==0] = random.rand(len(cohpts[ci==0]), 2)*array([wd, ht])
            # wrap points over the edges
            mod(cohpts, [wd, ht], cohpts)
            # fill in the points of the frame
            lum[array(cohpts[:,0], dtype='ubyte'),array(cohpts[:,1], dtype='ubyte'), i] = dot_color

            ranpts += ranvel
            ri = mod(ri + 1, duration)
            ranpts[ri==0] = random.rand(len(ranpts[ri==0]), 2)*array([wd, ht])
            ranspd, ranang = random.normal(speed, speed/10., (len(ri)-count_nonzero(ri))), random.uniform(0,2*pi, (len(ri)-count_nonzero(ri)))
            ranvel[ri==0,0]*= cos(ranang)
            ranvel[ri==0,1]*= sin(ranang)
            mod(ranpts, [wd, ht], ranpts)
            # fill in the points of the frame
            lum[array(ranpts[:,0], dtype='ubyte'),array(ranpts[:,1], dtype='ubyte'), i] = dot_color

        # make a list of images
        if self.fast:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:,:,(f+2,f,f+1)].tostring()) for f in arange(0, num_frames*3, 3)]
        else:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:,:,(f,f,f)].tostring()) for f in arange(0, num_frames*3, 3)]
            
        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def choose_animation(self, num):
        '''Assign one of the already generated bar stims as the current
        display animation.'''
        self.image = self.animations[num]


        
if __name__=='__main__':
    import holocube.hc5 as hc
    project = 0
    bg = [1., 1., 1., 1.]
    bg = [0., 0., 0., 1.]
    near, far = .01, 1.
    randomize = False
    hc.window.start(project=project, bg_color=bg, near=near, far=far)
    w = hc.window

    num_frames = 120

    sp = kinetogram_class(hc.window, fast=False)

    # a series of sfs
    cohs = linspace(0,1,6)
    vels = array([[0,.05], [0,-.05]])
    density = 1
    duration = 50

    sp.add_kinetogram(0,0,0, density=density, duration=duration,
                      coherence=cohs[5], velocity=vels[0],
                      sd=.35, num_frames=num_frames)

