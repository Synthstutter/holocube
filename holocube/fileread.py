# serial control of arduino

from numpy import *
from pyglet import *
import os.path

class Filereader():
    '''Read and write from a data file (like fictrac).'''

    def __init__(self):
        self.curr_xpos = 0.
        self.curr_zpos = 0.
        self.curr_heading = 0.

    def start(self, file_name='fictrac_cl.data'):
        if file_name=='dummy':
            self.file = None
            self.read = self.dummy_read_file
        else:
            self.file = open(os.path.expanduser(file_name), 'r')
            self.read = self.read_file
        
    def dummy_read_file(self, channel=0):
        self.curr_xpos -= 0.01*cos(self.curr_heading)
        self.curr_zpos -= 0.01*sin(self.curr_heading)
        self.curr_heading += random.randn()*.02
    
    def read_file(self, channel=0):
        self.file.seek(0)
        data = [float(s.strip(' ,')) for s in self.file.readlines(1)[0].split()]
        self.curr_xpos = data[1]
        self.curr_ypos = data[2]
        self.curr_heading = data[3]

    def xpos(self):
        self.read()
        return self.curr_xpos

    def zpos(self):
        self.read()
        return self.curr_zpos

    def heading(self):
        self.read()
        return self.curr_heading

