# run a series of exps with pyg and ard

from numpy import *
import pyglet
from pyglet.gl import *
from pyglet.window import key
import inspect
from os.path import isdir
from os import listdir

def spikelist(num, llen, dist=10):
    sl = zeros((llen))
    sl[arange(0,num)*dist] = 1
    return sl


class Scheduler():
    '''Schedules a set of experiments with rest presentations in
    between.'''

    def __init__(self):
        self.testing = False
        self.resting = False
        self.exps = []
        self.rest = [] #default rest procedure
        self.tests = [[]] #the first inner parens for the rest procedure
        self.order = []
        self.curr_num = 0 #iterative number of tests done so far
        self.curr_test = []
        self.curr_test_len = 0
        self.num_tests = 0
        self.ind = 0
        self.names = []
        self.aborted = False
    
    def start(self, window, randomize=True, freq=120, expdir='experiments/'):

        self.freq = freq

        self.window = window
        self.window.add_key_action(key._0, self.abort, True)

        self.randomize = randomize #do we randomize experiment order

    def save_exp(self, rest_time=1):
        name = inspect.getouterframes(inspect.currentframe())[1][1] #get the file path that called save_exp
        name = name.rsplit('/')[-1].rsplit('.')[0]                  #get its name without '.py'
        if name == 'rest': #the stimulus between experiments
            self.rest.append(self.tests)
        else: #regular experiment
            self.names.append(name)
            self.exps.append(self.tests)
            self.tests = [[]] #the first inner parens for the rest procedure
            ind = len(self.exps)-1
            curr_key = [key._1, key._2, key._3, key._4, key._5,
                        key._6, key._7, key._8, key._9][ind]

            if ind<=8:
                self.window.add_key_action(curr_key, self.print_exp, ind)
                self.window.add_key_action(curr_key, self.load_exp, ind)
                self.window.add_key_action(curr_key, self.begin, True)
            if ind==0: print '\nScheduler key assignments:'
            print key.symbol_string(curr_key) + ' - ' + name
        

    def load_exp(self, num):
        if num >= len(self.exps):
            print 'Only %d expirments loaded so far.'%len(self.exps)
        else:
            self.tests = self.exps[num]
            self.num_tests = len(self.tests)

    def add_test(self, *func_args):
        self.tests.append([])
        for i in range(0,len(func_args),3):
            func, args, seq = func_args[i], func_args[i+1], func_args[i+2]
            self.tests[-1].append([func, args, seq])
        self.num_tests = len(self.tests-1)

    def add_rest(self, *func_args):
        for i in range(0,len(func_args),3):
            func, args, seq = func_args[i], func_args[i+1], func_args[i+2]
            self.tests[0].append([func, args, seq])

    def test_to_rest(self, dt=0.0):
        if self.testing: pyglet.clock.unschedule(self.test)
        if not self.rest_bar.visible: self.rest_bar.add()
        
        self.resting = True
        if self.curr_num < self.num_tests: #if tests are left to perform
            self.curr_num += 1             #schedule the next one
            test_num = self.order[self.curr_num-1]
            pyglet.clock.schedule_once(self.rest_to_test, self.rest_time, test_num)
        else: print 'done  ##########################################################'
        pyglet.clock.schedule_interval(self.rest, 1./self.freq) #start the rest


    def rest(self, dt=0.0):
        # execute the interval functions
        for rest in self.curr_rest_interval:
            rest[0](rest[1][self.ind])
        for rest in self.curr_rest_funcs:
            rest[0](rest[1]())
        self.ind += 1
        if self.ind == self.curr_rest_len: #are we done?
            # execute the ending functions
            for rest in self.curr_rest_end:
                rest[0](rest[1])
            # back to resting
            self.rest_to_rest()


    def rest_to_test(self, dt=0.0, test_num=0):
        # get rid of the resting schedule interval and the visible bar
        if self.resting: pyglet.clock.unschedule(self.rest)
        if self.rest_bar.visible: self.rest_bar.remove()

        # start the test
        print test_num
        self.testing = True

        # divide up the functions to execute
        curr_test = self.tests[test_num]
        self.curr_test_start =    [test for test in curr_test if test[2]== 0 or test[2]=='start']
        self.curr_test_end   =    [test for test in curr_test if test[2]==-1 or test[2]=='end']
        self.curr_test_interval = [test for test in curr_test if (test[2]== 1 or test[2]=='middle') and hasattr(test[1], '__iter__')]
        self.curr_test_funcs    = [test for test in curr_test if (test[2]== 1 or test[2]=='middle') and hasattr(test[1], '__call__')]
        self.curr_test_len = len(self.curr_test_interval[0][1])

        # execute the starting functions
        for test in self.curr_test_start:
            test[0](test[1])
        
        self.ind = 0
        pyglet.clock.schedule_interval(self.test, 1./self.freq)


    def test(self, dt=0.0):
        # execute the interval functions
        for test in self.curr_test_interval:
            test[0](test[1][self.ind])
        for test in self.curr_test_funcs:
            test[0](test[1]())
        self.ind += 1
        if self.ind == self.curr_test_len: #are we done?
            # execute the ending functions
            for test in self.curr_test_end:
                test[0](test[1])
            # back to resting
            self.test_to_rest()
    

    def begin(self, state=True):
        '''Start the experiment, by setting the order of the trial
        and setting the current trial to 0, and  executing test_to_rest()'''
        if state:
            if self.randomize:
                self.order = random.permutation(len(self.tests))
            else:
                self.order = arange(len(self.tests) - 1) #-1 for the rest
            self.curr_num = 1 #put rests in position 0
            print 'begin ##########################################################'
            self.test_to_rest()
        else:
            self.abort()
 
    def abort(self, state=True):
        print 'Aborting...'
        self.curr_num = self.num_tests
        self.aborted = True

    def print_exp(self, ind=0):
        print '\nExperiment ' + str(ind + 1) + ' - ' + self.names[ind]

