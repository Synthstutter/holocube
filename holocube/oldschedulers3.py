# run a series of exps with pyg and ard

import pyglet
from pyglet.gl import * #this will overwrite 'resize' if it comes first
from pyglet.window import key
import inspect
from os.path import isdir
from os import listdir
from numpy import *

def spikelist(num, llen, dist=10):
    sl = zeros((llen))
    sl[arange(0,num)*dist] = 1
    return sl

def expnum_sig(num, llen, dist=10, mult=255):
    sl = zeros((llen,3), dtype='int')+mult/3
    sl[(arange(0,num)+1)*dist,0] = mult
    sl[-1] = [0,0,0]
    return sl

def expnum(num, llen, dist=10, col_1=255, col_2=127):
    sl = zeros((llen,3), dtype='int')
    sl[(arange(0,num)+2)*dist,0] = col_2
    sl[0] = [col_1, 0, 0]
    sl[-2] = [col_1, 0, 0]
    sl[-1] = [0,0,0]
    return sl
        
def expflash(num, llen, dist=10, col1=255, col2=127):
    sl = zeros((llen), dtype='int')
    sl[(arange(0,num)+2)*dist] = col_2
    sl[0] = col_1
    sl[-2] = col_1
    sl[-1] = 0
    return sl

class Experiment():
    '''Hold all the parameters of an experiment.'''
    def __init__(self):
        self.name = ''
        self.tests = [] #each test is a list of start, during, and end funct and their parameters
        self.num_tests = 0
        self.rest = [] #rest is like a single test to be performed in between each test
        self.experiment = [] #this is just start and end funcs to execute before and after the whole experiment

    def __repr__(self):
        return '%s (experiment class)'%(self.name)

    def add_test(self, numframes, *func_args):
        '''Add a test (list of funcs and args), to the list of tests.'''
        new_test = [numframes]
        for i in range(len(func_args)):
            func, args, seq = func_args[i]
            new_test.append([func, atleast_1d(args), seq])
        self.tests.append(new_test)
        self.num_tests = len(self.tests)

    def add_rest(self, numframes, *func_args):
        '''Put the special test called rest into its own slot.'''
        self.rest.append(numframes)
        for i in range(len(func_args)):
            func, args, seq = func_args[i]
            self.rest.append([func, atleast_1d(args), seq])

    def add_exp_commands(self, *func_args):
        '''Add a list of funcs and args performed at the start and end of the whole
        experiment.'''
        new_exp_commands = [0]
        for i in range(len(func_args)):
            func, args, seq = func_args[i]
            new_exp_commands.append([func, atleast_1d(args), seq])
        self.experiment.append(new_exp_commands)
            
    def test_commands(self, test_num=0, command_type='start'):
        '''Return the commands of the test that are of the given type, start, func
        (during), interval (during), or end.'''
        if test_num == -1 or test_num == 'rest':
            curr_test = self.rest
        else:
            curr_test = self.tests[test_num]
            
        numframes = curr_test[0]

        if command_type=='start':
            commands = [command for command in curr_test[1:] if command[2]== 0 or command[2]=='start']
        elif command_type=='func':
            commands = [command for command in curr_test[1:] if (command[2]== 1 or command[2]=='during') and hasattr(command[1][0], '__call__')]
        elif command_type=='interval':
            commands = [command for command in curr_test[1:] if (command[2]== 1 or command[2]=='during') and not hasattr(command[1][0], '__call__')]
        elif command_type=='end':
            commands = [command for command in curr_test[1:] if command[2]== -1 or command[2]=='end']
        else:
            print 'command not found.'
            return -1
        return commands

    def test_len(self, test_num=0):
        '''return the number of frames in a given test'''
        if test_num == -1 or test_num == 'rest':
            curr_test = self.rest
        else:
            curr_test = self.tests[test_num]
        numframes = curr_test[0]
        return numframes

    # eventually, the experiment instance should execute its own functions, not the scheduler
    def execute(test_num, command_type):
        pass

class Scheduler():
    '''schedules a set of experiments with bar tracking in between.'''

    def __init__(self):
        self.testing = False
        self.resting = False
        self.aborted = False
        self.experiments = [] #holds the instances of class Experiment
        self.curr_experiment = None
        self.curr_experiment_ind = -1
        self.tests = []
        self.rests = []
        self.order = []
        self.curr_test_ind = 0 #iterative number of tests done so far
        self.curr_rest_ind = 0
        self.curr_test_len = 0
        self.num_rests = 0
        self.frame_ind = 0     #the frame number in a rest or test
        self.names = []
    
    def start(self, window, rest_time=1.0, randomize=True, freq=120):

        '''Start the scheduler (not an experiment)'''

        self.freq = freq
        self.rest_time = rest_time
        self.rest_frames = rest_time*freq

        self.window = window
        self.window.add_key_action(key._0, self.abort_experiment, True)
        self.window.add_key_action(key.U, self.sched_print, True)
        self.window.add_key_action(key.QUOTELEFT, self.change_rest, True)

        self.randomize = randomize #do we randomize experiment order


    def save_exp(self, rest_time=1):
        name = inspect.getouterframes(inspect.currentframe())[1][1] #get the file path that called save_exp
        name = name.rsplit('/')[-1].rsplit('.')[0]                  #get its name without '.py'
        self.names.append(name)

        # experiment class
        self.curr_experiment.name = name

        # there is at least one test, so
        # this is not a rest file, so assign a key
        if self.curr_experiment.tests != []: 
            self.experiments.append(self.curr_experiment)
            ind = len(self.experiments)-1
            curr_key = [key._1, key._2, key._3, key._4, key._5,
                        key._6, key._7, key._8, key._9][ind]
            if ind<=8:
                self.window.add_key_action(curr_key, self.begin_experiment, ind)
            if ind==0: print '\nScheduler key assignments:'
            print key.symbol_string(curr_key) + ' - ' + name

        # since this is not a test file and there is at least one
        # rest, add it to rests and make it the default rest
        elif self.curr_experiment.rest != []:
            self.rests.append(self.curr_experiment)
            print 'rest - ' + name

        self.curr_experiment = None #clear it for the next save
        

    def add_test(self, *func_args):
        self.tests.append([])
        longest = 0 #when numframes isn't given, figure it out and append it to the start
        for i in range(0,len(func_args),3):
            func, args, seq = func_args[i], atleast_1d(func_args[i+1]), func_args[i+2]
            if hasattr(func_args[i+1], '__iter__'):
                longest = max(longest, len(func_args[i+1]))
        numframes = longest
        # now add the test to the experiment class
        if self.curr_experiment == None:
            self.curr_experiment = Experiment()
        func_args = array(func_args).reshape(-1,3)
        self.curr_experiment.add_test(numframes, *func_args) 

    def new_test(self, numframes, *func_args):
        if self.curr_experiment == None:
            self.curr_experiment = Experiment()
        self.curr_experiment.add_test(numframes, *func_args)

    def new_rest(self, numframes, *func_args):
        if self.curr_experiment == None:
            self.curr_experiment = Experiment()
        self.curr_experiment.add_rest(numframes, *func_args)

    def test_to_rest(self, dt=0.0):

        '''Unschedule the test procedure, fill in the rest funcs and
        schedule the rest procedure.'''

        # get rid of the testing schedule interval
        if self.testing:
            pyglet.clock.unschedule(self.show_test_frame)

        # start the rest
        self.resting = True
        self.testing = False

        #if tests are left to perform:
        if self.curr_test_ind + 1 < self.experiments[self.curr_experiment_ind].num_tests: 
            self.curr_test_ind += 1             #schedule the next one

            # and the current exp has no rest procedure 
            if self.experiments[self.curr_experiment_ind].rest == []:
                # use the scheduler rest procedure
                curr_exp = self.rests[self.curr_rest_ind]
                # and the default rest time
                self.curr_rest_len = self.rest_frames

            # if the curr exp has a rest procedure
            else:
                # use it and its specified rest time
                curr_exp = self.experiments[self.curr_experiment_ind]
                self.curr_rest_len =curr_exp.test_len('rest')

        # otherwise tests are finished
        else:
            # in case we got here by aborting
            self.aborted = False
            #use the between-exp rest pattern
            curr_exp = self.rests[self.curr_rest_ind] 
            # use the scheduler rest time, probably inf
            self.curr_rest_len =curr_exp.test_len('rest')
            print 'end ___________________________________________________________'

        # grab the types of functions from Experiment class
        self.curr_rest_start = curr_exp.test_commands('rest', 'start')
        self.curr_rest_funcs = curr_exp.test_commands('rest', 'func')
        self.curr_rest_interval = curr_exp.test_commands('rest', 'interval')
        self.curr_rest_end = curr_exp.test_commands('rest', 'end')
        # execute the starting functions
        for command, arg, kind in self.curr_rest_start:
            command(arg)
        # schedule the show_rest_frame procedure to run the during functions
        self.frame_ind = 0
        pyglet.clock.schedule_interval(self.show_rest_frame, 1./self.freq)


    def show_rest_frame(self, dt=0.0):
        '''Called continuously during a rest (between tests or between
        experiments)'''
        # execute the interval functions
        for command, arg, kind in self.curr_rest_interval:
            command(take(arg, [self.frame_ind],0,mode='wrap')[0])
        for command, arg, kind in self.curr_rest_funcs:
            command(take(arg,[self.frame_ind],0,mode='wrap')[0]())
        self.frame_ind += 1

        # do i need this?
        if self.frame_ind >= self.curr_rest_len or self.aborted: #are we done?
            # execute the ending functions
            for command, arg, kind in self.curr_rest_end:
                command(arg)
            # back to testing, the test num to schedule is not the
            # curr_test_ind itself, but whichever test this indicates
            # in the possibly randomized order
            self.rest_to_test(test_num=self.order[self.curr_test_ind])
 

    def rest_to_test(self, dt=0.0, test_num=0):
        
        '''Unschedule the rest procedure, run the rest_end functions,
        fill in the curr_test funcs and schedule the test procedure.'''

        # get rid of the resting schedule interval and the visible bar
        if self.resting:
            pyglet.clock.unschedule(self.show_rest_frame)
            
        # start the test
        self.testing = True
        self.aborted = False

        curr_exp =  self.experiments[self.curr_experiment_ind]

        print test_num, '   ', self.curr_test_ind + 1, ' of ', curr_exp.num_tests

        # Experiment class
        self.curr_test_len = curr_exp.test_len(test_num)
        self.curr_test_start = curr_exp.test_commands(test_num, 'start')
        self.curr_test_funcs = curr_exp.test_commands(test_num, 'func')
        self.curr_test_interval = curr_exp.test_commands(test_num, 'interval')
        self.curr_test_end = curr_exp.test_commands(test_num, 'end')

        # execute the starting functions
        for test in self.curr_test_start:
            test[0](test[1])

        # schedule the test procedure to run the during functions
        self.frame_ind = 0
        pyglet.clock.schedule_interval(self.show_test_frame, 1./self.freq)


    def show_test_frame(self, dt=0.0):
        '''Called continuously during a test.'''
        # execute the interval functions
        for command, arg, kind in self.curr_test_interval:
            command(take(arg, [self.frame_ind],0,mode='wrap')[0])
        for command, arg, kind in self.curr_test_funcs:
            command(take(arg,[self.frame_ind],0,mode='wrap')[0]())
        self.frame_ind += 1
        if self.frame_ind >= self.curr_test_len or self.aborted: #are we done?
            # execute the ending functions
            for command, arg, kind in self.curr_test_end:
                command(arg)
            # back to resting
            self.test_to_rest()
    

    def begin_experiment(self, ind):

        '''Start an experiment'''
        
        print '\nExperiment ' + str(ind + 1) + ' - ' + self.experiments[ind].name

        self.curr_experiment_ind = ind
        num_tests = self.experiments[ind].num_tests

        if self.randomize: #do we randomize the exp order?
            self.order = random.permutation(num_tests)
        else:              #or present them in ascending order?
            self.order = arange(num_tests)

        print 'begin order', self.order

        # start 
        self.curr_test_ind = -1
        print 'begin ##########################################################'
        self.test_to_rest()

 
    def abort_experiment(self, state=True):
        # set the curr_test_ind to the last test
        if self.resting:
            print 'Aborting...'
            self.curr_test_ind = self.experiments[self.curr_experiment_ind].num_tests-1
        if self.testing:
            print 'Aborting...'
            self.curr_test_ind = self.experiments[self.curr_experiment_ind].num_tests
        self.aborted=True

    # def abort_experiment(self, state=True):
    #     print 'Aborting...'
    #     self.aborted=True
    #     # set the curr_test_ind to the last test
    #     if self.resting:
    #         print 'here1'
    #         pyglet.clock.unschedule(self.show_rest_frame)
    #     elif self.testing:
    #         print 'here2'
    #         pyglet.clock.unschedule(self.show_test_frame)
            
    #     self.curr_test_ind = 0
    #     curr_exp = self.rests[self.curr_rest_ind] 
    #     # use the scheduler rest time, probably inf
    #     self.curr_rest_len =curr_exp.test_len('rest')
    #     # grab the types of functions from Experiment class
    #     self.curr_rest_start = curr_exp.test_commands('rest', 'start')
    #     self.curr_rest_funcs = curr_exp.test_commands('rest', 'func')
    #     self.curr_rest_interval = curr_exp.test_commands('rest', 'interval')
    #     self.curr_rest_end = curr_exp.test_commands('rest', 'end')
    #     # execute the starting functions
    #     for command, arg, kind in self.curr_rest_start:
    #         command(arg)
    #     # schedule the show_rest_frame procedure to run the during functions
    #     self.frame_ind = 0
    #     pyglet.clock.schedule_interval(self.show_rest_frame, 1./self.freq)
    #     self.resting = True
    #     self.testing = False

    def change_rest(self, state=True):
        '''If there are multiple rest stimuli, switch to the next one.'''
        self.curr_rest_ind = mod(self.curr_rest_ind + 1, len(self.rests))
        print self.rests[self.curr_rest_ind].name
        self.test_to_rest()

    def sched_print(self, state=True):
        print '1scheduler parameters:'
        print 'self.testing', self.testing
        print 'self.resting', self.resting
        print 'self.aborted', self.aborted
        print 'self.frame_ind', self.frame_ind
        print 'self.curr_experiment', self.curr_experiment
        print 'self.curr_experiment_ind', self.curr_experiment_ind

        print 'self.order', self.order
        print 'self.curr_test_ind', self.curr_test_ind

        print ' self.curr_test_len', self.curr_test_len
        print 'self.num_rests',self.num_rests 
        print 'self.names',self.names
        print 'self.rests',self.rests
        
        

