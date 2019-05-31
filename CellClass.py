from neuron import h
from neuron import gui

class CellClass(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.soma = h.Section(name='soma', cell=self)
        self.soma.L = 10
        self.soma.nseg = 1
        self.soma.diam = 30
        self.soma.insert('hh')
        self.soma.Ra = 100
        self.soma.cm = 10
        
        #settings for synapse
#         self.syn = h.ExpSyn(0.8,sec=self.soma)
#         self.syn.tau = 5

        # for excitatory inputs
        self.AMPA = h.ExpSyn(0.5,sec=self.soma)
        self.AMPA.tau = 5
        self.AMPA.e = 0 # this is above resting potential, which causes depolarization
        
        # for inhibitory inputs
        self.GABA = h.ExpSyn(0.5,sec=self.soma)
        self.GABA.tau = 10
        self.GABA.e = -80   # this is below resting potential, which causes hyperpolarization

        #STDP constant
        d=1
        p=1
        self.LR = 0.000015#0.00001 stdp?
#         self.LR = 0.000005#0.00001 hebb?
        self.syn_STDP = h.ExpSynSTDP(0.8,sec=self.soma)
        self.syn_STDP.tau = 5
        self.syn_STDP.d = d#LTD
        self.syn_STDP.p = p#LTP
        self.syn_STDP.verbose=0
        self.syn_STDP.LR = self.LR
        
    def setWeightRange(self, minWeight,maxWeight):
        self.syn_STDP.minWeight = minWeight
        self.syn_STDP.maxWeight = maxWeight
        
    def setLearningState(self,state):
        self.syn_STDP.learning = state
        
    def setLR(self,num):
        self.syn_STDP.LR = self.LR*num