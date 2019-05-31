import scipy
import matplotlib.pyplot as plt
#from CellClass import *
from CellClass import *
from StateClass import *
from random import *
import math
import pickle
import numpy


class Controller(object):
    def __init__(self,dim_stim,dim):
        self.variables = StateClass()
        
        self.variables.maxWeight = 0.005 #maximum weight synaptic connections can have
        self.variables.minWeight = 0  #minimum weight synaptic connections can have
        self.variables.dim = dim  #number of cells on one side in a layer (ie the number of cells are dim*dim)
        self.variables.learningState = 1  #0: no synaptic modifications / 1: synaptic modifications with STDP
        self.variables.tstop = 300 #duration for each trial ([ms])
        self.variables.maxWeightsMulti=1.5#1.5 #toL23
        self.variables.maxWeightsMulti2=3#2.5# #toL5
        self.resultFolderName = "results";
         
        #h.dt = 1  # time-step ([ms]) -- PROBLEM: this value cannot be changed?? (stays 0.025)
        self.variables.hz = 40 #maximum frequency of the input signal. PROBLEM: If I increase too much, post-synaptic cell never get activated.. 
        
        delay_rgb2channel = 1#+ uniform(-1,1)
        delay_ToL4 = 1
        delay_C2ToL23 = 1
        delay_L4ToL23 = 1
        delay_L23ToL5 = 1
        delay_lateral_pot1 = 1#0.1
        delay_lateral_pot2 = 2#0.2
        delay_lateral_pot3 = 3#0.3
        delay_lateral_dep = 4#0.4#0.6
        
        weight_rgb2channel = 0.005#0.01
        weight_ToL4 = self.variables.maxWeight
        weight_L4ToL23 = self.variables.maxWeight*self.variables.maxWeightsMulti
        weight_C2ToL23 = self.variables.maxWeight*self.variables.maxWeightsMulti
        weight_L23ToL5 = self.variables.maxWeight*self.variables.maxWeightsMulti2
        weight_lateral_pot1 = self.variables.maxWeight*0.9#*0.8*0.9
        weight_lateral_pot2 = self.variables.maxWeight*0.5#*0.5*0.9
        weight_lateral_pot3 = self.variables.maxWeight*0.3#*0.4*0.9
        weight_lateral_dep = self.variables.maxWeight*1.5#0.9#*0.8*0.9
        
        #to set range of the initial weights relative to (self.variables.maxWeight-self.variables.minWeight)
        initRandMinRange = 0 
        initRandMaxRange = 1
        self.variables.dim_stim = dim_stim

        self.r_stims = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.g_stims = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.b_stims = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.steadyActivation = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.variables.delay_r = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.variables.delay_g = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.variables.delay_b = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
#         self.variables.delay_steady = [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
#         self.variables.steady_state_activation = 0.2

        self.noiseLv = 0.3;
#         self.noiseSteady = 1;

        for y in range(self.variables.dim_stim):
            for x in range(self.variables.dim_stim):
                self.r_stims[y][x] = h.NetStim(0.5)
                self.r_stims[y][x].noise = self.noiseLv#0.1
                self.variables.delay_r[y][x] = random(); #starting timing
                self.g_stims[y][x] = h.NetStim(0.5)
                self.g_stims[y][x].noise = self.noiseLv#0.1
                self.variables.delay_g[y][x] = random();
                self.b_stims[y][x] = h.NetStim(0.5)
                self.b_stims[y][x].noise = self.noiseLv#0.1
                self.variables.delay_b[y][x] = random();
#                 self.steadyActivation[y][x] = h.NetStim(0.5);
#                 self.steadyActivation[y][x].noise = 0#0.1
#                 self.steadyActivation[y][x].noise = self.noiseSteady#0.1
#                 self.steadyActivation[y][x].number = 1e9
#                 self.variables.delay_steady[y][x] = random()*25;
#                 self.steadyActivation[y][x].start = self.variables.delay_steady[y][x];
#                 self.variables.delay_steady[y][x] = random();





        self.rds_r =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.rds_g =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.rds_b =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
#         self.rds_steady =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        
        self.variables.seed_r =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.variables.seed_g =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        self.variables.seed_b =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
#         self.variables.seed_steady =  [[0 for x in xrange(self.variables.dim_stim)] for x in xrange(self.variables.dim_stim)] 
        
        for y in range(self.variables.dim_stim):
            for x in range(self.variables.dim_stim):
                self.rds_r[y][x]= h.Random()
                self.rds_r[y][x].negexp(1) # must have this line! -- set random # generator using negexp(1) - avg interval in NetStim
                self.variables.seed_r[y][x] = random() # your random number seed (should differ for each NetStim)
                self.rds_r[y][x].MCellRan4(self.variables.seed_r[y][x],self.variables.seed_r[y][x]) # initialize here
                self.r_stims[y][x].noiseFromRandom(self.rds_r[y][x])  # use random # generator for this NetStim
                
                self.rds_g[y][x] = h.Random()
                self.rds_g[y][x].negexp(1) # must have this line! -- set random # generator using negexp(1) - avg interval in NetStim
                self.variables.seed_g[y][x] = random() # your random number seed (should differ for each NetStim)
                self.rds_g[y][x].MCellRan4(self.variables.seed_g[y][x],self.variables.seed_g[y][x]) # initialize here
                self.g_stims[y][x].noiseFromRandom(self.rds_g[y][x])  # use random # generator for this NetStim
                
                self.rds_b[y][x] = h.Random()
                self.rds_b[y][x].negexp(1) # must have this line! -- set random # generator using negexp(1) - avg interval in NetStim
                self.variables.seed_b[y][x] = random() # your random number seed (should differ for each NetStim)
                self.rds_b[y][x].MCellRan4(self.variables.seed_b[y][x],self.variables.seed_b[y][x]) # initialize here
                self.b_stims[y][x].noiseFromRandom(self.rds_b[y][x])  # use random # generator for this NetStim
#                 
#                 self.rds_steady[y][x] = h.Random()
#                 self.rds_steady[y][x].negexp(1) # must have this line! -- set random # generator using negexp(1) - avg interval in NetStim
#                 self.variables.seed_steady[y][x] = random() # your random number seed (should differ for each NetStim)
#                 self.rds_steady[y][x].MCellRan4(self.variables.seed_steady[y][x],self.variables.seed_steady[y][x]) # initialize here
#                 self.steadyActivation[y][x].noiseFromRandom(self.rds_steady[y][x])  # use random # generator for this NetStim
#                 
#     








         
        #init layer to store cells
        self.L_channels = []    #L layer
        self.C1_channels = []   #C1 layer
        self.C2_channels = []   #C2 layer
        self.cortexLayer_4 = []   #output layer
        self.cortexLayer_2_3 = []
        self.cortexLayer_5 = []
#         self.cortexLayer_5 = []
        
        #to record firing counts in the cortexLayer_4 when exposed to different stimuli 
        self.firingSums = []
        
        #init NetCon for C1, C2, L, cortexLayer_4
        self.NetCons = []   #to store Synaptic connections where weights are fixed (rgb to L, C1, C2; lateral connections in cortexLayer_4)
        self.NetCons_STDP_LtoL4 = []    #to store synaptic connections of L
        self.NetCons_STDP_C1toL4 = []   #to store synaptic connections of C1
        self.NetCons_STDP_C2toL23 = []   #to store synaptic connections of C2
        self.NetCons_STDP_L4toL23 = []
        self.NetCons_STDP_L23toL5 = []
          

        
#         
        #init topologically corresponding synaptic connections (rgb to L, C1, C2; L, C1, C2 to the cortexLayer_4)
        for y in range(dim):
            for x in range(dim):
                index = y*dim + x
                self.L_channels.append(CellClass(x,y))
                self.C1_channels.append(CellClass(x,y))
                self.C2_channels.append(CellClass(x,y))
                self.cortexLayer_4.append(CellClass(x,y))
                self.cortexLayer_2_3.append(CellClass(x,y))
                self.cortexLayer_5.append(CellClass(x,y))
                
                self.cortexLayer_4[index].setWeightRange(self.variables.minWeight,self.variables.maxWeight) #set min/maxWeights
                self.cortexLayer_2_3[index].setWeightRange(self.variables.minWeight,self.variables.maxWeight*self.variables.maxWeightsMulti) #set min/maxWeights
                self.cortexLayer_5[index].setWeightRange(self.variables.minWeight,self.variables.maxWeight*self.variables.maxWeightsMulti2) #set min/maxWeights
                
                y_stim = int(y*self.variables.dim_stim/self.variables.dim);
                x_stim = int(x*self.variables.dim_stim/self.variables.dim);

                #Constructing synaptic connections between rgb and L, C1, C2 based on color oppornent theory
                #weights are varied by multiplying by uniform(0,2)
                delay = delay_rgb2channel
                weightConst = weight_rgb2channel
                #L NetCon init (luminance - LGN-M)
                weightTmp = weightConst*uniform(0.5,2);
#                 weightTmp2 = weightConst*uniform(0.5,2)*0.5;#steadyState?
                self.NetCons.append(h.NetCon(self.r_stims[y_stim][x_stim],self.L_channels[index].AMPA,0,delay,weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.L_channels[index].AMPA,0,delay,weightTmp2))
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.g_stims[y_stim][x_stim],self.L_channels[index].AMPA,0,delay,weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.L_channels[index].AMPA,0,delay,weightTmp2))#source, dest, threshold, delay, weight
                
                
                #C1 NetCon init (red-green - LGN-P)
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.r_stims[y_stim][x_stim],self.C1_channels[index].AMPA,0,delay,weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.C1_channels[index].AMPA,0,delay,weightTmp2))#source, dest, threshold, delay, weight
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.g_stims[y_stim][x_stim],self.C1_channels[index].GABA,0,delay,1.5*weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.C1_channels[index].GABA,0,delay,weightTmp2))#source, dest, threshold, delay, weight
                
                #C2 NetCon init (blue-yellow - LGN k-path)
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.r_stims[y_stim][x_stim],self.C2_channels[index].AMPA,0,delay,1*weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.C2_channels[index].AMPA,0,delay,1*weightTmp2))#source, dest, threshold, delay, weight
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.g_stims[y_stim][x_stim],self.C2_channels[index].AMPA,0,delay,1*weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.C2_channels[index].AMPA,0,delay,1*weightTmp2))#source, dest, threshold, delay, weight
                weightTmp = weightConst*uniform(0.5,2);
                self.NetCons.append(h.NetCon(self.b_stims[y_stim][x_stim],self.C2_channels[index].GABA,0,delay,3*weightTmp))#source, dest, threshold, delay, weight
#                 self.NetCons.append(h.NetCon(self.steadyActivation[y_stim][x_stim],self.C2_channels[index].GABA,0,delay,2*weightTmp2))#source, dest, threshold, delay, weight
                
                
                #Constructing synaptic connections between L, C1 and CortexLayer_4
                weightConst = weight_ToL4
                delayConst = delay_ToL4# + uniform(-2,2)
                #L to cortex
                tmpNetCon = h.NetCon(self.L_channels[index].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.L_channels[index].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)
                self.NetCons_STDP_LtoL4.append(tmpNetCon)
                #C1 to cortex
                tmpNetCon = h.NetCon(self.C1_channels[index].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.C1_channels[index].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C1toL4.append(tmpNetCon)
                
                
                
                #Constructing synaptic connections between CortexLayer_4, C2 and CortexLayer_2_3
                #C2 to cortexLayer_2_3
                weightConst=weight_C2ToL23
                delayConst = delay_C2ToL23
                tmpNetCon = h.NetCon(self.C2_channels[index].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.C2_channels[index].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C2toL23.append(tmpNetCon)
                
                #CortexLayer_4 to cortexLayer_2_3
                weightConst=weight_L4ToL23
                delayConst = delay_L4ToL23
                tmpNetCon = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[index].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L4toL23.append(tmpNetCon)
                
                
                #Constructing synaptic connections between CortexLayer_2_3 to CortexLayer_5
                #L23 to cortexLayer_5
                weightConst=weight_L23ToL5
                delayConst = delay_L23ToL5
                tmpNetCon = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[index].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L23toL5.append(tmpNetCon)
                
                
                
        ##init feed-forward connections from 3 more neighboring cells in the preceding layers (L, C1, C2) to the cortex layer (could be removed)
        ##  ....
        ##  .TN.  T:topologically corresponding cell
        ##  .NN.  N:neighboring cells
        ##  ....
        ##
        ##*could be increased the number from 3 to 8 (all the surrounding neighbors)
        for y in range(dim):
            for x in range(dim):
                #LGN-monochrome path to L4
                weightConst = weight_ToL4
                delayConst = delay_ToL4
                tmpNetCon = h.NetCon(self.L_channels[((y+1)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.L_channels[((y+1)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_LtoL4.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.L_channels[y*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.L_channels[y*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_LtoL4.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.L_channels[((y+1)%dim)*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.L_channels[((y+1)%dim)*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_LtoL4.append(tmpNetCon)
                 
                 
                #LGN-R/G-paht to L4
                tmpNetCon = h.NetCon(self.C1_channels[((y+1)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.C1_channels[((y+1)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C1toL4.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.C1_channels[y*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.C1_channels[y*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C1toL4.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.C1_channels[((y+1)%dim)*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_4[index].syn_STDP, sec = self.C1_channels[((y+1)%dim)*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C1toL4.append(tmpNetCon)
                 
                
                
                
                #LGN-K-path to L_2/3
                weightConst=weight_L4ToL23
                tmpNetCon = h.NetCon(self.C2_channels[((y+1)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.C2_channels[((y+1)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C2toL23.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.C2_channels[y*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.C2_channels[y*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C2toL23.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.C2_channels[((y+1)%dim)*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.C2_channels[((y+1)%dim)*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_C2toL23.append(tmpNetCon)
                
                
                #L4 to L2/3
                tmpNetCon = h.NetCon(self.cortexLayer_4[((y+1)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[((y+1)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L4toL23.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.cortexLayer_4[y*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[y*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L4toL23.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.cortexLayer_4[((y-1+dim)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[((y-1+dim)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L4toL23.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.cortexLayer_4[y*dim+(x-1+dim)%dim].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[y*dim+(x-1+dim)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L4toL23.append(tmpNetCon)
                
#                 tmpNetCon = h.NetCon(self.cortexLayer_4[((y+1)%dim)*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_2_3[index].syn_STDP, sec = self.cortexLayer_4[((y+1)%dim)*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
#                 tmpNetCon.threshold = 0
#                 tmpNetCon.delay = delayConst# + uniform(-2,2)
#                 tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
#                 self.NetCons_STDP_L4toL23.append(tmpNetCon)
                
                
                
                #L2/3 to L5
                weightConst=weight_L23ToL5
                tmpNetCon = h.NetCon(self.cortexLayer_2_3[((y+1)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[((y+1)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L23toL5.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.cortexLayer_2_3[y*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[y*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L23toL5.append(tmpNetCon)
                
                tmpNetCon = h.NetCon(self.cortexLayer_2_3[((y-1+dim)%dim)*dim+x].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[((y-1+dim)%dim)*dim+x].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L23toL5.append(tmpNetCon)
                 
                tmpNetCon = h.NetCon(self.cortexLayer_2_3[y*dim+(x-1+dim)%dim].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[y*dim+(x-1+dim)%dim].soma)#source, dest, threshold, delay, weight
                tmpNetCon.threshold = 0
                tmpNetCon.delay = delayConst# + uniform(-2,2)
                tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
                self.NetCons_STDP_L23toL5.append(tmpNetCon)

                 
#                 tmpNetCon = h.NetCon(self.cortexLayer_2_3[((y+1)%dim)*dim+(x+1)%dim].soma(0.5)._ref_v,self.cortexLayer_5[index].syn_STDP, sec = self.cortexLayer_2_3[((y+1)%dim)*dim+(x+1)%dim].soma)#source, dest, threshold, delay, weight
#                 tmpNetCon.threshold = 0
#                 tmpNetCon.delay = delayConst# + uniform(-2,2)
#                 tmpNetCon.weight[0] = weightConst*uniform(initRandMinRange,initRandMaxRange)#*random()
#                 self.NetCons_STDP_L23toL5.append(tmpNetCon)
                
                
        ##Lateral connections in cortex layer 
        for y in range(dim):
            for x in range(dim):
                index = y*dim + x
                    
                #Lateral potent.
                ##
                ##   .P.   S:source
                ##   PSP   P:potentiation
                ##   .P.
                ##
                #layer V1_4
                weightTemp = weight_lateral_pot1
                delayConst = delay_lateral_pot1

                index_tmp = y*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = y*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y+1)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                #layer V1_23
                index_tmp = y*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = y*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y+1)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #layer V1_5
                index_tmp = y*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = y*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y+1)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
                 
                #Lateral potent 2
                ##
                ##   P.P    S:source
                ##   .S.    P:potentiation
                ##   P.P
                ##
                #layer V1_4
                weightTemp = weight_lateral_pot2
                delayConst = delay_lateral_pot2
                #diagonal
                index_tmp = ((y+1)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = ((y+1)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x+1)%dim
                #print ((y-1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y-1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                #layer V1_23
                index_tmp = ((y+1)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = ((y+1)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x+1)%dim
                #print ((y-1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y-1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #layer V1_5
                index_tmp = ((y+1)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                index_tmp = ((y+1)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x+1)%dim
                #print ((y-1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
              
                index_tmp = ((y-1+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y-1)%dim),(x-1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                #Lateral potent 3
                ##   PPP
                ##  P...P    S:source
                ##  P.S.P    P:potentiation
                ##  P...P
                ##   PPP
                #layer V1_4
                weightTemp = weight_lateral_pot3
                delayConst = delay_lateral_pot3
                #top_l
                index_tmp = ((y-2+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_m
                index_tmp = ((y-2+dim)%dim)*dim+x
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_r
                index_tmp = ((y-2+dim)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_t
                index_tmp = ((y-1+dim)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_m
                index_tmp = y*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #right_b
                index_tmp = ((y+1)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_r
                index_tmp = ((y+2)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_m
                index_tmp = ((y+2)%dim)*dim+(x)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #bottom_l
                index_tmp = ((y+2)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                                
                #left_b
                index_tmp = ((y+1)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_m
                index_tmp = ((y)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_t
                index_tmp = ((y-1+dim)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].AMPA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                
                
                #layer V1_23
                #top_l
                index_tmp = ((y-2+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_m
                index_tmp = ((y-2+dim)%dim)*dim+x
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_r
                index_tmp = ((y-2+dim)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_t
                index_tmp = ((y-1+dim)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_m
                index_tmp = y*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #right_b
                index_tmp = ((y+1)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_r
                index_tmp = ((y+2)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_m
                index_tmp = ((y+2)%dim)*dim+(x)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #bottom_l
                index_tmp = ((y+2)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                                
                #left_b
                index_tmp = ((y+1)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_m
                index_tmp = ((y)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_t
                index_tmp = ((y-1+dim)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].AMPA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                #layer V1_5
                #top_l
                index_tmp = ((y-2+dim)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_m
                index_tmp = ((y-2+dim)%dim)*dim+x
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #top_r
                index_tmp = ((y-2+dim)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_t
                index_tmp = ((y-1+dim)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
             
                #right_m
                index_tmp = y*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #right_b
                index_tmp = ((y+1)%dim)*dim+(x+2)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_r
                index_tmp = ((y+2)%dim)*dim+(x+1)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)

                #bottom_m
                index_tmp = ((y+2)%dim)*dim+(x)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #bottom_l
                index_tmp = ((y+2)%dim)*dim+(x-1+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                                
                #left_b
                index_tmp = ((y+1)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_m
                index_tmp = ((y)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                #left_t
                index_tmp = ((y-1+dim)%dim)*dim+(x-2+dim)%dim
                #print ((y+1)%dim),(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].AMPA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                


                #Lateral depression
                ##   DDD
                ##  D...D
                ## D.....D   S:source
                ## D..S..D   D:depression
                ## D.....D
                ##  D...D
                ##   DDD
                weightTemp = weight_lateral_dep*1.7#2
                delayConst = delay_lateral_dep
                    
                #L4
                index_tmp = y*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y+1)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
 
            
                index_tmp = y*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+1)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)           
                 
                 
             
                index_tmp = ((y+3)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)     
                 
                      
             
                index_tmp = ((y-3+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y-3+dim)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
  
                index_tmp = ((y-3+dim)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                index_tmp = ((y-2+dim)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y-2+dim)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_4[index].soma(0.5)._ref_v, self.cortexLayer_4[index_tmp].GABA, sec=self.cortexLayer_4[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                
                #L2_3
                weightTemp = weight_lateral_dep*1.5#1.7
                
                index_tmp = y*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y+1)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
 
            
                index_tmp = y*dim+(x-3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+1)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)           
                 
                 
             
                index_tmp = ((y+3)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)     
                 
                      
             
                index_tmp = ((y-3+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y-3+dim)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
  
                index_tmp = ((y-3+dim)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                index_tmp = ((y-2+dim)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y-2+dim)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_2_3[index].soma(0.5)._ref_v, self.cortexLayer_2_3[index_tmp].GABA, sec=self.cortexLayer_2_3[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                #layer V1_5
                weightTemp = weight_lateral_dep * 1
                
                index_tmp = y*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y+1)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x+3)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                 
 
            
                index_tmp = y*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+1)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y-1+dim)%dim)*dim+(x-3+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)           
                 
                 
             
                index_tmp = ((y+3)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)
                 
                index_tmp = ((y+3)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp 
                self.NetCons.append(nc_tmp)     
                 
                      
             
                index_tmp = ((y-3+dim)%dim)*dim+x
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
 
                index_tmp = ((y-3+dim)%dim)*dim+(x+1)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
  
                index_tmp = ((y-3+dim)%dim)*dim+(x-1+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                
                index_tmp = ((y-2+dim)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y-2+dim)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x-2+dim)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
                index_tmp = ((y+2)%dim)*dim+(x+2)%dim
                nc_tmp = h.NetCon(self.cortexLayer_5[index].soma(0.5)._ref_v, self.cortexLayer_5[index_tmp].GABA, sec=self.cortexLayer_5[index].soma)
                nc_tmp.delay=delayConst#*(random()+0.001)
                nc_tmp.threshold=0
                nc_tmp.weight[0]=weightTemp
                self.NetCons.append(nc_tmp)
                
    
    def initExtra(self):
        self.variables.trainingColors = numpy.zeros(3*self.variables.nTrainings).reshape((self.variables.nTrainings, 3))

    def myInitNetStims (self):
#         print 'in myInitNetStims'
        
        for y in range(self.variables.dim_stim):
            for x in range(self.variables.dim_stim):
                self.rds_r[y][x].MCellRan4(self.variables.seed_r[y][x],self.variables.seed_r[y][x])
                self.rds_r[y][x].negexp(1)
                
                self.rds_g[y][x].MCellRan4(self.variables.seed_g[y][x],self.variables.seed_g[y][x])
                self.rds_g[y][x].negexp(1)
                
                self.rds_b[y][x].MCellRan4(self.variables.seed_b[y][x],self.variables.seed_b[y][x])
                self.rds_b[y][x].negexp(1)
                
#                 self.rds_steady[y][x].MCellRan4(self.variables.seed_steady[y][x],self.variables.seed_steady[y][x])
#                 self.rds_steady[y][x].negexp(1)
           
    #function to save the current state of the network so that it could be loaded by runMe2.py to test it with different stimuli
    def saveWeightsAndDelays(self, itr):
        self.variables.NetCons_Weights = []
        self.variables.NetCons_Delays = []
        self.variables.NetCons_STDP_LtoL4_Weights = []
        self.variables.NetCons_STDP_LtoL4_Delays = []
        self.variables.NetCons_STDP_C1toL4_Weights = []
        self.variables.NetCons_STDP_C1toL4_Delays = []
        self.variables.NetCons_STDP_C2toL23_Weights = []
        self.variables.NetCons_STDP_C2toL23_Delays = []
        self.variables.NetCons_STDP_L4toL23_Weights = []
        self.variables.NetCons_STDP_L4toL23_Delays = []
        self.variables.NetCons_STDP_L23toL5_Weights = []
        self.variables.NetCons_STDP_L23toL5_Delays = []
        
        for ind in range(len(self.NetCons)):
            self.variables.NetCons_Weights.append(self.NetCons[ind].weight[0])
            self.variables.NetCons_Delays.append(self.NetCons[ind].delay)
        for ind in range(len(self.NetCons_STDP_LtoL4)):
            self.variables.NetCons_STDP_LtoL4_Weights.append(self.NetCons_STDP_LtoL4[ind].weight[0])
            self.variables.NetCons_STDP_LtoL4_Delays.append(self.NetCons_STDP_LtoL4[ind].delay)
            self.variables.NetCons_STDP_C1toL4_Weights.append(self.NetCons_STDP_C1toL4[ind].weight[0])
            self.variables.NetCons_STDP_C1toL4_Delays.append(self.NetCons_STDP_C1toL4[ind].delay)
            self.variables.NetCons_STDP_C2toL23_Weights.append(self.NetCons_STDP_C2toL23[ind].weight[0])
            self.variables.NetCons_STDP_C2toL23_Delays.append(self.NetCons_STDP_C2toL23[ind].delay)
        for ind in range(len(self.NetCons_STDP_L4toL23)):
            self.variables.NetCons_STDP_L4toL23_Weights.append(self.NetCons_STDP_L4toL23[ind].weight[0])
            self.variables.NetCons_STDP_L4toL23_Delays.append(self.NetCons_STDP_L4toL23[ind].delay)
            self.variables.NetCons_STDP_L23toL5_Weights.append(self.NetCons_STDP_L23toL5[ind].weight[0])
            self.variables.NetCons_STDP_L23toL5_Delays.append(self.NetCons_STDP_L23toL5[ind].delay)
        f = open(self.resultFolderName+"/Network_"+str(itr)+".obj", "wb")
        pickle.dump(self.variables, f)
        f.close()
    
    #function to load **.obj files that store network state information
    def loadWeightsAndDelays(self, filename, folder = "results"):
        f = open(folder+"/"+filename)
        self.variables = pickle.load(f)
        f.close()
        for ind in range(len(self.NetCons)):
            self.NetCons[ind].weight[0] = self.variables.NetCons_Weights[ind]
            self.NetCons[ind].delay = self.variables.NetCons_Delays[ind]
        for ind in range(len(self.NetCons_STDP_LtoL4)):
            self.NetCons_STDP_LtoL4[ind].weight[0] = self.variables.NetCons_STDP_LtoL4_Weights[ind]
            self.NetCons_STDP_LtoL4[ind].delay = self.variables.NetCons_STDP_LtoL4_Delays[ind]
            self.NetCons_STDP_C1toL4[ind].weight[0] = self.variables.NetCons_STDP_C1toL4_Weights[ind]
            self.NetCons_STDP_C1toL4[ind].delay = self.variables.NetCons_STDP_C1toL4_Delays[ind]
            self.NetCons_STDP_C2toL23[ind].weight[0] = self.variables.NetCons_STDP_C2toL23_Weights[ind]
            self.NetCons_STDP_C2toL23[ind].delay = self.variables.NetCons_STDP_C2toL23_Delays[ind]   
        for ind in range(len(self.NetCons_STDP_L4toL23)):  
            self.NetCons_STDP_L4toL23[ind].weight[0] = self.variables.NetCons_STDP_L4toL23_Weights[ind]
            self.NetCons_STDP_L4toL23[ind].delay = self.variables.NetCons_STDP_L4toL23_Delays[ind]        
            self.NetCons_STDP_L23toL5[ind].weight[0] = self.variables.NetCons_STDP_L23toL5_Weights[ind]
            self.NetCons_STDP_L23toL5[ind].delay = self.variables.NetCons_STDP_L23toL5_Delays[ind]              
        
    
    #set input that is currently presented to the network
    def setInput(self,inputStim,duration=1):#inputStim[r][g][b]
        #spike input
        delay = 0#self.variables.tstop * (1-duration)/2
        eps = 0.0001
        noDelay = 0;
        
        if(noDelay==0):
            for y in range(self.variables.dim_stim):
                for x in range(self.variables.dim_stim):
                    r = (inputStim[y][x].r +  inputStim[y][x].g * 0.7 + inputStim[y][x].b * 0.25)/(1+0.7+0.25);
                    g = (inputStim[y][x].r * 0.7 + inputStim[y][x].g + inputStim[y][x].b * 0.25)/(1+0.7+0.25);
                    b = inputStim[y][x].b
                    
                    self.r_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*r+eps#*3#(average) number of spikes
                    self.r_stims[y][x].interval = self.variables.tstop/self.r_stims[y][x].number# ms (mean) time between spikes
                    if(delay+self.r_stims[y][x].interval*self.variables.delay_r[y][x]>self.variables.tstop*duration):
                        self.r_stims[y][x].number = 0
                        self.r_stims[y][x].start = self.variables.tstop
                    else:
                        self.r_stims[y][x].start = (delay+self.r_stims[y][x].interval*self.variables.delay_r[y][x])
                        self.r_stims[y][x].number = (self.variables.tstop*duration - self.r_stims[y][x].start)/self.r_stims[y][x].interval
                        
                    self.g_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*g+eps#*3#(average) number of spikes
                    self.g_stims[y][x].interval = self.variables.tstop/self.g_stims[y][x].number# ms (mean) time between spikes
                    if(delay+self.g_stims[y][x].interval*self.variables.delay_g[y][x]>self.variables.tstop*duration):
                        self.g_stims[y][x].number = 0
                        self.g_stims[y][x].start = self.variables.tstop
                    else:
                        self.g_stims[y][x].start = delay+self.g_stims[y][x].interval*self.variables.delay_g[y][x]
                        self.g_stims[y][x].number = (self.variables.tstop*duration - self.g_stims[y][x].start)/self.g_stims[y][x].interval
                     
                    self.b_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*b+eps#*3#(average) number of spikes
                    self.b_stims[y][x].interval = self.variables.tstop/self.b_stims[y][x].number# ms (mean) time between spikes
                    if (delay+self.b_stims[y][x].interval*self.variables.delay_b[y][x]>self.variables.tstop*duration):
                        self.b_stims[y][x].number = 0
                        self.b_stims[y][x].start = self.variables.tstop
                    else:
                        self.b_stims[y][x].start = (delay+self.b_stims[y][x].interval*self.variables.delay_b[y][x])
                        self.b_stims[y][x].number = (self.variables.tstop*duration - self.b_stims[y][x].start)/self.b_stims[y][x].interval
                    
    #                 self.steadyActivation[y][x].number = self.variables.hz*self.variables.tstop/1000*self.variables.steady_state_activation
    #                 self.steadyActivation[y][x].interval = self.variables.tstop/self.steadyActivation[y][x].number# ms (mean) time between spikes
    #                 self.steadyActivation[y][x].start = (delay+self.steadyActivation[y][x].interval*self.variables.delay_steady[y][x])%(self.variables.tstop)
              
    #         self.r_stim.number = self.variables.hz*self.variables.tstop/1000*(self.variables.steady_state_activation+(1-self.variables.steady_state_activation)*r)#*3#(average) number of spikes
    #         self.r_stim.interval = self.variables.tstop/self.r_stim.number# ms (mean) time between spikes
    #         self.r_stim.start = self.r_stim.interval/2#*random()
    #          
    #         self.g_stim.number = self.variables.hz*self.variables.tstop/1000*(self.variables.steady_state_activation+(1-self.variables.steady_state_activation)*g)#*3#(average) number of spikes
    #         self.g_stim.interval = self.variables.tstop/self.g_stim.number# ms (mean) time between spikes
    #         self.g_stim.start = self.g_stim.interval/2#*random()
    #          
    #         self.b_stim.number = self.variables.hz*self.variables.tstop/1000*(self.variables.steady_state_activation+(1-self.variables.steady_state_activation)*b)#*3#(average) number of spikes
    #         self.b_stim.interval = self.variables.tstop/self.b_stim.number# ms (mean) time between spikes
    #         self.b_stim.start = self.b_stim.interval/2#*random()
        else:
            for y in range(self.variables.dim_stim):
                for x in range(self.variables.dim_stim):
                    r = (inputStim[y][x].r +  inputStim[y][x].g * 0.7 + inputStim[y][x].b * 0.25)/(1+0.7+0.25);
                    g = (inputStim[y][x].r * 0.7 + inputStim[y][x].g + inputStim[y][x].b * 0.25)/(1+0.7+0.25);
                    b = inputStim[y][x].b
                    
                    self.r_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*r+eps#*3#(average) number of spikes
                    self.r_stims[y][x].interval = self.variables.tstop/self.r_stims[y][x].number# ms (mean) time between spikes
                    if(delay>self.variables.tstop*duration):
                        self.r_stims[y][x].number = 0
                        self.r_stims[y][x].start = self.variables.tstop
                    else:
                        self.r_stims[y][x].start = (delay)
                        self.r_stims[y][x].number = (self.variables.tstop*duration - self.r_stims[y][x].start)/self.r_stims[y][x].interval
                        
                    self.g_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*g+eps#*3#(average) number of spikes
                    self.g_stims[y][x].interval = self.variables.tstop/self.g_stims[y][x].number# ms (mean) time between spikes
                    if(delay>self.variables.tstop*duration):
                        self.g_stims[y][x].number = 0
                        self.g_stims[y][x].start = self.variables.tstop
                    else:
                        self.g_stims[y][x].start = delay
                        self.g_stims[y][x].number = (self.variables.tstop*duration - self.g_stims[y][x].start)/self.g_stims[y][x].interval
                     
                    self.b_stims[y][x].number = self.variables.hz*self.variables.tstop/1000*b+eps#*3#(average) number of spikes
                    self.b_stims[y][x].interval = self.variables.tstop/self.b_stims[y][x].number# ms (mean) time between spikes
                    if (delay>self.variables.tstop*duration):
                        self.b_stims[y][x].number = 0
                        self.b_stims[y][x].start = self.variables.tstop
                    else:
                        self.b_stims[y][x].start = (delay)
                        self.b_stims[y][x].number = (self.variables.tstop*duration - self.b_stims[y][x].start)/self.b_stims[y][x].interval            
    
    
    def saveColor(self,r,g,b,itr):
        self.variables.trainingColors[itr,0] = r;
        self.variables.trainingColors[itr,1] = g;
        self.variables.trainingColors[itr,2] = b;
        
    
    #function to set Learning rate (learning rate decreases proportion to the training iterations)
    def setLR(self,progress):
        for index in range(self.variables.dim*self.variables.dim):
            self.cortexLayer_4[index].setLR(progress)
#         for index in range(self.variables.dim*self.variables.dim):
#             self.cortexLayer_2_3[index].setLR(1-progress)
            
    #function to start recording
    def recordChannelVols(self):
        #record voltage dynamics of neuron[0] in each channels 
        self.vvolt_channel_L = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_channel_L.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_channel_L[index].record(self.L_channels[index].soma(0.5)._ref_v)
         
        self.vvolt_channel_C1 = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_channel_C1.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_channel_C1[index].record(self.C1_channels[index].soma(0.5)._ref_v)
         
        self.vvolt_channel_C2 = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_channel_C2.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_channel_C2[index].record(self.C2_channels[index].soma(0.5)._ref_v)

    
    
    def recordVols(self):
        #record voltage dynamics of all neurons in the cortex layer
        self.vvolt_cortexL_4 = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_cortexL_4.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_cortexL_4[index].record(self.cortexLayer_4[index].soma(0.5)._ref_v)
            
        self.vvolt_cortexL_2_3 = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_cortexL_2_3.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_cortexL_2_3[index].record(self.cortexLayer_2_3[index].soma(0.5)._ref_v)
            
        self.vvolt_cortexL_5 = []
        for index in range(self.variables.dim*self.variables.dim):
            self.vvolt_cortexL_5.append(h.Vector(int(self.variables.tstop / h.dt) + 1))  # make a Vector 
            self.vvolt_cortexL_5[index].record(self.cortexLayer_5[index].soma(0.5)._ref_v)
                
    def weightNormalization(self):
        #find Max
        LtoL4_avg = 0
        C1toL4_avg = 0
        C2toL23_avg = 0
        L4toL23_avg = 0
        L23toL5_avg = 0
        
        #calculate average weight
        for index in range(len(self.NetCons_STDP_LtoL4)):#self.variables.dim*self.variables.dim):
            LtoL4_avg+=self.NetCons_STDP_LtoL4[index].weight[0]
            C1toL4_avg+=self.NetCons_STDP_C1toL4[index].weight[0]
            C2toL23_avg+=self.NetCons_STDP_C2toL23[index].weight[0]
        for index in range(len(self.NetCons_STDP_L4toL23)):#self.variables.dim*self.variables.dim):
            L4toL23_avg+=self.NetCons_STDP_L4toL23[index].weight[0]
            L23toL5_avg+=self.NetCons_STDP_L23toL5[index].weight[0]
        LtoL4_avg/=len(self.NetCons_STDP_LtoL4)
        C1toL4_avg/=len(self.NetCons_STDP_C1toL4)
        C2toL23_avg/=len(self.NetCons_STDP_C2toL23)
        L4toL23_avg/=len(self.NetCons_STDP_L4toL23)
        L23toL5_avg/=len(self.NetCons_STDP_L23toL5)
        
#         print "avg="+str([LtoL4_avg, C1toL4_avg, C2toL23_avg]) + "ideal=" +str((self.variables.maxWeight-self.variables.minWeight)/2)
        
        #calculate the differences between the ideal average values and actual average values
        LtoL4_mod = (self.variables.maxWeight-self.variables.minWeight)/2-LtoL4_avg
        C1toL4_mod = (self.variables.maxWeight-self.variables.minWeight)/2-C1toL4_avg
        C2toL23_mod = (self.variables.maxWeight*self.variables.maxWeightsMulti-self.variables.minWeight)/2-C2toL23_avg    
        L4toL23_mod = (self.variables.maxWeight*self.variables.maxWeightsMulti-self.variables.minWeight)/2-L4toL23_avg    
        L23toL5_mod = (self.variables.maxWeight*self.variables.maxWeightsMulti2-self.variables.minWeight)/2-L23toL5_avg     
        
#         print "shift=" + str([LtoL4_mod, C1toL4_mod, C2toL23_mod])      
        
        #adjust the weight values to make the average is around 50%
        for index in range(len(self.NetCons_STDP_LtoL4)):
            tmp = self.NetCons_STDP_LtoL4[index].weight[0]+LtoL4_mod
            if(tmp>=self.variables.maxWeight):
                self.NetCons_STDP_LtoL4[index].weight[0] = self.variables.maxWeight
#                 LtoL4_mod+=(tmp-self.variables.maxWeight)/(len(self.NetCons_STDP_LtoL4)-index)
            elif(tmp<=self.variables.minWeight):
                self.NetCons_STDP_LtoL4[index].weight[0] = self.variables.minWeight
#                 LtoL4_mod-=(self.variables.minWeight-tmp)/(len(self.NetCons_STDP_LtoL4)-index)
            else:
                self.NetCons_STDP_LtoL4[index].weight[0] = tmp
            
            tmp = self.NetCons_STDP_C1toL4[index].weight[0]+C1toL4_mod
            if(tmp>=self.variables.maxWeight):
                self.NetCons_STDP_C1toL4[index].weight[0] = self.variables.maxWeight
#                 C1toL4_mod+=(tmp-self.variables.maxWeight)/(len(self.NetCons_STDP_C1toL4)-index)
            elif(tmp<=self.variables.minWeight):
                self.NetCons_STDP_C1toL4[index].weight[0] = self.variables.minWeight
#                 C1toL4_mod-=(self.variables.minWeight-tmp)/(len(self.NetCons_STDP_C1toL4)-index)
            else:
                self.NetCons_STDP_C1toL4[index].weight[0] = tmp  
            
            tmp = self.NetCons_STDP_C2toL23[index].weight[0]+C2toL23_mod
            if(tmp>=self.variables.maxWeight*self.variables.maxWeightsMulti):
                self.NetCons_STDP_C2toL23[index].weight[0] = self.variables.maxWeight*self.variables.maxWeightsMulti
#                 C2toL23_mod+=(tmp-self.variables.maxWeight*self.variables.maxWeightsMulti)/(len(self.NetCons_STDP_C2toL23)-index)
            elif(tmp<=self.variables.minWeight):
                self.NetCons_STDP_C2toL23[index].weight[0] = self.variables.minWeight
#                 C2toL23_mod-=(self.variables.minWeight-tmp)/(len(self.NetCons_STDP_C2toL23)-index)
            else:
                self.NetCons_STDP_C2toL23[index].weight[0] = tmp
                    
        for index in range(len(self.NetCons_STDP_L4toL23)):   
            tmp = self.NetCons_STDP_L4toL23[index].weight[0]+L4toL23_mod
            if(tmp>=self.variables.maxWeight*self.variables.maxWeightsMulti):
                self.NetCons_STDP_L4toL23[index].weight[0] = self.variables.maxWeight*self.variables.maxWeightsMulti
#                 L4toL23_mod+=(tmp-self.variables.maxWeight*self.variables.maxWeightsMulti)/(len(self.NetCons_STDP_L4toL23)-index)
            elif(tmp<=self.variables.minWeight):
                self.NetCons_STDP_L4toL23[index].weight[0] = self.variables.minWeight
#                 L4toL23_mod-=(self.variables.minWeight-tmp)/(len(self.NetCons_STDP_L4toL23)-index)
            else:
                self.NetCons_STDP_L4toL23[index].weight[0] = tmp   
                
            tmp = self.NetCons_STDP_L23toL5[index].weight[0]+L23toL5_mod
            if(tmp>=self.variables.maxWeight*self.variables.maxWeightsMulti2):
                self.NetCons_STDP_L23toL5[index].weight[0] = self.variables.maxWeight*self.variables.maxWeightsMulti2
#                 L23toL5_mod+=(tmp-self.variables.maxWeight*self.variables.maxWeightsMulti2)/(len(self.NetCons_STDP_L23toL5)-index)
            elif(tmp<=self.variables.minWeight):
                self.NetCons_STDP_L23toL5[index].weight[0] = self.variables.minWeight
#                 L23toL5_mod-=(self.variables.minWeight-tmp)/(len(self.NetCons_STDP_L23toL5)-index)
            else:
                self.NetCons_STDP_L23toL5[index].weight[0] = tmp            
            
        
            
#                 
#     def weightNormalization2(self):
#             #find Max
#             LtoL4_div = 0
#             C1toL4_div = 0
#             C2toL23_div = 0
#             L4toL23_div = 0
#             L23toL5_div = 0
#             
#             #calculate average weight
#             for index in range(len(self.NetCons_STDP_LtoL4)):#self.variables.dim*self.variables.dim):
#                 LtoL4_div+=pow(self.NetCons_STDP_LtoL4[index].weight[0],2)
#                 C1toL4_div+=pow(self.NetCons_STDP_C1toL4[index].weight[0],2)
#                 C2toL23_div+=pow(self.NetCons_STDP_C2toL23[index].weight[0],2)
#             for index in range(len(self.NetCons_STDP_L4toL23)):#self.variables.dim*self.variables.dim):
#                 L4toL23_div+=pow(self.NetCons_STDP_L4toL23[index].weight[0],2)
#                 L23toL5_div+=pow(self.NetCons_STDP_L23toL5[index].weight[0],2)
#             
#             #adjust the weight values to make the average is around 50%
#             for index in range(len(self.NetCons_STDP_LtoL4)):
#                 self.NetCons_STDP_LtoL4[index].weight[0]= self.NetCons_STDP_LtoL4[index].weight[0]/math.sqrt(LtoL4_div)*self.variables.maxWeight
#                 self.NetCons_STDP_C1toL4[index].weight[0] = self.NetCons_STDP_C1toL4[index].weight[0]/math.sqrt(C1toL4_div)*self.variables.maxWeight
#                 self.NetCons_STDP_C2toL23[index].weight[0] = self.NetCons_STDP_C2toL23[index].weight[0]/math.sqrt(C2toL23_div)*self.variables.maxWeight*self.variables.maxWeightsMulti
#                         
#             for index in range(len(self.NetCons_STDP_L4toL23)):   
#                 self.NetCons_STDP_L4toL23[index].weight[0] = self.NetCons_STDP_L4toL23[index].weight[0]/math.sqrt(L4toL23_div)*self.variables.maxWeight*self.variables.maxWeightsMulti
#                 self.NetCons_STDP_L23toL5[index].weight[0] = self.NetCons_STDP_L23toL5[index].weight[0]/math.sqrt(L23toL5_div)*self.variables.maxWeight*self.variables.maxWeightsMulti2
#                 
    
    def saveSpikeDetails(self,r,g,b,itr):
        #output voltage dynamics of each cells in the cortex layer
        f = open(self.resultFolderName+"/L4_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_cortexL_4)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_cortexL_4[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_cortexL_4[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/L23_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_cortexL_2_3)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_cortexL_2_3[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_cortexL_2_3[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()
        
        
        f = open(self.resultFolderName+"/L5_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_cortexL_5)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_cortexL_5[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_cortexL_5[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()
        
    def saveChannelSpikeDetails(self,r,g,b,itr):
        #output voltage dynamics of each cells in the cortex layer
        f = open(self.resultFolderName+"/L_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_channel_L)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_channel_L[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_channel_L[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/C1_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_channel_C1)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_channel_C1[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_channel_C1[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/C2_firing_"+str(itr)+"_"+str([r, g, b])+".txt", "w")
        for index in range(len(self.vvolt_channel_C2)):
            f.write(str(r)+str(g)+str(b)+",")
            for t in range(len(self.vvolt_channel_C2[index])):
                if t>0:
                    f.write(",")
                if (self.vvolt_channel_C2[index][t]>0):
                    f.write(str(1))
                else:
                    f.write(str(0))
            f.write("\n")
        f.write("\n")
        f.close()

    #save FR data (if a stimulus undergoes transformation; i.e., input contains colour regarded as the same but slightly different in this case)
    def outputFR_trans(self,r,g,b,itr):
        #output spike counts of each cell
        f = open(self.resultFolderName+"/trans_L23_FR_"+str(itr)+"_"+str([r, g, b])+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L23[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/trans_L4_FR_"+str(itr)+"_"+str([r, g, b])+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L4[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/trans_L5_FR_"+str(itr)+"_"+str([r, g, b])+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L5[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()
    
    #save FR data (without any transformation)
    def outputFR(self,itr):
        #output spike counts of each cell
        f = open(self.resultFolderName+"/L4_FR_"+str(itr)+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L4[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/L23_FR_"+str(itr)+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L23[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()
        
        f = open(self.resultFolderName+"/L5_FR_"+str(itr)+".txt", "a")
        for y in range(self.variables.dim):
            for x in range(self.variables.dim):
                index = y*self.variables.dim+x
                f.write(str(self.spikeCount_L5[y][x]))
                if(index != self.variables.dim*self.variables.dim-1):
                    f.write(",")
        f.write("\n")
        f.close()    
        
    
    def updateAllSpikeCount(self):#including channels
        self.updateSpikeCount()
        dim = self.variables.dim
        self.spikeCount_L = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_channel_L[index][_i]>0:
                        if (_i>0 and self.vvolt_channel_L[index][_i-1]>0):
                            continue
                        self.spikeCount_L[y][x] += 1#self.vvolt_cortexL_4[index][_i]
                        
        self.spikeCount_C1 = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_channel_C1[index][_i]>0:
                        if (_i>0 and self.vvolt_channel_C1[index][_i-1]>0):
                            continue
                        self.spikeCount_C1[y][x] += 1#self.vvolt_cortexL_4[index][_i]
                        
        self.spikeCount_C2 = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_channel_C2[index][_i]>0:
                        if (_i>0 and self.vvolt_channel_C2[index][_i-1]>0):
                            continue
                        self.spikeCount_C2[y][x] += 1#self.vvolt_cortexL_4[index][_i]
   
        
    
    #function to count number of spikes 
    def updateSpikeCount(self):
        dim = self.variables.dim
        self.spikeCount_L4 = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_cortexL_4[index][_i]>0:
                        if (_i>0 and self.vvolt_cortexL_4[index][_i-1]>0):
                            continue
                        self.spikeCount_L4[y][x] += 1#self.vvolt_cortexL_4[index][_i]
                        
        self.spikeCount_L23 = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_cortexL_2_3[index][_i]>0:
                        if (_i>0 and self.vvolt_cortexL_2_3[index][_i-1]>0):
                            continue
                        self.spikeCount_L23[y][x] += 1#self.vvolt_cortexL_4[index][_i]
                        
        self.spikeCount_L5 = [[0 for x in xrange(dim)] for x in xrange(dim)] 
        for y in range(dim):
            for x in range(dim):
                index = y*dim+x 
                for _i in range(int(self.variables.tstop / h.dt) + 1):
                    if self.vvolt_cortexL_5[index][_i]>0:
                        if (_i>0 and self.vvolt_cortexL_5[index][_i-1]>0):
                            continue
                        self.spikeCount_L5[y][x] += 1#self.vvolt_cortexL_4[index][_i]
                        
                        
    def hebbUpdate(self):#todo sometime in the future
        self.updateAllSpikeCount()
        #C1
        #update the weights based on the firing count at the end of each iteration
        
        
        
        
        
            
    #to change synaptic modifications are active or not
    def setLearningStates(self,state):#state=0: no learning, state=1:learning is active
        self.learningState = state
        for index in range(self.variables.dim*self.variables.dim):
            self.cortexLayer_4[index].setLearningState(state)
            self.cortexLayer_2_3[index].setLearningState(state)
            self.cortexLayer_5[index].setLearningState(state)
    
    #to plot voltage changes with neuron simulator's GUI
    def drawGraph(self):
        g = h.Graph()
        self.vvolt_channel_L[0].plot(g, h.dt, 1, 1)#black
        self.vvolt_channel_C1[0].plot(g, h.dt, 2, 1)#red
        self.vvolt_channel_C2[0].plot(g, h.dt, 3, 1)#blue
        self.vvolt_cortexL_4[0].plot(g, h.dt, 4, 1)#light green
        
        g.exec_menu("View = plot")  # set Graph view to what was drawn
        raw_input("Press Enter to exit...")
    
    def run(self):
        h.tstop = self.variables.tstop
        fihns = h.FInitializeHandler(0, self.myInitNetStims) # create an object
        h.run()
