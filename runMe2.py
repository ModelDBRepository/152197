from Controller import *
import matplotlib.pyplot as plt
import pylab
from time import time
###########
##@summary: This is to test the network saved in **.obj, which is generated during training by rumMe.py
##@author: Akihiro Eguchi
##Aug 13, 2013

nTrainings = 2001
ndim = 30
dim_stim = 10

controller = Controller(dim_stim,ndim)
resultFolderName = "results";
controller.resultFolderName = resultFolderName;
transOn = 1
singleColor = 1
alteringInput = 0#not fully implemented yet
itr = 2000

class InputStim:
    r=0
    g=0
    b=0
    def setRGB(self,r,g,b):
        self.r= r
        self.g= g
        self.b= b

inputStim = [[InputStim() for x in xrange(dim_stim)] for x in xrange(dim_stim)]

controller.loadWeightsAndDelays("Network_"+str(itr)+".obj",resultFolderName)
controller.setLearningStates(0)#stop synaptic modifications
controller.variables.tstop = 300;
nThreads = 7;
controller.pc = h.ParallelContext()
controller.pc.nthread(nThreads)

if(alteringInput):
    controller.AlteringInputInit();


weightTemp_LtoL4 = []
weightTemp_C1toL4 = []
weightTemp_C2toL23 = []
weightTemp_L4toL23 = []
weightTemp_L23toL5 = []
for index in range(len(controller.NetCons_STDP_LtoL4)):
    weightTemp_LtoL4.append(controller.NetCons_STDP_LtoL4[index].weight[0])
    weightTemp_C1toL4.append(controller.NetCons_STDP_C1toL4[index].weight[0])
    weightTemp_C2toL23.append(controller.NetCons_STDP_C2toL23[index].weight[0])
for index in range(len(controller.NetCons_STDP_L4toL23)):
    weightTemp_L4toL23.append(controller.NetCons_STDP_L4toL23[index].weight[0])
    weightTemp_L23toL5.append(controller.NetCons_STDP_L23toL5[index].weight[0])


plt.subplot(5, 1, 1)
plt.hist(weightTemp_L23toL5, bins=100, range=[0, 0.005*3]);
plt.xlim(0,0.005*3)
plt.xlabel("synaptic weights between V1_L23 and V1_L5")
plt.ylabel("number of synapses")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
# plt.show();

plt.subplot(5, 1, 2)
plt.hist(weightTemp_C2toL23, bins=100, range=[0, 0.005*1.5]);
plt.xlim(0,0.005*1.5)
plt.xlabel("synaptic weights between C2 and V1_L23")
plt.ylabel("number of synapses")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)

plt.subplot(5, 1, 3)
plt.hist(weightTemp_L4toL23, bins=100, range=[0, 0.005*1.5]);
plt.xlim(0,0.005*1.5)
plt.xlabel("synaptic weights between V1_L4 and V1_L23")
plt.ylabel("number of synapses")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)


plt.subplot(5, 1, 4)
plt.hist(weightTemp_C1toL4, bins=100, range=[0, 0.005]);
plt.xlim(0,0.005)
plt.xlabel("synaptic weights between C1 and V1_L4")
plt.ylabel("number of synapses")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
# plt.show();

plt.subplot(5, 1, 5)
plt.hist(weightTemp_LtoL4, bins=100, range=[0, 0.005]);
plt.xlim(0,0.005)
plt.xlabel("synaptic weights between L and V1_L4")
plt.ylabel("number of synapses")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
plt.show();



#0.5 0 0.5
#0 0 1
#0 1 1
#0 1 0
#1 1 0
#1 0.5 0
#1 0 0
#1 0 1

# purple    0.5 0 1
# blue      0 0 1 
# light-green 0 1 0
# light-blue 0 1 1
# red         1 0 0
# pink        1 0 1
# yellow      1 1 0
# orange      1 0.5 0
if(singleColor==1):
    fig1 = plt.gcf()
    plt.clf()
    for r in range(2):
        r_bak = r
        for g in range(2):
            g_bak = g
            for b in range(2):
                r2 = r;
                g2 = g;
                b2 = b;
                if(r==0 and g==0 and b==0):
                    r2=0.5
                    g2=0
                    b2=1
  
                if(r==1 and g==1 and b==1):
                    r2=1
                    g2=0.5
                    b2=0
                
                for y in range(dim_stim):
                    for x in range(dim_stim):
                        inputStim[y][x].setRGB(r2,g2,b2)
                if(alteringInput):
                    controller.setAlteringInput(inputStim, 0.25)
                else:
                    controller.setInput(inputStim,0.8)
                controller.recordVols()
#                 controller.recordChannelVols()
                controller.run()
                                
                controller.updateSpikeCount()
#                 controller.outputFR(itr)
#                 controller.saveSpikeDetails(r,g,b,itr)
#                 controller.saveChannelSpikeDetails(r,g,b,itr)
                
                
             

                
                plt.subplot(8,4,r*4+g*2+b+1)
                plt.imshow(controller.spikeCount_L5,cmap=pylab.gray())
                plt.colorbar()
                
                plt.subplot(8,4,r*4+g*2+b+13)
                plt.imshow(controller.spikeCount_L23,cmap=pylab.gray())
                plt.colorbar()
                
                plt.subplot(8,4,r*4+g*2+b+25)
                plt.imshow(controller.spikeCount_L4,cmap=pylab.gray())
                plt.colorbar()
                
                
                
                
                
                if(transOn):
                    controller.outputFR_trans(r2,g2,b2,itr)
                    
                    #transformation: varies input with similar colours
                    modVal = 0.01
                    if r2 ==  0:
                        rMod = r2+modVal
                    else:
                        rMod = r2-modVal
                    if g2 ==  0:
                        gMod = g2+modVal
                    else:
                        gMod = g2-modVal
                    if b2 ==  0:
                        bMod = b2+modVal
                    else:
                        bMod = b2-modVal
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(rMod,g2,b2)
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(r2,gMod,b2)
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(r2,g2,bMod)                
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(rMod,gMod,b2)
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(r2,gMod,bMod)
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(rMod,g2,bMod)                
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(rMod,gMod,bMod)                
                    controller.setInput(inputStim)
                    controller.recordVols()
                    controller.run()
                    controller.updateSpikeCount()
                    controller.outputFR_trans(r2,g2,b2,itr)
    fig1.savefig(resultFolderName+"/"+str(itr),dpi=100)
    plt.show()
else:
#     controller.variables.tstop = 300
    fig1 = plt.gcf()
    plt.clf()
    b = 0
    for r in range(2):
        for g in range(2):
            if (r==g):
                continue
            for y in range(dim_stim):
                for x in range(dim_stim):
                    if y>dim_stim/3 and y<dim_stim*2/3 and x>dim_stim/3 and x>dim_stim*2/3:
                        inputStim[y][x].setRGB(1-r,1-g,b)
                    else:
                        inputStim[y][x].setRGB(r,g,b)
            controller.setInput(inputStim,0.3)
            controller.recordVols()
            controller.run()
            controller.updateSpikeCount()
            controller.outputFR(itr)    
            
            plt.subplot(2,1,r+1)
            plt.imshow(controller.spikeCount_L4,cmap=pylab.gray())
            plt.colorbar()
            
            #controller.drawGraph()
            controller.saveSpikeDetails(r,g,b,111110);
            
    fig1.savefig(resultFolderName+"/multiColTest300_normal"+str(itr),dpi=100)
            
controller.setLearningStates(1)#start synaptic modifications



    
# raw_input("Press Enter to exit...")
