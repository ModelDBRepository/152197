from Controller import *
import matplotlib.pyplot as plt
import pylab
import Image, glob
from time import time
import math



startingTime = time()
nTrainings = 2001
ndim = 30
dim_stim = 10
inputFromImages = 1
controller = Controller(dim_stim,ndim)
Hebb = 0

weights_LtoL4 = []
weights_C1toL4 = []
weights_C2toL23 = []
weights_L4toL23 = []
weights_L23toL5 = []


TimeAndthread = [-1 for x in xrange(16)]
#to specify number of thread employed
nThreads = 7;
controller.pc = h.ParallelContext()
controller.pc.nthread(nThreads)
nThreadTuning = 0

class InputStim:
    r=0
    g=0
    b=0
    def setRGB(self,r,g,b):
        self.r= r
        self.g= g
        self.b= b

inputStim = [[InputStim() for x in xrange(dim_stim)] for x in xrange(dim_stim)]

input_files = glob.iglob("./input/*.jpg")

inputImgs = []
for data in input_files:
    im = Image.open(data)
    tmp = im.load();
    inputImgs.append(tmp)
    
resultFolderName = "results";
controller.resultFolderName = resultFolderName;

controller.variables.nTrainings =  nTrainings;
controller.initExtra()
if(Hebb):
    controller.setLearningStates(0)
    
# controller.loadWeightsAndDelays("Network_1600.obj",resultFolderName)

for itr in range(nTrainings):
#     if itr<1601:
#         continue;
    #testing
    #saving weights
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
    weights_LtoL4.append(weightTemp_LtoL4)
    weights_C1toL4.append(weightTemp_C1toL4)
    weights_C2toL23.append(weightTemp_C2toL23)
    weights_L4toL23.append(weightTemp_L4toL23)
    weights_L23toL5.append(weightTemp_L23toL5)
    
    #output the weight dynamics
    if(itr%100==0):
        fig1 = plt.gcf()
        plt.clf()  
        plt.subplot(511)
        plt.plot(weights_L23toL5)
        plt.subplot(512)
        plt.plot(weights_L4toL23)
        plt.subplot(513)
        plt.plot(weights_C2toL23)
        plt.subplot(514)
        plt.plot(weights_C1toL4)
        plt.subplot(515)
        plt.plot(weights_LtoL4)
        fig1.savefig(resultFolderName + "/weightDynamics_L5_L4toL23_C2toL23_C1_L.png",dpi=100)
    
    #save networkstates
    if(itr%100==0):
        controller.saveWeightsAndDelays(itr)
        
    
    #test the network with 8 different colour input
    if(itr%100==0):
#     if(itr==2000):
        controller.setLearningStates(0)#stop synaptic modifications
        #plotting firing counts         
        fig1 = plt.gcf()
        plt.clf()
        for r in range(2):
            for g in range(2):
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
                    
                    
                    
                    startTime = time()
                    
                    for y in range(dim_stim):
                        for x in range(dim_stim):
                            inputStim[y][x].setRGB(r2,g2,b2)
                    controller.setInput(inputStim,0.8)
                    controller.recordVols()
                    if(itr==-1):
                        controller.recordChannelVols()
                    controller.run()
                    
                    controller.updateSpikeCount()
                    if(itr%100==0):
                        controller.outputFR(itr)
                    
                    
                    
                    
                    timeSpent = time()-startTime
                    
                    TimeAndthread[nThreads] = timeSpent 
                    if(nThreadTuning):
                        if(nThreads>1 and TimeAndthread[nThreads-1]<timeSpent):
                            nThreads = nThreads-1
                            nThreadTuning=0
                        else:
                            nThreads=nThreads+1
                        controller.pc.nthread(nThreads)
                        
                    if(itr%2000==-1):
                        controller.saveSpikeDetails(r2,g2,b2,itr)
                    if(itr==-1):
                        controller.saveChannelSpikeDetails(r2,g2,b2,itr)
                    
                    
                    plt.subplot(8,4,r*4+g*2+b+1)
                    plt.imshow(controller.spikeCount_L5,cmap=pylab.gray())
                    plt.colorbar()
                    
                    plt.subplot(8,4,r*4+g*2+b+13)
                    plt.imshow(controller.spikeCount_L23,cmap=pylab.gray())
                    plt.colorbar()
                    
                    plt.subplot(8,4,r*4+g*2+b+25)
                    plt.imshow(controller.spikeCount_L4,cmap=pylab.gray())
                    plt.colorbar()
                    #plt.show()

                    
                        
        fig1.savefig(resultFolderName+"/"+str(itr),format="png")#dpi=100, 
        if(~Hebb):
            controller.setLearningStates(1)#start synaptic modifications

    if(itr==nTrainings-1):
        break
    
    print itr
    startTime = time()
    #weight plot

    
#     controller.setLR(1-(itr/nTrainings))#set learning rate
#     for y in range(dim_stim):
#         for x in range(dim_stim):
#             inputStim[y][x].setRGB(random(),random(),random())
#             
    if(inputFromImages):
        loadedImg = inputImgs[int(len(inputImgs) * itr/nTrainings)]
        xBegin = 200*random()
        yBegin = 200*random()
        tmp_r_tot = 0
        tmp_g_tot = 0
        tmp_b_tot = 0
        for y in range(dim_stim):
            for x in range(dim_stim):
                tmp = loadedImg[x+xBegin,y+yBegin]
                inputStim[y][x].setRGB(tmp[0]/255.0,tmp[1]/255.0,tmp[2]/255.0)
                tmp_r_tot+=tmp[0]/255.0;
                tmp_g_tot+=tmp[1]/255.0;
                tmp_b_tot+=tmp[2]/255.0;
        controller.saveColor(tmp_r_tot/(dim_stim*dim_stim),tmp_g_tot/(dim_stim*dim_stim),tmp_b_tot/(dim_stim*dim_stim),itr)
    else:
        input_r = random();
        input_g = random();
        input_b = random();
        for y in range(dim_stim):
            for x in range(dim_stim):
                inputStim[y][x].setRGB(input_r,input_g,input_b)
        controller.saveColor(input_r,input_g,input_b,itr)
 
            
            #print (tmp[0]/255.0,tmp[1]/255.0,tmp[2]/255.0)
    
    
    
    
    controller.setInput(inputStim)
    controller.recordVols()
    if(Hebb):
        controller.recordChannelVols()
    controller.run()
    
    if(Hebb):
        controller.hebbUpdate()
    
    
    
    controller.weightNormalization()
#     controller.weightNormalization2()
    #controller.drawGraph()
    timeSpent = time()-startTime
    print "iteration time:"+str(timeSpent)+" with nThreads:"+str(nThreads)
    print "estimated remaining: at least "+str(timeSpent*(nTrainings-itr))+" + testing Time"
    
    TimeAndthread[nThreads] = timeSpent
    
#     if(nThreadTuning):
#         if(nThreads>1 and TimeAndthread[nThreads-1]<timeSpent):
#             nThreads = nThreads-1
#             nThreadTuning=0
#         else:
#             nThreads=nThreads+1
#         controller.pc.nthread(nThreads)
        
print (time() - startingTime)
# raw_input("Press Enter to exit...")
