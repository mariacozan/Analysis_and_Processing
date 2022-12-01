# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:34:07 2022

@author: maria
"""
import numpy as np
import pandas as pd
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import butter,filtfilt,medfilt
import csv
import re





#getting the F traces which are classified as cells by Suite2P (manually curated ROIs should be automatically saved)
def getcells(filePathF, filePathiscell):
    """
    This function returns the ROIs that are classified as cells. 
    Careful, only use this if you have manually curated the Suite2P data!

    Parameters
    ----------
    filePathF : string
        The path of where the fluorescence traces from Suite2P are located. 
        It will load the file as an array within the function.
        This should be an array of shape [x,y] where x is the number of ROIs and y the corresponding values of F intensity
        
    filePathiscell : string
        The path of where the iscell file from Suite2P is located.
        iscell should be an array of shape [x,y] where x is the number of ROIs and y is the classification confidence
        (values are boolean, 0 for not a cell, 1 for cell)
        cells is a 1D array [x] with the identify of the ROIs classified as cells in iscell

    Returns
    -------
    F_cells : array of float32 
        array of shape [x,y] where x is the same as the one in cells and y contains the corresponding F intensities

    """
    iscell = np.load(filePathiscell, allow_pickle=True)
    F = np.load(filePathF, allow_pickle=True)
    cells = np.where(iscell == 1)[0]
    F_cells = F[cells,:]
    
    return F_cells

#%%Liad's functions slightly adapted

#code from Liad, returns the metadata, remember to change the number of channels
def GetNidaqChannels(niDaqFilePath, numChannels):
    """
    

    Parameters
    ----------
    niDaqFilePath : string
        the path of the nidaq file.
    numChannels : int, optional
        Number of channels in the file. The default is 7.

    Returns
    -------
    niDaq : matrix
        the matrix of the niDaq signals [time X channels]

    """
    niDaq = np.fromfile(niDaqFilePath, dtype= np.float64)
    niDaq = np.reshape(niDaq,(int(len(niDaq)/numChannels),numChannels))
    return niDaq

def AssignFrameTime(frameClock,th = 0.5,plot=False):
    """
    The function assigns a time in ms to a frame time.
    
    Parameters:
    frameClock: the signal from the nidaq of the frame clock
    th : the threshold for the tick peaks, default : 3, which seems to work 
    plot: plot to inspect, default = False
    
    returns frameTimes (ms)
    """
    #Frame times
    # pkTimes,_ = sp.signal.find_peaks(-frameClock,threshold=th)
    # pkTimes = np.where(frameClock<th)[0]
    # fdif = np.diff(pkTimes)
    # longFrame = np.where(fdif==1)[0]
    # pkTimes = np.delete(pkTimes,longFrame)
    # recordingTimes = np.arange(0,len(frameClock),0.001)
    # frameTimes = recordingTimes[pkTimes]
    
    # threshold = 0.5
    pkTimes = np.where(np.diff(frameClock > th, prepend=False))[0]    
    # pkTimes = np.where(np.diff(np.array(frameClock > 0).astype(int),prepend=False)>0)[0]
       
    
    if (plot):
        f,ax = plt.subplots(1)
        ax.plot(frameClock)
        ax.plot(pkTimes,np.ones(len(pkTimes))*np.min(frameClock),'r*')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')
        
        
    return pkTimes


#function from Liad, detecting photodiode change
def DetectPhotodiodeChanges_old(photodiode,plot=True,lowPass=30,kernel = 101,fs=1000, waitTime=10000):
    """
    The function detects photodiode changes using a 'Schmitt Trigger', that is, by
    detecting the signal going up at an earlier point than the signal going down,
    the signal is filtered and smootehd to prevent nosiy bursts distorting the detection.W
    
    Parameters: 
    photodiode: the signal from the nidaq of the photodiode    
    lowPass: the low pass signal for the photodiode signal, default: 30,
    kernel: the kernel for median filtering, default = 101.
    fs: the frequency of acquisiton, default = 1000
    plot: plot to inspect, default = False   
    waitTime: the delay time until protocol start, default = 5000
    
    returns: st,et (ms) (if acq is 1000 Hz)
    
   ***** WHAT DOES ST, ET STAND FOR???*****
    """    
    
    b,a = sp.signal.butter(1, lowPass, btype='low', fs=fs)
    # sigFilt = photodiode
    sigFilt = sp.signal.filtfilt(b,a,photodiode)
    sigFilt = sp.signal.medfilt(sigFilt,kernel)
   
  
    maxSig = np.max(sigFilt)
    minSig = np.min(sigFilt)
    thresholdU = (maxSig-minSig)*0.2
    thresholdD = (maxSig-minSig)*0.4
    threshold =  (maxSig-minSig)*0.5
    
    # find thesehold crossings
    crossingsU = np.where(np.diff(np.array(sigFilt > thresholdU).astype(int),prepend=False)>0)[0]
    crossingsD = np.where(np.diff(np.array(sigFilt > thresholdD).astype(int),prepend=False)<0)[0]
    # crossingsU = np.delete(crossingsU,np.where(crossingsU<waitTime)[0])     
    # crossingsD = np.delete(crossingsD,np.where(crossingsD<waitTime)[0])   
    crossings = np.sort(np.unique(np.hstack((crossingsU,crossingsD))))
  
    
    if (plot):
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(photodiode,label='photodiode raw')
        ax.plot(sigFilt,label = 'photodiode filtered')        
        ax.plot(crossings,np.ones(len(crossings))*threshold,'g*')  
        ax.hlines([thresholdU],0,len(photodiode),'k')
        ax.hlines([thresholdD],0,len(photodiode),'k')
        # ax.plot(st,np.ones(len(crossingsD))*threshold,'r*')  
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')  
    
    return crossings

def DetectPhotodiodeChanges_new(photodiode,plot=False,kernel = 101,upThreshold = 0.2, downThreshold = 0.4,fs=1000, waitTime=5000):
    """
    The function detects photodiode changes using a 'Schmitt Trigger', that is, by
    detecting the signal going up at an earlier point than the signal going down,
    the signal is filtered and smootehd to prevent nosiy bursts distorting the detection.W
    
    Parameters: 
    photodiode: the signal from the nidaq of the photodiode    
    lowPass: the low pass signal for the photodiode signal, default: 30,
    kernel: the kernel for median filtering, default = 101.
    fs: the frequency of acquisiton, default = 1000
    plot: plot to inspect, default = False   
    waitTime: the delay time until protocol start, default = 5000
    
    returns: diode changes (s) up to the user to decide what on and off mean
    """    
    
    # b,a = sp.signal.butter(1, lowPass, btype='low', fs=fs)
    sigFilt = photodiode
    # sigFilt = sp.signal.filtfilt(b,a,photodiode)
    sigFilt = sp.signal.medfilt(sigFilt,kernel)
   
  
    maxSig = np.max(sigFilt)
    minSig = np.min(sigFilt)
    thresholdU = (maxSig-minSig)*upThreshold
    thresholdD = (maxSig-minSig)*downThreshold
    threshold =  (maxSig-minSig)*0.5
    
    # find thesehold crossings
    crossingsU = np.where(np.diff(np.array(sigFilt > thresholdU).astype(int),prepend=False)>0)[0]
    crossingsD = np.where(np.diff(np.array(sigFilt > thresholdD).astype(int),prepend=False)<0)[0]
    crossingsU = np.delete(crossingsU,np.where(crossingsU<waitTime)[0])     
    crossingsD = np.delete(crossingsD,np.where(crossingsD<waitTime)[0])   
    crossings = np.sort(np.unique(np.hstack((crossingsU,crossingsD))))
  
    
    if (plot):
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(photodiode,label='photodiode raw')
        ax.plot(sigFilt,label = 'photodiode filtered')        
        ax.plot(crossings,np.ones(len(crossings))*threshold,'g*')  
        ax.hlines([thresholdU],0,len(photodiode),'k')
        ax.hlines([thresholdD],0,len(photodiode),'k')
        # ax.plot(st,np.ones(len(crossingsD))*threshold,'r*')  
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')    

    return crossings


def GetStimulusInfo(filePath,props):
    """
    

    Parameters
    ----------
    filePath : str
        the path of the log file.
    props : array-like
        the names of the properties to extract.

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the props and their values.

    """
    
      

    StimProperties  = []
    
    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    
            
        for row in reader:
            a = []
            
            for p in range(len(props)):
                # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
                m = re.findall(props[p]+'=([a-zA-Z0-9_.-]*)', row[np.min([len(row)-1,p])])
                if (len(m)>0):
                    a.append(m[0])            
            if (len(a)>0):
                stimProps = {}
                for p in range(len(props)):
                    stimProps[props[p]] = a[p]
                StimProperties.append(stimProps)
    return StimProperties


def AlignStim(signal, time, eventTimes, window,timeUnit=1,timeLimit=1):
    aligned = [];
    t = [];
    dt = np.median(np.diff(time,axis=0))
    if (timeUnit==1):
        w = np.rint(window / dt).astype(int)
    else:
        w = window.astype(int)
    maxDur = signal.shape[0]
    if (window.shape[0] == 1): # constant window
        mini = np.min(w[:,0]);
        maxi = np.max(w[:,1]);
        tmp = np.array(range(mini,maxi));
        w = np.tile(w,((eventTimes.shape[0],1)))
    else:
        if (window.shape[0] != eventTimes.shape[0]):
            print('No. events and windows have to be the same!')
            return 
        else:
            mini = np.min(w[:,0]);
            maxi = np.max(w[:,1]);
            tmp = range(mini,maxi); 
    t = tmp * dt;
    aligned = np.zeros((t.shape[0],eventTimes.shape[0],signal.shape[1]))
    for ev in range(eventTimes.shape[0]):
    #     evInd = find(time > eventTimes(ev), 1);
        
        wst = w[ev,0]
        wet = w[ev,1]
        
        evInd = np.where(time>=eventTimes[ev])[0]
        if (len(evInd)==0): 
            continue
        else :
            # None
            # if dist is bigger than one second stop
            if (np.any((time[evInd[0]]-eventTimes[ev])>timeLimit)):
                continue
            
        st = evInd[0]+ wst #get start
        et = evInd[0] + wet  #get end        
        
        alignRange = np.array(range(np.where(tmp==wst)[0][0],np.where(tmp==wet-1)[0][0]+1))
        
       
        sigRange = np.array(range(st,et))
       
        valid = np.where((sigRange>=0) & (sigRange<maxDur))[0]
      
        aligned[alignRange[valid],ev,:] = signal[sigRange[valid],:];
    return aligned, t

#def DetectWheelMove(moveA,moveB,rev_res = 1024, total_track = 598.47,plot=True):
    """
    The function detects the wheel movement. 
    At the moment uses only moveA.    
    
    Parameters: 
    moveA,moveB: the first and second channel of the rotary encoder
    rev_res: the rotary encoder resoution, default =1024
    total_track: the total length of the track, default = 598.47 (mm)
    kernel: the kernel for median filtering, default = 101.
    
    plot: plot to inspect, default = False   
    
    returns: distance
    """
    
    
    # make sure all is between 1 and 0
    moveA /= np.max(moveA)
    moveA -= np.min(moveA)
    moveB /= np.max(moveB)
    moveB -= np.min(moveB)
    
    # detect A move
    ADiff = np.diff(moveA)
    Ast = np.where(ADiff >0.5)[0]
    Aet = np.where(ADiff <-0.5)[0]
    
    # detect B move
    BDiff = np.diff(moveB)
    Bst = np.where(BDiff >0.5)[0]
    Bet = np.where(BDiff <-0.5)[0]
    
    #Correct possible problems for end of recording
    if (len(Ast)>len(Aet)):
        Aet = np.hstack((Aet,[len(moveA)]))
    elif (len(Ast)<len(Aet)):
        Ast = np.hstack(([0],Ast))   
    
    
    dist_per_move = total_track/rev_res
    
    # Make into distance
    track = np.zeros(len(moveA))
    track[Ast] = dist_per_move
    
    distance = np.cumsum(track)
        
    if (plot):
        f,ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(moveA)
        # ax.plot(np.abs(ADiff))
        ax[0].plot(Ast,np.ones(len(Ast)),'k*')
        ax[0].plot(Aet,np.ones(len(Aet)),'r*')
        ax[0].set_xlabel('time (ms)')
        ax[0].set_ylabel('Amplitude (V)')
        
        ax[1].plot(distance)
        ax[1].set_xlabel('time (ms)')
        ax[1].set_ylabel('distance (mm)')
        
        ax[2].plot(track)
        ax[2].set_xlabel('time (ms)')
        ax[2].set_ylabel('Move')
    
    # movFirst = Amoves>Bmoves
    
    return distance

def running_info(filePath, th = 3, plot=False):
    with open(filePath) as file_name:
        csvChannels = np.loadtxt(file_name, delimiter=",")
    arduinoTime = csvChannels[:,-1]
    arduinoTimeDiff = np.diff(arduinoTime,prepend=True)
    normalTimeDiff = np.where(arduinoTimeDiff>-100)[0]
    csvChannels = csvChannels[normalTimeDiff,:]
    # convert time to second (always in ms)
    arduinoTime = csvChannels[:,-1]/1000 
    # Start arduino time at zero
    arduinoTime-=arduinoTime[0]
    csvChannels = csvChannels[:,:-1]
    numChannels = csvChannels.shape[1]
    if (plot):
        f,ax = plt.subplots(numChannels,sharex=True)
        for i in range(numChannels):
            ax[i].plot(arduinoTime,csvChannels[:,i])
            
    
    return csvChannels,arduinoTime


def DetectWheelMove(moveA,moveB,timestamps,rev_res = 1024, total_track = 59.847, plot=False):
    """
    The function detects the wheel movement. 
    At the moment uses only moveA.   
    [[ALtered the minimum from 0 to 5 because of the data from 04/08/22 -M]]
    
    Parameters: 
    moveA,moveB: the first and second channel of the rotary encoder
    rev_res: the rotary encoder resoution, default =1024
    total_track: the total length of the track, default = 59.847 (cm)
    kernel: the kernel for median filtering, default = 101.
    
    plot: plot to inspect, default = False   
    
    returns: velocity[cm/s], distance [cm]
    """
    #introducing thresholoding in case the non movement values are not 0, 5 was the biggest number for now
    
    th_index = moveA<5
    moveA[th_index] = 0
    th_index = moveB<5
    moveB[th_index] = 0
    moveA = np.round(moveA).astype(bool)
    moveB = np.round(moveB).astype(bool)
    counterA = np.zeros(len(moveA))
    counterB = np.zeros(len(moveB))
    
    # detect A move
    risingEdgeA = np.where(np.diff(moveA>0,prepend=True))[0]
    risingEdgeA = risingEdgeA[moveA[risingEdgeA]==1]
    risingEdgeA_B = moveB[risingEdgeA]
    counterA[risingEdgeA[risingEdgeA_B==0]]=1
    counterA[risingEdgeA[risingEdgeA_B==1]]=-1    
    

    
    # detect B move
    risingEdgeB = np.where(np.diff(moveB>0,prepend=True))[0]#np.diff(moveB)
    risingEdgeB = risingEdgeB[moveB[risingEdgeB]==1]
    risingEdgeB_A = moveB[risingEdgeB]
    counterA[risingEdgeB[risingEdgeB_A==0]]=-1
    counterA[risingEdgeB[risingEdgeB_A==1]]=1    
    


    dist_per_move = total_track/rev_res
    
    instDist = counterA*dist_per_move
    distance = np.cumsum(instDist)
    
    averagingTime = int(np.round(1/np.median(np.diff(timestamps))))
    sumKernel = np.ones(averagingTime)
    tsKernel = np.zeros(averagingTime)
    tsKernel[0]=1
    tsKernel[-1]=-1
    
    # take window sum and convert to cm
    distWindow = np.convolve(instDist,sumKernel,'same')
    # count time elapsed
    timeElapsed = np.convolve(timestamps,tsKernel,'same')
    
    velocity = distWindow/timeElapsed
    # if (plot):
    #     f,ax = plt.subplots(3,1,sharex=True)
    #     ax[0].plot(moveA)
    #     # ax.plot(np.abs(ADiff))
    #     ax[0].plot(Ast,np.ones(len(Ast)),'k*')
    #     ax[0].plot(Aet,np.ones(len(Aet)),'r*')
    #     ax[0].set_xlabel('time (ms)')
    #     ax[0].set_ylabel('Amplitude (V)')
        
    #     ax[1].plot(distance)
    #     ax[1].set_xlabel('time (ms)')
    #     ax[1].set_ylabel('distance (mm)')
        
    #     ax[2].plot(track)
    #     ax[2].set_xlabel('time (ms)')
    #     ax[2].set_ylabel('Move')
    
    # movFirst = Amoves>Bmoves
    
    return velocity, distance

def Get_Stim_Identity(log, reps, protocol_type, types_of_stim):
    """
    

    Parameters
    ----------
    log : array 
        contains the log of stimuli, assumes the order of the columns is "Ori", "SFreq", "TFreq", "Contrast".
    reps : integer
        how many times a stimulus was repeated.
    protocol_type : string
        DESCRIPTION. The options are :
            - "simple" which refers to the protocol which only shows 12 types of orietnations
            - "TFreq": protocol with different temp frequencies
            - "SFreq": protocol with different spatial frequencies
            = "Contrast": protocol with different contrasts.
    types_of_stim : integer
        DESCRIPTION. 
        Refers to the different types of stimuli shown. 
        Assumes that for "simple", types of stim is 12 becuase 12 different orientations are shown.
        For all the others, it is assumed to be 24 because there are 4 different orientartions and 6 different variations of parameters
        

    Returns
    -------
    an array of shape (types_of_stim, reps) if protocol was "simple"
    an array of shape(4, reps, 6) for all other protocols (if 4 different orientations and 6 different other parameters).

    """



#the angles of the stim
#in the case of 20 iterations, given that for simple gratings protocol 12 orientations are shown, the total stimuli shown is 240
    
    if types_of_stim == 12:
            angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
            TFreq =np.array([2])
            SFreq = np.array([0.08])
            contrast = np.array([1])
        #for other gratings protocols such as temp freq etc, this number should be double
    elif types_of_stim == 24:
                angles = np.array([0, 90, 180, 270])
                
                TFreq = np.array([0.5, 1, 2, 4, 8, 16]) 
                SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
                contrast = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])
    #what each angle means
    # 0 degrees is vertical to the left, 
    #90 is horizontal down, 
    #180 is vertical to the right and 
    #270 is horizontal up
    #with these 4 orientations can test orientation and direction selectivity
    #reps = how many repetitions of the same stim we have


#getting a 3D array with shape(orientation, repeats, TFreq/SFreq)

    #all_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
    #all_SFreq = np.zeros((angles.shape[0], reps, SFreq.shape[0])).astype(int)
    all_parameters = np.zeros((angles.shape[0], TFreq.shape[0], reps)).astype(int)
    #all_oris = np.zeros((angles.shape[0], reps)).astype(int)
    
    for angle in range(angles.shape[0]):
        
            
        if protocol_type == "TFreq":
            for freq in range(TFreq.shape[0]):
                specific_TF = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq]) & (log[:,3] == 1)) [0]
                all_parameters[angle, freq, :] = specific_TF
        
        if protocol_type == "SFreq":
            for freq in range(SFreq.shape[0]):
                specific_SF = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq]) & (log[:,3] == 1)) [0]
                all_parameters[angle, freq, :] = specific_SF
        
        if protocol_type == "Contrast":
            for freq in range(TFreq.shape[0]):
                specific_contrast = np.where((log[:,0] == angles[angle]) & (log[:,3] == contrast[freq])) [0]            
                all_parameters[angle, freq, :] = specific_contrast
                
        # if protocol_type == "simple":
        #     specific_P = np.where((log[:,0] == angles[angle])) [0]
        #     all_oris[angle, :] = specific_P
    #return all_oris
    return all_parameters
        
def behaviour_reps (log, types_of_stim,reps, protocol_type, speed, time, stim_on, stim_off):
    
    """
    Takes the stim on values and the stim off values which tell you the exact time
    Then uses this to find the value in the running data which gives you a vector that contains all the values within that period 
    Decides within the loop if 90% of the values are above a certain threshold then assign to each rep a 0 or 1 value 
    Make separate arrays which contain the indices like in all_oris but split into running and rest arrays
    Then can use these values to plot separate parts of the traces (running vs not running)

    Parameters
    ----------
    log : array 
        contains the log of stimuli, assumes the order of the columns is "Ori", "SFreq", "TFreq", "Contrast".
    types_of_stim : integer
        DESCRIPTION: 
        Refers to the different types of stimuli shown. 
        Assumes that for "simple", types of stim is 12 becuase 12 different orientations are shown.
        For all the others, it is assumed to be 24 because there are 4 different orientartions and 6 different variations of parameters
    reps : integer
        how many times a stimulus was repeated.
    protocol_type : string
        The options are :
            - "simple" which refers to the protocol which only shows 12 types of orietnations
            - "TFreq": protocol with different temp frequencies
            - "SFreq": protocol with different spatial frequencies
            = "contrast": protocol with different contrasts.
    speed : 1D array
        the speed throughout the whole experiment.
    time : 1D array
        The corrected time at which the behaviour occured.
    Both of the above are outputs from Liad's function "DetectWheelMove" and duinoDelayCompensation
    stim_on : 1D array
        from photodiode, time at which stimulus appears.
    stim_off : 1D array
        same as above but when stim disappears.

    Returns
    -------
    two arrays of shape (types_of_stim, reps) if protocol was "simple" 
    two arrays of shape(4, reps, 6) for all other protocols (if 4 different orientations and 6 different other parameters)
    (one for running trials, one for rest trials)
    """
         
    stim_on_round = np.around(stim_on, decimals = 2)
    stim_off_round = np.around(stim_off, decimals = 2)





    speed_time = np.stack((time, speed)).T
   
    for rep in range(stim_on.shape[0]-1):
            start = np.where(stim_on_round[rep] == speed_time[:,0])[0]
            stop = np.where(stim_off_round[rep] == speed_time[:,0])[0]
            interval = speed_time[start[0]:stop[0], 1]
            running_bool = np.argwhere(interval>1)
            plt.plot(interval)
            if running_bool.shape[0]/interval.shape[0]>0.9:
                a = 1
            else:
                a = 0
            if a ==1:
                #now appending the final column of the log with 1 if the above turns out true
                    log[rep,4] = 1



    if types_of_stim == 12:
        angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
        TFreq =np.array([2])
        SFreq = np.array([0.08])
        contrast = np.array([1])
        #for other gratings protocols such as temp freq etc, this number should be double
    elif types_of_stim == 24:
        angles = np.array([0, 90, 180, 270])
       
        TFreq = np.array([0.5, 1, 2, 4, 8, 16]) 
        SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
        contrast = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])
    
   
    """
    for running
    """      
    #running = np.ones((4, 6,  30))*np.nan
    running = []

    #creates a list of arrays by looking in the log file and sorting the indices based on the desired angles, freq 
    #and if there is a 0 or a 1 in the final column 
    for angle in range(angles.shape[0]):
                if protocol_type == "SFreq":
                    for freq in range(TFreq.shape[0]):
                        
                        specific_SF_r = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq]) & (log[:,3] == 1) & (log[:,4] ==1)) [0]
                        #running[angle, freq,:] = specific_SF_r
                        running.append(specific_SF_r)
                        
                if protocol_type == "TFreq":
                    for freq in range(TFreq.shape[0]):
                        
                        specific_TF_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq]) & (log[:,3] == 1) & (log[:,4] ==1)) [0]
                        running.append(specific_TF_r)
                        #running[angle, freq,:] = specific_TF_r
                        
                if protocol_type == "Contrast":
                    for freq in range(TFreq.shape[0]):
                        specific_contrast_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq]) & (log[:,4] ==1)) [0]
                        running.append(specific_contrast_r)
                        #running[angle, freq, :] = specific_contrast_r
                elif protocol_type == "simple": 
                    
                    specific_P_r = np.where((log[:,0] == angles[angle]) & (log[:,4] ==1)) [0]
                    
                    running.append(specific_P_r)
                    
    
        
    """
    for rest
    """      
    rest = []
        
    for angle in range(angles.shape[0]):
            if protocol_type == "SFreq":
                for freq in range(TFreq.shape[0]):
                    
                    specific_SF_re = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq]) & (log[:,4] ==0)) [0]
                    rest.append(specific_SF_re)
                    
            if protocol_type == "TFreq":
                for freq in range(TFreq.shape[0]):
                    
                    specific_TF_re = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq]) & (log[:,4] ==0)) [0]
                    rest.append(specific_TF_re)
                    
            if protocol_type == "Contrast":
                for freq in range(TFreq.shape[0]):
                    specific_contrast_re = np.where((log[:,0] == angles[angle]) & (log[:,3] == contrast[freq]) & (log[:,4] ==0)) [0]
                    rest.append(specific_contrast_re)
            elif protocol_type == "simple": 
                
                specific_P_re = np.where((log[:,0] == angles[angle]) & (log[:,4] ==0)) [0]
                
                rest.append(specific_P_re)
    return running, rest  