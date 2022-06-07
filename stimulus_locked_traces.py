# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:18:13 2022

@author: maria
"""

"""
Plan for stimulus locked trace:
    1. get the time at which photodiode is going from on to off, see function below
    2. check frame clock and see at which frame number we had a stimulus appear
    3. use this frame number to check the F trace and align these two
    ! also need to remember that the traces are at 6Hz so need to convert both to a common unit (seconds?) before aligning stuff
    4. take for ex 50 frames before and 50 frames after the stim and plot this
    5. now check the csv file with the stim identity info
    6. group traces for one type of stimulus, then average these responses for the response to same stim for same neuron
    (use a general linear model??)
    
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import butter,filtfilt,medfilt
path= 'Z:/RawData/Hedes/2022-03-23/1/NiDaqInput0.bin'

def GetMetadataChannels(niDaqFilePath, numChannels = 7):
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


def DetectPhotodiodeChanges(photodiode,plot=False,lowPass=30,kernel = 101,fs=1000, waitTime=5000):
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
    thresholdD = (maxSig-minSig)*0.8
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

meta = GetMetadataChannels(path, numChannels=4)
#plotting metadata
tmeta= meta.T
fig, axs = plt.subplots(4, squeeze=True)
axs[0].plot(tmeta[0, 14500:15000])
axs[0].title.set_text("Photodiode")
axs[1].plot(tmeta[1, 14500:15000])
axs[1].title.set_text("Frame Clock")
axs[2].plot(tmeta[2, 14500:15000])
axs[2].title.set_text("Pockel feedback")
axs[3].plot(tmeta[3, 14500:15000])
axs[3].title.set_text("Piezo")

for ax in axs.flat:
    ax.label_outer()
    


"""
gives when the photodiode went from on to off, so at the onset of the stimulus?
"""
# photodiode = meta[:,0]
# photodiode_change = DetectPhotodiodeChanges(photodiode,plot=False,lowPass=30,kernel = 101,fs=1000, waitTime=5000)

"""
getting the frame clock data and checking which number corresponds to the photodiode change, 
although probably enough to just use the photodiode since they are aligned well (onset of stim coincides with frame onset)
"""
frame_clock= meta[:,1]
plt.plot(frame_clock)


"""
getting the signal, for now using the raw F
"""
animal=  'Hedes'
date= '2022-03-23'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_number= '1'

filePathF='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//F.npy'
signal= np.load(filePathF, allow_pickle=True)

""" need to make sure frames and photdiode change times have the same unit (seconds)?"""
# tmeta= meta.T
# photodiode1= tmeta[0]
# crossings= photodiode(photodiode1)

#Liad's code for aligning stim
# def AlignStim(signal, time, eventTimes, window,timeUnit=1,timeLimit=1):
#     aligned = [];
#     t = [];
#     dt = np.median(np.diff(time,axis=0))
#     if (timeUnit==1):
#         w = np.rint(window / dt).astype(int)
#     else:
#         w = window.astype(int)
#     maxDur = signal.shape[0]
#     if (window.shape[0] == 1): # constant window
#         mini = np.min(w[:,0]);
#         maxi = np.max(w[:,1]);
#         tmp = np.array(range(mini,maxi));
#         w = np.tile(w,((eventTimes.shape[0],1)))
#     else:
#         if (window.shape[0] != eventTimes.shape[0]):
#             print('No. events and windows have to be the same!')
#             return 
#         else:
#             mini = np.min(w[:,0]);
#             maxi = np.max(w[:,1]);
#             tmp = range(mini,maxi); 
#     t = tmp * dt;
#     aligned = np.zeros((t.shape[0],eventTimes.shape[0],signal.shape[1]))
#     for ev in range(eventTimes.shape[0]):
#     #     evInd = find(time > eventTimes(ev), 1);
        
#         wst = w[ev,0]
#         wet = w[ev,1]
        
#         evInd = np.where(time>=eventTimes[ev])[0]
#         if (len(evInd)==0): 
#             continue
#         else :
#             # None
#             # if dist is bigger than one second stop
#             if (np.any((time[evInd[0]]-eventTimes[ev])>timeLimit)):
#                 continue
            
#         st = evInd[0]+ wst #get start
#         et = evInd[0] + wet  #get end        
        
#         alignRange = np.array(range(np.where(tmp==wst)[0][0],np.where(tmp==wet-1)[0][0]+1))
        
       
#         sigRange = np.array(range(st,et))
       
#         valid = np.where((sigRange>=0) & (sigRange<maxDur))[0]
      
#         aligned[alignRange[valid],ev,:] = signal[sigRange[valid],:];
#     return aligned, t