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


"""
getting the signal, for now using the raw F
"""
animal=  'Hedes'
date= '2022-03-23'
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '1'
plane_number = '1'
#IMPORTANT: SPECIFY THE FRAME RATE
frame_rate = 6
filePathF='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//F.npy'
filePathops= 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//ops.npy'
filePathmeta= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//iscell.npy'

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
iscell = np.load(filePathiscell, allow_pickle=True)
cells = np.where(iscell == 1)[0]
signal_cells = getcells(filePathF= filePathF, filePathiscell= filePathiscell)

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

meta = GetMetadataChannels(filePathmeta, numChannels=4)
#plotting metadata
# tmeta= meta.T
# fig, axs = plt.subplots(4, squeeze=True)
# axs[0].plot(tmeta[0, 14500:15000])
# axs[0].title.set_text("Photodiode")
# axs[1].plot(tmeta[1, 14500:15000])
# axs[1].title.set_text("Frame Clock")
# axs[2].plot(tmeta[2, 14500:15000])
# axs[2].title.set_text("Pockel feedback")
# axs[3].plot(tmeta[3, 14500:15000])
# axs[3].title.set_text("Piezo")

# for ax in axs.flat:
#     ax.label_outer()
    


"""
1.Step: get the times when the stimulus appeared so when the photodiode went from on to off
"""
photodiode = meta[:,0]
photodiode_change = DetectPhotodiodeChanges(photodiode,plot=False,lowPass=30,kernel = 101,fs=1000, waitTime=5000)

"""
2.Step: get stim onset time as frame number
output from 1. gives a matrix with the time at which stim. appeared, but this is at a sampling rate of 1000Hz so need to divide values by 1000
"""
#photodiode change= pdc
pdc_secs= photodiode_change/1000
#this value should now be y where (x,y) is the shape of the F matrix

"""
***Next task***: obtain the F values 100ms before and 500ms after the frame specified in 2
"""
# """
# 2. Find the right frame number which correspnds to the time the stimulus came on
# getting the frame clock data and checking which number corresponds to the photodiode change, 
# although probably enough to just use the photodiode since they are aligned well (onset of stim coincides with frame onset)
# """
frame_clock= meta[:,1]
# plt.plot(frame_clock)



""" need to make sure frames and photdiode change times have the same unit (seconds)?
So now need to create a sort of LUT which helps me identify which frame identity corresponds to the seconds --> simply divide the total frame number by the frame rate (6 most of the time)"""

#checking the lemgth of time of the photodiode acq and the experiment length (just first exp though)
length_time = len(photodiode)/1000
#loading ops file to get length of first experiment
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()

#printing data path to know which data was analysed
key_list = list(ops.values())
print(key_list[88])
print("frames per folder:",ops["frames_per_folder"])
exp= np.array(ops["frames_per_folder"])

#experiment number: 0=1; 1=2, etc ((maybe write a function for this))
exp_n = 1
length_frames = exp[exp_n]/frame_rate

if np.rint(length_frames) == np.rint(length_time):
    print("length is the same")
else:
    print("difference is:",abs(length_time-length_frames))
    
"""
Next task: aligning the frames with stim, maybe create a 2D matrix with both?
"""
#getting the photodiode times in the same unit as frames
pdc_frames = photodiode_change/1000*frame_rate

"""2022-06-20:
- use np.where to obtain the F traces 1s before and 1s after the stim onset
- to do this, create a matrix where all the values from pdc_frames are actually intervals
- create a for loop which creates snippets of the big F trace with the intervals being the intervals created from the previous point
"""

#stimulus aligned trace for one cell and 1 stimulus
steps = 2/(2*frame_rate)
range_of_window = np.arange(-1, 1, steps)
signal_exp1 = np.float64(signal[:,0:exp[1]])

#ROI= np.random.choice(cells)

stim = 100
stim_str = str(stim)
#stim_traces = np.zeros()
#for m in range(pdc_frames.shape[0]):
for ROI in range(cells.shape[0]):
    ROI_str= str(ROI)
    ROI_is = "ROI number "+ROI_str+"."
    signal_perROI = signal_exp1[ROI, :]
    one_window_b = int(pdc_frames[stim]-frame_rate)
    one_window_f = int(pdc_frames[stim]+frame_rate)
    
    F_on_stim = signal_perROI[one_window_b:one_window_f]
    fig, ax = plt.subplots()
    ax.plot(range_of_window, F_on_stim)
    ax.set_xlabel('Time(s)', fontsize= 16)
    ax.set_ylabel('F intensity', fontsize= 16)
    max_height = np.max(F_on_stim) +500
    plt.text(10, max_height, ROI_is)
    filePath_stim_aligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//'+experiment+'//stim'+stim_str+'ROI'+ROI_str+'.png'
                
    # plt.savefig(filePath_stim_aligned)
    

"""
2022-06-21:
    Now that the stim-aligned trace can be plotted, need to:
        - generalise it to all stimuli times (currently only for one of them for all cells)
        - append these separate traces into an array/s which contain:
                per cell all the traces aligned to a stimulus
                on top of that all the traces of the cells for all the stimuli (3D array?)
                
"""

# for n in pdc_frames:
    

"""
defining the window for the stimulus
"""
# window= np.array([-1, 1])

# #Liad's code for aligning stim
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

# # #AlignStim(signal= signal, time= frame_clock, eventTimes= photodiode_change, window= window)

# # #window is an array like (-0.5s to -.5 secs)