# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:22:51 2022

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


#code from Liad, returns the metadata, remember to change the number of channels
def GetMetadataChannels(niDaqFilePath, numChannels):
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
def DetectPhotodiodeChanges(photodiode,plot=True,lowPass=30,kernel = 101,fs=1000, waitTime=10000):
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




#getting the signal, for now using the raw F

animal=  'Glaucus'
date= '2022-06-30'
#note: if experiment type not known, put 'suite2p' instead
experiment = '1'
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '0'
log_number = '0'
plane_number = '1'
#IMPORTANT: SPECIFY THE FRAME RATE
frame_rate = 15
#the total amount of seconds to plot
seconds = 5
#specify the cell for single cell plotting
cell = 75
res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
    
#loading ops file to get length of first experiment
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()

#loading ops file to get length of first experiment
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()

#printing data path to know which data was analysed
key_list = list(ops.values())
print(key_list[88])
print("frames per folder:",ops["frames_per_folder"])
exp= np.array(ops["frames_per_folder"])
#getting the first experiment, this is the length of the experiment in frames
exp1 = int(exp[0])
"""
Step 1: getting the cell traces I need, here the traces for the first experiment
"""

#getting the F trace of cells (and not ROIs not classified as cells) using a function I wrote
signal_cells = getcells(filePathF= filePathF, filePathiscell= filePathiscell).T

#
exp= np.array(ops["frames_per_folder"])
#getting the first experiment, this is the length of the experiment in frames
exp1 = int(exp[0])

#getting the fluorescence for the first experiment
first_exp_F = signal_cells[:, 0:exp1]

# to practice will work with one cell for now from one experiment

F_onecell = signal[cell, 0:exp1]
# fig,ax = plt.subplots()
# plt.plot(F_onecell) 

"""
Step 2: getting the times of the stimuli
"""

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = GetMetadataChannels(filePathmeta, numChannels=5)

#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = DetectPhotodiodeChanges(photodiode,plot= True,lowPass=30,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset

stim_on = photodiode_change[::2]
stim_on_minus1 = stim_on[0:-1]
# fig,ax = plt.subplots()
# ax.plot(stim_on) 
#getting the photodiode times in the same unit as frames
stim_times = stim_on/1000*frame_rate

#plotting the stim times with an example trace to visualise what I actually need
# fig,ax = plt.subplots(1)
# ax.scatter(x=stim_times, y =np.ones((stim_times.shape[0])), c="green")
# ax.plot(F_onecell, c= "blue")

#from this I simply need the traces one second before and 4 secs after

"""
Step 3: getting the identity of the stimuli
"""
#getting stimulus identity
Log_list = GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])
#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)
#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)
log_Ori = log[:,0].reshape(240,)
#for now just checking orientations
stim_times_Ori_H = np.stack(( log_Ori, stim_times)).astype(np.float64)
stim_times_Ori = stim_times_Ori_H.T

#the angles of the stim
angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
#what each angle means
# 0 degrees is vertical to the left, 
#90 is horizontal down, 
#180 is vertical to the right and 
#270 is horizontal up
#with these 4 orientations can test prientation and direction selectivity
#reps = how many repetitions of the same stim we have
reps = 20
all_P = np.zeros((reps, angles.shape[0])).astype(int)
angle_times = np.zeros((reps, angles.shape[0]))

#getting th indices for each repetition of the specific gratings
for i in range(angles.shape[0]):
    specific_P = np.where(stim_times_Ori == angles[i])[0]
    all_P[:, i] = specific_P
    
#getting the actual times in frames for each repetition and each angle
for i in range(angles.shape[0]):
    angle_once = np.array(stim_times[all_P[:,i]]).astype(int)
    angle_once = angle_once.reshape(20,)
    angle_times[:, i] = angle_once

#checking for one stimulus
angle = 5
repetition = 7
one_stim = int(angle_times[repetition,angle])
one_aligned = F_onecell[one_stim-frame_rate:one_stim+frame_rate*(seconds-1)]

#plotting the one trace
end =4
steps = frame_rate*seconds
range_of_window = np.linspace(-1, end, steps)

# fig,ax = plt.subplots(1)
# ax.plot(range_of_window,one_aligned)

#plotting all reps for the one angle
# fig,ax = plt.subplots(1)
# all_reps = np.zeros((steps,reps))
# for rep in range(angle_times.shape[0]):
#     one_stim = int(angle_times[rep,angle])
#     all_reps[:, rep] = F_onecell[one_stim-frame_rate:one_stim+frame_rate*(seconds-1)]
#     ax.plot(range_of_window,all_reps[rep])
    

"""
frame clock
"""

tmeta= meta.T
frame_clock = tmeta[1]
frame_times = AssignFrameTime(frame_clock, plot = False)
# frame_times1 = frame_times[1:]
frame_on = frame_times[::2]
frames_plane1 = frame_on[1::4]

window= np.array([-1, 4]).reshape(1,-1)

# #Liad's code for aligning stim
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


window= np.array([-1000, 4000]).reshape(1,-1)
aligned_all = AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)

# # #window is an array like (-0.5s to -.5 secs)

aligned = aligned_all[0]
mean_response = np.zeros((aligned.shape[0], reps, aligned.shape[2]))
#fig,ax = plt.subplots(all_P.shape[1])
#i = 0
# for cells in range(aligned.shape[2]):
#                    for angle in range(all_P.shape[0]):
#                        for reps  in range(all_P.shape[1]):
#                            aligned_one_angle = aligned[:,all_P[reps,angle] , 0]
#                            #ax.plot(aligned[:,all_P[reps,angle] , 0])
#                            mean_response[:, angle, cells] = aligned_one_angle
#now for one type of stim, just load the all_P to plot them!

cell = 75
savePath = 'D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//final_figures//30-60-120-150//cell'+str(cell)+'.png'
angles_str = ["30","60", "90", "120", "150", "180", "210", "240","270", "300", "330", "360"]
mean90 = aligned[:,all_P[:,0] , cell].mean(axis = 1)
mean180 = aligned[:,all_P[:,1] , cell].mean(axis = 1)
mean270 = aligned[:,all_P[:,3] , cell].mean(axis = 1)
mean360 = aligned[:,all_P[:,4] , cell].mean(axis = 1)
mean_all = np.stack((mean90, mean180, mean270, mean360))


mean = np.zeros((all_P.shape[0], aligned.shape[0]))
for angle in range(angles.shape[0]):
    mean[angle,:] = aligned[:,all_P[:,0] , 0].mean(axis = 1)
end =3
steps = aligned.shape[0]
range_of_window = np.linspace(-1, end, steps)
#plotting mean response for one angle
# fig,ax = plt.subplots()
# ax.plot(range_of_window, mean_all[0])
#plotting all for one angle
# fig,ax = plt.subplots()
# for reps  in range(all_P.shape[0]):
#     ax.plot(range_of_window, aligned[:,all_P[reps,2] , cell], "lightgray" )
#ax.plot(range_of_window, mean_all[0])
#for cell in range(aligned.shape[2]):
fig, axs = plt.subplots(2,2, squeeze=True)
#for angle in range(mean_all.shape[0]):
#for cell in range(aligned.shape[2]):
range_of_window = aligned_all[1]

for reps  in range(all_P.shape[0]):
    axs[0,0].plot(range_of_window, aligned[:,all_P[reps,2] , cell], "lightgray" )
    axs[0,0].plot(range_of_window, mean_all[0], "black")
    axs[0,0].set_title(str(angles_str[0]))
    axs[0,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
    axs[0,1].plot(range_of_window, aligned[:,all_P[reps,5] , cell], "lightgray" )
    axs[0,1].plot(range_of_window, mean_all[1], "black")
    axs[0,1].set_title(str(angles_str[1]))
    axs[0,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
    axs[1,0].plot(range_of_window, aligned[:,all_P[reps,8] , cell], "lightgray" )
    axs[1,0].plot(range_of_window, mean_all[2], "black")
    axs[1,0].set_title(str(angles_str[3]))
    axs[1,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
    axs[1,1].plot(range_of_window, aligned[:,all_P[reps,11] , cell], "lightgray" )
    axs[1,1].plot(range_of_window, mean_all[3], "black")
    axs[1,1].set_title(str(angles_str[4]))
    axs[1,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
    plt.savefig(savePath)

    # axs[0,0].title.set_text(str(degree[0]))
    # axs[1].plot(tmeta[1, start_short:end_short])
    # axs[1].title.set_text("Frame Clock")
    # axs[2].plot(tmeta[2, start_short:end_short])
    # axs[2].title.set_text("Pockel feedback")
    # axs[3].plot(tmeta[3, start_short:end_short])
    # axs[3].title.set_text("Piezo")
    # axs[4].plot(tmeta[3, start_short:end_short])
    # axs[4].title.set_text("Synchronisation signal")
plt.xlabel("Time(s)")
for ax in axs.flat:
    ax.label_outer()