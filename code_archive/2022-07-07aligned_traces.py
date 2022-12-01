# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:54:19 2022

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



#function from Liad, detecting photodiode change
def DetectPhotodiodeChanges(photodiode,plot=False,lowPass=30,kernel = 101,fs=1000, waitTime=10000):
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
savePath = 'D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//cell'+str(cell)+'.png'
    
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


#getting the F trace of cells (and not ROIs not classified as cells) using a function I wrote
signal_cells = getcells(filePathF= filePathF, filePathiscell= filePathiscell)

#below code just to check the F for all ROIs
# iscell = np.load(filePathiscell, allow_pickle=True)
# F = np.load(filePathF, allow_pickle=True)
# cells = np.where(iscell == 1)[0]


#getting the fluorescence for the first experiment
first_exp_F = signal_cells[:, 0:exp1]

# to practice will work with one cell for now from one experiment
#cell = 80
F_onecell = signal[cell, 0:exp1]



#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = GetMetadataChannels(filePathmeta, numChannels=5)

#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]
#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = DetectPhotodiodeChanges(photodiode,plot= False,lowPass=30,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
stim_on = photodiode_change[::2]

#getting the photodiode times in the same unit as frames
stim_times = stim_on/1000*frame_rate

#getting stimulus identity
Log_list = GetStimulusInfo (filePathlog, props = ["Ori", "SFreq"])
#converting the list of dictionaries into an array and adding the time of the stimulus
log = pd.DataFrame(Log_list).values
stim_on_df = pd.DataFrame(stim_times).values
stim_times_allP = np.hstack(( log, stim_on_df)).astype(np.float64)
Ori = stim_times_allP[:,0]
SFreq = stim_times_allP[:,1]
# getting the times of when the grating was at certain degrees 
# 0 degrees is vertical to the left, 
#90 is horizontal down, 
#180 is vertical to the right and 
#270 is horizontal up
degree = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
spatial = 0.08
#spatial = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
specific_P1 = np.where(np.logical_and(Ori == degree[1], SFreq == spatial))[0]
#reps = how many repetitionsof the same stim we have
reps = 20
all_P = np.zeros((reps, degree.shape[0])).astype(int)
deg_times = np.zeros((reps, degree.shape[0]))

for i in range(degree.shape[0]):
    specific_P = np.where(np.logical_and(Ori == degree[i], SFreq == spatial))[0]
    all_P[:, i] = specific_P
    #deg_times[:, i] = np.array(stim_on_df[all_P[:,i],:]).astype(int)
    
    

for i in range(degree.shape[0]):
    deg_once = np.array(stim_on_df[all_P[:,i]]).astype(int)
    deg_once = deg_once.reshape(20,)
    deg_times[:, i] = deg_once
    
deg_times = np.array(deg_times).astype(int)

# now have the precise times for one type of stimulus orientation across the iterations
all_stim_int = deg_times.astype(np.int64)[:,0]
#choose the degrees, from 0-11, 0=30, 1= 60 etc
n= 1
one_stim = int(deg_times[0,n])

stim1 = F_onecell[one_stim-frame_rate:one_stim+frame_rate*4]

degree_string= str(degree[n])
#plotting one response
# end =4
# steps = stim1.shape[0]
# range_of_window = np.linspace(-1, end, steps)
# fig ,ax = plt.subplots()
# ax.plot(range_of_window,stim1)
# ax.set_title(degree_string)
#creating an array wihich contains the traces from all the repetitions
#rows: the traces, columns: the repetitions
trace_length = stim1.shape[0]



deg_times1 = np.array(stim_on_df[specific_P1, :]).astype(int)
#for loop which goes through the F trace and adds the fragments specified to an array

#all_reps90 = np.zeros((60, reps))

#90
all_reps90 = np.zeros((trace_length, reps))
for n in range(18):
    stim = int(deg_times[n,2])
    before = stim-frame_rate
    after = stim+frame_rate*4
    all_reps90[:,n] = F_onecell[before:after]


mean_response90 = all_reps90.mean(axis=0)

end =3
steps = mean_response90.shape[0]
range_of_window = np.linspace(-1, end, steps)
fig,ax = plt.subplots()
for i in range(all_reps90.shape[0]):
    ax.plot(range_of_window, all_reps90[i], "lightgray", linewidth = 2)
ax.plot(range_of_window, mean_response90, "black", linewidth = 4)
ax.set_title(str(90))




# #180
# all_reps180 = np.zeros((trace_length, reps))
# for n in range(18):
#     stim = int(deg_times[n,6])
#     before = stim-frame_rate
#     after = stim+frame_rate*4
#     all_reps180[:,n] = F_onecell[before:after]
    
# mean_response180 = all_reps180.mean(axis=0)
    
# #270
# all_reps270 = np.zeros((trace_length, reps))
# for n in range(18):
#     stim = int(deg_times[n,8])
#     before = stim-frame_rate
#     after = stim+frame_rate*4
#     all_reps270[:,n] = F_onecell[before:after]
    
# mean_response270 = all_reps270.mean(axis=0)

# #360
# all_reps360 = np.zeros((trace_length, reps))
# for n in range(18):
#     stim = int(deg_times[n,10])
#     before = stim-frame_rate
#     after = stim+frame_rate*4
#     all_reps360[:,n] = F_onecell[before:after]

# mean_response360 = all_reps360.mean(axis=0)

# all_degrees = np.stack((all_reps90, all_reps180, all_reps270, all_reps360), axis = 0)
# #plotting the mean response with std
# # mean_response = all_reps.mean(axis=0)
# # std_response = all_reps.std(axis=0)
# end =3
# steps = mean_response90.shape[0]
# range_of_window = np.linspace(-1, end, steps)

# fig, axs = plt.subplots(2,2)
# for i in range(all_reps90.shape[0]):
#     axs[0,0].plot(range_of_window, all_reps90[i], "lightgray", linewidth = 2)
    
# axs[0,0].plot(range_of_window, mean_response90, "black", linewidth = 4)
# axs[0,0].set_title(str(90))

# for i in range(all_reps90.shape[0]):
#     axs[0,1].plot(range_of_window, all_reps180[i], "lightgray", linewidth = 2)
    
# axs[0,1].plot(range_of_window, mean_response180, "black", linewidth = 4)
# axs[0,1].set_title(str(180))

# for i in range(all_reps90.shape[0]):
#     axs[0,1].plot(range_of_window, all_reps270[i], "lightgray", linewidth = 2)
    
# axs[1,0].plot(range_of_window, mean_response270, "black", linewidth = 4)
# axs[1,0].set_title(str(270))

# for i in range(all_reps90.shape[0]):
#     axs[1,1].plot(range_of_window, all_reps360[i], "lightgray", linewidth = 2)
    
# axs[1,1].plot(range_of_window, mean_response360, "black", linewidth = 4)
# axs[1,1].set_title(str(360))
# # axs.plot(range_of_window, mean_response-std_response, "royalblue")
# # axs.plot(range_of_window, mean_response+std_response, "royalblue")
# plt.xlabel('Time(s)', fontsize= 16)
# #plt.ylabel('F intensity', fontsize= 16)
# plt.savefig(savePath)
    

"""
above for loop must have an extra dimension for all 12 degrees:
    for now have a 2D array of shape (75,20), will need a 3D array of shape (12,75,20)
"""
all_reps_all = np.zeros((degree.shape[0], trace_length, reps))
all_reps_one = np.zeros((trace_length, reps))
#all_reps_all=[]

#m=2


# for n in range(deg_times.shape[0]):
#     stim = int(deg_times[n,m])
#     before = stim-frame_rate
#     after = stim+frame_rate*4
#     all_reps_one[:,n] = F_onecell[before:after]
#     all_reps_all = np.stack((all_reps_one), axis = 0)
        
# for m in range(deg_times.shape[1]):
#     for n in range(deg_times.shape[0]):
#         stim = int(deg_times[n,m])
#         before = stim-frame_rate
#         after = stim+frame_rate*4
#         all_reps_one[:,n] = F_onecell[before:after]
#         all_reps_all[m,:,:] = np.stack((all_reps_one), axis = 0)
    

    
# #plan for more plotting:
# #for each cell need a plot with n quadrants (n being the number of orientations shown)
# #plot the mean and all iterations in each quadrant for each orientation
# #add the identity of the stim above each plot
# #to make this faster, create array which contains all the 
# fig,axs = plt.subplots(2,2)
# axs[0,0].plot(range_of_window, mean_response, "black", linewidth = 4)
# axs[0,1].plot


"""
Bugs to solve:
    - making the embedded for loop work to give me a 3d array for all orientations, all iterations for stim duration
    - making sure the last iterations don't give errors
    - in the end having a 4D (?) array with all orientation,etc for all cells! (think if this is a good idea)
    - automatically saving plots (one plot per cell with all the orientations)
    -make sure the average trace doesn't go to zero at the end always??
"""