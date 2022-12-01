# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:25:16 2022

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

#printing data path to know which data was analysed
key_list = list(ops.values())
print(key_list[88])
print("frames per folder:",ops["frames_per_folder"])
exp= np.array(ops["frames_per_folder"])
#getting the first experiment, this is the length of the experiment in frames
exp1 = int(exp[0])

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

#loading the F trace of cells into a variable
signal_cells = getcells(filePathF= filePathF, filePathiscell= filePathiscell)

iscell = np.load(filePathiscell, allow_pickle=True)
F = np.load(filePathF, allow_pickle=True)
cells = np.where(iscell == 1)[0]
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

def GetLogEntry(filePath,entryString):
    """
    

    Parameters
    ----------
    filePath : str
        the path of the log file.
    entryString : the string of the entry to look for

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the props and their values.

    """
    
    
    a = []   
    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                    
            m = re.findall('^'+entryString.lower()+'*', row[2].lower())
            if (len(m)>0):
                a.append(row)          
            
    if (len(a)==0):
        return []           
    return np.vstack(a)

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
  
meta = GetMetadataChannels(filePathmeta, numChannels=5)
  


"""
1.Step: get the times when the stimulus appeared so when the photodiode went from on to off
"""
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]
#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = DetectPhotodiodeChanges(photodiode,plot= False,lowPass=30,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
stim_on = photodiode_change[::2]
"""
2.Step: get stim onset time as frame number
output from 1. gives a matrix with the time at which stim. appeared, but this is at a sampling rate of 1000Hz so need to divide values by 1000
"""
#getting the photodiode times in the same unit as frames
stim_times = stim_on/1000*frame_rate

"""
3.Step: obtain the F values 2s before and 2s after the frame specified in 2
"""

#getting the fluorescence for the first experiment
first_exp_F = signal_cells[:, 0:exp1]
   



#creating the window for the plot only so it's shown in seconds
#end:define the duration of stim + 1
# end =3
# steps = 20
# range_of_window = np.arange(-1, end, steps)

stim = 100
stim_str = str(stim)


# for ROI in range(signal_cells.shape[0]):
#     ROI_str= str(ROI)
#     ROI_is = "ROI number "+ROI_str+"."
#     signal_perROI = first_exp_F[ROI, :]
#     one_window_b = int(stim_times[stim]-frame_rate)
#     one_window_f = int(stim_times[stim]+frame_rate)
    
#     F_on_stim = signal_perROI[one_window_b:one_window_f]
#     fig, ax = plt.subplots()
#     ax.plot(range_of_window, F_on_stim)
#     ax.set_xlabel('Time(s)', fontsize= 16)
#     ax.set_ylabel('F intensity', fontsize= 16)
#     max_height = np.max(F_on_stim) +500
#     plt.text(10, max_height, ROI_is)
#     filePath_stim_aligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//'+experiment+'//stim'+stim_str+'ROI'+ROI_str+'.png'
                
#     plt.savefig(filePath_stim_aligned)
    

"""
2022-06-23:
    Now need to append these separate traces into an array/s which contain:
                per cell all the traces aligned to a stimulus
                on top of that all the traces of the cells for all the stimuli (3D array?)
                
    How?
    For each stimulus in stim_times, I need to create an array which contains the fluorescence at for ex 1s before and 2s after for each cell
    so the dimenisions of the array would be (cells, the F trace across 3 secs, the stimuli)
    So need a for loop which goes through each cell (i.e. row) in the first_exp_F array then "slices" each row multiple times at intervals specified in the stim interval array below
    Then it takes the sliced bits and appends them into an array of shape (cells, the F trace across 3 secs, the stimuli)
                
"""
# cells = signal_cells.shape[0]
# #interval refers to how many seconds long the window will be
# interval = 3
# window_size = int(frame_rate*interval*2)
# stimuli = stim_times.shape[0]

# stim_aligned = np.zeros((cells, window_size, stimuli))



# for stim in range(stim_times.shape[0]):
    
#     for cells in range(first_exp_F.shape[0]):
#         before_stim = int(stim_times[stim]-frame_rate*interval)
#         after_stim = int(stim_times[stim]+frame_rate*interval)
#         stim_aligned[:, : , cells] = first_exp_F[:, before_stim:after_stim] 

# #creating an array with the stimulus interval times in case this would be needed?
# stim_before = np.array(stim_times - frame_rate*interval)
# stim_after = np.array(stim_times + frame_rate*interval)
# stim_interval = np.stack((stim_before, stim_after))

#need to 'translate' what this before and after time is in the matrix indices 
#(not as straightforward as getting the times and checking them in the F trace
# also an issue with the above shape of the matrices

"""
2022-07-04:
    to get the log of the stimuli, just need to use Liad's code "GetStimulusInfo"
    
"""
"""
BUG
"""
#%%somewhere here there is a bug (basically the stimulus times array should be the same
# length as the log list but it isn't for some reason)
# getting the times for all the orientations and/or other parameters
Log_list = GetStimulusInfo (filePathlog, props = ["Ori"])
#converting the list of dictionaries into an array and adding the time of the stimulus
log = pd.DataFrame(Log_list).values
stim_on_df = pd.DataFrame(stim_times).values
ori_times = np.hstack(( log, stim_on_df)).astype(np.float64)
#%%
# getting the times of when the grating was at certain degrees 
# 0 degrees is vertical to the left, 
#90 is horizontal down, 
#180 is vertical to the right and 
#270 is horizontal up
#degree = np.array([0, 90, 180, 270])
degree = 90
spatial = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
deg = np.where(ori_times == degree)[0]
#SFreq = np.where(ori_times == spatial)[0]
deg_times = stim_on_df[deg, :]

#deg_times = np.zeros(())
#for loop to append an array with all the times at the different degrees
# for angle in range(degree.shape[0]):
#     deg = np.where(ori_times == degree[angle])[0]
#     deg_times = stim_on_df[deg, :]
# to practice will work with one cell for now from one experiment
cell = 0
F_onecell = signal[cell, 0:exp1]

# now have the precise times of stimuli and one cell F trace
one_stim = int(deg_times[1])
all_stim_int = deg_times.astype(np.int64)[:,0]
stim1 = F_onecell[one_stim-frame_rate:one_stim+frame_rate*3]
test = [1]
# fig, ax = plt.subplots()
# ax.plot(range_of_window, stim1)
# ax.plot(test, '|')
# ax.set_xlabel('Time(s)', fontsize= 16)
# ax.set_ylabel('F intensity', fontsize= 16)

#creating an array wihich contains the traces from all the repetitions
#rows: the traces, columns: the repetitions
rows = stim1.shape[0]
columns = 20

#for loop which goes through the F trace and adds the fragments specified to an array
all_stim = np.zeros((rows, columns))
for n in range(all_stim_int.shape[0]):
    stim = int(deg_times[n])
    all_stim[:, n] = F_onecell[stim-frame_rate:stim+frame_rate*3]
    
mean_response = all_stim.mean(axis=0)
std_response = all_stim.std(axis=0)
end =3
steps = mean_response.shape[0]
range_of_window = np.linspace(-1, end, steps)
# fig, axs = plt.subplots()
# for i in all_stim:
#     axs.plot(range_of_window,i, linestyle = "dashed")
# axs.plot(range_of_window, mean_response, "green", linewidth = 4)
# axs.set_xlabel('Time(s)', fontsize= 16)
# axs.set_ylabel('F intensity', fontsize= 16)
fig, axs = plt.subplots()
axs.plot(range_of_window, mean_response, "blue", linewidth = 4)
axs.plot(range_of_window, mean_response-std_response, "royalblue")
axs.plot(range_of_window, mean_response+std_response, "royalblue")
axs.set_xlabel('Time(s)', fontsize= 16)
axs.set_ylabel('F intensity', fontsize= 16)


# fig, axs = plt.subplots()
# axs.plot(signal[0, 0:exp1])
# for n in range(deg_times.shape[0]):
#     axs.plot(deg_times[0,n], 'o')

    #plt.plot(np.mean(i),"red", linewidth = 4)
    
    
"""
2022-07-06: running speed infomration implementation
"""

def DetectWheelMove(moveA,moveB,rev_res = 1024, total_track = 598.47,plot=False):
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
def running_info(filePath):
    file = open(filePath)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
        
    return rows
running_behaviour = running_info(filePathArduino)
#wheel_movement = DetectWheelMove()