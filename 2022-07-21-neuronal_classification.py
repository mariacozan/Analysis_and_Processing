# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:56:29 2022

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
import functions2022_07_15 as fun

#getting the signal, for now using the raw F

animal=  'Hedes'
date= '2022-07-19'
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

res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
iscell = np.load(filePathiscell, allow_pickle=True)
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
#getting second experiment
exp2 = int(exp[1])
#getting experiment 3
if exp.shape[0] == 3:
    exp3 = int(exp[2])
"""
Step 1: getting the cell traces I need, here the traces for the first experiment
"""

#getting the F trace of cells (and not ROIs not classified as cells) using a function I wrote
signal_cells = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T
#%%
#


#getting the fluorescence for the first experiment
first_exp_F = signal_cells[:, 0:exp1]


# to practice will work with one cell for now from one experiment
cell = 33
F_onecell = signal[cell, 0:exp1]
# fig,ax = plt.subplots()
# plt.plot(F_onecell) 

"""
Step 2: getting the times of the stimuli
"""

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = fun.GetMetadataChannels(filePathmeta, numChannels=5)
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun.DetectPhotodiodeChanges(photodiode,plot= True,lowPass=30,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset

stim_on = photodiode_change[1::2]

# fig,ax = plt.subplots()
# ax.plot(stim_on) 




"""
Step 3: actually aligning the stimuli with the traces (using Liad's function)
"""

tmeta= meta.T
frame_clock = tmeta[1]
frame_times = fun.AssignFrameTime(frame_clock, plot = False)
# frame_times1 = frame_times[1:]
frame_on = frame_times[::2]
frames_plane1 = frame_on[1::4]
frames_plane2 = frame_on[2::4]

#window: specify the range of the window
window= np.array([-1000, 4000]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)


#aligned: thetraces for all the stimuli for all the cells
aligned = aligned_all[0]
#the actual time, usually 1 second before and 4 seconds after stim onset in miliseconds
time = aligned_all[1]

#%%
"""
Step 4: getting the identity of the stimuli
"""
#need to get the log info file extraction to work

#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["LightOn"])
#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)

#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)
#no of stimuli specifes the total amount of stim shown
nr_stimuli = aligned.shape[1]

#%%
#getting one neuron for testing and plotting of a random stimulus:
neuron = 3
one_neuron = aligned[:,:,neuron]
fig,ax = plt.subplots()
ax.plot(time,one_neuron[:,])
ax.axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
