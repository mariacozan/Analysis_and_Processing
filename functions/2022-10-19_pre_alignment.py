# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:38:09 2022

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
import Analysis_and_Processing.functions as fun
import Data.Bonsai.extract_data as fun_ext
import os
"""
more updated stim aligning protocol, uses more recent functions
"""

#getting the signal, for now using the raw F

animal=  'Giuseppina'
#animal = input("animal name ")
date= '2022-11-03'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
exp_no = int(experiment)
#experiment = input("experiment number(integer only) ")
#experiment_int = int(experiment)
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '2'
log_number = '2'
plane_number = '2'
plane_number_int = int(plane_number)
exp_name = 'SFreq'
reps = 30

#%%loading all suite2p files
#nr_planes = int(input("how many planes were imaged?"))
nr_planes = 4
#types of stim refers to the different combos of stim you can have depending on oris and parameters used (for simple gratings this is 12 and for all others it is 24)
types_of_stim = 24
res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#
#filePathF ='C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
#filePathops = 'C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#


signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
#filePathiscell = 'C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
iscell = np.load(filePathiscell, allow_pickle=True)
#loading ops file to get length of first experiment
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()

#loading ops file to get length of first experiment
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()

#printing data path to know which data was analysed
key_list = list(ops.values())
print("experiments ran through Suite2P", key_list[88])
print("frames per folder:",ops["frames_per_folder"])
exp= np.array(ops["frames_per_folder"])

#%%getting the F trace of cells 
#(and not ROIs not classified as cells) using a function I wrote
signal_cells_all = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T
#%%
if exp_no == 1:
    signal_cells = signal_cells_all[0:exp[exp_no-1], :]
elif exp_no == 2:
    signal_cells = signal_cells_all[exp[exp_no-2]:exp[exp_no-1]+exp[exp_no-2], :]

#%%metadata files
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'

#%%getting stimulus timing
# remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = fun_ext.GetNidaqChannels(filePathmeta)
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun_ext.DetectPhotodiodeChanges(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
print("please choose the relevant experiment below! stim on or off etc")

#%%plotting metadata vs trace
#for neuron in range(aligned.shape[2]):
for neuron in range(0,4):
    kernel = 101
    waitTime = 5000
    upThreshold = 0.2
    downThreshold = 0.4
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
    
    f,ax = plt.subplots(2,sharex= False)
    ax[0].plot(photodiode,label='photodiode raw')
    ax[0].plot(sigFilt,label = 'photodiode filtered')        
    ax[0].plot(crossings,np.ones(len(crossings))*threshold,'g*')  
    ax[0].hlines([thresholdU],0,len(photodiode),'k')
    ax[0].hlines([thresholdD],0,len(photodiode),'k')
            # ax.plot(st,np.ones(len(crossingsD))*threshold,'r*')  
    #ax.legend()
    ax[0].set_xlabel('time (ms)')
    ax[0].set_ylabel('Amplitude (V)') 
    
    ax[1].plot(signal_cells[:, neuron])
    
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ ''+res+'//plane'+plane_number+'//classification//all//cell'+str(neuron)+'.png')

#%%aligning stimulus

stim_on = photodiode_change[0::2]
stim_off = photodiode_change[1::2]

tmeta= meta.T
frame_clock = tmeta[1]
frame_on = fun_ext.AssignFrameTime(frame_clock, plot = False)
# frame_times1 = frame_times[1:]

#frame_on = frame_times[::2]
frames_plane1 = frame_on[plane_number_int::nr_planes]
#frames_plane2 = frame_on[plane_number_int::nr_planes]

#window: specify the range of the window
window= np.array([-1, 4]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells

aligned = aligned_all[0]


aligned_all_off = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_off, window= window,timeLimit=1000)
aligned_off = aligned_all_off[0]


time = aligned_all[1]
time_off = aligned_all_off[1]


#%%
fig, ax = plt.subplots(2, sharex=True, sharey= True)
ax[0].plot(frames_plane1,signal_cells[:,0])
ax[1].plot(frames_plane1,signal_cells[:,1])
# ax[2].plot(frames_plane1,signal_cells[:,2])
#ax.axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
#for event in range(aligned.shape[1]):
    #ev= np.where(frames_plane1>=stim_on[event])[0]
# for a in stim_on:
ax[0].vlines(stim_on, min(signal_cells[:, 2]), max(signal_cells[:,2]),  "red", linestyle="dashed", linewidth = 1)
ax[0].vlines(stim_off, min(signal_cells[:,2]), max(signal_cells[:,2]),  "blue", linestyle="dashed", linewidth = 1)
ax[1].vlines(stim_on, min(signal_cells[:,2]), max(signal_cells[:,2]),  "red", linestyle="dashed", linewidth = 1)
ax[1].vlines(stim_off, min(signal_cells[:,2]), max(signal_cells[:,2]),  "blue", linestyle="dashed", linewidth = 1)
# ax[2].vlines(stim_on, min(signal_cells[:,2]), max(signal_cells[:,2]),  "red", linestyle="dashed", linewidth = 1)
# ax[2].vlines(stim_off, min(signal_cells[:,2]), max(signal_cells[:,2]),  "blue", linestyle="dashed", linewidth = 1)

# import pickle
# pickle.dump(fig, open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_stim_all_cells.fig.pickle', 'wb'))

#%%making  heatmap
fig,axs = plt.subplots()
sns.heatmap(frames_plane1,signal_cells.T,cmap='bone')
axs.vline(stim_on[0], "red", linestyle="dashed", linewidth = 10)
#ax.vlines(stim_off, min(signal_cells[:,0].T), max(signal_cells[:,0].T),  "blue", linestyle="dashed", linewidth = 10)
#%%saving fig in an interactive way
import pickle
pickle.dump(fig, open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_stim.fig.pickle', 'wb'))
#%%loading figure
#only supports pyplot figures (so not seaborn probs)
import pickle
figx = pickle.load(open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_stim.fig.pickle', 'rb'))

figx.show() # Show the figure, edit it, etc.!
#%%
data = figx.axes[0].lines[0].get_data()




#%%saving the aligned traces for post alignment processing
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_gratings_good.npy', aligned)
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_gratings_off.npy', aligned_off)
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//time.npy', time)
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//time_off.npy', time_off)
#%%
#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])

#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)

#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//log.npy', log)

        
#%%stimulus identity as indices for the aligned trace
all_parameters = fun.Get_Stim_Identity(log = log, reps = 30, types_of_stim = 24, protocol_type = str(exp_name))
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_parameters.npy', all_parameters)


#%%behavioural data

#%%metadata files
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
#running data
#%%loading running data
running_behaviour = fun.running_info(filePathArduino, plot = True)
#%%
channels = running_behaviour[0]

forward = channels[:,0]
backward = channels [:,1]
time_stamps = running_behaviour[1]

WheelMovement = fun.DetectWheelMove(forward, backward, timestamps = time_stamps)

speed = WheelMovement[0]
fig,ax = plt.subplots(2)
ax[0].plot(speed)
ax[1].plot(forward)

pickle.dump(fig, open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//running_data.fig.pickle', 'wb'))
#%%
tmeta= meta.T
niTime = np.array(range(0,meta.shape[0]))/1000


#getting sync signal:
syncNiDaq = tmeta[-1]
syncArd = channels[:, -1]

corrected_time = fun_ext.arduinoDelayCompensation(nidaqSync = syncNiDaq ,ardSync = syncArd, niTimes = niTime ,ardTimes = time_stamps)
corrected_time = np.around(corrected_time, decimals = 2)

#%%need to add a column of zeros to log to be able to append a 1 to trials if the trial involved running
aligned1 = aligned[:, 0:720, :]
zero = np.zeros((aligned1.shape[1])).reshape(aligned1.shape[1], )
log = np.column_stack((log, zero))

reps_behaviour = fun.behaviour_reps(log = log, types_of_stim = 24, reps = reps, protocol_type = exp_name, speed = speed, time = corrected_time, stim_on = stim_on, stim_off = stim_off)
#the above function gives the reps for each orientation for running and for rest states

#%%
running = reps_behaviour[0]
running_array = np.array((running), dtype = object)

np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//running.npy', running_array)

rest = reps_behaviour[1]
rest_array = np.array((rest), dtype = object)

np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//running.npy', rest_array)

locomotion = np.stack((running_array, rest_array))
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//locomotion.npy', locomotion)
