# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:49:13 2022

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
import extract_data as fun_ext
"""
more updated stim aligning protocol, uses more recent functions
"""

#getting the signal, for now using the raw F

animal=  'Glaucus'
#animal = input("animal name ")
date= '2022-08-23'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '3'
#%%
#experiment = input("experiment number(integer only) ")
#experiment_int = int(experiment)
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '2'
log_number = '2'
plane_number = '1'
plane_number_int = int(plane_number)
#%%
# file_number = input("NiDaq file no. ")
# log_number = input("log file no. ")
# plane_number = input("imaging plane ")
# plane_number_int = int(plane_number)
#%%loading Suite2P output
#nr_planes = int(input("how many planes were imaged?"))
nr_planes = 4
#reps_init = 20
res = ''
#filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
#filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#
filePathF ='C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#

signal= np.load(filePathF, allow_pickle=True)
#filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
filePathiscell = 'C://Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
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
signal_cells = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T
# one = ops["frames_per_folder"][0]
# two = ops["frames_per_folder"][1] +one
# one_exp = signal_cells[one:two, 12]
# plt.plot(one_exp)
#%%getting the metadata


filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'

#depending on when the experiment was done, need to add on the frame clock since it starts from zero but suite2p sees it as continuous
#so we have two things to change: 


file_number_exp1 = "0"
file_number_exp2 = "1"
file_number_exp3 = "2"
file_meta_exp1 = 'Z://RawData//'+animal+ '//'+date+ '//1//NiDaqInput'+file_number_exp1+'.bin'
file_meta_exp2 = 'Z://RawData//'+animal+ '//'+date+ '//2//NiDaqInput'+file_number_exp2+'.bin'
file_meta_exp3 = 'Z://RawData//'+animal+ '//'+date+ '//3//NiDaqInput'+file_number_exp3+'.bin'
"""
Step 2: getting the times of the stimuli
"""

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)

meta_exp1 = fun_ext.GetNidaqChannels(file_meta_exp1, numChannels=5)
meta_exp2 = fun_ext.GetNidaqChannels(file_meta_exp2, numChannels=5)
meta_exp3 = fun_ext.GetNidaqChannels(file_meta_exp3, numChannels=5)

#getting the photodiode info, usually the first column in the meta array
photodiode_exp1 = meta_exp1[:,0]
photodiode_exp2 = meta_exp2[:,0]
photodiode_exp3 = meta_exp3[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
#%%
photodiode_change_exp1 = fun_ext.DetectPhotodiodeChanges(photodiode_exp1,plot= True,kernel = 101,fs=1000, waitTime=10000)
photodiode_change_exp2 = fun_ext.DetectPhotodiodeChanges(photodiode_exp2,plot= True,kernel = 101,fs=1000, waitTime=10000)
photodiode_change_exp3 = fun_ext.DetectPhotodiodeChanges(photodiode_exp3,plot= True,kernel = 101,fs=1000, waitTime=10000)
#%%
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset


tmeta_exp1 = meta_exp1.T
tmeta_exp2 = meta_exp2.T
tmeta_exp3 = meta_exp3.T

frame_clock_exp1 = tmeta_exp1[1]
frame_clock_exp2 = tmeta_exp2[1]
frame_clock_exp3 = tmeta_exp3[1]

frames_exp1 = fun_ext.AssignFrameTime(frame_clock_exp1, plot = False)
frames_exp2 = fun_ext.AssignFrameTime(frame_clock_exp2, plot = False)
frames_exp3 = fun_ext.AssignFrameTime(frame_clock_exp3, plot = False)

#%%

#adding up the frame clocks
addto_exp2 = frames_exp2 + frames_exp1[-1]
addto_exp3 = frames_exp3 + addto_exp2[-1]
frames_all = np.concatenate((frames_exp1, addto_exp2, addto_exp3))

#adding up the photodiode changes
photo_addto_exp2 = photodiode_change_exp2 + photodiode_change_exp1[-1]
photo_addto_exp3 = photodiode_change_exp3 + photo_addto_exp2[-1]
photodiode_all = np.concatenate((photodiode_change_exp1, photo_addto_exp2, photo_addto_exp3))

#photodiode_change_new = photodiode_change + added_time
print("please choose the relevant experiment below! stim on or off etc")
#%%

#because the newest function gives everythin in seconds but the align stim function does it in ms, need to convert it back 

stim_on = photodiode_all[0::2]
stim_off = photodiode_all[2::2]

# fig,ax = plt.subplots()
# ax.plot(stim_on) 







"""
Step 3: actually aligning the stimuli with the traces (using Liad's function)
"""



#frame_on = fun_ext.AssignFrameTime(frame_clock, plot = False)
# frame_times1 = frame_times[1:]

#frame_on = frame_times[::2]
frames_plane1 = frames_all[plane_number_int::nr_planes]
frames_plane2 = frames_all[plane_number_int::nr_planes]

#window: specify the range of the window
window= np.array([-1, 9]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= photodiode_all, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells

aligned_all_exp = aligned_all[0]

time = aligned_all[1]

#there are 13 types of stim and ten reps, need to plot every 13 
#%%

#getting the last experiment:
aligned = aligned_all_exp[:,-131:-1, :]

#%%for chirp only
#window: specify the range of the window
window_chirp = np.array([-1, 9]).reshape(1,-1)
aligned_chirp = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= photodiode_all, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells

aligned_all_chirp = aligned_all[0]

time_chirp = aligned_all_chirp[1]
#getting the last experiment:
aligned_chirp_only = aligned_all_chirp[:,-131:-1, :]

chirp = aligned_chirp_only[:, 5::13, :]
for neuron in range(aligned.shape[2]):
    fig, ax = plt.subplots()
    ax.plot(aligned_chirp_only[:, 5::13, neuron], c = "lightgray")
    ax.plot(aligned_chirp_only[:, 5::13, neuron].mean(axis = 1), c = "black")
    fig.text(0.5, 0.04, "Time(ms)", ha = "center") 
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ ''+res+'//plane'+plane_number+'//classification//chirp//cell'+str(neuron)+'.png')

#%%    

stim_type1 = [ "LightOn", "LightOnBlack", "GreyOn", "ChirpFrequency"]
stim_type2 =  ["GreyOn", "ChirpContrast", "GreyOn", "LightOnBlack"]
stim_type3 = ["BlueOn", "LightOnBlack", "GreenOn", "LightOnBlack"]
#%%
for neuron in range(aligned.shape[2]):
#for neuron in range(4,5):
    fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
    #for n in range(3):
    
    for stim in range(0,4):
        ax[0,stim].plot(time,aligned[:, stim::13,  neuron], c = "lightgray")
        ax[0,stim].plot(time,aligned[:, stim::13,  neuron].mean(axis = 1), c = "black")
        ax[0,stim].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
        ax[0,stim].axvline(x=3, c="blue", linestyle="dashed", linewidth = 1)
        ax[0,stim].set_title(str([stim_type1[stim]]))    

    for stim in range(4,8):
        ax[1,stim-4].plot(time,aligned[:, stim::13,  neuron], c = "lightgray")
        ax[1,stim-4].plot(time,aligned[:, stim::13,  neuron].mean(axis = 1), c = "black")
        ax[1,stim-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
        ax[1,stim-4].axvline(x=3, c="blue", linestyle="dashed", linewidth = 1)
        ax[1,stim-4].set_title(str([stim_type2[stim-4]]))        
                             
    for stim in range(8,12):
        ax[2,stim-8].plot(time,aligned[:, stim::13,  neuron], c = "lightgray")
        ax[2,stim-8].plot(time,aligned[:, stim::13,  neuron].mean(axis = 1), c = "black")
        ax[2,stim-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
        ax[2,stim-8].axvline(x=3, c="blue", linestyle="dashed", linewidth = 1)
        ax[2,stim-8].set_title(str([stim_type3[stim-8]]))
    fig.text(0.5, 0.04, "Time(ms)", ha = "center") 
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ ''+res+'//plane'+plane_number+'//classification//test//cell'+str(neuron)+'.png')

#%%plotting metadata vs trace
#for neuron in range(aligned.shape[2]):
for neuron in range(4,5):
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
    
    ax[1].plot(signal_cells[56109:63338, neuron])
    
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ ''+res+'//plane'+plane_number+'//classification//all//cell'+str(neuron)+'.png')
#%%
for neuron in range(4,5):
    fig,ax = plt.subplots(1)
    ax.plot(aligned[:, 1,  neuron])
  #%%  
plt.plot(aligned[:, 37,  neuron])