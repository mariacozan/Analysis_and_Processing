# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:44:14 2022

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

animal=  'Hedes'
#animal = input("animal name ")
date= '2022-08-05'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '1'
exp_name = 'SFreq'
#%%
#experiment = input("experiment number(integer only) ")
#experiment_int = int(experiment)
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '0'
log_number = '0'
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


filePathaligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_exp1.npy'
aligned_test = np.load(filePathaligned, allow_pickle=True)
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
#meta_exp3 = fun_ext.GetNidaqChannels(file_meta_exp3, numChannels=5)

#getting the photodiode info, usually the first column in the meta array
photodiode_exp1 = meta_exp1[:,0]
photodiode_exp2 = meta_exp2[:,0]
#photodiode_exp3 = meta_exp3[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
#%%
photodiode_change_exp1 = fun_ext.DetectPhotodiodeChanges(photodiode_exp1,plot= True,kernel = 101,fs=1000, waitTime=10000)
photodiode_change_exp2 = fun_ext.DetectPhotodiodeChanges(photodiode_exp2,plot= True,kernel = 101,fs=1000, waitTime=10000)
#photodiode_change_exp3 = fun_ext.DetectPhotodiodeChanges(photodiode_exp3,plot= True,kernel = 101,fs=1000, waitTime=10000)
#%%
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset


tmeta_exp1 = meta_exp1.T
tmeta_exp2 = meta_exp2.T
#tmeta_exp3 = meta_exp3.T

frame_clock_exp1 = tmeta_exp1[1]
frame_clock_exp2 = tmeta_exp2[1]
#frame_clock_exp3 = tmeta_exp3[1]

frames_exp1 = fun_ext.AssignFrameTime(frame_clock_exp1, plot = False)
frames_exp2 = fun_ext.AssignFrameTime(frame_clock_exp2, plot = False)
#frames_exp3 = fun_ext.AssignFrameTime(frame_clock_exp3, plot = False)

#%%

#adding up the frame clocks
addto_exp2 = frames_exp2 + frames_exp1[-1]
#addto_exp3 = frames_exp3 + addto_exp2[-1]
frames_all = np.concatenate((frames_exp1, addto_exp2))
#frames_all = np.concatenate((frames_exp1, addto_exp2, addto_exp3))

#adding up the photodiode changes
photo_addto_exp2 = photodiode_change_exp2 + photodiode_change_exp1[-1]
#photo_addto_exp3 = photodiode_change_exp3 + photo_addto_exp2[-1]
photodiode_all = np.concatenate((photodiode_change_exp1, photo_addto_exp2))
#photodiode_all = np.concatenate((photodiode_change_exp1, photo_addto_exp2, photo_addto_exp3))



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
window= np.array([-1, 4]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells

aligned_all_exp = aligned_all[0]

time = aligned_all[1]
#%%getting the specific aligned traces
length_1 = int(photodiode_change_exp1.shape[0]/2)
length_exp2 = int(photodiode_change_exp2.shape[0]/2)
length_2 = length_1 +length_exp2

aligned_exp1 = aligned_all_exp[:, 0:length_1, :]
aligned_exp2 = aligned_all_exp[:, length_1:length_2, :]

#saving the aligned traces as npy files
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_exp1.npy', aligned_exp1)

np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_exp2.npy', aligned_exp2)
#%%

"""
Step 4: getting the identity of the stimuli
"""
#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])
#%%
#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)

#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)

#%%

types_of_stim = 24
if types_of_stim == 12:
        angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
        angles_str = ["30","60", "90", "120", "150", "180", "210", "240","270", "300", "330", "360"]
    
        #for other gratings protocols such as temp freq etc, this number should be double
elif types_of_stim == 24:
        angles = np.array([0, 90, 180, 270])
        angles_str = ["0","90","180","270"]
        tfreq_str = ["0.5", "1", "2", "4", "8", "16"]
        sfreq_str = ["0.01", "0.02", "0.04", "0.08", "0.16", "0.32"]
        contrast_str = ["0","0.125", "0.25", "0.5", "0.75", "1"]
        
#%%stimulus identity
all_parameters = fun.Get_Stim_Identity(log = log, reps = 30, types_of_stim =24, protocol_type = "SFreq")

#%%behaviour
running_behaviour = fun.running_info(filePathArduino, plot = True)
channels = running_behaviour[0]

forward = channels[:,0]
backward = channels [:,1]
time_stamps = running_behaviour[1]

WheelMovement = fun.DetectWheelMove(forward, backward, timestamps = time_stamps)

speed = WheelMovement[0]
fig,ax = plt.subplots(2)
ax[0].plot(speed)
ax[1].plot(forward)

print("please put in the right metadata variable, script might need more editing")
meta = meta_exp1
tmeta= meta.T
niTime = np.array(range(0,meta.shape[0]))/1000


#getting sync signal:
syncNiDaq = tmeta[-1]
syncArd = channels[:, -1]

corrected_time = fun_ext.arduinoDelayCompensation(nidaqSync = syncNiDaq ,ardSync = syncArd, niTimes = niTime ,ardTimes = time_stamps)
corrected_time = np.around(corrected_time, decimals = 2)

#%%
#%%need to add a column of zeros to log to be able to append a 1 to trials if the trial involved running
aligned = aligned_exp1
#because the nidaq info has been modified, need to again choose the right chunks here
#eventually should concatenate the Arduino data as well
stim_on = stim_on[0:length_1]
stim_off = stim_off[0:length_1]
zero = np.zeros((aligned.shape[1])).reshape(aligned.shape[1], )
log = np.column_stack((log, zero))

reps_behaviour = fun.behaviour_reps(log = log, types_of_stim = 24, reps = 30, protocol_type = "SFreq", speed = speed, time = corrected_time, stim_on = stim_on, stim_off = stim_off)
#the above function gives the reps for each orientation for running and for rest states

#%%
running = reps_behaviour[0]
rest = reps_behaviour[1]


#%%reshaping for protocols other than simple gratings
# # reshaping the above to be sorted as (no. of angles, no. of freq, no. of reps for each)
# running = running_oris.reshape(4, 6, running_oris.shape[1]).astype("int64")
# rest = rest_oris.reshape(4, 6, rest_oris.shape[1]).astype("int64")

#%%creating the lists of parameters

    
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


#%%plotting for other gratings protocol
print("please choose right aligned traces!")
aligned = aligned_exp1
#for neuron in range(aligned.shape[2]):
for neuron in range(35,37):
    fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
    
    
#angle = 0
    for a in range(0,4):
        ax[0,a].set_title(str(angles_str[a]) + "degrees", loc = "right")
    for angle in range(0,6):
        
                    ax[angle,0].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle,0].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle,0].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle,0].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[angle,0].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle,0].xaxis.set_label_position('top')
                    #ax[angle,0].set_xlabel(str(sfreq_str[angle]), loc = "left")
                    ax[angle,0].set_xlabel(str(contrast_str[angle]), loc = "left")
                    
    for angle in range(6,12):
                    ax[angle-6,1].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-6,1].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-6,1].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-6,1].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-6,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[angle-6,1].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-6,1].xaxis.set_label_position('top')
                    
                    
    for angle in range(12,18):
                    ax[angle-12,2].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-12,2].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-12,2].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-12,2].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-12,2].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[angle-12,2].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-12,2].xaxis.set_label_position('top')
            
    
    for angle in range(18,24):
                    ax[angle-18,3].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-18,3].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-18,3].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-18,3].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-18,3].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[angle-18,3].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-18,3].xaxis.set_label_position('top')
                   
                   
    fig.text(0.5, 0.04, "Time(ms)", ha = "center")
                  
    
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//Contrast//all_oris//test//running_vs_rest/cell'+str(neuron)+'.png')


#%%plotting all orientations and all temp frequencies
aligned = aligned_exp1
all_TFreq = all_parameters
for neuron in range(aligned.shape[2]):
#for neuron in range(107,111):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(6):
                    ax[freq,angle].plot(time,aligned[:, all_TFreq[angle,freq, :], neuron], c = "lightgrey")
                    ax[freq,angle].plot(time, aligned[:,all_TFreq[angle,freq, :] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    ax[freq,0].set_title(str(tfreq_str[freq]))
                    #ax[freq,0].set_title(str(sfreq_str[freq]))
                    #ax[freq,0].set_title(str(contrast_str[freq]))
            plt.xlabel("Time(ms)")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//TFreq//all_oris//test2//cell'+str(neuron)+'.png')



