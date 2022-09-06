# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:29:36 2022

@author: maria
"""

"""
Tuning curves strategy:
    - get aligned traces as usual
    - for each rep, subtract baseline
    - then compute average over that 2s interval
    -do this for all different types of stim and plot the values in one graph
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
#%%
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

#%%
experiment = '2'
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
#%%getting the F trace of cells 
#(and not ROIs not classified as cells) using a function I wrote
signal_cells = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T
#%%

"""
Step 2: getting the times of the stimuli
"""

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = fun_ext.GetNidaqChannels(filePathmeta, numChannels=5)
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun_ext.DetectPhotodiodeChanges(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)
#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
print("please choose the relevant experiment below! stim on or off etc")


#%%

stim_on = photodiode_change[0::2]
stim_off = photodiode_change[2::2]


"""
Step 3: actually aligning the stimuli with the traces (using Liad's function)
"""

tmeta= meta.T
frame_clock = tmeta[1]
frame_on = fun_ext.AssignFrameTime(frame_clock, plot = False)
# frame_times1 = frame_times[1:]

#frame_on = frame_times[::2]
frames_plane1 = frame_on[plane_number_int::nr_planes]
frames_plane2 = frame_on[plane_number_int::nr_planes]

#window: specify the range of the window
window= np.array([-1, 4]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells

aligned = aligned_all[0]


time = aligned_all[1]
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
all_parameters = fun.Get_Stim_Identity(log = log, reps = 30, types_of_stim =24, protocol_type = "Contrast")

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

tmeta= meta.T
niTime = np.array(range(0,meta.shape[0]))/1000


#getting sync signal:
syncNiDaq = tmeta[-1]
syncArd = channels[:, -1]

corrected_time = fun_ext.arduinoDelayCompensation(nidaqSync = syncNiDaq ,ardSync = syncArd, niTimes = niTime ,ardTimes = time_stamps)
corrected_time = np.around(corrected_time, decimals = 2)

#%%
#%%need to add a column of zeros to log to be able to append a 1 to trials if the trial involved running
zero = np.zeros((aligned.shape[1])).reshape(aligned.shape[1], )
log = np.column_stack((log, zero))

reps_behaviour = fun.behaviour_reps(log = log, types_of_stim = 24, reps = 30, protocol_type = "Contrast", speed = speed, time = corrected_time, stim_on = stim_on, stim_off = stim_off)
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


#%%above function but outside the function
stim_on_round = np.around(stim_on, decimals = 2)
stim_off_round = np.around(stim_off, decimals = 2)


protocol_type = "SFreq"
types_of_stim = 24
reps = 30


speed_time = np.stack((corrected_time, speed)).T
rep_running = []
rep_stationary = []
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
                log_bool[rep,4] = a
        
log = log_bool    
#%%

#what I need is to get the indices of the big log file with all reps that are associated with running or not
#so need to assign a 1 or 0 to each rep
      
    #getting the log for running and for rest
#log_running = log[rep_running]
#log_rest = log[rep_stationary]

"""
for running
"""      
running = []
   
    
for angle in range(angles.shape[0]):
        if protocol_type == "SFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_SF_r = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq]) & (log[:,4] ==1)) [0]
                running.append(specific_SF_r)
                
        if protocol_type == "TFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_TF_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq]) & (log[:,4] ==1)) [0]
                running.append(specific_TF_r)
                
        if protocol_type == "Contrast":
            for freq in range(TFreq.shape[0]):
                specific_contrast_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq]) & (log[:,4] ==1)) [0]
                running.append(specific_contrast_r)
        elif protocol_type == "simple": 
            
            specific_P_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == 2) & (log[:,4] ==1)) [0]
            
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
            
            specific_P_re = np.where((log[:,0] == angles[angle]) & (log[:,2] == 2)  & (log[:,4] ==0)) [0]
            
            rest.append(specific_P_re)
#%%plotting for simple gratings protocol, needs to be tweaked because functions were changed
running_oris = reps_behaviour[0]
rest_oris = reps_behaviour[1]

for neuron in range(aligned.shape[2]):
#for neuron in range(0,1):
    fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
#angle = 0
    for angle in range(0,4):
                    ax[0,angle].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[0,angle].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum", alpha = 0.2)
                    ax[0,angle].plot(time, aligned[:,running_oris[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[0,angle].plot(time, aligned[:,rest_oris[angle] , neuron].mean(axis = 1), c = "purple")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[0,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
    for angle in range(4,8):
                    ax[1,angle-4].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[1,angle-4].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum", alpha = 0.2)
                    ax[1,angle-4].plot(time, aligned[:,running_oris[angle], neuron].mean(axis = 1), c = "teal")
                    ax[1,angle-4].plot(time, aligned[:,rest_oris[angle], neuron].mean(axis = 1), c = "purple")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[1,angle-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].set_title(str(angles_str[angle]))
    for angle in range(8,12):
                    ax[2,angle-8].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[2,angle-8].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum", alpha = 0.2)
                    ax[2,angle-8].plot(time, aligned[:,running_oris[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[2,angle-8].plot(time, aligned[:,rest_oris[angle] , neuron].mean(axis = 1), c = "purple")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgray")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[2,angle-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].set_title(str(angles_str[angle]))
                    
    fig.text(0.5, 0.04, "Time(ms)", ha = "center")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//running_v_rest//cell'+str(neuron)+'.png')

#%%plotting for other gratings protocol
for neuron in range(aligned.shape[2]):
#for neuron in range(40,42):
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
                    #ax[angle-12,2].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-12,2].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    #ax[angle-12,2].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
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
                    #ax[freq,0].set_title(str(tfreq_str[freq]))
                    #ax[freq,0].set_title(str(sfreq_str[freq]))
                    ax[freq,0].set_title(str(contrast_str[freq]))
            plt.xlabel("Time(ms)")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//Contrast//all_oris//test//cell'+str(neuron)+'.png')




#%%plotting for all oris during running vs rest states
#for neuron in range(aligned.shape[2]):
for neuron in range(23,25):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(all_parameters.shape[2]):
                    #while running_oris[freq] > 0:
                        ax[freq,angle].plot(time,aligned[:, running[angle,freq, :], neuron], c = "turquoise")
                        ax[freq,angle].plot(time,aligned[:,rest[angle,freq, :], neuron], c = "plum")
                        a = aligned[:,running[angle, freq, :], neuron]
                        ax[freq,angle].plot(time, np.median(a, axis = 1), c = "teal")
                        b = aligned[:,rest[angle, freq, :] , neuron]
                        ax[freq,angle].plot(time, np.median(b, axis = 1), c = "purple")
                        ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[0,angle].set_title(str(angles_str[angle]))
                        ax[freq,0].set_title(str(sfreq_str[freq]))
                    
            plt.xlabel("Time(ms)")
            #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'////SFreq//test//cell'+str(neuron)+'.png')
#angle+freq
#%%
neuron = 24      
fig,ax = plt.subplots(6,4, sharex = True, sharey = True)

for angle in range(0,4):
    for freq in range(all_parameters.shape[2]):
        #for rep in range (8,10):
                    #while running_oris[freq] > 0:
                        ax[freq,angle].plot(time,aligned[:, running[angle, freq, :], neuron], c = "turquoise")
                        ax[freq,angle].plot(time,aligned[:,rest[angle, :, freq], neuron], c = "plum")
                       # ax[freq,angle].plot(time, aligned[:,running[angle, :, freq], neuron].mean(axis = 1), c = "teal")
                        #ax[freq,angle].plot(time, aligned[:,rest[angle, :, freq] , neuron].mean(axis = 1), c = "purple")
                        ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[0,angle].set_title(str(angles_str[angle]))
                        ax[freq,0].set_title(str(sfreq_str[freq]))
#%%plotting all orientations and all spatial frequencies
#for neuron in range(aligned.shape[2]):
for neuron in range(23,25):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,3):
                for freq in range(all_parameters.shape[1]):
                    #for rep in range(.shape[2]):
                        #if running[angle,freq, rep] >0:
                            ax[angle, freq].plot(time,aligned[:, all_parameters[angle,freq, :], neuron], c = "lightgrey")
                            #ax[freq,angle].plot(time,aligned[:, running[angle,freq, :], neuron], c = "turquoise")
                            ax[angle, freq].plot(time, aligned[:,all_parameters[angle,freq, :] , neuron].mean(axis = 1), c = "black")
                            # ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                            # ax[0,angle].set_title(str(angles_str[angle]))
                            # ax[freq,0].set_title(str(sfreq_str[freq]))
                    
            plt.xlabel("Time(ms)")
            #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//SFreq//test//cell'+str(neuron)+'.png')

#%%
angle = 1
freq = 5
neuron = 24
fig,ax = plt.subplots()
ax.plot(time,aligned[:, all_parameters[angle,:,freq], neuron], c = "lightgrey")
ax.plot(time,aligned[:, 3, neuron], c = "red")
#%%
angle = 0
freq = 0
neuron  = 1

#fig,ax = plt.subplots(1, sharex = True, sharey = True)
plt.plot(aligned[:, all_parameters[angle,:,freq], neuron], c = "lightgrey")
#plt.plot(aligned[:,all_parameters[angle,0, freq] , neuron].mean(axis = 1), c = "black")
plt.axvline(x=15, c="red", linestyle="dashed", linewidth = 1)
#ax[0,angle].set_title(str(angles_str[angle]))
#ax[freq,0].set_title(str(sfreq_str[freq]))
                    
#%%
#now need to create arrays which contain the average intensity per rep with baseline (500ms before stim onset), so for each orientation need one single value
#this value will then be plotted according to the orientation and/or other parameter (spatial/temporal tuning curve)

angle = 0
neuron = 0

#for one angle and running data only:

one = aligned[:,running_oris[angle], neuron]
baseline = aligned[0:8,running_oris[angle], neuron].mean(axis = 0)
trace = aligned[8:,running_oris[angle], neuron].mean(axis = 0)
norm = trace - baseline
#%%
#for one angle and all oris

one = aligned[:,all_oris[angle], neuron]
baseline = aligned[0:8,all_oris[angle], neuron].mean(axis = 0)
trace = aligned[8:,all_oris[angle], neuron].mean(axis = 0)
norm = (trace - baseline).mean(axis = 0)
#%%
#for all neurons and angles
mean_values = np.zeros((angles.shape[0],aligned.shape[2]))
for neuron in range(aligned.shape[2]):
    fig,ax = plt.subplots(1, sharex = True, sharey = True)
    for angle in range(angles.shape[0]):
        baseline = aligned[0:8,all_oris[angle], neuron].mean(axis = 0)
        trace = aligned[8:,all_oris[angle], neuron].mean(axis = 0)
        norm = (trace - baseline).mean(axis = 0)
        mean_values[angle, neuron] = norm 
    
    ax.plot(angles,mean_values[:,neuron])
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//tuning_curves//cell'+str(neuron)+'.png', transparent = True)

#%%trying a function from online to plot mean and sem
import scipy

def plot_mean_and_sem(array, axis=0):
    mean = array.mean(axis=axis)
    sem_plus = mean + scipy.stats.sem(array, axis=axis)
    sem_minus = mean - scipy.stats.sem(array, axis=axis)
    
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    plt.plot(mean)
    
plot_mean_and_sem(aligned[17:,running_oris[2], 24])

#%%ori and direction tuning curves
import scipy
#the below code needs to be adapted to calculate the SEM and plot it with the rest
running_oris = reps_behaviour[0]
rest_oris = reps_behaviour[1]
#for all neurons and angles
running_values= np.zeros((angles.shape[0],aligned.shape[2]))
rest_values = np.zeros((angles.shape[0],aligned.shape[2]))
all_sem_p_running = np.zeros((angles.shape[0],aligned.shape[2]))
all_sem_m_running = np.zeros((angles.shape[0],aligned.shape[2]))
all_sem_p_rest = np.zeros((angles.shape[0],aligned.shape[2]))
all_sem_m_rest = np.zeros((angles.shape[0],aligned.shape[2]))
#for neuron in range(1):
for neuron in range(aligned.shape[2]):
    fig,ax = plt.subplots(1, sharex = True, sharey = True)
    
    for angle in range(angles.shape[0]):
        array_b = aligned[8:17,running_oris[angle], neuron]
        baseline = array_b.mean(axis = 0)
        array_t = aligned[17:,running_oris[angle], neuron]
        trace = array_t.mean(axis = 0)
        norm = (trace - baseline).mean(axis = 0)
        
        sem_plus = norm + scipy.stats.sem(trace, axis=0)
        sem_minus = norm - scipy.stats.sem(trace, axis=0)
        #sem_plus_b = norm + scipy.stats.sem(baseline, axis=0)
        #sem_minus_b = norm - scipy.stats.sem(baseline, axis=0)
        #sem_plus = sem_plus_t- sem_plus_b
        #sem_minus = sem_minus_t - sem_minus_b
        all_sem_p_running[angle, neuron] = sem_plus
        all_sem_m_running[angle, neuron] = sem_minus
        running_values[angle, neuron] = norm
        
        
    for angle in range(angles.shape[0]):
        array_b_r = aligned[8:17,rest_oris[angle], neuron]
        baseline_r = array_b_r.mean(axis = 0)
        array_t_r = aligned[17:,rest_oris[angle], neuron]
        trace_r = array_t_r.mean(axis = 0)
        norm_r = (trace_r - baseline_r).mean(axis = 0)
        
        sem_plus_r= norm_r + scipy.stats.sem(trace_r, axis=0)
        sem_minus_r = norm_r - scipy.stats.sem(trace_r, axis=0)
        # sem_plus_b = norm + scipy.stats.sem(baseline, axis=0)
        # sem_minus_b = norm - scipy.stats.sem(baseline, axis=0)
        # sem_plus = sem_plus_t- sem_plus_b
        # sem_minus = sem_minus_t - sem_minus_b
        all_sem_p_rest[angle, neuron] = sem_plus_r
        all_sem_m_rest[angle, neuron] = sem_minus_r
        rest_values[angle, neuron] = norm_r
      
    ax.scatter(angles,running_values[:,neuron], c = "teal")
    ax.plot(angles,running_values[:,neuron], c = "teal")
    ax.fill_between(angles, all_sem_p_running[:,neuron], all_sem_m_running[:,neuron], alpha=0.5, color = "teal")
    ax.scatter(angles,rest_values[:,neuron], c = "purple")
    ax.plot(angles,rest_values[:,neuron], c = "purple")
    ax.fill_between(angles, all_sem_p_rest[:,neuron], all_sem_m_rest[:,neuron], alpha=0.5, color = "purple")
    fig.text(0.5, 0.04, "Degrees", ha = "center")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//tuning_curves//behaviour//cell'+str(neuron)+'.png')

#%%freq/contrast tuning curves
running_oris = reps_behaviour[0]
rest_oris = reps_behaviour[1]
#for all neurons and angles
running_values= np.zeros((angles.shape[0], SFreq.shape[0], aligned.shape[2]))
rest_values = np.zeros((angles.shape[0], SFreq.shape[0],aligned.shape[2]))
for neuron in range(1):
#for neuron in range(aligned.shape[2]):
    fig,ax = plt.subplots(4, sharex = True, sharey = True)
    
    for angle in range(angles.shape[0]):
        for freq in range(SFreq.shape[0]):
            baseline = aligned[0:8,running_oris[angle+freq], neuron].mean(axis = 0)
            trace = aligned[8:,running_oris[angle+freq], neuron].mean(axis = 0)
            norm = (trace - baseline).mean(axis = 0)
            running_values[angle, freq, neuron] = norm
    # for angle in range(angles.shape[0]):
    #     baseline = aligned[0:8,rest_oris[angle], neuron].mean(axis = 0)
    #     trace = aligned[8:,rest_oris[angle], neuron].mean(axis = 0)
    #     norm = (trace - baseline).mean(axis = 0)
    #     rest_values[angle, neuron] = norm
    # ax.scatter(angles,running_values[:,neuron], c = "teal")
    # ax.plot(angles,running_values[:,neuron], c = "teal")
    # ax.scatter(angles,rest_values[:,neuron], c = "purple")
    # ax.plot(angles,rest_values[:,neuron], c = "purple")
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//tuning_curves//behaviour//cell'+str(neuron)+'.png')

#%%creating all the tuning curve data
import scipy
running_oris = reps_behaviour[0]
rest_oris = reps_behaviour[1]

neuron = 24
running_values0= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p0 = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m0 = np.zeros((SFreq.shape[0],aligned.shape[2]))

running_values90= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p90 = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m90 = np.zeros((SFreq.shape[0],aligned.shape[2]))

running_values180= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p180 = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m180 = np.zeros((SFreq.shape[0],aligned.shape[2]))

running_values270= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p270 = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m270 = np.zeros((SFreq.shape[0],aligned.shape[2]))


rest_values0= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p0_r = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m0_r = np.zeros((SFreq.shape[0],aligned.shape[2]))

rest_values90= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p90_r = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m90_r = np.zeros((SFreq.shape[0],aligned.shape[2]))

rest_values180= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p180_r = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m180_r = np.zeros((SFreq.shape[0],aligned.shape[2]))

rest_values270= np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_p270_r = np.zeros((SFreq.shape[0],aligned.shape[2]))
all_sem_m270_r = np.zeros((SFreq.shape[0],aligned.shape[2]))


for neuron in range(aligned.shape[2]):
    #for angle 0
    #running
    for freq in range(SFreq.shape[0]):
            baseline0 = aligned[8:17, running_oris[0:6][freq], neuron].mean(axis = 0)
            trace0 = aligned[17:,running_oris[0:6][freq], neuron].mean(axis = 0)
            norm0 = (trace0 - baseline0).mean(axis = 0)
            sem_plus0 = norm0 + scipy.stats.sem(trace0, axis=0)
            sem_minus0 = norm0 - scipy.stats.sem(trace0, axis=0)
            all_sem_p0[freq, neuron] = sem_plus0
            all_sem_m0[freq, neuron] = sem_minus0
            
            running_values0[freq, neuron] = norm0
    #rest        
    for freq in range(SFreq.shape[0]):
            baseline0_r = aligned[8:17, rest_oris[0:6][freq], neuron].mean(axis = 0)
            trace0_r = aligned[17:,rest_oris[0:6][freq], neuron].mean(axis = 0)
            norm0_r = (trace0_r - baseline0_r).mean(axis = 0)
            sem_plus0_r = norm0_r + scipy.stats.sem(trace0_r, axis=0)
            sem_minus0_r = norm0_r - scipy.stats.sem(trace0_r, axis=0)
            all_sem_p0_r[freq, neuron] = sem_plus0_r
            all_sem_m0_r[freq, neuron] = sem_minus0_r
            
            rest_values0[freq, neuron] = norm0_r
    #angle90        
            
    for freq in range(SFreq.shape[0]):
            baseline90 = aligned[8:17, running_oris[6:12][freq], neuron].mean(axis = 0)
            trace90 = aligned[17:,running_oris[6:12][freq], neuron].mean(axis = 0)
            norm90 = (trace90 - baseline90).mean(axis = 0)
            sem_plus90 = norm90 + scipy.stats.sem(trace90, axis=0)
            sem_minus90 = norm90 - scipy.stats.sem(trace90, axis=0)
            all_sem_p90[freq, neuron] = sem_plus90
            all_sem_m90[freq, neuron] = sem_minus90
            
            running_values90[freq, neuron] = norm90
            
    for freq in range(SFreq.shape[0]):
            baseline90_r = aligned[8:17, rest_oris[6:12][freq], neuron].mean(axis = 0)
            trace90_r = aligned[17:,rest_oris[6:12][freq], neuron].mean(axis = 0)
            norm90_r = (trace90_r - baseline90_r).mean(axis = 0)
            sem_plus90_r = norm90_r + scipy.stats.sem(trace90_r, axis=0)
            sem_minus90_r = norm90_r - scipy.stats.sem(trace90_r, axis=0)
            all_sem_p90_r[freq, neuron] = sem_plus90_r
            all_sem_m90_r[freq, neuron] = sem_minus90_r
            
            rest_values90[freq, neuron] = norm90_r
    #angle180
    
    for freq in range(SFreq.shape[0]):
            baseline180 = aligned[8:17, running_oris[12:18][freq], neuron].mean(axis = 0)
            trace180 = aligned[17:,running_oris[12:18][freq], neuron].mean(axis = 0)
            norm180 = (trace180 - baseline180).mean(axis = 0)
            sem_plus180 = norm180 + scipy.stats.sem(trace180, axis=0)
            sem_minus180 = norm180 - scipy.stats.sem(trace180, axis=0)
            all_sem_p180[freq, neuron] = sem_plus180
            all_sem_m180[freq, neuron] = sem_minus180
            
            running_values180[freq, neuron] = norm180
            
    for freq in range(SFreq.shape[0]):
            baseline180_r = aligned[8:17, rest_oris[12:18][freq], neuron].mean(axis = 0)
            trace180_r = aligned[17:,rest_oris[12:18][freq], neuron].mean(axis = 0)
            norm180_r = (trace180_r - baseline180_r).mean(axis = 0)
            sem_plus180_r = norm180_r + scipy.stats.sem(trace180_r, axis=0)
            sem_minus180_r = norm180_r - scipy.stats.sem(trace180_r, axis=0)
            all_sem_p180_r[freq, neuron] = sem_plus180_r
            all_sem_m180_r[freq, neuron] = sem_minus180_r
            
            rest_values180[freq, neuron] = norm180_r
    #angle270
        
    for freq in range(SFreq.shape[0]):
            baseline270 = aligned[8:17, running_oris[18:24][freq], neuron].mean(axis = 0)
            trace270 = aligned[17:,running_oris[18:24][freq], neuron].mean(axis = 0)
            norm270 = (trace270 - baseline270).mean(axis = 0)
            sem_plus270 = norm270 + scipy.stats.sem(trace270, axis=0)
            sem_minus270 = norm270 - scipy.stats.sem(trace270, axis=0)
            all_sem_p270[freq, neuron] = sem_plus270
            all_sem_m270[freq, neuron] = sem_minus270
            
            running_values270[freq, neuron] = norm270
            
    for freq in range(SFreq.shape[0]):
            baseline270_r = aligned[8:17, rest_oris[18:24][freq], neuron].mean(axis = 0)
            trace270_r = aligned[17:,rest_oris[18:24][freq], neuron].mean(axis = 0)
            norm270_r = (trace270_r - baseline270_r).mean(axis = 0)
            sem_plus270_r = norm270_r + scipy.stats.sem(trace270_r, axis=0)
            sem_minus270_r = norm270_r - scipy.stats.sem(trace270_r, axis=0)
            all_sem_p270_r[freq, neuron] = sem_plus270_r
            all_sem_m270_r[freq, neuron] = sem_minus270_r
            
            rest_values270[freq, neuron] = norm270_r            
 #%% plotting frequency tuning curves
#running_values = np.stack((running_values0, running_values90, running_values180, running_values270))
#for neuron in range(aligned.shape[2]):
for neuron in range(24,25):    
    fig,ax = plt.subplots(2,2, sharex = True, sharey = True)
    
    ax[0,0].scatter(SFreq,running_values0[:,neuron], c = "teal")
    ax[0,0].plot(SFreq,running_values0[:,neuron], c = "teal")
    ax[0,0].fill_between(SFreq, all_sem_p0[:,neuron], all_sem_m0[:,neuron], alpha=0.5, color = "teal")
    ax[0,0].scatter(SFreq,rest_values0[:,neuron], c = "purple")
    ax[0,0].plot(SFreq,rest_values0[:,neuron], c = "purple")
    ax[0,0].fill_between(SFreq, all_sem_p0_r[:,neuron], all_sem_m0_r[:,neuron], alpha=0.5, color = "purple")
    ax[0,0].set_title(str(angles_str[0]) + " degrees", loc = "center")
    
    ax[1,0].scatter(SFreq,running_values90[:,neuron], c = "teal")
    ax[1,0].plot(SFreq,running_values90[:,neuron], c = "teal")
    ax[1,0].fill_between(SFreq, all_sem_p90[:,neuron], all_sem_m90[:,neuron], alpha=0.5, color = "teal")
    ax[1,0].scatter(SFreq,rest_values90[:,neuron], c = "purple")
    ax[1,0].plot(SFreq,rest_values90[:,neuron], c = "purple")
    ax[1,0].fill_between(SFreq, all_sem_p90_r[:,neuron], all_sem_m90_r[:,neuron], alpha=0.5, color = "purple")
    ax[1,0].set_title(str(angles_str[1]) + " degrees", loc = "center")
    
    ax[0,1].scatter(SFreq,running_values180[:,neuron], c = "teal")
    ax[0,1].plot(SFreq,running_values180[:,neuron], c = "teal")
    ax[0,1].fill_between(SFreq, all_sem_p180[:,neuron], all_sem_m180[:,neuron], alpha=0.5, color = "teal")
    ax[0,1].scatter(SFreq,rest_values180[:,neuron], c = "purple")
    ax[0,1].plot(SFreq,rest_values180[:,neuron], c = "purple")
    ax[0,1].fill_between(SFreq, all_sem_p180_r[:,neuron], all_sem_m180_r[:,neuron], alpha=0.5, color = "purple")
    ax[0,1].set_title(str(angles_str[2]) + " degrees", loc = "center")
    
    ax[1,1].scatter(SFreq,running_values270[:,neuron], c = "teal")
    ax[1,1].plot(SFreq,running_values270[:,neuron], c = "teal")
    ax[1,1].fill_between(SFreq, all_sem_p270[:,neuron], all_sem_m270[:,neuron], alpha=0.5, color = "teal")
    ax[1,1].scatter(SFreq,rest_values270[:,neuron], c = "purple")
    ax[1,1].plot(SFreq,rest_values270[:,neuron], c = "purple")
    ax[1,1].fill_between(SFreq, all_sem_p270_r[:,neuron], all_sem_m270_r[:,neuron], alpha=0.5, color = "purple")
    ax[1,1].set_title(str(angles_str[3]) + " degrees", loc = "center")
    
    fig.text(0.5, 0.04, "Frequency(cycles/sec)", ha = "center")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//SFreq//all_oris//running_vs_rest//tuning_curves//cell'+str(neuron)+'.png')


#now need to plot the mean for each freq for all those separte angles
#running_values= np.zeros((angles.shape[0], SFreq.shape[0], aligned.shape[2]))
#baseline = aligned[0:8,running_oris[0:6], neuron].mean(axis = 0)
#trace = aligned[8:,running_oris[0:6], neuron].mean(axis = 0)
#norm = (trace - baseline).mean(axis = 0)
#running_values[angle, freq, neuron] = norm

#%%plotting the ori traces

#for neuron in range(aligned.shape[2]):
for neuron in range(0,1):
    fig,ax = plt.subplots(4,4, sharex = True, sharey = True)
#angle = 0
    for angle in range(0,4):
                    ax[0,angle].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[0,angle].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum")
                    ax[0,angle].plot(time, aligned[:,running_oris[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[0,angle].plot(time, aligned[:,rest_oris[angle] , neuron].mean(axis = 1), c = "purple")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[0,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
    for angle in range(4,8):
                    ax[1,angle-4].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[1,angle-4].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum")
                    ax[1,angle-4].plot(time, aligned[:,running_oris[angle], neuron].mean(axis = 1), c = "teal")
                    ax[1,angle-4].plot(time, aligned[:,rest_oris[angle], neuron].mean(axis = 1), c = "purple")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[1,angle-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].set_title(str(angles_str[angle]))
    for angle in range(8,12):
                    ax[2,angle-8].plot(time,aligned[:,running_oris[angle], neuron], c = "turquoise")
                    ax[2,angle-8].plot(time,aligned[:,rest_oris[angle], neuron], c = "plum")
                    ax[2,angle-8].plot(time, aligned[:,running_oris[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[2,angle-8].plot(time, aligned[:,rest_oris[angle] , neuron].mean(axis = 1), c = "purple")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgray")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[2,angle-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].set_title(str(angles_str[angle]))
    for angle in range(angles.shape[0]):
        baseline = aligned[0:8,running_oris[angle], neuron].mean(axis = 0)
        trace = aligned[8:,running_oris[angle], neuron].mean(axis = 0)
        norm = (trace - baseline).mean(axis = 0)
        running_values[angle, neuron] = norm
    for angle in range(angles.shape[0]):
        baseline = aligned[0:8,rest_oris[angle], neuron].mean(axis = 0)
        trace = aligned[8:,rest_oris[angle], neuron].mean(axis = 0)
        norm = (trace - baseline).mean(axis = 0)
        rest_values[angle, neuron] = norm
    ax[3,0].plot(angles,running_values[:,neuron], c = "teal")
    ax[3,0].plot(angles,rest_values[:,neuron], c = "purple")
    plt.xlabel("Time(ms)")
   # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//running_v_rest//cell'+str(neuron)+'.png')




#%%creating histograms which show the preferred frequency across the population
#from tuning curves need to determine the frequency that the neurons are most tuned to (lookk at the running_values variables!)
#across the 6 values forthe different freqencies, need to rank them based on highest response etc
#getting the index of the max response for each neuron into a matrix so will go from a (6,91) msatrix to (1,91)
#then will get how often this specifc index occurs and plot as a histogram
