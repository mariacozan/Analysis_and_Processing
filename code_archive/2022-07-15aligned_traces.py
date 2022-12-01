# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:32:12 2022

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

#getting the signal, for now using the raw F

animal=  'Hedes'
#animal = input("animal name ")
date= '2022-08-05'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
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

res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#
#filePathF ='C://Temporary_Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
#filePathops = 'C://Temporary_Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'#

filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
#filePathiscell = 'C://Temporary_Suite2P_output//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'
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
plt.plot(signal_cells[:,0])
#%%

"""
Step 2: getting the times of the stimuli
"""

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = fun_ext.GetNidaqChannels(filePathmeta, numChannels=5, plot = True)
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]
#%%
#using the function from above to put the times of the photodiode changes (in milliseconds!)
#photodiode_change = photodiode_change[1:,]
photodiode_change = fun_ext.DetectPhotodiodeChanges(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)

#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
print("please choose the relevant experiment below! stim on or off etc")
#%%

stim_on = photodiode_change[0::2]
stim_off = photodiode_change[2::2]



# fig,ax = plt.subplots()
# ax.plot(stim_on) 




"""
Step 3: actually aligning the stimuli with the traces (using Liad's function)
"""

tmeta= meta.T
frame_clock = tmeta[1]
frame_times = fun.AssignFrameTime(frame_clock, plot = True)

# frame_times1 = frame_times[1:]

frame_on = frame_times[::2]
#frame_on= frame_on/1000
frames_plane1 = frame_on[plane_number_int::nr_planes]
frames_plane2 = frame_on[plane_number_int::nr_planes]

#window: specify the range of the window
window= np.array([-1000, 4000]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)
#aligned_all_new = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on_new, window= window,timeLimit=1000)


#aligned: thetraces for all the stimuli for all the cells
aligned = aligned_all[0]
#aligned_new = aligned_all_new[0]
#the actual time, usually 1 second before and 4 seconds after stim onset in miliseconds
time = aligned_all[1]
#%%
#getting one neuron for testing and plotting of a random stimulus but also plotting all the reps (relevant for off response):
neuron = 0
#for neuron in range(aligned.shape[2]):
fig,ax = plt.subplots()
one_neuron = aligned[:,:,neuron]

ax.plot(time,one_neuron, c = "lightgray")
    # ax.plot(time,one_neuron.mean(axis = 1), c = "black")
    # ax.axvline(x=0, c="blue", linestyle="dashed", linewidth = 1)
    # ax.axvline(x=2000, c="red", linestyle="dashed", linewidth = 1)
    # plt.xlabel("Time(ms)")
    # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//off_response//cell'+str(neuron)+'.png')
#%%

"""
Step 4: getting the identity of the stimuli
"""
#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])
#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)

#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)
#no of stimuli specifes the total amount of stim shown
nr_stimuli = aligned.shape[1]
#log_Ori takes the first column of the log array because that corresponds to the first elelment in props in the GetStimulusInfo function above
log_Ori = log[:,0].reshape(nr_stimuli,)

#%%

"""
Step 5: getting the iterations that correspond to the same stim identity, here just angle
"""
nr_stimuli = 480
#the angles of the stim
#in the case of 20 iterations, given that for simple gratings protocol 12 orientations are shown, the total stimuli shown is 240
if nr_stimuli == 240:
    angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    angles_str = ["30","60", "90", "120", "150", "180", "210", "240","270", "300", "330", "360"]
    #for other gratings protocols such as temp freq etc, this number should be double
elif nr_stimuli == 480:
    angles = np.array([0, 90, 180, 270])
    angles_str = ["0","90","180","270"]
    tfreq_str = ["0.5", "1", "2", "4", "8", "16"]
    sfreq_str = ["0.01", "0.02", "0.04", "0.08", "0.16", "0.32"]
    contrast_str = ["0","0.125", "0.25", "0.5", "0.75", "1"]
else:
    print("Check the exact stimulus protocol, total stimuli numbers don't match current settings")
#what each angle means
# 0 degrees is vertical to the left, 
#90 is horizontal down, 
#180 is vertical to the right and 
#270 is horizontal up
#with these 4 orientations can test orientation and direction selectivity
#reps = how many repetitions of the same stim we have
if nr_stimuli == 240:
    reps = 20
    #because the same orientations are shown for each freq parameter, need 6x more reps
elif nr_stimuli == 480:
    reps = 120
all_P = np.zeros((reps, angles.shape[0])).astype(int)
angle_times = np.zeros((reps, angles.shape[0]))




#%%

"""
Step 5b: if available, seeing how the different temp/spatial/contrast freq affect responses
"""
#first getting the angles available, usually only 4 when trying other parameters
#angles = np.array([0, 90, 180, 270])
#Temp freq
if nr_stimuli == 480:
    TFreq = np.array([0.5, 1, 2, 4, 8, 16]) 
    SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    contrast = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])
if nr_stimuli == 240:
    TFreq =np.array([2])
    SFreq = np.array([0.08])
    contrast = np.array([1])


#%% for all angles
#getting a 3D array with shape(orientation, repeats, TFreq/SFreq)
# reps = 20
# types_of_stim = 24
# protocol_type = "TFreq"

def Get_Stim_Identity(log, reps, protocol_type, types_of_stim):

      

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
                    
    all_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
    all_SFreq = np.zeros((angles.shape[0], reps, SFreq.shape[0])).astype(int)
    all_parameters = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
    all_contrast = np.zeros((angles.shape[0], reps, contrast.shape[0])).astype(int)
    all_oris = np.zeros((angles.shape[0], reps)).astype(int)  
    
    for angle in range(angles.shape[0]):
        if protocol_type == "TFreq":
                for freq in range(TFreq.shape[0]):
                    specific_TF = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq])) [0]
                    all_parameters[angle, :, freq] = specific_TF
            
        if protocol_type == "SFreq":
                for freq in range(SFreq.shape[0]):
                    specific_SF = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq])) [0]
                    all_parameters[angle, :, freq] = specific_SF
            
        if protocol_type == "Contrast":
                for freq in range(TFreq.shape[0]):
                    specific_contrast = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq])) [0]            
                    all_parameters[angle, :, freq] = specific_contrast
        return all_parameters
    
#%%
reps = 20
all_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
all_SFreq = np.zeros((angles.shape[0], reps, SFreq.shape[0])).astype(int)
all_parameters = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
all_contrast = np.zeros((angles.shape[0], reps, contrast.shape[0])).astype(int)
all_oris = np.zeros((angles.shape[0], reps)).astype(int)  
#if nr_stimuli == 480:
for angle in range(angles.shape[0]):
    if nr_stimuli == 480:
            for freq in range(TFreq.shape[0]): #and j in range(TFreq.shape[0]):
                specific_TF = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq])) [0]
                #specific_SF = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq])) [0]
               # specific_contrast = np.where((log[:,0] == angles[angle]) & (log[:,3] == contrast[freq])) [0]
                all_TFreq[angle, :, freq] = specific_TF
                #all_SFreq[angle, :, freq] = specific_SF
                #all_contrast[angle, :, freq] = specific_contrast
    elif nr_stimuli == 240: 
        #elif protocol_type == "simple":
            
            specific_P = np.where((log[:,0] == angles[angle]) & (log[:,2] == 2)) [0]
            
            all_oris[angle, :] = specific_P
            
#%%


identity = Get_Stim_Identity(log = log, reps = 20, types_of_stim = 24, protocol_type = "TFreq")

"""
Step 5c: plotting at all temp freq
"""
#%%
#for one orientation but check if the plots are actually good!
neuron = 0
#for neuron in range(aligned.shape[2]):

fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
    #TFreq 1
#ori = 0
for ori in range(angles.shape[0]):
        for freq in range(all_TFreq.shape[2]):
            for rep in all_TFreq[ori, :,freq]:
                ax[freq,ori].plot(time,aligned[:,all_TFreq[ori,:,freq], neuron], c = "lightgrey")
                ax[freq,ori].plot(time, aligned[:,all_TFreq[ori,:,freq] , neuron].mean(axis = 1), c = "black")
                ax[freq,ori].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                ax[0,ori].set_title(str(angles_str[ori]))
    # for rep in all_TFreq[0]:
    #         ax[0,0].plot(time,aligned[:,rep, neuron], c = "lightgrey")
    #         ax[0,0].plot(time, aligned[:,all_TFreq[0,:,0] , neuron].mean(axis = 1), c = "black")
    #         ax[0,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
    #         ax[0,0].set_title(str(angles_str[0]))
        
#%%for different orientations but same freq

#for freq in TFreq:
freq = 2
Ori_0 = np.where((log[:, 0] == 0) & (log[:,2] == freq))[0]
    
Ori_90 = np.where((log[:, 0] == 90) & (log[:,2] == freq))[0]
Ori_180 = np.where((log[:, 0] == 180) & (log[:,2] == freq))[0]
    #starting from the second iteration because the first one (which is the first ever stim shown, doesn't give any response, not even noise)
Ori_180 = Ori_180[1:]
Ori_270 = np.where((log[:, 0] == 270) & (log[:,2] == freq))[0]
#Ori_360 = np.where(log_Ori == 360)[0]
#if nr_stimuli == 240:
#for neuron in range(aligned.shape[2]):
for neuron in range(8):
    fig,ax = plt.subplots(2,2, sharex = True, sharey = True)

        
    for rep in Ori_0:
                    ax[0,0].plot(time,aligned[:,Ori_0, neuron], c = "lightgrey")
                    ax[0,0].plot(time, aligned[:,Ori_0 , neuron].mean(axis = 1), c = "black")
                    ax[0,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,0].set_title(str(angles_str[0]))
    for rep in Ori_90:
                    ax[0,1].plot(time,aligned[:,Ori_90, neuron], c = "lightgrey")
                    ax[0,1].plot(time, aligned[:,Ori_90 , neuron].mean(axis = 1), c = "black")
                    ax[0,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,1].set_title(str(angles_str[1]))   
    for rep in Ori_180:
                    ax[1,0].plot(time,aligned[:,Ori_180, neuron], c = "lightgrey")
                    ax[1,0].plot(time, aligned[:,Ori_180 , neuron].mean(axis = 1), c = "black")
                    ax[1,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,0].set_title(str(angles_str[2]))
    for rep in Ori_270:
                    ax[1,1].plot(time,aligned[:,Ori_270, neuron], c = "lightgrey")
                    ax[1,1].plot(time, aligned[:,Ori_270 , neuron].mean(axis = 1), c = "black")
                    ax[1,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,1].set_title(str(angles_str[3]))
                
    plt.xlabel("Time(ms)")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//TFreq'+str(freq)+'//cell'+str(neuron)+'.png')
    
#%%for same orientation but different freq

ori = 0
Freq_05 = np.where((log[:, 0] == ori) & (log[:,2] == 0.5))[0]
    
Freq_1 = np.where((log[:, 0] == ori) & (log[:,2] == 1))[0]
Freq_2 = np.where((log[:, 0] == ori) & (log[:,2] == 2))[0]
    
Freq_4 = np.where((log[:, 0] == ori) & (log[:,2] == 4))[0]
Freq_8 = np.where((log[:, 0] == ori) & (log[:,2] == 8))[0]
Freq_16 = np.where((log[:, 0] == ori) & (log[:,2] == 16))[0]
#Ori_360 = np.where(log_Ori == 360)[0]
#if nr_stimuli == 240:
for neuron in range(aligned.shape[2]):
    #for neuron in range(8):
        fig,ax = plt.subplots(2,3, sharex = True, sharey = True)
    
            
        for rep in Freq_05:
                        ax[0,0].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[0,0].plot(time, aligned[:,Freq_05 , neuron].mean(axis = 1), c = "black")
                        ax[0,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[0,0].set_title(str(tfreq_str[0]))
        for rep in Freq_1:
                        ax[0,1].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[0,1].plot(time, aligned[:,Freq_1 , neuron].mean(axis = 1), c = "black")
                        ax[0,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[0,1].set_title(str(tfreq_str[1]))   
        for rep in Freq_2:
                        ax[0,2].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[0,2].plot(time, aligned[:,Freq_2 , neuron].mean(axis = 1), c = "black")
                        ax[0,2].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[0,2].set_title(str(tfreq_str[2]))
        for rep in Freq_4:
                        ax[1,0].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[1,0].plot(time, aligned[:,Freq_4 , neuron].mean(axis = 1), c = "black")
                        ax[1,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[1,0].set_title(str(tfreq_str[3]))
        for rep in Freq_8:
                        ax[1,1].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[1,1].plot(time, aligned[:,Freq_8 , neuron].mean(axis = 1), c = "black")
                        ax[1,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[1,1].set_title(str(tfreq_str[4]))                
                        
        for rep in Freq_16:
                        ax[1,2].plot(time,aligned[:,rep, neuron], c = "lightgrey")
                        ax[1,2].plot(time, aligned[:,Freq_16 , neuron].mean(axis = 1), c = "black")
                        ax[1,2].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                        ax[1,2].set_title(str(tfreq_str[5]))            
        plt.xlabel("Time(ms)")
        plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//ori'+str(ori)+'cell'+str(neuron)+'.png')
    
#%%plotting all orientations if protocol was simple gratings
#for neuron in range(aligned.shape[2]):
for neuron in range(0,1):
       
            fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                    # ax[0,angle].plot(time,aligned[:,all_oris[angle, 0:3], neuron], c = "peachpuff")
                    # ax[0,angle].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[0,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].axvline(x=2000, c="blue", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
            for angle in range(4,8):
                    # ax[1,angle-4].plot(time,aligned[:,all_oris[angle,0:3], neuron], c = "peachpuff")
                    # ax[1,angle-4].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
                    # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[1,angle-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].axvline(x=2000, c="blue", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].set_title(str(angles_str[angle]))
            for angle in range(8,12):
                    # ax[2,angle-8].plot(time,aligned[:,all_oris[angle,0:3], neuron], c = "peachpuff")
                    # ax[2,angle-8].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
                    # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgray")
                    ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[2,angle-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].axvline(x=2000, c="blue", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].set_title(str(angles_str[angle]))
                    
            plt.xlabel("Time(ms)")
           # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//cell'+str(neuron)+'.png')

#%%plotting all orientations and all temp frequencies
#for neuron in range(aligned.shape[2]):
for neuron in range(10,12):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(all_TFreq.shape[2]):
                    ax[freq,angle].plot(time,aligned[:, all_TFreq[angle,:,freq], neuron], c = "lightgrey")
                    ax[freq,angle].plot(time, aligned[:,all_TFreq[angle,:, freq] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    ax[freq,0].set_title(str(tfreq_str[freq]))
                    
            plt.xlabel("Time(ms)")
          #  plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//cell'+str(neuron)+'.png')

#%%plotting all orientations and all spatial frequencies
for neuron in range(aligned.shape[2]):
#for neuron in range(3,7):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(all_SFreq.shape[2]):
                    ax[freq,angle].plot(time,aligned[:, all_SFreq[angle,:,freq], neuron], c = "lightgrey")
                    ax[freq,angle].plot(time, aligned[:,all_SFreq[angle,:, freq] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    ax[freq,0].set_title(str(sfreq_str[freq]))
                    
            plt.xlabel("Time(ms)")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//cell'+str(neuron)+'.png')
            
#%%plotting all orientations and all contrasts
for neuron in range(aligned.shape[2]):
#for neuron in range(10,15):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(all_contrast.shape[2]):
                    ax[freq,angle].plot(time,aligned[:, all_contrast[angle, freq, :], neuron], c = "lightgrey")
                    ax[freq,angle].plot(time, aligned[:,all_contrast[angle, freq, :] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    ax[freq,0].set_title(str(contrast_str[freq]))
                    
            plt.xlabel("Time(ms)")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ ''+res+'//plane'+plane_number+'//Contrast//test//cell'+str(neuron)+'.png')