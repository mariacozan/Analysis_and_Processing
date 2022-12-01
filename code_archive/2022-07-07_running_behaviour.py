# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:20:22 2022

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
import extract_data as fun_ext
import functions2022_07_15 as fun
import extract_data_old as fun_ext_old
import scipy as sp

animal=  'Hedes'
date= '2022-08-04'
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '1'
log_number = '1'
plane_number = '1'
nr_planes = 4
plane_number_int = int(plane_number)
#IMPORTANT: SPECIFY THE FRAME RATE
frame_rate = 15
reps = 30
res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+log_number+'.csv'
#filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput0.csv'

#filePathArduino = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//ArduinoInput'+file_number+'.csv'
signal= np.load(filePathF, allow_pickle=True)

filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'

iscell = np.load(filePathiscell, allow_pickle=True)

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
#%%
test = np.loadtxt(r'C://MyPrograms//ArduinoInput0.csv',delimiter=',')
#%%
"""
function below gives the 5 channels of the arduinoL first column is running forwards, backwards, camera ticks 1, camera ticks 2, sync signal
"""
running_behaviour = fun.running_info(filePathArduino, plot = True)
channels = running_behaviour[0]
#%%
forward = channels[:,0]
backward = channels [:,1]
time_stamps = running_behaviour[1]

WheelMovement = fun.DetectWheelMove(forward, backward, timestamps = time_stamps)
#%%
"""
speed given is cm/s
"""
speed = WheelMovement[0]
fig,ax = plt.subplots(2)
ax[0].plot(speed)
ax[1].plot(forward)

#%%
"""
now need to determine
- what counts as movement and what doesn't (in Sylvia's paper: >/= 1cm/s; also in the Saleem paper? (everything starts at 1cm/s although or one plot it starts at 3cm/s))
- when this actually happened (need to synch with NiDaq)
- the alignment with the traces

"""
#alignment with NiDaq

#getting metadata info, remember to choose the right number of channels!! for most recent data it's 5 (for data in March but after thr 16th it's 4 and 7 before that)
meta = fun_ext.GetNidaqChannels(filePathmeta, numChannels=5,plot=True)

#%%
tmeta= meta.T
niTime = np.array(range(0,meta.shape[0]))/1000


#getting sync signal:
syncNiDaq = tmeta[-1]
syncArd = channels[:, -1]
    
#%%

corrected_time = fun_ext.arduinoDelayCompensation(nidaqSync = syncNiDaq ,ardSync = syncArd, niTimes = niTime ,ardTimes = time_stamps)
corrected_time = np.around(corrected_time, decimals = 2)

#now have the corrected time at which running data was recorded




#%%getting the F trace of cells 
#(and not ROIs not classified as cells) using a function I wrote
signal_cells = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T
# no2 = signal_cells[0:exp[1]]
# #plotting the running speed trace for the whole experiment with a neuronal trace
# fig,ax = plt.subplots(2)
# ax[0].plot(corrected_time, speed, c = "black")
# #ax[1].plot(no2[:, 9])
#%%

"""
Step 2: getting the times of the stimuli
"""


#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

#using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun_ext.DetectPhotodiodeChanges(photodiode,plot= True, waitTime=10000)


#the above is indiscriminate photodiode change, when it's on even numbers that is the stim onset
#converting to miliseconds because the align stim function takes ms

print("please choose the relevant experiment below! stim on or off etc")
#%%



stim_on = photodiode_change[0::2]
stim_off = photodiode_change[2::2]
# fig,ax = plt.subplots()
# ax.plot(stim_on) 




"""
Step 3: actually aligning the stimuli with the traces (using Liad's function)
"""

frame_clock = tmeta[1]
#frame_on = fun_ext.AssignFrameTime(frame_clock, plot = True)
#below checking the old function
frame_times = fun.AssignFrameTime(frame_clock, plot = True)
frame_on = frame_times[::2]
frame_on= frame_on/1000

frames_plane1 = frame_on[plane_number_int::nr_planes]
frames_plane2 = frame_on[plane_number_int::nr_planes]

#window: specify the range of the window
window= np.array([-1, 4]).reshape(1,-1)
aligned_on_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)
aligned_off_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_off, window= window,timeLimit=1000)

#aligned: thetraces for all the stimuli for all the cells
aligned = aligned_on_all[0]
#aligned = aligned_off_all[0]

#the actual time, usually 1 second before and 4 seconds after stim onset in miliseconds
time = aligned_on_all[1]

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

#log_Ori takes the first column of the log array because that corresponds to the first elelment in props in the GetStimulusInfo function above
#log_Ori = log[:,0].reshape(nr_stimuli,)

#%%
#no of stimuli specifes the total amount of stim shown
nr_stimuli = aligned.shape[1]
"""
Step 5: getting the iterations that correspond to the same stim identity, here just angle
"""
#the angles of the stim
#in the case of 20 iterations, given that for simple gratings protocol 12 orientations are shown, the total stimuli shown is 240
if nr_stimuli == 12*reps:
    angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    angles_str = ["30","60", "90", "120", "150", "180", "210", "240","270", "300", "330", "360"]
    #for other gratings protocols such as temp freq etc, this number should be double
elif nr_stimuli == 24*reps:
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
if nr_stimuli == 24*reps:
    TFreq = np.array([0.5, 1, 2, 4, 8, 16]) 
    SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    contrast = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])
if nr_stimuli == 12*reps:
    TFreq =np.array([2])
    SFreq = np.array([0.08])
    contrast = np.array([1])



#%%old plan
"""
now need to get the running traces for all iterations and match them to the same neural traces

Plan:
    1.known values: speed at a corresponding time which is aligned with the NiDaq time; we know the time at which the stimulus was on
    2.the speed values need to be averaged every 100ms
    3.for every iteration, need to assign running versus not running (if 90% of values are >1, then running)
        a.to do this, need to get the values for a given window of time (like the one we have now :-1 to 4s or just during stim pres?)
"""
"""
Averaging over 100ms

"""

#this below doesn't work because every 100ms doesn't correspond to every 100 element
speed_mean = np.mean(speed[:(len(speed)//100)*100].reshape(-1,100), axis=1)
plt.plot(speed)
#trying something else to get the index which is every 0.1s but doesn't work as it should
#this basically looks at the first element of the array with speed and time and appends indices which are 0.1s later to a list
#doesn't work as intended though
a = []
range_s = np.arange(start = speed_time[0,0]+0.1, stop = speed_time[0,0]+2, step = 0.1)
for n in range_s:
    temp = np.where(speed_time[:,0] == n) [0]
    a.append(temp)
    
#all_it_running = np.zeros((reps, 2471))
#for iteration in range(reps):
iteration = 5
stim_start = np.where(stim_on_round[iteration] == speed_time[:,0])[0]
stim_before = np.where(speed_time[stim_start[0],0]-1 ==speed_time[:,0])[0]
stim_end = np.where(speed_time[stim_start[0],0]+4 ==speed_time[:,0])[0]
    
one_stim_behaviour = np.array((speed_time[stim_before[0]:stim_end[0],1]))
    #all_it_running[iteration, :] = one_stim_behaviour

time_behaviour = np.array(range(0,one_stim_behaviour.shape[0]))/1000
    
fig,ax = plt.subplots(2)
ax[0].plot(one_stim_behaviour, c = "black")
#ax[1].plot(time,aligned[:, iteration, 5] )

for neuron in range(aligned.shape[2]):
#for neuron in range(107,111):       
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
#%%

reps_behaviour = fun.behaviour_reps(log = log, types_of_stim = 12,reps = 30, protocol_type = "simple", speed = speed, time = corrected_time, stim_on = stim_on, stim_off = stim_off)

#%%
"""
we now know the time at which a certain speed occurs (in secs ), what we now need is to know exactly at which time which iteration occured
we do know that because we gave that as an input to the alignment function (the stim_on variable!)
"""


#!! need to average across 100ms for these speed values since it's too upsampled to make sense


stim_on_round = np.around(stim_on, decimals = 2)
stim_off_round = np.around(stim_off, decimals = 2)


#%%
#strategy: take the stim on values and the stim off values which tell you the exact time
#, then use this to find the value in the running data which gives you a vector that contains all the values within that period 
#then decide within the loop if 90%of the values are above a certain threshold then assign to each rep a 0 or 1 value 
#make separate arrays which contain the indices like in all_oris but with a 1 or 0 assigned
#then can use these values to plot separate parts of the traces (running vs not running)
#also make an option with super fast running?
speed_time = np.stack((corrected_time, speed)).T
rep_running = []
rep_stationary = []
for rep in range(stim_on.shape[0]-1):
    start = np.where(stim_on_round[rep] == speed_time[:,0])[0]
    stop = np.where(stim_off_round[rep] == speed_time[:,0])[0]
    interval = speed_time[start[0]:stop[0], 1]
    running = np.argwhere(interval>1)
    if running.shape[0]/interval.shape[0]>0.9:        
            rep_running.append(rep)
    else:
            rep_stationary.append(rep)

rep_running = np.array(rep_running).astype("int64")
rep_stationary = np.array(rep_stationary).astype("int64")

#%% for all angles

reps = 30
#getting a 3D array with shape(orientation, repeats, TFreq/SFreq)
all_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
#all_SFreq = np.zeros((angles.shape[0], reps, SFreq.shape[0])).astype(int)
#all_contrast = np.zeros((angles.shape[0], reps, contrast.shape[0])).astype(int)
all_oris = np.zeros((angles.shape[0], reps)).astype(int)

for angle in range(angles.shape[0]):
    if nr_stimuli == 24*reps:
        for freq in range(TFreq.shape[0]): #and j in range(TFreq.shape[0]):
            specific_TF = np.where((log[:,0] == angles[angle]) & (log[:,2] == TFreq[freq])) [0]
           # specific_SF = np.where((log[:,0] == angles[angle]) & (log[:,1] == SFreq[freq])) [0]
            #specific_contrast = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq])) [0]
            all_TFreq[angle, :, freq] = specific_TF
           # all_SFreq[angle, :, freq] = specific_SF
            #all_contrast[angle, :, freq] = specific_contrast
    elif nr_stimuli == 12*reps: 
        
        specific_P = np.where((log[:,0] == angles[angle]) & (log[:,2] == 2)) [0]
        
        all_oris[angle, :] = specific_P
#%% 

reps = 30 
nr_stimuli = 720     
#getting the log for running and for rest
log_running = log[rep_running]
log_rest = log[rep_stationary]
"""
for running
"""      
#running_SFreq = np.zeros((angles.shape[0], reps, SFreq.shape[0])).astype(int)
running_SFreq = []
#all_contrast = np.zeros((angles.shape[0], reps, contrast.shape[0])).astype(int)
running_oris = []

for angle in range(angles.shape[0]):
    if nr_stimuli == 24*reps:
        for freq in range(TFreq.shape[0]): 
            #specific_TF_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,2] == TFreq[freq])) [0]
            specific_SF_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,1] == SFreq[freq])) [0]
            #specific_contrast_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq])) [0]
            #running_TFreq[angle, :, freq] = specific_TF_r
            running_SFreq.append(specific_SF_r)
            #running_SFreq[angle, :, freq] = specific_SF_r
            #running_contrast[angle, :, freq] = specific_contrast_r
    elif nr_stimuli == 12*reps: 
        
        specific_P_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,2] == 2)) [0]
        
        running_oris.append(specific_P_r)

#%%
stim_on_round = np.around(stim_on, decimals = 2)
stim_off_round = np.around(stim_off, decimals = 2)


protocol_type = "SFreq"


speed_time = np.stack((corrected_time, speed)).T
rep_running = []
rep_stationary = []
for rep in range(stim_on.shape[0]-1):
        start = np.where(stim_on_round[rep] == speed_time[:,0])[0]
        stop = np.where(stim_off_round[rep] == speed_time[:,0])[0]
        interval = speed_time[start[0]:stop[0], 1]
        running = np.argwhere(interval>1)
        if running.shape[0]/interval.shape[0]>0.9:        
                rep_running.append(rep)
        else:
                rep_stationary.append(rep)
    
rep_running = np.array(rep_running).astype("int64")
rep_stationary = np.array(rep_stationary).astype("int64")            
      
    #getting the log for running and for rest
log_running = log[rep_running]
log_rest = log[rep_stationary]
    
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
"""
for running
"""      
running = []
   
    
for angle in range(angles.shape[0]):
        if protocol_type == "SFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_SF_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,1] == SFreq[freq])) [0]
                running.append(specific_SF_r)
                
        if protocol_type == "TFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_TF_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,2] == TFreq[freq])) [0]
                running.append(specific_TF_r)
                
        if protocol_type == "Contrast":
            for freq in range(TFreq.shape[0]):
                specific_contrast_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq])) [0]
                running.append(specific_contrast_r)
        elif protocol_type == "simple": 
            
            specific_P_r = np.where((log_running[:,0] == angles[angle]) & (log_running[:,2] == 2)) [0]
            
            running.append(specific_P_r)
    
"""
for rest
"""      
rest = []
    
for angle in range(angles.shape[0]):
        if protocol_type == "SFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_SF_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,1] == SFreq[freq])) [0]
                rest.append(specific_SF_re)
                
        if protocol_type == "TFreq":
            for freq in range(TFreq.shape[0]):
                
                specific_TF_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,2] == TFreq[freq])) [0]
                rest.append(specific_TF_re)
                
        if protocol_type == "Contrast":
            for freq in range(TFreq.shape[0]):
                specific_contrast_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,3] == contrast[freq])) [0]
                rest.append(specific_contrast_re)
        elif protocol_type == "simple": 
            
            specific_P_re = np.where((log_running[:,0] == angles[angle]) & (log_running[:,2] == 2)) [0]
            
            rest.append(specific_P_re)
#%%

"""
for rest
"""      
#rest_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)
rest_SFreq = []
#rest_contrast = np.zeros((angles.shape[0], reps, contrast.shape[0])).astype(int)
rest_oris = []

for angle in range(angles.shape[0]):
    if nr_stimuli == 24*reps:
        for freq in range(TFreq.shape[0]): #and j in range(TFreq.shape[0]):
            specific_TF_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,2] == TFreq[freq])) [0]
            specific_SF_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,1] == SFreq[freq])) [0]
            #specific_contrast_r = np.where((log[:,0] == angles[angle]) & (log[:,2] == contrast[freq])) [0]
            #rest_TFreq[angle, :, freq] = specific_TF_re
            rest_SFreq.append(specific_SF_re)
            #rest_contrast[angle, :, freq] = specific_contrast_re
    elif nr_stimuli == 12*reps: 
        
        specific_P_re = np.where((log_rest[:,0] == angles[angle]) & (log_rest[:,2] == 2)) [0]
        
        rest_oris.append(specific_P_re)

#%%plotting all orientations and all spatial frequencies
#%%plotting all orientations if protocol was simple gratings
#for neuron in range(aligned.shape[2]):
for neuron in range(0,5):
       
            fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                    # ax[0,angle].plot(time,aligned[:,all_oris[angle, :3], neuron], c = "peachpuff")
                    # ax[0,angle].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    # ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    # ax[0,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    # ax[0,angle].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
            #         ax[0,angle].set_title(str(angles_str[angle]))
            for angle in range(4,8):
            #         # ax[1,angle-4].plot(time,aligned[:,all_oris[angle,0:3], neuron], c = "peachpuff")
            #         # ax[1,angle-4].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
            #         # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
            #         # ax[1,angle-4].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    ax[1,angle-4].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[1,angle-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].set_title(str(angles_str[angle]))
            for angle in range(8,12):
            #         # ax[2,angle-8].plot(time,aligned[:,all_oris[angle,0:3], neuron], c = "peachpuff")
            #         # ax[2,angle-8].plot(time,aligned[:,all_oris[angle,16:19], neuron], c = "plum")
            #         # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,0:3] , neuron].mean(axis = 1), c = "coral")
            #         # ax[2,angle-8].plot(time, aligned[:,all_oris[angle,16:19] , neuron].mean(axis = 1), c = "purple")
                    ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgray")
                    ax[2,angle-8].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[2,angle-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].set_title(str(angles_str[angle]))
                    
            plt.xlabel("Time(ms)")
           # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//cell'+str(neuron)+'.png')

#%%
#trying for one angle and neuron only
neuron = 5
angle = 0
one_ori = all_oris[angle,:].astype("int64")


#run = np.zeros(())
run = []
for rep in range(one_ori.shape[0]):
    running_one = np.where(rep_running == one_ori[rep])[0]
    #run = np.extract(running_one, rep_running)
    run.append(running_one)
    #run[rep,:] = running_one

run = np.array(run, dtype = object)
run =  [x for x in run if x]
run = np.stack((run))
run = np.reshape(run, (run.shape[0],))

rest = []
for rep in range(one_ori.shape[0]):
    rest_one = np.where(rep_stationary == one_ori[rep])[0]
    #run = np.extract(running_one, rep_running)
    rest.append(rest_one)
    #run[rep,:] = running_one

rest = np.array(rest, dtype = object)
rest =  [x for x in rest if x]
rest = np.stack((rest))
rest = np.reshape(rest, (rest.shape[0],))
#the above gives me the index in the rep_running array which corresponds to that one angle
#the values of the rep running correpond to all the stim reps which were categorised as running
    
#need the index of rep_running which is part of the specified orientation array
#%%for simple gratings protocol

for neuron in range(aligned.shape[2]):
#for neuron in range(0,1):
    fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
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
                    
    plt.xlabel("Time(ms)")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//all_oris//running_v_rest//cell'+str(neuron)+'.png')

#%%
#%%plotting all orientations and all spatial frequencies
for neuron in range(aligned.shape[2]):
#for neuron in range(107,111):       
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
           # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//highpass100//cell'+str(neuron)+'.png')            