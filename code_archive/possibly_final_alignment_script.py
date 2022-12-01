# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:27:43 2022

@author: maria
"""

"""
Goal: 
    1.to optimise and streamline the alignment of the traces and the separation of them into during running vs rest 
    2.to make the script cleaner with functions instead of all written out
Strategy: 
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
import seaborn as sns
sns.set()

#%%input needed:specify experiment details

animal=  'Iris'
#animal = input("animal name ")
date= '2022-10-12'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
exp_name = 'Simple'
file_number = '1'
log_number = '1'
plane_number = '0'
plane_number_int = int(plane_number)
nr_planes = 1
repetitions = 30

#%%loading the Suite2P output

#Suite2P
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//ops.npy'
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//iscell.npy'

#below options in case data is on the C drive
# CfilePathF ='C://Suite2P_output//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//F.npy'
# CfilePathops = 'C://Suite2P_output//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//ops.npy'
# CfilePathiscell = 'C://Suite2P_output//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//iscell.npy'

#%%
CfilePathvalues ='C://Suite2P_output//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//Values.csv'
#Cvalues = np.loadtxt(CfilePathvalues,delimiter=',')


ImageJ = []
with open('C://Suite2P_output//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//Values.csv', 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        ImageJ.append(row)
        
a = np.array(ImageJ).astype(float)

#%%
plt.plot(a)

#%%
#loading files
# signal= np.load(filePathF, allow_pickle=True)
# iscell = np.load(filePathiscell, allow_pickle=True)
# ops =  np.load(filePathops, allow_pickle=True)
# ops = ops.item()

Csignal= np.load(CfilePathF, allow_pickle=True)
Ciscell = np.load(CfilePathiscell, allow_pickle=True)
Cops =  np.load(CfilePathops, allow_pickle=True)
Cops = Cops.item()

#printing data path to know which data was analysed
# key_list = list(ops.values())
# print("experiments ran through Suite2P", key_list[88])
# print("frames per folder:",ops["frames_per_folder"])
# exp= np.array(ops["frames_per_folder"])
Cexp= np.array(Cops["frames_per_folder"])

filePathaligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_exp1.npy'
#aligned_test = np.load(filePathaligned, allow_pickle=True)

#%%getting the F trace of cells 
#(and not ROIs not classified as cells) using a function I wrote


#signal_cells = fun.getcells(filePathF= filePathF, filePathiscell= filePathiscell).T

Csignal_cells = fun.getcells(filePathF= CfilePathF, filePathiscell= CfilePathiscell).T


#%%loading the metadata
#as of 2022-09-26 (the date this file was created) and previously from the experiments done after March, there are three files to load:
    #1.the log file which gives the stimulus identity
    #2. the metadata file (with 5 channels, bin file): photodiode (when stim changes), frame clock, piezo movement, pockels, sync with Arduino
    #3. the Arduino data file (with 6 channels, csv file): forward, backward, camera1, camera2, sync, time
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'

#%%
#if I want to align traces for the second, third etc experiment, I need to create one continuous metadata file because the Suite2P data is continuous

file_number_exp1 = "0"
file_number_exp2 = "1"
file_number_exp3 = "2"
file_meta_exp1 = 'Z://RawData//'+animal+ '//'+date+ '//1//NiDaqInput'+file_number_exp1+'.bin'
#file_meta_exp2 = 'Z://RawData//'+animal+ '//'+date+ '//2//NiDaqInput'+file_number_exp2+'.bin'
#file_meta_exp3 = 'Z://RawData//'+animal+ '//'+date+ '//3//NiDaqInput'+file_number_exp3+'.bin'

meta_exp1 = fun_ext.GetNidaqChannels(file_meta_exp1, numChannels=5)
#meta_exp2 = fun_ext.GetNidaqChannels(file_meta_exp2, numChannels=5)
#meta_exp3 = fun_ext.GetNidaqChannels(file_meta_exp3, numChannels=5)

#getting the photodiode info, usually the first column in the meta array
photodiode_exp1 = meta_exp1[:,0]
#photodiode_exp2 = meta_exp2[:,0]
#photodiode_exp3 = meta_exp3[:,0]

#%%plotting of full cell trace and photodiode
neuron = 2
f,ax = plt.subplots(2,sharex=False)

ax[0].plot(photodiode_exp1)
#ax[1]. plot(Csignal_cells[1980:3960,0])
ax[1]. plot(Csignal_cells[:,neuron])
#ax[1]. plot(a[:,1])
plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//Contrast//0cell'+str(neuron)+'.png')
#%%calculating the times the photodiode detected a change for all experiments (in this case up to 3)
photodiode_change_exp1 = fun_ext.DetectPhotodiodeChanges(photodiode_exp1,plot= True,kernel = 101,fs=1000, waitTime=10000)
#photodiode_change_exp2 = fun_ext.DetectPhotodiodeChanges(photodiode_exp2,plot= True,kernel = 101,fs=1000, waitTime=10000)
#photodiode_change_exp3 = fun_ext.DetectPhotodiodeChanges(photodiode_exp3,plot= True,kernel = 101,fs=1000, waitTime=10000)

#tranverse of the metadata array used because below function takes that

frame_clock_exp1 = meta_exp1.T[1]
#frame_clock_exp2 = meta_exp2.T[1]
#frame_clock_exp3 = tmeta_exp3[1]

frames_exp1 = fun_ext.AssignFrameTime(frame_clock_exp1, plot = False)
#frames_exp2 = fun_ext.AssignFrameTime(frame_clock_exp2, plot = False)
#frames_exp3 = fun_ext.AssignFrameTime(frame_clock_exp3, plot = False)



#adding up the frame clocks
#addto_exp2 = frames_exp2 + frames_exp1[-1]
#addto_exp3 = frames_exp3 + addto_exp2[-1]
#frames_all = np.concatenate((frames_exp1, addto_exp2))
#frames_all = np.concatenate((frames_exp1, addto_exp2, addto_exp3))

#adding up the photodiode changes
#photo_addto_exp2 = photodiode_change_exp2 + photodiode_change_exp1[-1]
#photo_addto_exp3 = photodiode_change_exp3 + photo_addto_exp2[-1]
#photodiode_all = np.concatenate((photodiode_change_exp1, photo_addto_exp2))
#photodiode_all = np.concatenate((photodiode_change_exp1, photo_addto_exp2, photo_addto_exp3))



#photodiode_change_new = photodiode_change + added_time
print("please choose the relevant experiment below! stim on or off etc")
#%%aliging the traces with the stimulus

#stim_on = photodiode_all[0::2]
#stim_off =  photodiode_all[1::2]

Cstim_on = photodiode_change_exp1[0::2]
Cstim_off = photodiode_change_exp1[1::2]

#the frames for the specified plane
#frames_plane1 = frames_all[plane_number_int::nr_planes]
Cframes_plane1 = frames_exp1[plane_number_int::nr_planes]
#frames_plane2 = frames_all[plane_number_int::nr_planes]

#window: specify the range of the window in seconds
window= np.array([-5, 5]).reshape(1,-1)
#aligned_all = fun.AlignStim(signal= signal_cells, time= frames_plane1, eventTimes= stim_on, window= window,timeLimit=1000)
Caligned_all = fun.AlignStim(signal= Csignal_cells, time= frames_exp1, eventTimes= Cstim_on, window= window,timeLimit=1000)
#aligned: the traces for all the stimuli for all the cells

#aligned_all_exp = aligned_all[0]
Caligned_all_exp = Caligned_all[0]
#the time is important for converting how many imaging frames represent the time window specified
#time = aligned_all[1]
Ctime = Caligned_all[1]
#%%
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//time.npy', Ctime)
#%%getting the specific aligned traces for each type of experiment
length_1 = int(photodiode_change_exp1.shape[0]/2)
length_exp2 = int(photodiode_change_exp2.shape[0]/2)
length_2 = length_1 +length_exp2


#aligned_exp1 = aligned_all_exp[:, 1:length_1, :]
#aligned_exp2 = aligned_all_exp[:, length_1:length_2, :]

stim_on_exp1 = stim_on[1:length_1]
stim_off_exp1 = stim_off[length_1:length_2]
#saving the aligned traces as npy files
#%%
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_gratings.npy', Caligned_all_exp)

#np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_exp2.npy', aligned_exp2)

"""
might be a good idea to stop this script here and use the saved aligned traces in another script for the further analysis
"""

#%%determining stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])

#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)
#%%
np.save('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//log.npy', log)
#%%the range of parameters (hard-coded, need to check in csv files put into Bonsai)
types_of_stim = 24
#the above number refers to the amount of combinations one can have with the range of parameters
if types_of_stim == 12:
        angles = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
        angles_str = ["30","60", "90", "120", "150", "180", "210", "240","270", "300", "330", "360"]
        TFreq =np.array([2])
        SFreq = np.array([0.08])
        contrast = np.array([1])
    
        #for other gratings protocols such as temp freq etc, this number should be double
elif types_of_stim == 24:
        angles = np.array([0, 90, 180, 270])
        angles_str = ["0","90","180","270"]
        tfreq_str = ["0.5", "1", "2", "4", "8", "16"]
        sfreq_str = ["0.01", "0.02", "0.04", "0.08", "0.16", "0.32"]
        contrast_str = ["0","0.125", "0.25", "0.5", "0.75", "1"]
        TFreq = np.array([0.5, 1, 2, 4, 8, 16]) 
        SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
        contrast = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])

        
#function which gives a 3D array with the indices in the aligned array (orientations, parameters (freq, etc), repetitions)       
all_parameters = fun.Get_Stim_Identity(log = log, reps = repetitions, types_of_stim =24, protocol_type = exp_name)

#%%loading behavioural data
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

tmeta= meta_exp2.T
niTime = np.array(range(0,meta_exp2.shape[0]))/1000


#getting sync signal:
syncNiDaq = tmeta[-1]
syncArd = channels[:, -1]

corrected_time = fun_ext.arduinoDelayCompensation(nidaqSync = syncNiDaq ,ardSync = syncArd, niTimes = niTime ,ardTimes = time_stamps)
corrected_time = np.around(corrected_time, decimals = 2)

#%%
#%%need to add a column of zeros to log to be able to append a 1 to trials if the trial involved running
zero = np.zeros((Caligned_all_exp.shape[1])).reshape(Caligned_all_exp.shape[1], )
log = np.column_stack((log, zero))

reps_behaviour = fun.behaviour_reps(log = log, types_of_stim = 24, reps = repetitions, protocol_type = exp_name, speed = speed, time = corrected_time, stim_on = Cstim_on, stim_off = Cstim_off)
#the above function gives the reps for each orientation for running and for rest states

#%%
running = reps_behaviour[0]
rest = reps_behaviour[1]


#%%plotting all orientations and all temp frequencies
aligned = Caligned_all_exp
time = Ctime

for neuron in range(aligned.shape[2]):
#for neuron in range(4,5):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                for freq in range(6):
                    ax[freq,angle].plot(time,aligned[:, all_parameters[angle,freq, :], neuron], c = "lightgrey")
                    ax[freq,angle].plot(time, aligned[:,all_parameters[angle,freq, :] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    #ax[freq,0].set_title(str(tfreq_str[freq]))
                    #ax[freq,0].set_title(str(sfreq_str[freq]))
                    #ax[freq,0].set_title(str(contrast_str[freq]))
                    ax[freq,0].set_ylabel(str(contrast_str[freq]), loc = "center")
            #plt.xlabel("neuron_"+str(neuron))
            plt.yticks([])  
            fig.text(0.5, 0.04, "Time(s)     ROI-"+str(neuron), ha = "center")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//Contrast//all_oris//cell'+str(neuron)+'.png')
            plt.close()

#%%plotting metadata vs trace
#for neuron in range(aligned_exp1.shape[2]):
for neuron in range(4,5):
    
    
    
    f,ax = plt.subplots()
    ax.plot(signal_cells[exp[0]:exp[0]+exp[1], neuron], label= 'traces', color = 'red')
    ax.legend()
    #ax.plot(signal_cells[0:exp[0], neuron])
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//TFreq//whole_trace2//cell'+str(neuron)+'.png')
    #plt.close()

#%%trying out seaborn
import seaborn as sns

aligned = Caligned_all_exp
for neuron in range(4,5):
    #for angle in range(0,6):
            sns.relplot(
             kind="line",
            x=time, y= aligned[:, 0:20, neuron],
            facet_kws=dict(sharex=True),
        )
            
#%% 
multi = sns.FacetGrid()
multi.map(sns.relplot,x=time, y= aligned[:, 0, neuron],
            facet_kws=dict(sharex=True),
        )
#%%plotting for other gratings protocol
#aligned = Caligned_all_exp
#import cProfile, pstats, io
#pr = cProfile.Profile()
#pr.enable()
#for neuron in range(aligned.shape[2]):



for neuron in range(7,8):
    fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
    #plt.legend(lines, labels ,loc = 'upper right', fontsize = 8)

    
#angle = 0
    for a in range(0,4):
        ax[0,a].set_title(str(angles_str[a]) + " degrees", loc = "right")
    for angle in range(0,6):
        
                    ax[angle,0].plot(time,aligned[:,running[angle], neuron], label = 'running', c = "turquoise")
                    ax[angle,0].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle,0].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle,0].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    #ax[angle,0].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle,0].xaxis.set_label_position('top')
                    ax[angle,0].set_ylabel(str(tfreq_str[angle])+ " c/s", loc = "center", fontsize = 8)
                    #ax[angle,0].set_ylabel(str(sfreq_str[angle])+ " c/d", loc = "center", fontsize = 8)

                    #ax[angle,0].set_xlabel(str(contrast_str[angle]), loc = "left")
                    
    for angle in range(6,12):
                    ax[angle-6,1].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-6,1].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-6,1].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-6,1].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-6,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    #ax[angle-6,1].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-6,1].xaxis.set_label_position('top')
                    
                    
    for angle in range(12,18):
                    ax[angle-12,2].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-12,2].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-12,2].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-12,2].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-12,2].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    #ax[angle-12,2].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-12,2].xaxis.set_label_position('top')
            
    
    for angle in range(18,24):
                    ax[angle-18,3].plot(time,aligned[:,running[angle], neuron], c = "turquoise")
                    ax[angle-18,3].plot(time,aligned[:,rest[angle], neuron], c = "plum", alpha = 0.2)
                    ax[angle-18,3].plot(time, aligned[:,running[angle] , neuron].mean(axis = 1), c = "teal")
                    ax[angle-18,3].plot(time, aligned[:,rest[angle] , neuron].mean(axis = 1), c = "purple")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                    #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[angle-18,3].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    #ax[angle-18,3].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[angle-18,3].xaxis.set_label_position('top')
                   
    #plt.xlabel("ROI-"+str(neuron), x = 1.4, y = -1)       
     
    fig.text(0.5, 0.04, "Time(s)     ROI-"+str(neuron), ha = "center")
    plt.yticks([])           
    
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//TFreq//all_oris//running_vs_rest/cell'+str(neuron)+'.png')
    #plt.close()
    
#%%