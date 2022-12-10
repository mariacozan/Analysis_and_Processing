# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:31:00 2022

@author: maria

"""
import numpy as np
import pandas as pd
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
import scipy 
from scipy.signal import butter,filtfilt,medfilt
import csv
import re
import functions2022_07_15 as fun
import extract_data as fun_ext
import seaborn as sns
sns.set()

#%%input needed:specify experiment details

animal=  'Giuseppina'
#animal = input("animal name ")
date= '2022-11-03'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
exp_name = 'SFreq'
file_number = '1'
log_number = '1'
plane_number = '1'
plane_number_int = int(plane_number)
nr_planes = 4
repetitions = 30
types_of_stim = 24
#%%

filePathaligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_gratings_good.npy'
filePathtime = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//time.npy'
filePathparameters = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_parameters.npy'
aligned_array = np.load(filePathaligned, allow_pickle=True)
time = np.load(filePathtime, allow_pickle=True)
filePathlogarray = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//log.npy'
log = np.load(filePathlogarray, allow_pickle = True) 
all_parameters = np.load(filePathparameters, allow_pickle=True)
filePathlocomotion = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//locomotion.npy'
locomotion = np.load(filePathlocomotion, allow_pickle=True)
# #%%
# a0 = np.where((log[:,0] == 0))[0]
# a180 = np.where((log[:,0] == 180))[0]

#%%

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
#%%
plt.plot(time, aligned_array[:,4,0])

#%%
aligned = aligned_array
#for neuron in range(aligned.shape[2]):
#for neuron in range(4,5):
neuron = 2       
fig,ax = plt.subplots(1, sharex = True, sharey = True)
    
    #for n in range(0,40):
ax.plot(time,aligned[:,:, neuron], c = "darkgrey")
ax.plot(time,aligned[:,:, neuron].mean(axis = 1), c = "black")

#ax.plot(time, aligned[:,:, neuron].mean(axis = 1), c = "black")
ax.axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
ax.axvline(x=0.5, c="blue", linestyle="dashed", linewidth = 1)
#ax.axvline(, c="red", linestyle="dashed", linewidth = 1)        
#plt.xlabel("neuron_"+str(neuron))
 
fig.text(0.5, 0.04, "Time(s)     ROI-"+str(neuron), ha = "center")


#plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//Contrast//cell'+str(neuron)+'.png')
  
import pickle
pickle.dump(fig, open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//cell'+str(neuron)+'.pickle', 'wb'))
#%%plotting traces to check if they are right
aligned = aligned_array
for neuron in range(aligned.shape[2]):
#for neuron in range(7,8):       
            fig,ax = plt.subplots(6,4, sharex = True, sharey = True)
        
            for angle in range(0,4):
                
                for freq in range(6):
                    ax[freq,0].set_ylabel(str(sfreq_str[freq])+ " c/s", loc = "center", fontsize = 8)
                    ax[freq,angle].plot(time,aligned[:, all_parameters[angle,freq, :], neuron], c = "darkgrey")
                    ax[freq,angle].plot(time, aligned[:,all_parameters[angle,freq, :] , neuron].mean(axis = 1), c = "black")
                    ax[freq,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
                    #ax[freq,0].set_title(str(tfreq_str[freq]))
                    ax[freq,0].set_title(str(sfreq_str[freq]))
                    #ax[freq,0].set_title(str(contrast_str[freq]))
                    #ax[freq,0].set_ylabel(str(contrast_str[freq]), loc = "center")
            #plt.xlabel("neuron_"+str(neuron))
            plt.yticks([])
            #plt.xticks(ticks = [0.5,1, 2, 4, 8, 12, 16])
            fig.text(0.5, 0.04, "Time(s)     ROI-"+str(neuron), ha = "center")
            plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_traces//cell'+str(neuron)+'.png')
            plt.close()
#%% plotting for orientation responses
   
#for neuron in range(aligned.shape[2]):
for neuron in range(10,11):
    fig,ax = plt.subplots(3,4, sharex = True, sharey = True)
#angle = 0
    for angle in range(0,4):
                    ax[0,angle].plot(time,aligned[:,all_parameters[angle], neuron], c = "darkgrey")
                    ax[0,angle].plot(time,aligned[:,all_parameters[angle], neuron].mean(axis = 1), c = "black")
                    ax[0,angle].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[0,angle].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[0,angle].set_title(str(angles_str[angle]))
    for angle in range(4,8):
                    ax[1,angle-4].plot(time, aligned[:,all_parameters[angle,:] , neuron], c = "darkgrey")
                    ax[1,angle-4].plot(time, aligned[:,all_parameters[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[1,angle-4].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[1,angle-4].set_title(str(angles_str[angle]))
    for angle in range(8,12):
                    ax[2,angle-8].plot(time, aligned[:,all_parameters[angle,:] , neuron], c = "darkgrey")
                    ax[2,angle-8].plot(time, aligned[:,all_parameters[angle,:] , neuron].mean(axis = 1), c = "black")
                    ax[2,angle-8].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                    ax[2,angle-8].set_title(str(angles_str[angle]))
                    
    plt.yticks([])
            #plt.xticks(ticks = [0.5,1, 2, 4, 8, 12, 16])
    fig.text(0.5, 0.04, "Time(s)     ROI-"+str(neuron), ha = "center")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_traces//cell'+str(neuron)+'.png')
    #plt.close()




#%%tuning_curves analysis

#getting the mean and st dev for all the parameters


all_p = all_parameters
range_of_parameters = TFreq

values= np.zeros((all_parameters.shape[0], range_of_parameters.shape[0],aligned.shape[2]))
all_sem_p = np.zeros((all_parameters.shape[0],range_of_parameters.shape[0],aligned.shape[2]))
all_sem_m = np.zeros((all_parameters.shape[0],range_of_parameters.shape[0],aligned.shape[2]))





for neuron in range(aligned.shape[2]):
    for angle  in range(all_parameters.shape[0]):
        
        for freq in range(range_of_parameters.shape[0]):
                baseline0 = aligned[8:17, all_p[angle, freq,:], neuron].mean(axis = 0)
                trace0 = aligned[17:,all_p[angle,freq, :], neuron].mean(axis = 0)
                norm0 = (trace0 - baseline0).mean(axis = 0)
                sem_plus0 = norm0 + scipy.stats.sem(trace0, axis=0)
                sem_minus0 = norm0 - scipy.stats.sem(trace0, axis=0)
                all_sem_p[angle,freq, neuron] = sem_plus0
                all_sem_m[angle, freq, neuron] = sem_minus0
                
                values[angle, freq, neuron] = norm0

#%%mean and st dev for ori tuning curves
import scipy 
all_p = all_parameters
values= np.zeros((all_parameters.shape[0],aligned.shape[2]))
all_sem_p = np.zeros((all_parameters.shape[0],aligned.shape[2]))
all_sem_m = np.zeros((all_parameters.shape[0],aligned.shape[2]))
    
for neuron in range(aligned.shape[2]):
    for angle  in range(all_parameters.shape[0]):
        
                baseline0 = aligned[8:17, all_p[angle,:], neuron].mean(axis = 0)
                trace0 = aligned[17:,all_p[angle, :], neuron].mean(axis = 0)
                norm0 = (trace0 - baseline0).mean(axis = 0)
                sem_plus0 = norm0 + scipy.stats.sem(trace0, axis=0)
                sem_minus0 = norm0 - scipy.stats.sem(trace0, axis=0)
                all_sem_p[angle, neuron] = sem_plus0
                all_sem_m[angle, neuron] = sem_minus0
                
                values[angle, neuron] = norm0
#%%plotting tuning curves
for neuron in range(aligned.shape[2]):
#for neuron in range(4,5):    
    fig,ax = plt.subplots(2,2, sharex = True, sharey = True)
    
    ax[0,0].scatter(range_of_parameters,values[0, :,neuron], c = "black")
    ax[0,0].plot(range_of_parameters,values[0, :,neuron], c = "black")
    ax[0,0].fill_between(range_of_parameters, all_sem_p[0, :,neuron], all_sem_m[0, :,neuron], alpha=0.5, color = "gray")
    ax[0,0].set_title(str(angles_str[0]) + " degrees", loc = "center")
    
    ax[1,0].scatter(range_of_parameters,values[1, :,neuron], c = "black")
    ax[1,0].plot(range_of_parameters,values[1, :,neuron], c = "black")
    ax[1,0].fill_between(range_of_parameters, all_sem_p[1, :,neuron], all_sem_m[1, :,neuron], alpha=0.5, color = "gray")
    ax[1,0].set_title(str(angles_str[1]) + " degrees", loc = "center")
    
    ax[0,1].scatter(range_of_parameters,values[2, :,neuron], c = "black")
    ax[0,1].plot(range_of_parameters,values[2, :,neuron], c = "black")
    ax[0,1].fill_between(range_of_parameters, all_sem_p[2, :,neuron], all_sem_m[2, :,neuron], alpha=0.5, color = "gray")
    ax[0,1].set_title(str(angles_str[2]) + " degrees", loc = "center")
    
    ax[1,1].scatter(range_of_parameters,values[3, :,neuron], c = "black")
    ax[1,1].plot(range_of_parameters,values[3, :,neuron], c = "black")
    ax[1,1].fill_between(range_of_parameters, all_sem_p[3, :,neuron], all_sem_m[3, :,neuron], alpha=0.5, color = "gray")
    ax[1,1].set_title(str(angles_str[3]) + " degrees", loc = "center")
    
    
    fig.text(0.5, 0.04, "                Frequency(cycles/sec)      ROI-"+str(neuron), ha = "center")
    plt.xticks(ticks = [0.5,1, 2, 4, 8, 12, 16])
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//tuning_curves//cell'+str(neuron)+'.png')
    plt.close()
    
#%%plotting ori curves
#for neuron in range(aligned.shape[2]):
for neuron in range(0,1):
    fig,ax = plt.subplots(1,  sharex = True, sharey = True)
    #for angle in range(angles.shape[0]):
    ax.scatter(angles,values[:,neuron], c = "black")
    ax.plot(angles,values[:,neuron], c = "black")
    ax.fill_between(angles, all_sem_p[:,neuron], all_sem_m[:,neuron], alpha=0.5, color = "darkgrey")
    fig.text(0.5, 0.04, "Degrees", ha = "center")
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//'+res+'//plane'+plane_number+'//'+exp_name+'//tuning_curves//cell'+str(neuron)+'.png')

#for radial plots add: subplot_kw={'projection': 'polar'},
    
#%%getting max values for population distribution temp freq
np.where(values[angle,:,neuron] == np.max(values[angle,:,neuron]))[0]


max_resp = np.zeros((all_p.shape[0], aligned.shape[2]))

for neuron in range(aligned.shape[2]):
    for angle in range(all_p.shape[0]):
        max_temp = np.array(np.where(values[angle,:,neuron] == np.max(values[angle,:,neuron]))[0])
        if max_temp.shape[0] > 0:
            max_temp = max_temp[0]
        max_resp[angle,neuron] = max_temp
    
if exp_name == "TFreq":
    
    max_resp[max_resp == 0] = 0.5
    max_resp[max_resp == 3] = 4
    max_resp[max_resp == 4] = 8
    max_resp[max_resp == 5] = 16
    
if exp_name == "SFreq":
    max_resp[max_resp == 0] = 0.01
    max_resp[max_resp == 1] = 0.02
    max_resp[max_resp == 2] = 0.04
    max_resp[max_resp == 3] = 0.08
    max_resp[max_resp == 4] = 0.16
    max_resp[max_resp == 5] = 0.32

#%%max resp for ori curves

#np.where(values[angle,neuron] == np.max(values[angle,neuron]))[0]


max_resp = np.zeros((aligned.shape[2]))

for neuron in range(aligned.shape[2]):
    for angle in range(all_p.shape[0]):
        max_temp = np.array(np.where(values[:,neuron] == np.max(values[:,neuron]))[0])
        if max_temp.shape[0] > 0:
            max_temp = max_temp[0]
        max_resp[neuron] = max_temp
        
max_resp[max_resp == 0] = 30
max_resp[max_resp == 1] = 60
max_resp[max_resp == 2] = 90
max_resp[max_resp == 3] = 120
max_resp[max_resp == 4] = 150
max_resp[max_resp == 5] = 180
max_resp[max_resp == 6] = 210
max_resp[max_resp == 7] = 240
max_resp[max_resp == 8] = 270
max_resp[max_resp == 9] = 300
max_resp[max_resp == 10] = 330
max_resp[max_resp == 11] = 360
     #%%   
fig,axs = plt.subplots()
sns.histplot(max_resp)
axs.set_title("Orientation and direction tuning curve", loc = "center")
plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//population_preference.png')
#axs.set_xlabel(angles)
#%%plotting population distributions for T Freq
fig,axs = plt.subplots(2,2, sharex = True, sharey = True)
sns.histplot(max_resp[0,:], log_scale = 2, ax = axs[0,0], binwidth = 0.15).set_title(str(angles_str[0])+" degrees")
ax[0,0].set_title(str(angles_str[0]) + " degrees", loc = "center")
sns.histplot(max_resp[1,:], log_scale = 2, ax = axs[0,1], binwidth = 0.15).set_title(str(angles_str[1])+" degrees")
ax[0,1].set_title(str(angles_str[1]) + " degrees", loc = "center") 
sns.histplot(max_resp[2,:], log_scale = 2, ax = axs[1,0], binwidth = 0.15).set_title(str(angles_str[2])+" degrees")
ax[1,0].set_title(str(angles_str[2]) + " degrees", loc = "center")
sns.histplot(max_resp[3,:], log_scale = 2, ax = axs[1,1], binwidth = 0.15).set_title(str(angles_str[3])+" degrees")
ax[1,1].set_title(str(angles_str[3]) + " degrees", loc = "center")
fig.text(0.5, 0.01, "                Temporal Frequency      ", ha = "center")
plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//population_preference_improved.png')

#%%plotting population distributions for S Freq
fig,axs = plt.subplots(2,2, sharex = True, sharey = True)
sns.histplot(max_resp[0,:], log_scale = 2, ax = axs[0,0], binwidth = 0.15).set_title(str(angles_str[0])+" degrees")
ax[0,0].set_title(str(angles_str[0]) + " degrees", loc = "center")
sns.histplot(max_resp[1,:], log_scale = 2, ax = axs[0,1], binwidth = 0.15).set_title(str(angles_str[1])+" degrees")
ax[0,1].set_title(str(angles_str[1]) + " degrees", loc = "center") 
sns.histplot(max_resp[2,:], log_scale = 2, ax = axs[1,0], binwidth = 0.15).set_title(str(angles_str[2])+" degrees")
ax[1,0].set_title(str(angles_str[2]) + " degrees", loc = "center")
sns.histplot(max_resp[3,:], log_scale = 2, ax = axs[1,1], binwidth = 0.15).set_title(str(angles_str[3])+" degrees")
ax[1,1].set_title(str(angles_str[3]) + " degrees", loc = "center")
fig.text(0.5, 0.01, "                Spatial Frequency      ", ha = "center")
plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//population_preference.png')


#%%saving fig in an interactive way
import pickle
pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
#%%loading figure
#only supports pyplot figures (so not seaborn probs)
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show() # Show the figure, edit it, etc.!
#%%
data = figx.axes[0].lines[0].get_data()





