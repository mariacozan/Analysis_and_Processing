# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:57:51 2023

@author: maria
"""

"""
Figure creation
"""

#raw full traces

"""function which plots:
    -the locomotion trace
    -the pupil trace
    -neuron example traces of my choosing
    
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import Analysis_and_Processing.functions.functions2022_07_15 as fun
import scipy
#%%
#defining path
animal=  'Hedes'
date= '2022-08-05'

exp_nr= 1
experiment= str(exp_nr)

#NDIN is the number in the NiDaq binary file, bear in mind this is not always experiment number - 1, always double check
#NDIN= exp_nr-1
#in case number is not exp number - 1 then put it in manually here:
NDIN = exp_nr-1
NDIN = 0
NiDaqInputNo= str(NDIN)
ADIN = 0
ArduinoInputNo = str(ADIN)
filePathInput='Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//NiDaqInput'+NiDaqInputNo+'.bin'
#need to add custom titles for plots
filePathOutput2s= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_2seconds-interval.png'
filePathOutput500ms= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_500ms-interval.png'

filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+ArduinoInputNo+'.csv'
filePathOutputArduino= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//ArduinoData500ms.png'
filePathOutputspeed= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//speed.png'



#%%speed

running_behaviour = fun.running_info(filePathArduino, plot = True)
channels = running_behaviour[0]

forward = channels[:,0]
backward = channels [:,1]
time_stamps = running_behaviour[1]

WheelMovement = fun.DetectWheelMove(forward, backward, timestamps = time_stamps)


arduinoTime = channels[:,-1]/1000 
# Start arduino time at zero
arduinoTime-=arduinoTime[0]
    
speed = WheelMovement[0]
#fig,ax = plt.subplots()
#ax.plot(speed)
#plt.savefig(filePathOutputspeed)

#%%pupil area
#note: before importing csv file I deleted the first and third row which only contained irrelevant info
#import csv file
video_number = str(0)
csv_path = r'D://DLC-projects//csv_output//'+animal+ '//'+date+ '//'+experiment+'//Video'+video_number+'DLC_resnet101_FaceNov22shuffle1_99000.csv'
import pandas as pd

df = pd.read_csv(csv_path)
#print(df)

#get columns from df which correspond to pupil points
x_vertical = df.loc[:, ['top','bottom']]
y_vertical = df.loc[:,['top.1', 'bottom.1']]
confidence = df.loc[:,['top.2']]
exclude = np.where(np.array(confidence)<0.9)[0]

distance_all_vertical = np.zeros(df.shape[0])


for n in range(df.shape[0]):
    distance_vertical = math.sqrt((float(x_vertical.iloc[n,0]) - float(x_vertical.iloc[n,1]))**2 + (
        float(y_vertical.iloc[n,0]) - float(y_vertical.iloc[n,1]))**2)
    distance_all_vertical[n] = distance_vertical


distance_all_vertical[exclude]= 0
distance_all_vertical = distance_all_vertical[distance_all_vertical>0]
    
x_horizontal = df.loc[:, ['left','right']]
y_horizontal = df.loc[:,['left.1', 'right.1']]

distance_all_horizontal = np.zeros(df.shape[0])

for n in range(df.shape[0]):
    distance_horizontal = math.sqrt((float(x_horizontal.iloc[n,0]) - float(x_horizontal.iloc[n,1]))**2 + (
        float(y_horizontal.iloc[n,0]) - float(y_horizontal.iloc[n,1]))**2)
    distance_all_horizontal[n] = distance_horizontal

distance_all_horizontal[exclude]= 0
distance_all_horizontal = distance_all_horizontal[distance_all_horizontal>0]
#calculate based on these coordinates whhat the area is per each frame

#calculate area
Area = (distance_all_vertical/2)*(distance_all_horizontal/2)*math.pi

#%%neural traces

exp_name = 'SFreq'
file_number = '0'
log_number = '0'
plane_number = '1'
plane_number_int = int(plane_number)
nr_planes = 4
repetitions = 30
types_of_stim = 24

filePathaligned = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//aligned_gratings_good.npy'
filePathtime = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//time.npy'
filePathparameters = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_parameters.npy'
filePathlogarray = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//log.npy'
filePathsignal = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//signal_cells.npy'

aligned_array = np.load(filePathaligned, allow_pickle=True)
time = np.load(filePathtime, allow_pickle=True)
log = np.load(filePathlogarray, allow_pickle = True) 
all_parameters = np.load(filePathparameters, allow_pickle=True)
signal_cells = np.load(filePathsignal, allow_pickle=True)


#%%downsampling the speed array and pupil array to match the signal_cells

from scipy.interpolate import interp1d

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

#downsampling to the length of the traces
downsampled_speed = downsample(speed,signal_cells.shape[0])
downsampled_area = downsample(Area, signal_cells.shape[0])

#%%
fig,ax= plt.subplots(2)
ax[0].plot(downsampled_speed)
ax[0].set_ylabel("speed")
ax[1].plot(downsampled_area)
ax[1].set_ylabel("pupil")

#%%time
frame_rate = 14.5425
#getting time in minutes

time_all = np.array(range(0,signal_cells.shape[0]))/(frame_rate*60)

#%%plotting



#for neuron in range(signal_cells.shape[1]):
for neuron in range(169,170):
    fig,ax = plt.subplots(5,  sharex = True, sharey = False)
    
    ax[0].plot(time_all,downsampled_area, c = "lightseagreen")
    ax[0].set_ylabel("pupil")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].plot(time_all,downsampled_speed, c = "chocolate")
    ax[1].set_ylabel("running"
                     "\n"
                     "(cm/s)")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[2].plot(time_all,signal_cells[:,11], c = "black")
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[3].plot(time_all,signal_cells[:,17], c = "darkblue")
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[4].plot(time_all,signal_cells[:,45], c = "darkmagenta")
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['right'].set_visible(False)
    #ax[2].set_ylabel("ROI"+str(neuron))
    ax[2].set_ylabel("Neuron 1")
    ax[3].set_ylabel("Neuron 2")
    ax[4].set_ylabel("Neuron 3")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].axes.yaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].axes.yaxis.set_ticklabels([])
    ax[2].axes.yaxis.set_ticklabels([])
    ax[2].axes.xaxis.set_ticklabels([])
    ax[3].axes.xaxis.set_ticklabels([])
    ax[3].axes.yaxis.set_ticklabels([])
    ax[4].axes.yaxis.set_ticklabels([])
    plt.xlabel("Time(minutes)")
    #plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//traces_and_behaviour//cell'+str(neuron)+'.png')
    #plt.close()

#%%plotting heatmaps
import seaborn as sns

signal_cells_swap = np.swapaxes(signal_cells,0, 1)


signal_cells_T = signal_cells.T
#%%
#plt.figure()
norm_speed = downsampled_speed/np.argmax(downsampled_speed,0)
#i = np.argsort(signal_cells[169,:])
speed_sort = np.argsort(downsampled_speed)
signal_cells_sorted = signal_cells_swap[:,speed_sort]
norm_signal_cells = signal_cells_sorted/np.nanmax(signal_cells_sorted,0)

#%%heatmap
plt.figure()
ax = sns.heatmap(
    #signal_cells_T/np.nanmax(signal_cells_T,0)
    norm_signal_cells, 
    vmin=-1, vmax=1, center=0,
    cmap= "coolwarm",
    #sns.diverging_palette(20, 220, n=200),
    square=False, yticklabels= "auto", xticklabels= False
)

#%%plotting traces

filePath_running_reps = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//reps_running.npy'
filePath_rest_reps = 'D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//reps_rest.npy'

running = list(np.load(filePath_running_reps, allow_pickle=True))
rest = list(np.load(filePath_rest_reps, allow_pickle=True))


aligned = aligned_array
running_oris = running
rest_oris = rest

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
        
        
#for neuron in range(57,58):
fig,ax = plt.subplots(6,4, sharex = True, sharey = False)

neuron1 = 32
neuron2 = 16
neuron3 = 85
neuron4 = 31
#angle = 0
for a in range(0,4):
    ax[0,0].set_title("increased", loc = "center")
    ax[0,1].set_title("decreased", loc = "center")
    ax[0,2].set_title("no change", loc = "center")
    ax[0,3].set_title("baseline"
                      "\n"
                      "increase", loc = "center")
    #ax[0,a].set_title(str(angles_str[a])+"degrees")
for angle in range(0,6):
                #ax[angle,0].set_ylabel(str(sfreq_str[angle])+ " c/s", loc = "center", fontsize = 8)
                #ax[angle,0].plot(time,aligned[:,running[angle], neuron1], c = "plum")
                #ax[angle,0].plot(time,aligned[:,rest[angle], neuron1], c = "turquoise", alpha = 0.2)
                ax[angle,0].plot(time, aligned[:,running[angle] , neuron1].mean(axis = 1), c = "mediumvioletred")
                ax[angle,0].plot(time, aligned[:,rest[angle] , neuron1].mean(axis = 1), c = "teal")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                #ax[angle,0].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                #ax[angle,0].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                ax[angle,0].xaxis.set_label_position('top')
                ax[angle,0].set_yticks([])
                
                #ax[angle,0].set_xlabel(str(contrast_str[angle]), loc = "left")
                
for angle in range(6,12):
                ax[angle-6,1].plot(time,aligned[:,running[angle], neuron2], c = "plum")
                ax[angle-6,1].plot(time,aligned[:,rest[angle], neuron2], c = "turquoise", alpha = 0.2)
                ax[angle-6,1].plot(time, aligned[:,running[angle] , neuron2].mean(axis = 1), c = "mediumvioletred")
                ax[angle-6,1].plot(time, aligned[:,rest[angle] , neuron2].mean(axis = 1), c = "teal")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                ax[angle-6,1].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                ax[angle-6,1].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                ax[angle-6,1].xaxis.set_label_position('top')
                ax[angle-6,1].set_yticks([])
                
for angle in range(12,18):
                ax[angle-12,2].plot(time,aligned[:,running[angle], neuron3], c = "plum")
                ax[angle-12,2].plot(time,aligned[:,rest[angle], neuron3], c = "turquoise", alpha = 0.2)
                ax[angle-12,2].plot(time, aligned[:,running[angle] , neuron3].mean(axis = 1), c = "mediumvioletred")
                ax[angle-12,2].plot(time, aligned[:,rest[angle] , neuron3].mean(axis = 1), c = "teal")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                ax[angle-12,2].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                ax[angle-12,2].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                ax[angle-12,2].xaxis.set_label_position('top')
                ax[angle-12,2].set_yticks([])

for angle in range(18,24):
    #for angle_true in range(6,12):
                ax[angle-18,3].plot(time,aligned[:,running[angle], neuron4], c = "plum")
                ax[angle-18,3].plot(time,aligned[:,rest[angle], neuron4], c = "turquoise", alpha = 0.2)
                ax[angle-18,3].plot(time, aligned[:,running[angle] , neuron4].mean(axis = 1), c = "mediumvioletred")
                ax[angle-18,3].plot(time, aligned[:,rest[angle] , neuron4].mean(axis = 1), c = "teal")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron], c = "lightgrey")
                #ax[0,angle].plot(time, aligned[:,all_oris[angle,:] , neuron].mean(axis = 1), c = "black")
                ax[angle-18,3].axvline(x=0, c="red", linestyle="dashed", linewidth = 1)
                ax[angle-18,3].axvline(x=2, c="blue", linestyle="dashed", linewidth = 1)
                ax[angle-18,3].xaxis.set_label_position('top')
                ax[angle-18,3].set_yticks([])
                handles, labels = ax[angle-18,3].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center')
                
plt.yticks([])                     
fig.text(0.5, 0.04, "Time(s)", ha = "center")
              
    
    # plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//SFreq//all_oris//running_vs_rest_moreROIs//cell'+str(neuron)+'.png')
    # plt.close()

#%%tuning curve figures





SFreq = TFreq
running_oris = running
rest_oris = rest
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
##running_values = np.stack((running_values0, running_values90, running_values180, running_values270))
for neuron in range(aligned.shape[2]):
#for neuron in range(0,1):    
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
    
    fig.text(0.5, 0.04, "                Frequency(cycles/sec)      ROI-"+str(neuron), ha = "center")
    plt.savefig('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//all_oris//running_vs_rest_moreROIs//tuning_curves_all//cell'+str(neuron)+'.png')
    plt.close()
