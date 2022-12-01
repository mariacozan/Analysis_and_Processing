# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:33:22 2022

@author: maria
"""

"""
pipeline for extracting DLC output and processing it to detect:
    pipul size area and change in movement for whiskerpad and nose
"""

import math
import numpy as np
import matplotlib.pyplot as plt
#import csv file

csv_path = "D://DLC-projects//csv_output//Giuseppina//2022-11-03//Video7DLC_resnet101_FaceNov22shuffle1_99000.csv"
import pandas as pd

df = pd.read_csv(r'D://DLC-projects//csv_output//Giuseppina//2022-11-03//Video7DLC_resnet101_FaceNov22shuffle1_99000.csv')
#print(df)

#%%
#get columns from df which correspond to pupil points


x_vertical = df.loc[:, ['top','bottom']]
y_vertical = df.loc[:,['top.1', 'bottom.1']]

distance_all_vertical = np.zeros(df.shape[0])

for n in range(df.shape[0]):
    distance_vertical = math.sqrt((float(x_vertical.iloc[n,0]) - float(x_vertical.iloc[n,1]))**2 + (
        float(y_vertical.iloc[n,0]) - float(y_vertical.iloc[n,1]))**2)
    distance_all_vertical[n] = distance_vertical
    
x_horizontal = df.loc[:, ['left','right']]
y_horizontal = df.loc[:,['left.1', 'right.1']]

distance_all_horizontal = np.zeros(df.shape[0])

for n in range(df.shape[0]):
    distance_horizontal = math.sqrt((float(x_horizontal.iloc[n,0]) - float(x_horizontal.iloc[n,1]))**2 + (
        float(y_horizontal.iloc[n,0]) - float(y_horizontal.iloc[n,1]))**2)
    distance_all_horizontal[n] = distance_horizontal


#calculate based on these coordinates whhat the area is per each frame

#calculate area
#%%

Area = (distance_all_vertical/2)*(distance_all_horizontal/2)*math.pi

#%%
#import functions2022_07_15 as fun

def running_info(filePath, th = 3, plot=False):
    with open(filePath) as file_name:
        csvChannels = np.loadtxt(file_name, delimiter=",")
    arduinoTime = csvChannels[:,-1]
    arduinoTimeDiff = np.diff(arduinoTime,prepend=True)
    normalTimeDiff = np.where(arduinoTimeDiff>-100)[0]
    csvChannels = csvChannels[normalTimeDiff,:]
    # convert time to second (always in ms)
    arduinoTime = csvChannels[:,-1]/1000 
    # Start arduino time at zero
    arduinoTime-=arduinoTime[0]
    csvChannels = csvChannels[:,:-1]
    numChannels = csvChannels.shape[1]
    if (plot):
        f,ax = plt.subplots(numChannels,sharex=True)
        for i in range(numChannels):
            ax[i].plot(arduinoTime,csvChannels[:,i])
            
    
    return csvChannels,arduinoTime

animal=  'Giuseppina'
#animal = input("animal name ")

date= '2022-11-03'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '3'
exp_no = int(experiment)
#experiment = input("experiment number(integer only) ")
#experiment_int = int(experiment)
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '2'
log_number = '2'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'


running_behaviour = running_info(filePathArduino, plot = True)

#%%

def DetectWheelMove(moveA,moveB,timestamps,rev_res = 1024, total_track = 59.847, plot=False):
    """
    The function detects the wheel movement. 
    At the moment uses only moveA.   
    [[ALtered the minimum from 0 to 5 because of the data from 04/08/22 -M]]
    
    Parameters: 
    moveA,moveB: the first and second channel of the rotary encoder
    rev_res: the rotary encoder resoution, default =1024
    total_track: the total length of the track, default = 59.847 (cm)
    kernel: the kernel for median filtering, default = 101.
    
    plot: plot to inspect, default = False   
    
    returns: velocity[cm/s], distance [cm]
    """
    #introducing thresholoding in case the non movement values are not 0, 5 was the biggest number for now
    
    th_index = moveA<5
    moveA[th_index] = 0
    th_index = moveB<5
    moveB[th_index] = 0
    moveA = np.round(moveA).astype(bool)
    moveB = np.round(moveB).astype(bool)
    counterA = np.zeros(len(moveA))
    counterB = np.zeros(len(moveB))
    
    # detect A move
    risingEdgeA = np.where(np.diff(moveA>0,prepend=True))[0]
    risingEdgeA = risingEdgeA[moveA[risingEdgeA]==1]
    risingEdgeA_B = moveB[risingEdgeA]
    counterA[risingEdgeA[risingEdgeA_B==0]]=1
    counterA[risingEdgeA[risingEdgeA_B==1]]=-1    
    

    
    # detect B move
    risingEdgeB = np.where(np.diff(moveB>0,prepend=True))[0]#np.diff(moveB)
    risingEdgeB = risingEdgeB[moveB[risingEdgeB]==1]
    risingEdgeB_A = moveB[risingEdgeB]
    counterA[risingEdgeB[risingEdgeB_A==0]]=-1
    counterA[risingEdgeB[risingEdgeB_A==1]]=1    
    


    dist_per_move = total_track/rev_res
    
    instDist = counterA*dist_per_move
    distance = np.cumsum(instDist)
    
    averagingTime = int(np.round(1/np.median(np.diff(timestamps))))
    sumKernel = np.ones(averagingTime)
    tsKernel = np.zeros(averagingTime)
    tsKernel[0]=1
    tsKernel[-1]=-1
    
    # take window sum and convert to cm
    distWindow = np.convolve(instDist,sumKernel,'same')
    # count time elapsed
    timeElapsed = np.convolve(timestamps,tsKernel,'same')
    
    velocity = distWindow/timeElapsed
    # if (plot):
    #     f,ax = plt.subplots(3,1,sharex=True)
    #     ax[0].plot(moveA)
    #     # ax.plot(np.abs(ADiff))
    #     ax[0].plot(Ast,np.ones(len(Ast)),'k*')
    #     ax[0].plot(Aet,np.ones(len(Aet)),'r*')
    #     ax[0].set_xlabel('time (ms)')
    #     ax[0].set_ylabel('Amplitude (V)')
        
    #     ax[1].plot(distance)
    #     ax[1].set_xlabel('time (ms)')
    #     ax[1].set_ylabel('distance (mm)')
        
    #     ax[2].plot(track)
    #     ax[2].set_xlabel('time (ms)')
    #     ax[2].set_ylabel('Move')
    
    # movFirst = Amoves>Bmoves
    
    return velocity, distance
channels = running_behaviour[0]

forward = channels[:,0]
backward = channels [:,1]
time_stamps = running_behaviour[1]

WheelMovement = DetectWheelMove(forward, backward, timestamps = time_stamps)

speed = WheelMovement[0]