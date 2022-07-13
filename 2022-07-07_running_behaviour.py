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

def GetMetadataChannels(niDaqFilePath, numChannels):
    """
    

    Parameters
    ----------
    niDaqFilePath : string
        the path of the nidaq file.
    numChannels : int, optional
        Number of channels in the file. The default is 7.

    Returns
    -------
    niDaq : matrix
        the matrix of the niDaq signals [time X channels]

    """
    niDaq = np.fromfile(niDaqFilePath, dtype= np.float64)
    niDaq = np.reshape(niDaq,(int(len(niDaq)/numChannels),numChannels))
    return niDaq

def DetectWheelMove(moveA,moveB,rev_res = 1024, total_track = 598.47,plot=True):
    """
    The function detects the wheel movement. 
    At the moment uses only moveA.    
    
    Parameters: 
    moveA,moveB: the first and second channel of the rotary encoder
    rev_res: the rotary encoder resoution, default =1024
    total_track: the total length of the track, default = 598.47 (mm)
    kernel: the kernel for median filtering, default = 101.
    
    plot: plot to inspect, default = False   
    
    returns: distance
    """
    
    
    # make sure all is between 1 and 0
    moveA /= np.max(moveA)
    moveA -= np.min(moveA)
    moveB /= np.max(moveB)
    moveB -= np.min(moveB)
    
    # detect A move
    ADiff = np.diff(moveA)
    Ast = np.where(ADiff >0.5)[0]
    Aet = np.where(ADiff <-0.5)[0]
    
    # detect B move
    BDiff = np.diff(moveB)
    Bst = np.where(BDiff >0.5)[0]
    Bet = np.where(BDiff <-0.5)[0]
    
    #Correct possible problems for end of recording
    if (len(Ast)>len(Aet)):
        Aet = np.hstack((Aet,[len(moveA)]))
    elif (len(Ast)<len(Aet)):
        Ast = np.hstack(([0],Ast))   
    
    
    dist_per_move = total_track/rev_res
    
    # Make into distance
    track = np.zeros(len(moveA))
    track[Ast] = dist_per_move
    
    distance = np.cumsum(track)
        
    if (plot):
        f,ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(moveA)
        # ax.plot(np.abs(ADiff))
        ax[0].plot(Ast,np.ones(len(Ast)),'k*')
        ax[0].plot(Aet,np.ones(len(Aet)),'r*')
        ax[0].set_xlabel('time (ms)')
        ax[0].set_ylabel('Amplitude (V)')
        
        ax[1].plot(distance)
        ax[1].set_xlabel('time (ms)')
        ax[1].set_ylabel('distance (mm)')
        
        ax[2].plot(track)
        ax[2].set_xlabel('time (ms)')
        ax[2].set_ylabel('Move')
    
    # movFirst = Amoves>Bmoves
    
    return distance

def running_info(filePath):
    file = open(filePath)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
        
    return rows

animal=  'Bellinda'
date= '2022-07-08'
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '1'
log_number = '1'
plane_number = '1'
#IMPORTANT: SPECIFY THE FRAME RATE
frame_rate = 15
res = ''
filePathF ='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//F.npy'
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//ops.npy'
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'
filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'
filePathArduino = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//ArduinoInput'+file_number+'.csv'
signal= np.load(filePathF, allow_pickle=True)
filePathiscell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+res+'suite2p//plane'+plane_number+'//iscell.npy'

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

running_behaviour = np.array(running_info(filePathArduino)).astype('float64')
wheel_movement = DetectWheelMove(running_behaviour[:,0],running_behaviour[:,1])

#code gives the absolute distance travelled overall during the whole session 
#need to get the velocity but in a meaningful way
#sampling at 1000Hz so divide into 100ms chunks and then calculate mm/ms velocity
#so then remember to convert to cm/s!!