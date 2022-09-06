# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:24:42 2022

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
import cProfile


animal=  'Hedes'
date= '2022-07-21'
#note: if experiment type not known, put 'suite2p' instead
experiment = '2'
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '1'
log_number = '1'
plane_number = '1'
#IMPORTANT: SPECIFY THE FRAME RATE
frame_rate = 15
#the total amount of seconds to plot
seconds = 5
#specify the cell for single cell plotting

filePathlog =  'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//Log'+log_number+'.csv'

#%%
#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])
cProfile.run('re.compile("foo|bar")')
#%%
#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)
#log[0] is the degrees, log[1] would be spatial freq etc (depending on the order in the log list)
#no of stimuli specifes the total amount of stim shown
# nr_stimuli = aligned.shape[1]
# #log_Ori takes the first column of the log array because that corresponds to the first elelment in props in the GetStimulusInfo function above
log_Ori = log[:,0].reshape(480,)

#first getting the angles available, usually only 4 when trying other parameters
angles = np.array([0, 90, 180, 270])
#Temp freq
TFreq = np.array([0.5, 1, 2, 4, 8, 16])
SFreq = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
#%%
#getting indices for the same type of stim, ie same orientation and same temp frep
#for one angle
reps = 20
all_oneP_TFreq = np.zeros((reps, TFreq.shape[0])).astype(int)

log_TFreq = log[:,2]
#for angle in range(angles.shape[0]):
angle = 0
for freq in range(TFreq.shape[0]): #and j in range(TFreq.shape[0]):
    specific_P = np.where((log[:,0] == angles[angle]) & (log_TFreq == TFreq[freq])) [0]
    all_oneP_TFreq[:, freq] = specific_P
        #all_TFreq = 
    
    
#%% for all angles
#getting a 3D array with shape(orientation, repeats, TFreq)
reps = 20
all_TFreq = np.zeros((angles.shape[0], reps, TFreq.shape[0])).astype(int)


for angle in range(angles.shape[0]):
    
    for freq in range(TFreq.shape[0]): #and j in range(TFreq.shape[0]):
        specific_P = np.where((log[:,0] == angles[angle]) & (log_TFreq == TFreq[freq])) [0]
        all_TFreq[angle, :, freq] = specific_P
        


cProfile.run('re.compile("foo|bar")')