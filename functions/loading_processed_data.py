# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:15:58 2023

@author: maria
"""

"""
analysis of pre-processed data

"""
import numpy as np
import pandas as pd
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import butter,filtfilt,medfilt
import csv
import re
import Analysis_and_Processing.functions.functions2022_07_15 as fun
import Data.Bonsai.extract_data as fun_ext
import os
import Analysis_and_Processing.functions.functions2022_07_15 as fun


animal=  'Glaucus'
date= '2022-08-18'

#getting the preprocessed data

filePathdff ='Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.dff.npy'
filePath_calcium_timestamps ='Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.timestamps.npy'
filePath_gratingsSF = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.npy'
filePath_gratingsSF_updated = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.updated.npy'
filePath_gratings_st = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.npy'
filePath_gratings_st_updated = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.updated.npy'
filePath_gratings_ori = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.npy'
filePath_gratings_ori_updated = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.updated.npy'
filePath_gratings_contrast_updated = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.contrast.updated.npy'
filePath_gratingsTF_updated = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.temporalF.updated.npy'


calcium_dff= np.load(filePathdff, allow_pickle=True)
calcium_timestamps = np.load(filePath_calcium_timestamps, allow_pickle=True)
gratings_SF = np.load(filePath_gratingsSF, allow_pickle=True)
gratings_SF_updated = np.load(filePath_gratingsSF_updated, allow_pickle=True)
gratings_st = np.load(filePath_gratings_st, allow_pickle=True)
gratings_st_updated = np.load(filePath_gratings_st_updated, allow_pickle=True)
gratings_ori = np.load(filePath_gratings_ori, allow_pickle=True)
gratings_ori_updated = np.load(filePath_gratings_ori_updated, allow_pickle=True)
gratings_contrast_updated = np.load(filePath_gratings_contrast_updated, allow_pickle=True)
gratings_TF_updated = np.load(filePath_gratingsTF_updated, allow_pickle=True)

#%% delete faulty photodiode changes

#if photodiode changes are more than the amount of stimuli shown, need to find the faulty one and remove from all stim arrays

# Load photodiode 
experiment = str(1)
file_number = str(0)
filePathmeta = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'

meta = fun.GetNidaqChannels(filePathmeta, numChannels=5)
#getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

# Get photodiode changes
photodiode_change = fun.DetectPhotodiodeChanges_new(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)
#%%
photodiode_change_updated = np.delete(np.delete(photodiode_change, 365),366)

photodiode_start = photodiode_change_updated[::2]

# Check the exact frame where it went bad: in the plot created above from function DetectPhotodiodeChanges
# for this example: 364,365,366 but double check, need to delete all from gratings-st and the other ones are fine

# Delete this datapoint in all the files
gratings_st_updated = np.delete(gratings_st_updated, 366)

#%% to check, plot the aligned traces with old and updated info

# Get only the data from that experiment

#loading ops file to get length of first experiment
plane_number = str(1)
filePathops = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//plane'+plane_number+'//ops.npy'#

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

#%% aligning traces

tmeta= meta.T
frame_clock = tmeta[1]
frame_on = fun_ext.assign_frame_time(frame_clock, plot = False)

plane_number = 1
nr_planes = 4
frames_plane1 = frame_on[plane_number::nr_planes]

window= np.array([-1, 4]).reshape(1,-1)
aligned_all = fun.AlignStim(signal= calcium_dff[0:exp[0]], time= frames_plane1, eventTimes= gratings_st_updated[0:photodiode_start.shape[0]-1], window= window,timeLimit=1000)

traces = aligned_all[0]
time = aligned_all[1]

#%%combining the datasets

log = np.stack((gratings_ori_updated, gratings_SF_updated, gratings_TF_updated, gratings_contrast_updated)).T
log_exp = log[0:photodiode_start.shape[0]]


exp_name = "SFreq"
all_parameters = fun.Get_Stim_Identity(log = log, reps = 20, types_of_stim = 24, protocol_type = str(exp_name))
#%%
zero = np.where(gratings_ori[0:photodiode_start.shape[0]] == 90)[0]

plt.plot(traces[:, zero,0])


