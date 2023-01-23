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

animal=  'Glaucus'
date= '2022-08-18'

#getting the preprocessed data

filePathdff ='Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.dff.npy'
filePath_calcium_timestamps ='Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.timestamps.npy'

filePath_gratingsSF = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.npy'
filePath_gratingsTF = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.temporalF.npy'
filePath_gratingscontrast = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.contrast.npy'
filePath_gratingsori = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.npy'

filePath_gratings_st = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.npy'
filePath_gratings_et = 'Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.et.npy'


calcium_dff= np.load(filePathdff, allow_pickle=True)
calcium_timestamps = np.load(filePath_calcium_timestamps, allow_pickle=True)

gratings_start =  np.load(filePath_gratings_st, allow_pickle=True)
gratings_end =  np.load(filePath_gratings_et, allow_pickle=True)
 
gratings_TF = np.load(filePath_gratingsTF, allow_pickle=True)
gratings_SF = np.load(filePath_gratingsSF, allow_pickle=True)
gratings_contrast = np.load(filePath_gratingscontrast, allow_pickle=True)
gratings_ori = np.load(filePath_gratingsori, allow_pickle=True)

#%% delete faulty photodiode changes


#def delete_faulty_photodiode_changes(
#         filePathNiDaq, faulty_frame_start, faulty_frame_end, 
#         filePath_gratings_st, filePath_gratings_et, 
#         filePath_gratingsSF, filePath_gratingsTF,
#         filePath_gratingscontrast, filePath_gratingsori
# ):
# if photodiode changes are more than the amount of stimuli shown, need to find the faulty one and remove from all stim arrays
experiment = str(1)
file_number = str(0)
filePathNiDaq = 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+'//NiDaqInput'+file_number+'.bin'


meta = fun.GetNidaqChannels(filePathNiDaq, numChannels=5)
# getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

# using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun.DetectPhotodiodeChanges_new(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)

#%%
# Get faulty frame: need to manually check the trace
faulty_frame_start = 1657789
faulty_frame_end = 1657636
faulty_change_start = int(np.where(photodiode_change == faulty_frame_start)[0]/2)
faulty_change_end = int(np.where(photodiode_change == faulty_frame_end)[0]/2)
gratings_start =  np.load(filePath_gratings_st, allow_pickle=True)
gratings_end =  np.load(filePath_gratings_et, allow_pickle=True)
 
gratings_TF = np.load(filePath_gratingsTF, allow_pickle=True)
gratings_SF = np.load(filePath_gratingsSF, allow_pickle=True)
gratings_contrast = np.load(filePath_gratingscontrast, allow_pickle=True)
gratings_ori = np.load(filePath_gratingsori, allow_pickle=True)

gratings_start_updated = np.delete(gratings_start,faulty_change_end)
gratings_end_updated = np.delete(gratings_start,faulty_change_end)
gratings_TF_updated = np.delete(gratings_TF,faulty_change_start)
gratings_SF_updated = np.delete(gratings_SF,faulty_change_start)
gratings_contrast_updated = np.delete(gratings_contrast,faulty_change_start)
gratings_ori_updated = np.delete(gratings_ori,faulty_change_start)

np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.updated.npy',gratings_start_updated)
np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.et.updated.npy',gratings_end_updated)
np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.temporalF.updated.npy',gratings_TF_updated)
np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.updated.npy',gratings_SF_updated)
np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.contrast.updated.npy',gratings_contrast_updated)
np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.updated.npy',gratings_ori_updated)

    
    
    