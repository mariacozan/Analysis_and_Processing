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
import functions2022_07_15 as fun
#import Data.Bonsai.extract_data as fun_ext
import os

animal=  'Glaucus'
date= '2022-08-18'

#getting the preprocessed data

filePathdff ='C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.dff.npy'
filePath_calcium_timestamps ='C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//calcium.timestamps.npy'

filePath_gratingsSF = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.npy'
filePath_gratingsTF = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.temporalF.npy'
filePath_gratingscontrast = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.contrast.npy'
filePath_gratingsori = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.npy'

filePath_gratings_st = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.npy'
filePath_gratings_et = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.et.npy'


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
filePathNiDaq = 'C://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//NiDaqInput'+file_number+'.bin'


meta = fun.GetNidaqChannels(filePathNiDaq, numChannels=5)
# getting the photodiode info, usually the first column in the meta array
photodiode = meta[:,0]

# using the function from above to put the times of the photodiode changes (in milliseconds!)
photodiode_change = fun.DetectPhotodiodeChanges_new(photodiode,plot= True,kernel = 101,fs=1000, waitTime=10000)

#%%
# Get faulty frame: need to manually check the trace
faulty_change_start = 366
faulty_change_end = 367
# faulty_change_start = int(np.where(photodiode_change == faulty_frame_start)[0]/2)
# faulty_change_end = int(np.where(photodiode_change == faulty_frame_end)[0]/2)
# gratings_start =  np.load(filePath_gratings_st, allow_pickle=True)
# gratings_end =  np.load(filePath_gratings_et, allow_pickle=True)
 
# gratings_TF = np.load(filePath_gratingsTF, allow_pickle=True)
# gratings_SF = np.load(filePath_gratingsSF, allow_pickle=True)
# gratings_contrast = np.load(filePath_gratingscontrast, allow_pickle=True)
# gratings_ori = np.load(filePath_gratingsori, allow_pickle=True)

# using np.delete to delete the first error and second error 
#(but because of the way that np. delete works, this will be the same value if the subsequent value needs to be deleted from original)
gratings_start_updated = np.delete(np.delete(gratings_start,faulty_change_start),faulty_change_start)

gratings_end_updated = np.delete(np.delete(gratings_end,faulty_change_start),faulty_change_start)
gratings_TF_updated = np.delete(gratings_TF,faulty_change_start)
gratings_SF_updated = np.delete(gratings_SF,faulty_change_start)
gratings_contrast_updated = np.delete(gratings_contrast,faulty_change_start)
gratings_ori_updated = np.delete(gratings_ori,faulty_change_start)

# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.st.updated.npy',gratings_start_updated)
# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.et.updated.npy',gratings_end_updated)
# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.temporalF.updated.npy',gratings_TF_updated)
# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.spatialF.updated.npy',gratings_SF_updated)
# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.contrast.updated.npy',gratings_contrast_updated)
# np.save('Z://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//suite2p//PreprocessedFiles//gratings.ori.updated.npy',gratings_ori_updated)

    
    
    