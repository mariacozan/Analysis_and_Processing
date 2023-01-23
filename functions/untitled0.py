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

animal=  'Giuseppina'
date= '2022-11-16'

#getting the preprocessed data

filePathdff ='D://PreprocessedFiles//'+animal+ '//'+date+ '//PreprocessedFiles//calcium.dff.npy'
calcium_dff= np.load(filePathdff, allow_pickle=True)

#%%
plt.plot(calcium_dff[:,6])