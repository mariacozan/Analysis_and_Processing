# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:31:56 2022

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
import functions2022_07_15 as fun
import pandas as pd

#code to write: extract trace from csv file (exported csv file from ImageJ plot z axis profile)
filePath = 'D://Values.csv'
filePathlog = 'D://Log9.csv'
#getting stimulus identity
Log_list = fun.GetStimulusInfo (filePathlog, props = ["Ori", "SFreq", "TFreq", "Contrast"])

#converting the list of dictionaries into an array and adding the time of the stimulus
#worked easiest by using pandas dataframe
log = np.array(pd.DataFrame(Log_list).values).astype(np.float64)

with open(filePath) as file_name:
        trace = np.loadtxt(file_name, delimiter =",")



plt.plot(trace[:,1])

