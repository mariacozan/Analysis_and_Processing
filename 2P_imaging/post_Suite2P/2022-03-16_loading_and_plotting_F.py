# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:10:03 2022

@author: maria
"""

import numpy as np
import matplotlib.pyplot as plt


#adjust file locations as needed, there will be the same set of data for every plane
F= np.load('C:/Users/maria/Documents/GitHub/internal/2P_imaging/Suite2p/Suite2P_output/Hedes//2022-03-23//F.npy', allow_pickle=True)
Fneu= np.load('C:/Users/maria/Documents/GitHub/internal/2P_imaging/Suite2p/Suite2P_output/Hedes//2022-03-23//Fneu.npy', allow_pickle=True)
ops= np.load('C:/Users/maria/Documents/GitHub/internal/2P_imaging/Suite2p/Suite2P_output/Hedes//2022-03-23//ops.npy', allow_pickle=True)
ops= ops.item()


#plotting an example trace
#n is the ROI you want to plot
n= 19
#fr is the frame rate per plane, usually 5.996 for 512 resolution and 5 planes and 7.27 for 256 resolution and 8 planes
fr= 6
#converting frames  to time
plt.plot(np.array(range(len(F[n])))/fr,F[n], c="b")
plt.plot(np.array(range(len(F[n])))/fr,Fneu[n], c="r")
plt.xlabel("Time(s)")
plt.ylabel("Fluorescence Intensity")
