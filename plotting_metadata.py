# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:49:39 2022

@author: maria
"""


import numpy as np
import matplotlib.pyplot as plt
path= 'Z:/RawData/Hedes/2022-03-14/1/NiDaqInput0.bin'

def GetMetadataChannels(niDaqFilePath, numChannels = 7):
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

meta= GetMetadataChannels(path, numChannels=7)
tmeta= meta.T
 
fig, axs = plt.subplots(4)
axs[0].plot(tmeta[0, 0:60000])
axs[1].plot(tmeta[1, 0:60000])
axs[2].plot(tmeta[2, 0:60000])
axs[3].plot(tmeta[3, 0:60000])

for ax in axs.flat:
    ax.label_outer()

"""
Questions:
    sampling rate still 1000/s?
    which rows correspond to which input?
    1:Photodiode
    2:FrameClock
    3:Pockel feedback
    4:Piezo
"""
