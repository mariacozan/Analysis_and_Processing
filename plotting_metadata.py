# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:49:39 2022

@author: maria
"""


import numpy as np
import matplotlib.pyplot as plt
path= 'Z:/RawData/Hedes/2022-03-14/1/NiDaqInput0.bin'

def GetMetadataChannels(niDaqFilePath, numChannels = 4):
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
#plotting for  
fig, axs = plt.subplots(7)
axs[0].plot(tmeta[0, 50000:52000])
axs[0].title.set_text("Photodiode")
axs[1].plot(tmeta[1, 50000:52000])
axs[1].title.set_text("Frame Clock")
axs[2].plot(tmeta[2, 50000:52000])
axs[2].title.set_text("Pockel feedback")
axs[3].plot(tmeta[3, 50000:52000])
axs[3].title.set_text("Piezo")
axs[4].plot(tmeta[4, 50000:52000])
axs[4].title.set_text("Wheel-F")
axs[5].plot(tmeta[5, 50000:52000])
axs[5].title.set_text("Wheel-B")
axs[6].plot(tmeta[6, 50000:52000])
axs[6].title.set_text("Camera")
for ax in axs.flat:
    ax.label_outer()
    
fig, axs = plt.subplots(7, squeeze=True)
axs[0].plot(tmeta[0, 14500:15000])
axs[0].title.set_text("Photodiode")
axs[1].plot(tmeta[1, 14500:15000])
axs[1].title.set_text("Frame Clock")
axs[2].plot(tmeta[2, 14500:15000])
axs[2].title.set_text("Pockel feedback")
axs[3].plot(tmeta[3, 14500:15000])
axs[3].title.set_text("Piezo")
axs[4].plot(tmeta[4, 14500:15000])
axs[4].title.set_text("Wheel-F")
axs[5].plot(tmeta[5, 14500:15000])
axs[5].title.set_text("Wheel-B")
axs[6].plot(tmeta[6, 14500:15000])
axs[6].title.set_text("Camera")
for ax in axs.flat:
    ax.label_outer()
    
#plt.imsave("'Z:/RawData/Hedes/2022-03-14/1/metadata_2seconds.png", )

# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i,50000:52000], c="b")
    
    
# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i,14500:15000], c="b")
   
   

"""

    1:Photodiode
    2:FrameClock
    3:Pockel feedback
    4:Piezo
    5:wheel-F
    6:Wheel-B
    7:Camera
"""
