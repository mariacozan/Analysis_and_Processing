# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:49:39 2022

@author: maria
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


#defining path
animal=  'Bellinda'
date= '2022-06-10'

exp_nr=1
experiment= str(exp_nr)

#NDIN is the number in the NiDaq binary file, bear in mind this is not always experiment number - 1, always double check
#NDIN= exp_nr-1
#in case number is not exp number - 1 then put it in manually here:
NDIN = 2
NiDaqInputNo= str(NDIN)

filePathInput='Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//NiDaqInput'+NiDaqInputNo+'.bin'
#need to add custom titles for plots
filePathOutput2s= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_2seconds-interval.png'
filePathOutput500ms= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_500ms-interval.png'
path= 'Z:/RawData/Eos/2022-05-04/1/NiDaqInput0.bin'

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
#specify how many channels there are in the binary file, check in bonsai script
numChannels= 5
start_short = 14000
end_short = 14500
start_long = 50000
end_long = 52000
meta= GetMetadataChannels(filePathInput, numChannels=numChannels)
tmeta= meta.T
#plotting for7 channels with titles

if numChannels == 7:
    fig1, axs = plt.subplots(7)
    axs[0].plot(tmeta[0, start_long:end_long])
    axs[0].title.set_text("Photodiode")
    axs[1].plot(tmeta[1, start_long:end_long])
    axs[1].title.set_text("Frame Clock")
    axs[2].plot(tmeta[2, start_long:end_long])
    axs[2].title.set_text("Pockel feedback")
    axs[3].plot(tmeta[3, start_long:end_long])
    axs[3].title.set_text("Piezo")
    axs[4].plot(tmeta[4, start_long:end_long])
    axs[4].title.set_text("Wheel-F")
    axs[5].plot(tmeta[5, start_long:end_long])
    axs[5].title.set_text("Wheel-B")
    axs[6].plot(tmeta[6, start_long:end_long])
    axs[6].title.set_text("Camera")
    plt.xlabel("Time(ms)")
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
        
    plt.savefig(filePathOutput2s)
        
    fig2, axs2 = plt.subplots(7, squeeze=True)
    axs2[0].plot(tmeta[0, start_short:end_short])
    axs2[0].title.set_text("Photodiode")
    axs2[1].plot(tmeta[1, start_short:end_short])
    axs2[1].title.set_text("Frame Clock")
    axs2[2].plot(tmeta[2, start_short:end_short])
    axs2[2].title.set_text("Pockel feedback")
    axs2[3].plot(tmeta[3, start_short:end_short])
    axs2[3].title.set_text("Piezo")
    axs2[4].plot(tmeta[4, start_short:end_short])
    axs2[4].title.set_text("Wheel-F")
    axs2[5].plot(tmeta[5, start_short:end_short])
    axs2[5].title.set_text("Wheel-B")
    axs2[6].plot(tmeta[6, start_short:end_short])
    axs2[6].title.set_text("Camera")
    plt.xlabel("Time(ms)")
    for ax in axs2.flat:
        ax.label_outer()
        
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    
    plt.savefig(filePathOutput500ms)

#plotting for 4 channels with titles
if numChannels == 4:
    fig, axs = plt.subplots(4)
    axs[0].plot(tmeta[0, start_long:end_long])
    axs[0].title.set_text("Photodiode")
    axs[1].plot(tmeta[1, start_long:end_long])
    axs[1].title.set_text("Frame Clock")
    axs[2].plot(tmeta[2, start_long:end_long])
    axs[2].title.set_text("Pockel feedback")
    axs[3].plot(tmeta[3, start_long:end_long])
    axs[3].title.set_text("Piezo")
    plt.xlabel("Time(ms)")
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
        
    plt.savefig(filePathOutput2s)
        
        
    fig, axs = plt.subplots(4, squeeze=True)
    axs[0].plot(tmeta[0, start_short:end_short])
    axs[0].title.set_text("Photodiode")
    axs[1].plot(tmeta[1, start_short:end_short])
    axs[1].title.set_text("Frame Clock")
    axs[2].plot(tmeta[2, start_short:end_short])
    axs[2].title.set_text("Pockel feedback")
    axs[3].plot(tmeta[3, start_short:end_short])
    axs[3].title.set_text("Piezo")
    plt.xlabel("Time(ms)")
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    
    plt.savefig(filePathOutput500ms)

# the newest metadata contains 5 channels, the last one being a synchronisation signal to see if the arduino and the NiDAQ times match
if numChannels == 5:
    fig, axs = plt.subplots(5)
    axs[0].plot(tmeta[0, start_long:end_long])
    axs[0].title.set_text("Photodiode")
    axs[1].plot(tmeta[1, start_long:end_long])
    axs[1].title.set_text("Frame Clock")
    axs[2].plot(tmeta[2, start_long:end_long])
    axs[2].title.set_text("Pockel feedback")
    axs[3].plot(tmeta[3, start_long:end_long])
    axs[3].title.set_text("Piezo")
    axs[4].plot(tmeta[3, start_long:end_long])
    axs[4].title.set_text("Synchronisation signal")
    plt.xlabel("Time(ms)")
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
        
    plt.savefig(filePathOutput2s)
        
        
    fig, axs = plt.subplots(5, squeeze=True)
    axs[0].plot(tmeta[0, start_short:end_short])
    axs[0].title.set_text("Photodiode")
    axs[1].plot(tmeta[1, start_short:end_short])
    axs[1].title.set_text("Frame Clock")
    axs[2].plot(tmeta[2, start_short:end_short])
    axs[2].title.set_text("Pockel feedback")
    axs[3].plot(tmeta[3, start_short:end_short])
    axs[3].title.set_text("Piezo")
    axs[4].plot(tmeta[3, start_long:end_long])
    axs[4].title.set_text("Synchronisation signal")
    plt.xlabel("Time(ms)")
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    
    plt.savefig(filePathOutput500ms)

#more optimised code for plotting for any number of channels but doesn't have titles for subplots yet

# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i,0:15000], c="b")
    
    
# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i,14500:15000], c="b")

# plt.savefig("")

#plotting the full length
# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i], c="b")
    
    
# fig, axs = plt.subplots(tmeta.shape[0], sharex=True)

# for i in range(tmeta.shape[0]):
    
#     axs[i].plot(tmeta[i], c="b")




"""

    1:Photodiode
    2:FrameClock
    3:Pockel feedback
    4:Piezo
    5:wheel-F
    6:Wheel-B
    7:Camera
"""

