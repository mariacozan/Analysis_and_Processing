# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:07:22 2022

@author: maria
"""

"""
Getting the piezo signal and converting volts to distance to determine the true distance between planes
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statistics as stats

#defining path
animal=  'Hedes'
date= '2022-06-28'
planes = 5
exp_nr=1
experiment= str(exp_nr)

#NDIN is the number in the NiDaq binary file, bear in mind this is not always experiment number - 1, always double check
#NDIN= exp_nr-1
#in case number is not exp number - 1 then put it in manually here:
NDIN = 0
NiDaqInputNo= str(NDIN)

filePathInput='Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//NiDaqInput'+NiDaqInputNo+'.bin'
#need to add custom titles for plots
filePathOutput2s= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_2seconds-interval.png'
filePathOutput500ms= 'Z://RawData//'+animal+ '//'+date+ '//'+experiment+ '//metadata_500ms-interval.png'
#path= 'Z:/RawData/Eos/2022-05-04/1/NiDaqInput0.bin'

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

def AssignFrameTime(frameClock,th = 0.5,plot=False):
    """
    The function assigns a time in ms to a frame time.
    
    Parameters:
    frameClock: the signal from the nidaq of the frame clock
    th : the threshold for the tick peaks, default : 3, which seems to work 
    plot: plot to inspect, default = False
    
    returns frameTimes (ms)
    """
    #Frame times
    # pkTimes,_ = sp.signal.find_peaks(-frameClock,threshold=th)
    # pkTimes = np.where(frameClock<th)[0]
    # fdif = np.diff(pkTimes)
    # longFrame = np.where(fdif==1)[0]
    # pkTimes = np.delete(pkTimes,longFrame)
    # recordingTimes = np.arange(0,len(frameClock),0.001)
    # frameTimes = recordingTimes[pkTimes]
    
    # threshold = 0.5
    pkTimes = np.where(np.diff(frameClock > th, prepend=False))[0]    
    # pkTimes = np.where(np.diff(np.array(frameClock > 0).astype(int),prepend=False)>0)[0]
       
    
    if (plot):
        f,ax = plt.subplots(1)
        ax.plot(frameClock)
        ax.plot(pkTimes,np.ones(len(pkTimes))*np.min(frameClock),'r*')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')
        
        
    return pkTimes

"""IMPORTANT: specify how many channels there are in the binary file, check in bonsai script and metadata plots!"""
numChannels= 5
meta= GetMetadataChannels(filePathInput, numChannels=numChannels)
tmeta= meta.T
frame_clock = tmeta[1]
frame_times = AssignFrameTime(frame_clock)
piezo = tmeta[3]

start = 5000
end = 7000


# fig1, axs = plt.subplots(2)
# axs[0].plot(tmeta[1, start:end])
# axs[0].title.set_text("Frame Clock")
# axs[1].plot(tmeta[3, start:end])
# axs[1].title.set_text("Piezo")



f,ax = plt.subplots(1)
ax.plot(tmeta[1, start:end], c = "green")
ax.plot(tmeta[3, start:end],c = "red")
ax.set_xlabel('time (ms)')
ax.set_ylabel('Amplitude (V)')
        


"""
What do I need from this? 
The end goal is to determine the distance between each plane 
What is known?
- values of 5V means the absolute distance moved was 400um
- the piezo signal sampled at 1000Hz showing the voltage values throught times
- the frame clock signal also sampled at 1000Hz which shows the voltage values of 5V when a frame was recorded
- based on Liad's code I also have the frame_times, the time at which a frame was recorded
- So I need to obtain the voltages for the middle 3 frames (so if starting from index 0, indexes 1, 2 and 3)
"""



# write a for loop to get the distance values, but the below is inefficient (although a similar approach using arrays
# instead could also be applicable to the stim aligning (I mean it's literally the same principle))
distance_plane = []
for frames in frame_times:
    distance_plane.append(piezo[frames])

distance_plane = np.array(distance_plane)

# plotting the absolute distance values for the time at which the end and the start of a frame was recorded for 100ms
# f,ax = plt.subplots(1)
# ax.plot(distance_plane[0:100], "-o")
# ax.set_xlabel('Time(ms)')
# ax.set_ylabel('distance from top plane')
#now have all the distance values for when a frame was written

#need to then take the even values sequentially to get the more values at the start of each frame
max_volts = np.amax(tmeta[3])
#specify the maximum distance (in um) travelled for each experiment (need to determine during experiments!)
max_distance = 90

#per 1 volts, the distance travelled is:
unit = max_distance/max_volts
d_start_frame = distance_plane[::2]*unit

f,ax = plt.subplots(1)
ax.plot(d_start_frame[0:100], "-o")
ax.set_xlabel('Time(ms)')
ax.set_ylabel('distance from top plane')

#check in the third figure generated which frames correspond to the linear parts
n =3
d_1 = abs(d_start_frame[n]) - abs(d_start_frame[n-1])
d_2 = abs(d_start_frame[n+1]) - abs(d_start_frame[n])
#d_3 = abs(d_start_frame[n+2]) - abs(d_start_frame[n+1])
distances = [d_1, d_2]
mean_d = stats.mean(distances)
print(mean_d)

#in theory the above should work but it doesn't make much sense, especially when the piezo signal is negative

##old code below, just checking max value and dividing by planes and multiplying times 80
#max_value = np.amax(tmeta[3])
# min_value = np.amin(tmeta[3])

"""
2022-07-06: reworking this
for the data from yesterday, the max distance travelled was 90um (from 20-110um deep, steps of 30)
the max voltage for that point is 1.85...
so then dividing 90 by the max voltage will give the distance per 1V which is 48.65um
-->it worked
"""

# print(min_value, max_value)

# #values should be between 0 and 5V (and 5V means a distance of 400um moved) but aren't
# V= max_value
# Vtodistance = V*80/planes
# print(Vtodistance)

#values used to be between 0-1 before May, then changed to lower numbers

#but this is not an issue, can simply 

# compare two recordings with different planes (if voltage change is the same)