# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:09:56 2022

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

path= 'D://Suite2Pprocessedfiles//Eos//2022-05-04//suite2p//plane1//ops.npy'

F= np.load('D://Suite2Pprocessedfiles//Eos//2022-05-04//suite2p//plane1//F.npy', allow_pickle=True)
Fneu = np.load('D://Suite2Pprocessedfiles//Eos//2022-05-04//suite2p//plane1//Fneu.npy', allow_pickle=True)
# spks = np.load(path, allow_pickle=True)
# stat = np.load(path, allow_pickle=True)
ops =  np.load(path, allow_pickle=True)
ops = ops.item()

#plan: plot F of neurons on top of each other and then plot the Z 
ops_list= list(ops.values())
zcorr= np.array(ops_list[132])

#plt.plot(zcorr)

Z= np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
#plt.plot(Z)


fig, axs = plt.subplots(F.shape[0]+1, sharex=True)

for i in range(F.shape[0]):
    
    axs[i].plot(F[i], c="green")
    axs[i].plot(Fneu[i], c="magenta")


axs[-1].plot(Z, c="blue")

for ax in axs.flat:
    ax.label_outer()
# axs.set_xlabel('frames')
# axs.set_title('Fluroescence traces +Neuropil and Z movement')

    