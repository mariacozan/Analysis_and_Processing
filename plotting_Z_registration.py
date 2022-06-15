# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:09:56 2022

@author: maria
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

#from D drive
animal=  'Eos'
date= '2022-02-28'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_number= '1'

filePathops='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//ops.npy'
filePathF='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//F.npy'
filePathFneu='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//Fneu.npy'
filePathcell = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//iscell.npy'
# spks = np.load(path, allow_pickle=True)
# stat = np.load(path, allow_pickle=True)
ops =  np.load(filePathops, allow_pickle=True)
ops = ops.item()
F= np.load(filePathF, allow_pickle=True)
Fneu = np.load(filePathFneu, allow_pickle=True)
iscell= np.load(filePathcell, allow_pickle=True)
cells= np.where(iscell == 1)[0]

F_cells = F[cells,:]


#plan: plot F of neurons on top of each other and then plot the Z 
ops_list= list(ops.values())
zcorr= np.array(ops_list[132])

#plt.plot(zcorr)

Z= np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
Z= Z.astype(float)
#plt.plot(Z)

# plotting all
# fig, axs = plt.subplots(F.shape[0]+1, sharex=True)

# for i in range(F.shape[0]):
    
#     axs[i].plot(F[i], c="green")
#     axs[i].plot(Fneu[i], c="magenta")


# axs[-1].plot(Z, c="blue")

# for ax in axs.flat:
#     ax.label_outer()
# axs.set_xlabel('frames')
# axs.set_title('Fluroescence traces +Neuropil and Z movement')

#plotting 1 ROI
fig, axs = plt.subplots(3, sharex=True)
#choose ROI
n = 34
n_str= str(n)
axs[0].plot(np.array(range(len(F[n])))/6, Z, c="blue")   
axs[0].set_ylabel('distance(um)', fontsize= 12)
axs[1].plot(np.array(range(len(F[n])))/6, F[n], c="green")
axs[1].set_ylabel('raw F', fontsize= 15)
axs[2].plot(np.array(range(len(F[n])))/6, Fneu[n], c="magenta")
axs[2].set_ylabel('raw F (np)', fontsize= 15)
plt.rc('xtick',labelsize=15)
plt.rc('ytick', labelsize=15)
plt.subplots_adjust(wspace=0.7, hspace=0.7)



for ax in axs.flat:
    ax.label_outer()
    

ax.set_xlabel('Time(s)', fontsize= 16)


filePathplot= 'D://Z-analysis//'+animal+ '//'+date+ '//trace_plot_for_ROI'+n_str+'.png'
plt.savefig(filePathplot)

    

# for i in range(F.shape[0]+1):
#     i_str=str(i)
#     fig, axs = plt.subplots(3, sharex=True)
#     for i in range(F.shape[0]):
    
#         fig, axs = plt.subplots(3, sharex=True)

    
#         axs[0].plot(F[i], c="green")
#         axs[1].plot(Fneu[i], c="magenta")


#     axs[2].plot(Z, c="blue")
# for ax in axs.flat:
#     ax.label_outer()
#     filePathplot= 'D://Z-analysis//'+animal+ '//'+date+ '//trace_plot_for_ROI'+i_str+'.png'
#     plt.savefig(filePathplot)
    
    
    
