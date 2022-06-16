# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:25:20 2022

@author: maria
"""

"""
seeing how correlated the cells are between planes

"""
import os
import numpy as np
import matplotlib.pyplot as plt


#from D drive
animal=  'Hedes'
date= '2022-03-23'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_a= '1'
plane_b= '2'
#filePathops='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//ops.npy'
filePathFa='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_a+'//F.npy'
filePathFb='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_b+'//F.npy'
#filePathFneu='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//Fneu.npy'
filePathcella = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_a+'//iscell.npy'
filePathcellb = 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_b+'//iscell.npy'

# spks = np.load(path, allow_pickle=True)
# stat = np.load(path, allow_pickle=True)
#ops =  np.load(filePathops, allow_pickle=True)
#ops = ops.item()
Fa= np.load(filePathFa, allow_pickle=True)
Fb=  np.load(filePathFb, allow_pickle=True)
#Fneu = np.load(filePathFneu, allow_pickle=True)
iscella= np.load(filePathcella, allow_pickle=True)
iscellb= np.load(filePathcellb, allow_pickle=True)
cellsa= np.where(iscella == 1)[0]
cellsb= np.where(iscellb == 1)[0]
F_cellsa = Fa[cellsa,:]
F_cellsb = Fb[cellsb,:]
#example: ROI 4 from plane 1 and ROI 12 from plane 2
cell_a = Fa[4]
cell_b = Fb[12]
plt.scatter(cell_a, cell_b)
#computing the overall correlation of each trace to the other trace in the plane
#for i in range(Fa.shape[0]):
    