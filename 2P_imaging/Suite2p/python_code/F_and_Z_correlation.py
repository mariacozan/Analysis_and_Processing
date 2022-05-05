# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:35:40 2022

@author: maria
"""

""" 
Z correlation plan:
    - load F trace
    -load Z (make a function to create the Z from the Z corr)
    do Pearson's correlation
    plot the corr results
    do corr analysis (ro)
    
"""
    
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

def z_trace(opspath):
    """
    loads the ops file which contains the z correlation info, this is then transformed into a Z trace using a guassian filter
    (partially taken from Suite2P source code)
    

    Parameters
    ----------
    path : string
        provide the path of the suite2p ops file


    Returns
    -------
    Z : array of int64
        returns the Z trace

    """
    ops =  np.load(opspath, allow_pickle=True)
    ops = ops.item()
    ops_list= list(ops.values())
    zcorr= np.array(ops_list[132])
    Z= np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
    return Z

path='D://Suite2Pprocessedfiles//Hedes//2022-03-24//suite2p//plane1//ops.npy'


F= np.load('D://Suite2Pprocessedfiles//Hedes//2022-03-24//suite2p//plane1//F.npy', allow_pickle=True)
Fneu = np.load('D://Suite2Pprocessedfiles//Hedes//2022-03-24//suite2p//plane1//Fneu.npy', allow_pickle=True)


Ztrace= z_trace(opspath=path)
FandZ= np.stack((Ztrace, F[0]))

#calculating the spearman's correlation (has to be non-parametric because discrete data) between the Z trace and the fluorescence
corr_list= []

for n in range(F.shape[0]):
   coeff, p= sp.stats.spearmanr(Ztrace, F[n])
   corr_list.append(coeff)

#taking the r values which are above 0.2 and getting the identity of the ROI which the value belongs to, appending this to a list
correlated= []
for ROI,coeff in enumerate(corr_list):
    if coeff>0.2 or coeff<-0.2:
        correlated.append(ROI)
        
#choose ROI
n=20

# fig, axs = plt.subplots(3, sharex=True)

# axs[0].plot(F[n], c="green")
# axs[1].plot(Fneu[n], c="magenta")
# axs[2].plot(Ztrace, c="blue")

# for ax in axs.flat:
#     ax.label_outer()

#scatterplot of Z trace vs F with the r value as an inset

#determine location of the inset depending on max F



max_F= np.amax(F[n])
max10p_F=np.amax(F[n])/9
corr_number= str(corr_list[n])
r= "r="+corr_number+"."
plt.scatter(Ztrace, F[n])
plt.text(np.mean(Ztrace),max_F+max10p_F, r, fontsize= 10)






        
   




   






   
