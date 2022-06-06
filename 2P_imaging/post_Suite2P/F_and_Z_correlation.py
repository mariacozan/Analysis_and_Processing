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

#from D drive
animal=  'Hedes'
date= '2022-03-23'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_number= '1'

filePathops='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//ops.npy'
filePathF='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//F.npy'
filePathFneu='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//Fneu.npy'




F= np.load(filePathF, allow_pickle=True)
Fneu = np.load(filePathFneu, allow_pickle=True)


Ztrace= z_trace(opspath=filePathops)
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
n=0
n_str= str(n)

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
scatterplot = plt.scatter(Ztrace, F[n])
plt.text(np.mean(Ztrace),max_F+max10p_F, r, fontsize= 10)



#save all the plots as pngs so it's easy to check them
# create folder for animal and date etc
#filePathplot= filePathops='D://Z-analysis//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'.png'
filePathplot= 'D://Z-analysis//'+animal+ '//'+date+ '//ROI'+n_str+'.png'
plt.savefig(filePathplot)
#do ANOVA analysis on these
#ANOVA= sp.kruskal()

# for ROI in range(F.shape[0]):
#     scatterplot= plt.scatter(Ztrace, F[n])
#     plt.savefig(filePathplot)

        
   




   






   
