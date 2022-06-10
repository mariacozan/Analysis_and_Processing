# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:07:57 2022

@author: maria
"""

import numpy as np

#checking the ops details of files
animal=  'Hedes'
date= '2022-03-23'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'

filePath='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane1//ops.npy'

ops =  np.load(filePath, allow_pickle=True)
ops = ops.item()

#printing data path to know which data was analysed
key_list = list(ops.values())
print(key_list[88])
print("frames per folder:",ops["frames_per_folder"])
first_exp= np.array(ops["frames_per_folder"][0])