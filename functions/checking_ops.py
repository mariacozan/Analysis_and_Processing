# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:07:57 2022

@author: maria
"""

import numpy as np

#checking the ops details of files
animal= 'Glaucus'
date= '2022-07-26'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane = '1'

filePath='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane+'//ops.npy'
#filePath='C://Temporary_Suite2P_output//'+animal+ '//'+date+ '//all//'+experiment+ '//plane'+plane+'//ops.npy'

ops =  np.load(filePath, allow_pickle=True)
ops = ops.item()

#printing data path to know which data was analysed
key_list = list(ops.values())
print(key_list[88])
print("frames per folder:",ops["frames_per_folder"], "frame rate per plane:", ops["fs"])
first_exp= np.array(ops["frames_per_folder"][0])
if "zcorr" in ops:
    print("Z correlation done")
else:
        print("Z correlation not done")