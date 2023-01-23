# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:41:14 2022

@author: maria
"""

"""
processing FaceMap data
"""

import numpy as np

#specify video number
no = str(3)
filePath = 'C://FaceMap//Video'+no+'_proc.npy'

#loading thr dictionary
FaceMap_data= np.load(filePath, allow_pickle=True)