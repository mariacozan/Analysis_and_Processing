# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:13:39 2022

@author: maria
"""

import numpy as np
import pandas as pd
import skimage
from skimage import io
from skimage import data
from skimage.util import img_as_float
import tifftools as tt
from pystackreg import StackReg
import os
#save as tiff!
def save_image(path, img, **kwargs):
    if "interpolation" in kwargs:
        io.imsave(path, img, interpolation=kwargs["interpolation"])
        print("Image stack saved to {}".format(path))
    else:
        io.imsave(path, img)
        print("Image stack saved to {}".format(path))
        
        
pathreg= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//test//ImageJ_reg//"
#pathreg2= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//test//ImageJ_reg//MAX_20.tif"

files= ["MAX_10.tif", "MAX_20.tif", "MAX_30.tif", "MAX_40.tif", "MAX_50.tif", "MAX_60.tif", "MAX_70.tif", "MAX_80.tif", "MAX_90.tif","MAX_100.tif"]

tif_list=[]

for file in files :
     TIF= skimage.io.imread(pathreg+file)
     tif= np.array(TIF)
     tif_list.append(tif)
     

combined_tifs= np.stack(tif_list)  

save_image(path= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//test//ImageJ_reg//combined.tif", img=combined_tifs)
         



