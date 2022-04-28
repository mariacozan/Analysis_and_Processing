# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:49:12 2022

@author: maria
"""

import numpy as np
import pandas as pd
import skimage

from skimage import io
from skimage import data
from skimage.util import img_as_float
#loading packages

path= "D://Tiff_stacks//Hedes//2022-03-23//file_00003_00001.tif"
TIF= skimage.io.imread(path)
#reading TIF
image = img_as_float(TIF)
#converting variable to float
image = np.array(TIF)
#then converting float into structured array

AVG_stack= image[0]

print(image.shape)
#confirm data type
avg_tif= np.average(image, axis=(1))
#average across all pixels

#save as tiff!
def save_image(path, img, **kwargs):
    if "interpolation" in kwargs:
        io.imsave(path, img, interpolation=kwargs["interpolation"])
        print("Image stack saved to {}".format(path))
    else:
        io.imsave(path, img)
        print("Image stack saved to {}".format(path))
        

save_image(path="D://Tiff_stacks//Hedes//2022-03-23//avg_file_00003_00001.tif", img=avg_tif)

from pystackreg import StackReg


img0 = io.imread("D://Tiff_stacks//Hedes//2022-03-23//avg_file_00003_00001.tif") # 3 dimensions : frames x width x height

sr = StackReg(StackReg.RIGID_BODY)

# register each frame to the previous (already registered) one
# this is what the original StackReg ImageJ plugin uses
# out_previous = sr.register_transform_stack(image, reference='previous')

# # register to first image
# out_first = sr.register_transform_stack(img0, reference='first')

# register to mean image
out_mean = sr.register_transform_stack(img0, reference='mean')

# # register to mean of first 10 images
# out_first10 = sr.register_transform_stack(img0, reference='first', n_frames=10)

# # calculate a moving average of 10 images, then register the moving average to the mean of
# # the first 10 images and transform the original image (not the moving average)
# out_moving10 = sr.register_transform_stack(img0, reference='first', n_frames=10, moving_average = 10)

save_image(path="D://Tiff_stacks//Hedes//2022-03-23//meanreg_file_00003_00001.tif", img=out_mean)