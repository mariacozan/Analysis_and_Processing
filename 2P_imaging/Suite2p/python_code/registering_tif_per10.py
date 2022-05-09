# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:19:06 2022

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

#save as tiff!
def save_image(path, img, **kwargs):
    if "interpolation" in kwargs:
        io.imsave(path, img, interpolation=kwargs["interpolation"])
        print("Image stack saved to {}".format(path))
    else:
        io.imsave(path, img)
        print("Image stack saved to {}".format(path))
        
def registration(stack, reference='first'):
    """
    https://pypi.org/project/pystackreg/
    Note that this function performs TRANSLATION transformation (proven to work best)
    other types of transformation: RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR

    Parameters
    ----------
    stack : array
    the unregistered array
    
    reference : string
    what kind of reference image to use, options:
        - first (default, proven to work best)
        - previous
        - mean

    Returns
    -------
    reg : array
        registered array

    """
    sr = StackReg(StackReg.TRANSLATION)
    reg= sr.register_transform_stack(stack, reference='first')
    return reg



"""
plan: registering 10 planes, then moving on to the next 10
end result: one registered tiff file 
-->if input has format (a, b, c, d) where
a = planes
b = samples per plane
c = x resolution
d = y resolution

code will return a registred z stack with the format  (a, b, c) where 
a = planes
b = x resolution
c = y resolution

How it does it:
    - reads tiff into an array
    - goes through each plane with its b amount of samples iteratively using a for loop
    - does the registration for these samples using the PyStackReg function
    - gets the mean from these registred samples
    - appends these into another array of shape (planes, x resolution, y resolution)

"""
#specifying the paths
drive= 'D://Tiff_stacks'
animal=  'Hedes'
date= '2022-03-30'
unreg_name= 'file_00005_00001'

filePath=drive+'//'+animal+ '//'+date+ '//'+unreg_name+'.tif'


reg_stack_name= "reg_z-stack"
path_reg= filePath+ reg_stack_name



#reading the tif
TIF= skimage.io.imread(filePath)
#then converting TIF into structured array
image = np.array(TIF)


#loop to create the registered arrays with the out_first function that I created based on the module pystackreg
planes = image.shape[0]
resolutionx = image.shape[2]
resolutiony= image.shape[3]
meanreg_arrays=np.zeros((planes, resolutionx, resolutiony))

for i in range(image.shape[0]):
    reg_arrays = registration(image[i,:,:,:])
    meanreg_arrays[i,:,:] = np.mean(reg_arrays, axis=0)


#save the registered array as a tiff
save_image(path= path_reg, img=meanreg_arrays)