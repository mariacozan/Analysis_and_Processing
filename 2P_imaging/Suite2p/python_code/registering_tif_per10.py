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
        
def registration(stack, **kwargs):
    sr = StackReg(StackReg.TRANSLATION)
    reg= sr.register_transform_stack(stack, reference='first')
    return reg


dim4 = np.full((101,10,512,512), 0)

    
dim3=np.full((10,5,5), 1)
dim3b= np.full((10,5,5), 2)
combined= np.vstack((dim3, dim3b))


"""
registering 10 planes, then moving on to the next 10:
    - convert 4d array to 3d array of all the planes in sequence
    - loop through 10 stacked 3d array
    - do registration on these
    - create separate tiffs for the 10 samples
    - combine into one big tiff again

"""
#reading the tif
path= "D://Tiff_stacks//Eos//2022-05-04//file_00005_00001.tif"
TIF= skimage.io.imread(path)
#then converting TIF into structured array
image = np.array(TIF)

#converting 4d array into 3d for easier manipulation
combined_stack= np.vstack(image)
#confirm data type
print(combined_stack.shape)


#example of one plane
# split_stack= combined_stack[500:510,:, :]

# reg_50= out_first(stack=split_stack)
# reg_50_mean= np.mean(reg_50, axis=0)

#split all the stacks per plane into a list
split_stacks= np.array_split(combined_stack, 101)

"""
- create for loop which iterates through the split_stacks list, maybe need to change it from a different format (something that can be changed?)
- does the registration,
- does a Z project thing on it (i.e. gets the mean image) -->averaging over Z
- and saves each element as a list of arrays (here, should give 101 elements)
- these are then combined into one array (use np.stack?)
- then save this as a tif

"""
#loop to create the registered arrays with the out_first function that I created based on the module pystackreg
reg_arrays=[]
meanreg_arrays=[]
for array in split_stacks:
    reg_arrays= registration(array)
    meanreg_arrays.append(np.mean(reg_arrays, axis=0))
    
all_planes= np.stack(meanreg_arrays)
save_image(path= "D://Tiff_stacks//Eos//2022-05-04//reg_stack_test.tif", img=all_planes)
    
    



# path= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//test//fromarray.tif"
# # for element in split_stacks:
# # 3 dimensions : frames x width x height
# sr = StackReg(StackReg.TRANSLATION)
# # register to mean image
# out_first = sr.register_transform_stack(split_stack, reference='first')
# save_image(path=path, img=out_first)  
