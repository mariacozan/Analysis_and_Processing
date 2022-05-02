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

#save as tiff!
def save_image(path, img, **kwargs):
    if "interpolation" in kwargs:
        io.imsave(path, img, interpolation=kwargs["interpolation"])
        print("Image stack saved to {}".format(path))
    else:
        io.imsave(path, img)
        print("Image stack saved to {}".format(path))
        



dim4 = np.full((101,10,512,512), 0)

    
dim3=np.full((10,5,5), 1)
dim3b= np.full((10,5,5), 2)
combined= np.vstack((dim3, dim3b))

for i in range(10):
    avg= combined[i]

"""
registering 10 planes, then moving on to the next 10:
    - convert 4d array to 3d array of all the planes in sequence
    - loop through 10 stacked 3d array
    - do registration on these
    - create separate tiffs for the 10 samples
    - combine into one big tiff again

"""

path= "D://Tiff_stacks//Hedes//2022-03-23//file_00003_00001.tif"
TIF= skimage.io.imread(path)
#then converting TIF into structured array
image = np.array(TIF)


combined_stack= np.vstack(image)
#confirm data type
print(combined_stack.shape)



split_stack= combined_stack[0:9,:, :]


split_stacks= np.array_split(combined_stack, 101)

split_stacks1= split_stacks[1]

n= 20
path= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//" +n+ ".tif"
path2= "D://Tiff_stacks//Hedes//2022-03-23//split_stacks//reg_" +n+ ".tif"

# for element in split_stacks:
save_image(path=path, img=split_stacks[n])


from pystackreg import StackReg


img0 = io.imread(path) # 3 dimensions : frames x width x height

sr = StackReg(StackReg.RIGID_BODY)

# register to mean image
out_mean = sr.register_transform_stack(img0, reference='mean')
save_image(path=path2, img=out_mean)

#write a for loop to create the tiffs, do reg and combine them into one again