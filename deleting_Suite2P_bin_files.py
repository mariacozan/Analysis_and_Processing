# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:46:53 2022

@author: maria
"""

import os

directory= "D://Suite2Pprocessedfiles"


#for loop to go through the directory and delete the bin files
for file in directory:
    if file= "*.bin"
    os.remove(file)
    
#different options for choosing files with a certain extension obtained from the internet    
fileDir = r"C:\Test"
fileExt = r".txt"
_ for _ in os.listdir(fileDir) if _.endswith(fileExt)


import glob

targetPattern = r"D://Suite2Pprocessedfiles//**//*.bin"

for file in targetPattern:
    os.remove(file)