# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:46:53 2022

@author: maria
"""

import os

#easy but a bit more time consuming

#from C drive
# os.remove("C://Temporary_Suite2P_output//2022-02-21//suite2p//plane0//data.bin")
# os.remove("C://Temporary_Suite2P_output//2022-02-21//suite2p//plane0//data_chan2.bin")

#from D drive
animal=  'Glaucus'
date= '2022-03-16'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_number= '0'

filePath='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//data.bin'
os.remove(filePath)

#os.remove("D://Suite2Pprocessedfiles//SS109//2022-02-23//suite2pa//plane4//data_chan2.bin")


# #directory= "D://Suite2Pprocessedfiles"

# # for file in directory if _.endswith(fileExt):
# #     os.remove(files)
    
# # fileDir = r"C:\Test"
# # fileExt = r".txt"
# # _ for _ in os.listdir(fileDir) if _.endswith(fileExt)
# #for loop to go through the directory and delete the bin files
# for file in directory:
#     if file= "*.bin"
#     os.remove(file)
    
# #different options for choosing files with a certain extension obtained from the internet    
# fileDir = r"C:\Test"
# fileExt = r".txt"
# _ for _ in os.listdir(fileDir) if _.endswith(fileExt)



# # import glob


# # targetPattern = r"C:\Te**\*.txt"
# # glob.glob(targetPattern)

# targetPattern = r"D://Suite2Pprocessedfiles//**//*.bin"

# for file in targetPattern:
#     os.remove(file)
