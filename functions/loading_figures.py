# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:47:12 2022

@author: maria
"""
animal=  'ELias'
#animal = input("animal name ")
date= '2022-09-23'
#date = input("date ")
#note: if experiment type not known, put 'suite2p' instead
experiment = '1'
exp_no = int(experiment)
#experiment = input("experiment number(integer only) ")
#experiment_int = int(experiment)
#the file number of the NiDaq file, not alway experiment-1 because there might have been an issue with a previous acquisition etc
file_number = '0'
log_number = '0'
plane_number = '0'
plane_number_int = int(plane_number)
exp_name = 'Gratings'
ROI = 0
#%%loading figure
#only supports pyplot figures (so not seaborn probs)
import pickle
figx = pickle.load(open('D://Stim_aligned//'+animal+ '//'+date+ '//plane'+plane_number+'//'+exp_name+'//cell'+str(ROI)+'.pickle', 'rb'))

figx.show() # Show the figure, edit it, etc.!