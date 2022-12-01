# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:44:35 2022

@author: maria
"""

from Data.Bonsai import extract_data
niDaqFilePath = 'Z://RawData//Giuseppina//2022-11-03//1//'

extract_data.get_nidaq_channels(niDaqFilePath, plot=True)