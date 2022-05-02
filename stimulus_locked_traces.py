# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:18:13 2022

@author: maria
"""

"""
Plan for stimulus locked trace:
    1. get the time at which photodiode is going from on to off, see function below
    2. check frame clock and see at which frame number we had a stimulus appear
    3. use this frame number to check the F trace and align these two
    4. take for ex 50 frames before and 50 frames after the stim and plot this
    5. now check the csv file with the stim identity info
    6. group traces for one type of stimulus, then average these responses for the response to same stim for same neuron
    
"""