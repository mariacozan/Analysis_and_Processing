# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:56:36 2022

@author: maria
"""

"""
Neuropil correction and dF/F:
    a:Fc(t) = F(t) -a∙N(t). The correction factor was determined for each ROI as follows. 
    First, F and N were low-pass filtered using the 8thpercentile in a moving window of 180 s,resulting in F0an N0.
    The resulting traces Ff(t) = F(t)-F0(t) and Nf(t) = N(t)-N0(t) were then used to estimateaas described previously(Dipoppa et al., 2018).
    In short, Nf was linearly fitted to F fusing only time points when values of Ff were relatively low and thus unlikely to reflect neural spiking. 
    Fc was then low-pass filtered as above (8thpercentile in a moving window of 180 s) to determine Fc,0.
    These traces corrected for neuropil contamination were then used to determine DF/F = (Fc(t) - Fc,0(t)) / max(1, meant(Fc,0(t)).
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          