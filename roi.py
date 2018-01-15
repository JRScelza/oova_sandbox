#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:35:10 2017

@author: jeromescelza
"""

import cv2
import numpy as np
import os
import pandas as pd

from scipy.interpolate import UnivariateSpline
from scipy import signal


def region_loc(f):
    f_4 = os.path.expanduser(f)
    im = cv2.imread(f_4)
    variant = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    channel_1, channel_2, channel_3 = variant[:,:,0] ,variant[:,:,1] ,variant[:,:,2]

    x = range(0,1944)
    y = channel_3[:,1100]
    cv2.line(variant,(1000,0),(1000,1944),(0,0,255),15)
    
    
    signal2 = y
    signal_series = pd.Series(signal2)
    smooth_data = pd.rolling_mean(signal_series,10)
    smooth_set = pd.Series(smooth_data)

    spl = UnivariateSpline(x, y)
    tm  = signal.argrelextrema(spl(x), np.less)

    sm_factor = 5000
    
    while len(tm[0]) > 5:
        spl.set_smoothing_factor(sm_factor)
        tm  = signal.argrelextrema(spl(x), np.less)
        sm_factor = sm_factor + 200
    
    peakind = signal.find_peaks_cwt(spl(x), np.arange(10,200), noise_perc=20)
    
    t =[]
    
    for i in range(0, len(tm)):
        t.append(spl(x)[tm[i]])
        
    z =[]
    
    for i in range(0, len(peakind)):
        z.append(spl(x)[peakind[i]])
        
        
    #add LOOP for peaking back in so we know the inflection points
    
    #plt.plot(x, spl(x), 'r--')
    #plt.scatter(tm, t)
    #plt.show()
    #    
    #plt.plot(x, spl(x), 'r--')
    #plt.scatter(peakind, z)
    #plt.show()
    
    
    # Here we are looking to identify all of the peaks.  We are eliminating outliers lower than 50, because the
    #alog looks 50 pxl on both sides
    f_sets = []  
    
    peak_location_factor = 0  
     
    for i in tm[0]:
        if i > 50:
            f_der = np.gradient(spl(x)[i - 50 : i + 50])
            f_sets.append(f_der)
            
        else:
            peak_location_factor = 1
            continue
    
    peak_quants = []
    for i in range(0, len(f_sets)):
        peak_quants.append(max(f_sets[i]))
        
    
    ROI = peak_quants.index(max(peak_quants)) + peak_location_factor
        
    
    #Our final return is the location of the center of the "TEST LINE"
    B_C = tm[0][ROI]
    
    #********************************************************************
    #********************************************************************
    #**************************FX 2*********************************
    #********************************************************************
    #********************************************************************
    
    
    
    x = range(0,2592)
    y = channel_3[B_C - 100,:]
    
    signal2 = y
    
    
    signal_series = pd.Series(signal2)
    
    smooth_data = pd.rolling_mean(signal_series,10)
    
    smooth_set = pd.Series(smooth_data)
    
    
    
    
    spl = UnivariateSpline(x, y)
    tm  = signal.argrelextrema(spl(x), np.less)
    
    
    sm_factor = 50000
    
    spl.set_smoothing_factor(sm_factor)
    tm  = signal.argrelextrema(spl(x), np.less)
    
    #while len(tm[0]) > 5:
    #    spl.set_smoothing_factor(sm_factor)
    #    tm  = signal.argrelextrema(spl(x), np.less)
    #    sm_factor = sm_factor + 200ps
    
    spike = min(spl(x)) + 30
    
    spike_value = 0
    spike_ind = 0
    
    
    while spike_value < spike:
        spike_value = spl(x)[spike_ind]
        spike_ind = spike_ind + 1
    
    
    return(B_C , spike_ind)
    
    
    
        
    #cv2.rectangle(im,(spike_ind + 70 , B_C - 100),(spike_ind + 1350, B_C + 100),(0,255,0),10)
    #plt.imshow(im)



















