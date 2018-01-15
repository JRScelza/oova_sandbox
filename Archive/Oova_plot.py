#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:05:22 2017

@author: jeromescelza
"""
import requests
import json
import pprint
import pandas as pd
import datetime
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython.display import display
import time
import warnings


oova_list = [0, 10, 14, 21, 42, 60, 96, 137]
comp_high = [0,0,0,0,5,10,15,21]
comp_low = [0,0,0,0,0,5,11,18]


ind = np.arange(0, len(comp_low))
width = 0.35
spacing = .25
fig = plt.figure()

t = 0
fig, axs = plt.subplots(1,2, figsize=(25, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.3)

axs = axs.ravel()

for i in range(0,8):
    
#    ax1 = fig.subplot(int('22%s' % t))
    rects1 = axs[i].bar(ind, oova_list, width, color='c')
    rects2 = axs[i].bar(ind+spacing, comp_high, width, color='m')
    rects3 = axs[i].bar(ind+(2 * spacing), comp_low, width, color='b')
    
    axs[i].set_xlabel('Analyte Concentration (uIU/L)')
    axs[i].set_ylabel('Test Line Intensity')
    axs[i].set_title('Test Line Resolution using Different Capture Molecules')
    axs[i].set_xticks(ind + width / 2)
    axs[i].set_xticklabels(('0.00', '0.16', '0.30', '0.60', '1.25', '2.50', '5.00','10.00'), rotation=45)
    axs[i].legend((rects1[0], rects2[0], rects3[0]), ('Gold Nano-Shells', 'Gold Nano-Spheres', 'Latex Spheres'))
    
    
    t=t + 1