#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:58:59 2017

@author: jeromescelza
"""


import cv2
import numpy as np
import os
from roi import region_loc
import pickle
import matplotlib.pyplot as plt

#You can pull in a random image from variant_1 to test the code.  Through the time series the image line intensity of variant_1 gets darker and darker.  You can grab it at
#different points to "pretend" you have more test strips.

model_file = '/Users/jeromescelza/Box Sync/oova_sandbox/model'

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

p = load_obj(model_file)    
    

control_hist_file = '/Users/jeromescelza/Box Sync/oova_sandbox/control_hist.npy'
variant_1_HIST = np.load(control_hist_file, allow_pickle=True, fix_imports=True, encoding='ASCII')


image_number = 173
f_test = os.path.expanduser("~/Desktop/ImagesOova/variant_2_1/image%s.jpg" % image_number)
variant_test = cv2.imread(f_test)

bounds , spike_pt  = region_loc(f_test)

#ret,threshtest = cv2.threshold(variant_test,min_thresh,255,cv2.THRESH_BINARY)

#rows,cols = variant_1.shape[:2]
rows,cols = variant_test.shape[:2]



#cv2.rectangle(im,(spike_ind + 70 , B_C - 100),(spike_ind + 1350, B_C + 100),(0,255,0),10)

x = spike_pt + 70
y = bounds - 100
w = 1350
h = 200

mask = np.zeros(variant_test.shape[:2],np.uint8)
mask[y:y+h,x:x+w] = 255


#halfhsv = basehsv[rows/2:rows-1,cols/2:cols-1].copy()  # Take lower half of the base image for testing

#hbins = 180
#sbins = 255
#hrange = [0,180]
#srange = [0,256]
#ranges = hrange+srange  # ranges = [0,180,0,256]
#ranges=None

#histbase = cv2.calcHist(basehsv,[0],mask,[0,256],ranges)

j = 4

variant_test_HIST = cv2.calcHist([variant_test],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])


#res = cv2.bitwise_and(variant_5,variant_5,mask = mask)


#Creating a list of values that represents the dummy strip line intensity (0, 16.67, 33, etc...)
   
value = cv2.compareHist(variant_1_HIST,variant_test_HIST,j)

m = p['polynomial'][0]
b = p['polynomial'][1]

def poly_lin(value, m, b):
    solution = (value - b)/ m
    return(solution)

answer = poly_lin(value, m , b)
answer = float("{0:.2f}".format(answer))

cv2.rectangle(variant_test,(spike_pt + 70 , bounds - 100),(spike_pt + 1350, bounds + 100),(0,255,0),10)



plt.imshow(variant_test,'gray'),plt.title('%s' % answer )






