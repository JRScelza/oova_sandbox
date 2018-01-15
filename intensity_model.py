#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:20:46 2018

@author: jeromescelza
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from roi import region_loc

#Pulling in all of the images.  
#Image 179 of each set is the final picture taken 
#(gives the most amount of time to make sure the test strip is stable)

f_1 = os.path.expanduser("~/Desktop/ImagesOova/variant_1_1/image179.jpg")
f_2 = os.path.expanduser("~/Desktop/ImagesOova/variant_2_1/image179.jpg")
f_3 = os.path.expanduser("~/Desktop/ImagesOova/variant_3_1/image179.jpg")
f_4 = os.path.expanduser("~/Desktop/ImagesOova/variant_4_1/image179.jpg")
f_5 = os.path.expanduser("~/Desktop/ImagesOova/variant_5_1/image179.jpg")
f_6 = os.path.expanduser("~/Desktop/ImagesOova/variant_6_1/image179.jpg")
f_blank = os.path.expanduser("~/Desktop/ImagesOova/variant_blank_1/image179.jpg")

variant_1 = cv2.imread(f_1)
variant_2 = cv2.imread(f_2)
variant_3 = cv2.imread(f_3)
variant_4 = cv2.imread(f_4)
variant_5 = cv2.imread(f_5)
variant_6 = cv2.imread(f_6)
variant_blank = cv2.imread(f_blank)

min_thresh = 126

#ret,thresh1 = cv2.threshold(variant_1,min_thresh,255,cv2.THRESH_BINARY)
#ret,thresh2 = cv2.threshold(variant_2,min_thresh,255,cv2.THRESH_BINARY)
#ret,thresh3 = cv2.threshold(variant_3,min_thresh,255,cv2.THRESH_BINARY)
#ret,thresh4 = cv2.threshold(variant_4,min_thresh,255,cv2.THRESH_BINARY)
#ret,thresh5 = cv2.threshold(variant_5,min_thresh,255,cv2.THRESH_BINARY)
#ret,thresh6 = cv2.threshold(variant_6,min_thresh,255,cv2.THRESH_BINARY)
#ret,threshblank = cv2.threshold(variant_blank,min_thresh,255,cv2.THRESH_BINARY)

bounds , spike_pt  = region_loc(f_1)
rows,cols = variant_1.shape[:2]


x = spike_pt + 70
y = bounds - 100
w = 1350
h = 200

mask = np.zeros(variant_1.shape[:2],np.uint8)
mask[y:y+h,x:x+w] = 255

variant_1_HIST = cv2.calcHist([variant_1],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_2_HIST = cv2.calcHist([variant_2],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_3_HIST = cv2.calcHist([variant_3],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_4_HIST = cv2.calcHist([variant_4],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_5_HIST = cv2.calcHist([variant_5],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_6_HIST = cv2.calcHist([variant_6],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])
variant_blank_HIST = cv2.calcHist([variant_blank],[0, 1, 2],mask,[8,8,8],[0, 256, 0, 256, 0, 256])


x =[]

for i in range(0,6):
    x.append(100-(i*16.67))

j = 4 # this is the compare function number i am using, you can selcet 0=Correlation, 1=Chi-Square, 2=Intersection, 3=Bhattacharyya, 4=Hellinger
#for j in range(0,4):
    
Control_Comp = cv2.compareHist(variant_1_HIST,variant_1_HIST,j)
Control_variant2 = cv2.compareHist(variant_1_HIST,variant_2_HIST,j)
Control_variant3 = cv2.compareHist(variant_1_HIST,variant_3_HIST,j)
Control_variant4 = cv2.compareHist(variant_1_HIST,variant_4_HIST,j)
Control_variant5 = cv2.compareHist(variant_1_HIST,variant_5_HIST,j)
Control_variant6 = cv2.compareHist(variant_1_HIST,variant_6_HIST,j)
Control_variantblank = cv2.compareHist(variant_1_HIST,variant_blank_HIST,j)

y =[Control_Comp, Control_variant2, Control_variant3, Control_variant5, Control_variant4, Control_variant6]


fig, ax = plt.subplots()
#fit = np.polyfit(x, y, deg=1)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.scatter(x, y),plt.plot(x, p(x)) 
fig.show()


def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    correlation = np.corrcoef(x, y)[0,1]

     # r
    results['correlation'] = correlation
     # r-squared
    results['determination'] = correlation**2

    return results

correlation = polyfit(x, y, 1)['correlation']

print(polyfit(x, y, 1))










