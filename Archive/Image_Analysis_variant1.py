# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:23:28 2016

@author: jrs75
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import pandas as pd
#import easygui
import os


f = os.path.expanduser("~/Desktop/image10.jpg")
#f2 = os.path.expanduser("~/Desktop/image100.jpg")



img = cv2.imread(f,1)
#img2 = cv2.imread(f2,1)


x = 0
y = 900
w = 2000
h = 250


mask = np.zeros(img.shape[:2],np.uint8)
mask[y:y+h,x:x+w] = 255
res = cv2.bitwise_and(img,img,mask = mask)

#mask2 = np.zeros(img2.shape[:2],np.uint8)
#mask2[y:y+h,x:x+w] = 255
#res2 = cv2.bitwise_and(img2,img2,mask = mask2)

#mask = np.zeros(img.shape,np.uint8)
#mask[y:y+h,x:x+w] = img[y:y+h,x:x+w]


hist = cv2.calcHist([res],[0],None,[256],[0,255])
#hist2 = cv2.calcHist([res2],[0],None,[256],[0,255])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])


plt.hist(img.ravel(),256,[0,256]); plt.show()
plt.hist(img.ravel(),256,[0,256]); plt.show()

plt.plot(hist_full), plt.plot(hist_mask)

hist,bins = np.histogram(res.ravel(),256,[0,256])
#plt.hist(img2.ravel(),256,[0,256]); plt.show()

#plt.imshow(mask, 'gray')
#plt.imshow(res2, 'gray')


