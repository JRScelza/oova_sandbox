# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:49:48 2016

@author: jrs75
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\Users\jrs75\OneDrive\Pictures\test22.jpg',0)
 
 # create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask2 = np.zeros(img.shape[:2], np.uint8)

mask[0:40, 5:20] = 255
mask2[0:40, 42:60] = 255

masked_img = cv2.bitwise_and(img,img,mask = mask)
masked_img2 = cv2.bitwise_and(img,img,mask = mask2)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
hist_mask2 = cv2.calcHist([img],[0],mask2,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(masked_img,'gray')
plt.subplot(223), plt.imshow(masked_img2, 'gray')

plt.subplot(224), plt.plot(hist_mask), plt.plot(hist_mask2)

plt.xlim([0,256])
   
plt.show()

values = np.transpose(hist_mask)
bins = np.ones(256)
area = np.sum((bins)*values)

print(area)

values2 = np.transpose(hist_mask2)
bins2 = np.ones(256)
area2 = np.sum((bins2)*values2)

print(area2)
