#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:17:32 2017

@author: jeromescelza
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
from colorSpace import region_loc

f_t = os.path.expanduser("~/Desktop/ImagesOova/variant_4_1/image100.jpg")

variant_6 = cv2.imread(f_1)

im = variant_6

hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

x = 313 + 70
y = 1033 - 100
w = 1350
h = 200

mask = np.zeros(variant_6.shape[:2],np.uint8)
mask[y:y+h,x:x+w] = hsv_img[y:y+h,x:x+w,2]

#temp = hsv_img[250:400 , 900:1100 , 2 ]

plt.imshow(mask)

hist = cv2.calcHist([mask],[0],None,[256],[0,256])




















#hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#
##imgg = cv2.cvtColor(im , cv2.COLOR_HSV2BGR)
#
##working well on 1 through 3
#COLOR_MIN = np.array([55,10, 35],np.uint8)
#COLOR_MAX = np.array([90, 255, 200],np.uint8)
#
#
##Purple
##
##COLOR_MIN = np.array([138,50, 50],np.uint8)
##COLOR_MAX = np.array([158, 255, 200],np.uint8)
#
#
#
#mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
#
#frame_threshed = mask
#
#ret,thresh = cv2.threshold(frame_threshed,127,255,0)
#
#_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
##img_black = np.zeros((1944, 2592), dtype=np.uint8)
#
#
##color = cv2.cvtColor(img_black, cv2.COLOR_GRAY2BGR)
##img = cv2.drawContours(color, contours, -1, (255,255,255), 2)
#
##opening = cv2.morphologyEx(color, cv2.MORPH_OPEN, kernel)
#
##
##opening = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel)
##
##closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
#
#
#
#areas = [cv2.contourArea(c) for c in contours]
#
#max_index = np.argmax(areas)
#
#
##arr = np.array(areas)
##
##max_list = arr.argsort()[-5:][::-1]
#
#cnt=contours[max_index]
#
#x,y,w,h = cv2.boundingRect(cnt)
#
#h = h +30
#
#y = y - 30
#
#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),15)
#
#
#
#plt.imshow(frame_threshed)