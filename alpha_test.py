#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:58:59 2017

@author: jeromescelza
"""


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from roi import region_loc

#You can pull in a random image from variant_1 to test the code.  Through the time series the image line intensity of variant_1 gets darker and darker.  You can grab it at
#different points to "pretend" you have more test strips.

image_number = 100

f_test = os.path.expanduser("~/Desktop/ImagesOova/variant_4_1/image%s.jpg" % image_number)

bounds , spike_pt  = region_loc(f_test)

variant_test = cv2.imread(f_test)

min_thresh = 126

ret,threshtest = cv2.threshold(variant_test,min_thresh,255,cv2.THRESH_BINARY)


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

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange  # ranges = [0,180,0,256]
ranges=None

#histbase = cv2.calcHist(basehsv,[0],mask,[0,256],ranges)



variant_test_HIST = cv2.calcHist([threshtest],[2],mask,[256],[0,256])


#res = cv2.bitwise_and(variant_5,variant_5,mask = mask)


#Creating a list of values that represents the dummy strip line intensity (0, 16.67, 33, etc...)

y =[]

for i in range(0,6):
    y.append(100-(i*16.67))

#The following will use the native OpenCV functions to compare histograms
#The test histogram is variant_i
#The control histogram is always variant_1, that is our 100%
#Opencv can use a handful of Probablility correlation functions outlined in the doc below
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

j = 4 # this is the compare function number i am using, you can selcet 0=Correlation, 1=Chi-Square, 2=Intersection, 3=Bhattacharyya, 4=Hellinger
#for j in range(0,4):    

Control_varianttest = cv2.compareHist(variant_1_HIST,variant_test_HIST,j)








plt.subplot(231),plt.imshow(variant_test,'gray'),plt.title('%s' % p(Control_varianttest) )
plt.subplot(232),plt.imshow(res,'gray'),plt.title('REGION OF INTEREST')
plt.subplot(233),plt.imshow(thresh1,'gray'),plt.title('MORPH TRANSFORM')

plt.show()




cv2.rectangle(variant_test,(spike_pt + 70 , bounds - 100),(spike_pt + 1350, bounds + 100),(0,255,0),10)
plt.imshow(variant_test,'gray'),plt.title('%s' % p(Control_varianttest) )
plt.show()



variables = [thresh1,thresh2, thresh3,thresh5,thresh4,thresh6 ] 

#j = 231
#
#for i in variables:
#    fig, ax = plt.subplots()
#    plt.subplot(j),plt.imshow(i)
#    plt.imshow(i)
#    j = j +1
#
#
#cv2.waitKey(1)
#cv2.destroyAllWindows()
#cv2.waitKey(3)

#46 52 50
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#print(hsv_green)
#[[[40 29 52]]]










im = variant_6

hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

imgg = cv2.cvtColor(im , cv2.COLOR_HSV2BGR)

#working well on 1 through 3
#COLOR_MIN = np.array([55,10, 110],np.uint8)
#COLOR_MAX = np.array([80, 255, 200],np.uint8)


#Purple

COLOR_MIN = np.array([138,50, 50],np.uint8)
COLOR_MAX = np.array([158, 255, 200],np.uint8)









mask = cv2.inRange(imgg, COLOR_MIN, COLOR_MAX)

frame_threshed = mask

ret,thresh = cv2.threshold(frame_threshed,127,255,0)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#img_black = np.zeros((1944, 2592), dtype=np.uint8)


#color = cv2.cvtColor(img_black, cv2.COLOR_GRAY2BGR)
#img = cv2.drawContours(color, contours, -1, (255,255,255), 2)

#opening = cv2.morphologyEx(color, cv2.MORPH_OPEN, kernel)

#
#opening = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel)
#
#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)




areas = [cv2.contourArea(c) for c in contours]

max_index = np.argmax(areas)


#arr = np.array(areas)
#
#max_list = arr.argsort()[-5:][::-1]

cnt=contours[max_index]

x,y,w,h = cv2.boundingRect(cnt)

h = h +30

y = y - 30

cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),15)



#plt.imshow(frame_threshed)




