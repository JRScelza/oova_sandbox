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


f = os.path.expanduser("~/Desktop/ImagesOova/variant_1_1/image179.jpg")
f_2 = os.path.expanduser("~/Desktop/ImagesOova/variant_3_1/image179.jpg")


img = cv2.imread(f,1)
img2 = img

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
#
#h, s, v = cv2.split(hsv)
#v += 255
#final_hsv = cv2.merge((h, s, v))
#img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

#img = cv2.equalizeHist(img)
 
#--------------------------------------------------Contrast ALGO using CLHA

lab= cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
#cv2.imshow("lab",lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
#cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
#cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

img2 = final

#------------------------------------------------------------------threshold

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = gray

#blur = cv2.GaussianBlur(img,(5,5),0)
ret,thresh1 = cv2.threshold(img2,190,255,cv2.THRESH_BINARY)
img2 = thresh1

#------------------------------------------------------------------------------mask
     
mask = np.zeros(img2.shape[:2], np.uint8)
mask2 = np.zeros(img2.shape[:2], np.uint8)

mask[0:55, 10:20] = 255
mask2[0:55, 42:52] = 255

masked_img = cv2.bitwise_and(img2,img2,mask = mask)
masked_img2 = cv2.bitwise_and(img2,img2,mask = mask2)


mask3 = np.zeros(img.shape[:2], np.uint8)
mask4 = np.zeros(img.shape[:2], np.uint8)

mask3[0:55, 10:20] = 255
mask4[0:55, 42:52] = 255

masked_img3 = cv2.bitwise_and(img,img,mask = mask3)
masked_img4 = cv2.bitwise_and(img,img,mask = mask4)
---------------------------------------------------------------------------------mask
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img2],[0],mask,[256],[0,256])
hist_mask2 = cv2.calcHist([img2],[0],mask2,[256],[0,256])

hist_mask3 = cv2.calcHist([img],[0],mask,[256],[0,256])
hist_mask4 = cv2.calcHist([img],[0],mask2,[256],[0,256])
#----------------------------------------------------------If you want to see what is really happening, swap plots masked_img3/4_-/2

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(masked_img3,'gray')
plt.subplot(223), plt.imshow(thresh1, 'gray')
plt.subplot(224), plt.plot(hist_mask3), plt.plot(hist_mask4)
plt.xlim([0,256])
   
plt.show()

values = np.transpose(hist_mask)
#bins = np.ones(256)
#area = np.sum((bins)*values)

#print(area)

values2 = np.transpose(hist_mask2)
#bins2 = np.ones(256)
#area2 = np.sum((bins2)*values2)

#print(area2)

#print((area/area2)*100)

f = f
q = f_2

#print(np.shape(values2))
v = np.arange(0,256,1,dtype=int)
v = v.reshape(1,256)
v = pd.DataFrame(v)
#values2 = pd.DataFrame(values2)
#values = pd.DataFrame(values)

#values2.to_csv(f,sep=',')
#values.to_csv(q,sep=',')

area2 = np.sum(np.multiply(v,values2),axis=1)
area = np.sum(np.multiply(v,values),axis=1)


print(area)
print(area2)

print((area2/area)*100)

#LH = str(area2/area*100)

#easygui.msgbox("you're LH Value is: " + str(LH), title="LH Value")



