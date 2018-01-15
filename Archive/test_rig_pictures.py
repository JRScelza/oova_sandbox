#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 07:18:05 2017

@author: jeromescelza
"""

#I need a light source
#A box to mount the stuff
# I might be able to just use the camera equipment
#Raspberry pi code, maybe a monitor and a keyboard and a mouse
#marker each bag of each variant
# NOTE that the image folders will be creating in the directory that  the Pi script is running from

import os
import sys
import numpy as np

from picamera import PiCamera
from time import sleep


image_folder_name = input('Please provide Variant (e.x. 1_0)')


f = "~/Desktop/test22.jpg"

newpath = 'Variant_%s' %  image_folder_name

camera = PiCamera()
camera.start_preview()
sleep(10)
camera.stop_preview()

if not os.path.exists(newpath):
    os.makedirs(newpath)

Answer = input('Was that enough time to get things aligned? [y/n]')

if Answer == 'y':
	camera.start_preview()
	for i in range(5):
		sleep(5)
		camera.capture('/home/pi/Desktop/image%s.jpg' % i)
		camera.stop_preview()
else:
    sys.exit()
	 