# -*- coding: utf-8 -*-
"""
Crop_rat_in_video.py

Finds the centroid of the rat (ideally) and crops a sub-image around this position

@author: You
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tracking_library as tracking

# Reload modules
import importlib
importlib.reload(tracking)

# Specify video file name
video_path = '/home/kampff/LC/videos/pre.avi'
#video_path = 'E:/AK_33.2_test/2018_04_29-15_43/Video.avi'
# Open video
video = cv2.VideoCapture(video_path)

# Compute background (for first 10 minutes of a session, using approx. 50 frames)
background = tracking.compute_background_median(video, 0, 72000, 1440)

# Test cropping
importlib.reload(tracking)
test = tracking.crop_rat(video, background, 640)

# Display
#plt.imshow(background, cmap='gray')
#plt.show()

#FIN