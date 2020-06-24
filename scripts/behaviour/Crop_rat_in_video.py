# -*- coding: utf-8 -*-
"""
Track_rat_in_video.py

Finds the nose and tail coordinates for every frame of a behaviour video

@author: You
"""
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tracking_library as tracking

# Reload modules
import importlib
importlib.reload(tracking)

# Specify video file name
#video_path = '/home/kampff/LC/videos/pre.avi'
video_path = 'F:/Videogame_Assay/AK_49.2/2019_09_16-14_11/Video.avi'
# Open video
video = cv2.VideoCapture(video_path)

# Compute background (for first 10 minutes of a session, using approx. 50 frames)
background = tracking.compute_background_median(video, 0, 72000, 1440)

# Test crop movie making
importlib.reload(tracking)

# Specify outout movie/data folder and filename
output_folder = 'F:/Videogame_Assay/AK_49.2/2019_09_16-14_11/crop.avi'

# Create crop movie
test = tracking.crop_rat(video, background, 640, output_folder)

# Display
#plt.imshow(background, cmap='gray')
#plt.show()

# Cleanup
video.release()

#FIN


#33.2 28 41.2 23.02
#40.2 12.05
