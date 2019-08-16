# -*- coding: utf-8 -*-
"""
Track_rat_in_video.py

Finds the nose and tail coordinates for every frame of a behaviour video

@author: You
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
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

# Open video
video = cv2.VideoCapture(video_path)

# Compute background (for first 10 minutes of a session, using approx. 50 frames)
background = tracking.compute_background_median(video, 0, 72000, 1440)

# Test tracking
importlib.reload(tracking)
test = tracking.track_rat(video, background)

# Display
plt.imshow(background, cmap='gray')
plt.show()

#FIN