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


hardrive_path = r'F:/'
rat_summary_table_path = r'F:/Videogame_Assay/AK_40.2_Pt.csv'

Level_1 = Level_1_parser(rat_summary_table_path)
Level_2_pre = Level_2_pre_parser(rat_summary_table_path)
Level_2_post = Level_2_post_parser(rat_summary_table_path)


# Reload modules
import importlib
importlib.reload(tracking)


session_list = Level_2_pre


for session in session_list:
    video_path = os.path.join(hardrive_path, session +'/Video.avi')
    try:
        # Open video
        video = cv2.VideoCapture(video_path)
        
        # Compute background (for first 10 minutes of a session, using approx. 50 frames)
        background = tracking.compute_background_median(video, 0, 72000, 1440)
        
        # Test crop movie making
        importlib.reload(tracking)
        
        # Specify outout movie/data folder and filename
        output_folder = os.path.join(hardrive_path, session + '/crop')
        
        # Create crop movie
        test = tracking.crop_rat(video, background, 640, output_folder)
        
        # Cleanup
        video.release()
        
    except Exception:
        print('something_WRONG' + session)
        continue
     
              


             


