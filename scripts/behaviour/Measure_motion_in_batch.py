# -*- coding: utf-8 -*-
"""
Track_rat_in_video.py

measure normalized motion (pixel changing in each frame normalised by the area of detected blobs)

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
import parser_library as prs
# Reload modules


hardrive_path = r'F:/'
rat_summary_table_path ='F:/Videogame_Assay/AK_33.2_Pt.csv'



Level_0 = prs.Level_0_paths(rat_summary_table_path)
Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_3= prs.Level_3_pre_paths(rat_summary_table_path)


# Reload modules
import importlib
importlib.reload(tracking)


session_list = Level_2_pre[0]



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
        output_folder = os.path.join(hardrive_path, session)
        output_name = os.path.join(hardrive_path, session +'/motions_clips_42') #output_folder + '/motion'
        
        #measure motion 
        tracking.motion(video, background, output_name)
        
        # Cleanup
        video.release()
        
    except Exception:
        print('something_WRONG' + session)
        continue
     
              

video_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Clips/Clip42.avi'

test= 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/motions_clips_42.csv'
test_clip = np.genfromtxt(test, delimiter=',',usecols=0)            








