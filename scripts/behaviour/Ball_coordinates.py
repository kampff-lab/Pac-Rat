# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:14:13 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import behaviour_library as behaviour
import parser_library as prs
import cv2 




rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_33.2'

Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)

sessions_subset = Level_2_pre



for session in sessions_subset:
    try:
    
        script_dir = os.path.join(hardrive_path + session) 
        video_path = os.path.join(hardrive_path, session + '/Video.avi')
        main_folder = os.path.join(script_dir +'/Ball_frames')
        csv_dir_path = os.path.join(hardrive_path, session + '/events/')
        csv_path = 'Ball_coordinates.csv'

        if not os.path.isdir(main_folder):
           os.makedirs(main_folder)   
       
        csv_dir_path = os.path.join(hardrive_path, session + '/events/')
        trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv') 
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
       
        touch_idx = trial_idx[:,2]
            
        frame_collection = behaviour.frame_before_trials(main_folder,video_path,touch_idx)
        ball_coordinates = behaviour.centroid_ball(main_folder)
     
        np.savetxt(csv_dir_path + csv_path, ball_coordinates,delimiter=',',fmt='%s')
            
        
    except Exception: 
        print (session + '/error')
    continue       

            



            
            
         










