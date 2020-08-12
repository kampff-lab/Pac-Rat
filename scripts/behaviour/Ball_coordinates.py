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


import importlib
importlib.reload(prs)
importlib.reload(behaviour)
hardrive_path = r'F:/' 




rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardLevel_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)

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
    continue  rive_path = r'F:/' 


     

            



hardrive_path = r'F:/' 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']




#Level 2 saving trial idx in each session folder under events folder


def frame_before_trials(target_dir,video_path,event= 3, offset=50):
    
    
    trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
    trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
    
    event_idx = trial_idx[:,event]
    
    video = cv2.VideoCapture(video_path)
    success, image=video.read()
    success=True
    count = 0
    for i in event_idx:
        
        video.set(cv2.CAP_PROP_POS_FRAMES, i + offset)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image 
#





for count, rat in enumerate(rat_summary_table_path):
    
    
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
     
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
           
            
                
            frame_collection = frame_before_trials(main_folder,video_path,event= 3, offset=50)
            ball_coordinates = behaviour.centroid_ball(main_folder)
         
            np.savetxt(csv_dir_path + csv_path, ball_coordinates,delimiter=',',fmt='%s')
            print(session + 'done')
    
        except Exception: 
            print (session + '/error')
            continue  
            









