# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:33:43 2019

@author: KAMPFF-LAB-ANALYSIS3
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
import DLC_parser_library as DLC


hardrive_path = r'F:/'
rat_summary_table_path = hardrive_path + 'Videogame_Assay/AK_50.2_behaviour_only.csv'
#rat_ID = 'AK_33.2'

#select the path of the sessions of interest belonging to different levels
Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)

# Reload modules
import importlib
importlib.reload(DLC)


sessions_subset = Level_1

for session in sessions_subset:
    
    script_dir = os.path.join(hardrive_path + session) 
    target_folder = os.path.join(script_dir +'/DLC_corrected_coordinates')
    video_path = os.path.join(script_dir + '/Video.csv')
    video_csv_counter = np.genfromtxt(video_path, usecols = 1)
    video_lenght = len(video_csv_counter)
    
    
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder) 
    
    try:
        #correct the dlc coordinates based on the initial cropping (Nan are inserted if the dlc likelihood is not greater than 0.999)
        x_nan_nose, y_nan_nose = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
        x_nan_tail_base, y_nan_tail_base = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
        
        #np.savetxt(target_folder + '/x_nan_nose.csv', x_nan_nose, delimiter=',',fmt='%s')
        #np.savetxt(target_folder + '/y_nan_nose.csv', y_nan_nose, delimiter=',',fmt='%s')
        #np.savetxt(target_folder + '/x_nan_tail_base.csv', x_nan_tail_base, delimiter=',',fmt='%s')
        #np.savetxt(target_folder + '/y_nan_tail_base.csv', y_nan_tail_base, delimiter=',',fmt='%s')
        
        # stack x and y of the same body part and save them into a folder called 'DLC_corrected_coordinates' 
        np.savetxt(target_folder + '/nose_corrected_coordinates.csv', np.vstack((x_nan_nose,y_nan_nose)).T,delimiter=',', fmt='%s')
        np.savetxt(target_folder + '/tail_base_corrected_coordinates.csv', np.vstack((x_nan_tail_base,y_nan_tail_base)).T,delimiter=',', fmt='%s')


        x_nan_nose_lenght = len(x_nan_nose)
        y_nan_nose_lenght = len(y_nan_nose)
        x_nan_tail_base_lenght = len(x_nan_tail_base)
        x_nan_tail_base_lenght = len(y_nan_tail_base)
        
        #print the lenghts to be able to compare the lenght of the videos.csv with the final body parts coordinate. they should be same
        print(session + '_DONE')
        print(video_lenght)
        print(x_nan_nose_lenght)
        print(y_nan_nose_lenght)
        print(x_nan_tail_base_lenght)
        print(x_nan_tail_base_lenght)
        
    except Exception:
        print('something_WRONG' + session)
        continue
     
              


