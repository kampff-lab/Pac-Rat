# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
 
import os

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']




#base_path = 'F:/Videogame_Assay/AK_48.4/2019_07_29-16_33/'
#
## Load Video File Details
#video_csv_file_path = base_path + 'video.csv'
#video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)
#
## Load Sync File Details                       
#sync_file_path = base_path + 'sync.bin'
#sync_dtype = np.uint8
#fs = 30000
#frame_rate = 120.0
#sync_data = np.fromfile(sync_file_path,sync_dtype)
#frame_sync_data = np.int8(sync_data & 1)
#frame_transitions = np.diff(frame_sync_data)
##zero at the end to deal with tuples, takes the first element of it
#frame_starts = np.where(frame_transitions == 1)[0]
#                
## Get sample number for each video frame
#num_video_frames = len(video_counter)
#sync_indices_for_video_frames = (video_counter-video_counter[0]).astype(np.uint32)
#sample_for_each_video_frame = frame_starts[sync_indices_for_video_frames]
#np.savetxt(base_path + r'Analysis/samples_for_frames.csv', sample_for_each_video_frame, delimiter=',')
#


# FIN                                




for r, rat in enumerate(rat_summary_ephys[3:]): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
 
        directory = session_path +'/Analysis/'
        
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        video_csv_file_path = session_path + '/video.csv'
        video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)
        
        # Load Sync File Details                       
        sync_file_path = session_path + '/sync.bin'
        sync_dtype = np.uint8
        fs = 30000
        frame_rate = 120.0
        sync_data = np.fromfile(sync_file_path,sync_dtype)
        frame_sync_data = np.int8(sync_data & 1)
        frame_transitions = np.diff(frame_sync_data)
        #zero at the end to deal with tuples, takes the first element of it
        frame_starts = np.where(frame_transitions == 1)[0]
                        
        # Get sample number for each video frame
        num_video_frames = len(video_counter)
        sync_indices_for_video_frames = (video_counter-video_counter[0]).astype(np.uint32)
        sample_for_each_video_frame = frame_starts[sync_indices_for_video_frames]
        np.savetxt(directory+ 'samples_for_frames.csv', sample_for_each_video_frame, delimiter=',')
        print(session)
        










