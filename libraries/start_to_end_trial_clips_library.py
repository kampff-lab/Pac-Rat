# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:59:33 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import cv2
import numpy as np
import os
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import behaviour_library as behaviour 
import pandas as pd 

hardrive_path = r'F:/' 




def CLIPS_start_to_end_trial(sessions_subset):
    
    try:  
        
        for session in sessions_subset:
            
            start_idx = []
            end_idx = []
            
            #create folder to allocate the clips             
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(hardrive_path, session + '/events/')
            results_dir = os.path.join(script_dir +'/Clips')
            csv_path = 'Trial_idx.csv'

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            
        
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            trial_start_path = os.path.join(hardrive_path, session + '/events/' + 'TrialStart.csv')
            video_csv_path = os.path.join(hardrive_path, session + '/Video.csv')
            video_path = os.path.join(hardrive_path, session + '/Video.avi')
            
            
            start = behaviour.timestamp_CSV_to_pandas(trial_start_path)
            end = behaviour.timestamp_CSV_to_pandas(trial_end_path)
            video_time = behaviour.timestamp_CSV_to_pandas(video_csv_path)
            
            
            start_closest = behaviour.closest_timestamps_to_events(video_time, start)
            end_closest = behaviour.closest_timestamps_to_events(video_time, end)
            
            if len(start_closest)>len(end_closest):
                start_closest = start_closest[:-1]
            
            start_idx.append(start_closest)
            end_idx.append(end_closest)
            
            
            np.savetxt(csv_dir_path + csv_path , np.vstack((start_idx,end_idx)).T,delimiter=',', fmt='%s')
            #video stuff
                        
            inputVid = cv2.VideoCapture(video_path)
            inputWidth = int(inputVid.get(cv2.CAP_PROP_FRAME_WIDTH))
            inputHeight  =  int(inputVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Open Output movie file, then specify compression, frames per second and size
            fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
            fps = 120
            offset = 360 #3''
            # Specify clip parameters
            startFrame = np.array(start_closest) - offset
            endFrame = np.array(end_closest) + offset
            numFrames =  endFrame - startFrame


            # Set starting read position
            start = 0
            for e in range(len(numFrames[:])):
    
                start_frame = start + e
                inputVid.set(cv2.CAP_PROP_POS_FRAMES, startFrame[start_frame])
                if e < 10:
                    outputFilename= results_dir +'/Clip0%d.avi' %start_frame
                    outputVid = cv2.VideoWriter(outputFilename, fourcc, fps, (inputWidth, inputHeight))
                else:
                    outputFilename= results_dir +'/Clip%d.avi' %start_frame
                    outputVid = cv2.VideoWriter(outputFilename, fourcc, fps, (inputWidth, inputHeight))
                for i in range(numFrames[start_frame]):
                    ret, im = inputVid.read()
                    outputVid.write(im)
                print (i)
                outputVid.release()
             
        inputVid.release()
            
    except Exception: 
            print('error'+ session)
            pass   
    
            








