# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:31:25 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np


hardrive_path = r'F:/' 



#given a session it finds the file called TrialEnd.csv which contains the timestamp of each trial together with the outcome of that trial (Food or Missed), and 
#it return the list of reward and misses trial by searching the string in column one  



def trial_outcome(session):
    trial_end_path = os.path.join(hardrive_path, session + '/events/'+'TrialEnd.csv')
    RewardOutcome_file=np.genfromtxt(trial_end_path, usecols=[1], dtype= str)
    rewards = []
    misses = []
    for count, i in enumerate(RewardOutcome_file):
        if i =='Food':
            rewards.append(count)
        else:
            misses.append(count)
    return rewards, misses




def PLOT_trial_and_misses(sessions_subset):    
    success_trials = []
    missed_trials = []
    for session in sessions_subset: 
        try:
            rewards, misses = trial_outcome(session) #would take session instead
            success_trials.append(len(rewards))
            missed_trials.append(len(misses))
        except Exception: 
            print('error'+ session)
            continue
    return success_trials, missed_trials   


  
def PLOT_trial_per_min(sessions_subset):    
    total_trials = []
    session_length = []
    try:
        for session in sessions_subset: 
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            RewardOutcome_file = np.genfromtxt(trial_end_path, usecols=[1], dtype= str)
        
            counter_csv = os.path.join(hardrive_path, session + '/Video.csv')
            counter = np.genfromtxt(counter_csv, usecols = 1)
        
            tot_frames = counter[-1] - counter[0]
            session_length_minutes = tot_frames/120/60
            
            rewards, misses = trial_outcome(session)
            tot_trials = len(rewards + misses)
            total_trials.append(tot_trials)
            session_length.append(session_length_minutes)
    except Exception: 
            print('error'+ session)
            pass
        
    return total_trials, session_length  

























#
#def idx_Event(trial_end_path):
#    Ball_on = os.path.join(hardrive_path,session + '/events/'+'BallOn.csv')
#    ball_timestamps = pd.read_csv(Ball_on, header = None, parse_dates=[0])
#    video_path = os.path.join(hardrive_path,session + '/Video.csv')
#    video_timestamps = pd.read_csv(video_path, delimiter=' ', header = None,usecols=[0], parse_dates=[0])
#
#
#
#
#def closest_timestamps_to_events(timestamp_csv, event_list):
#    nearest  = []
#    for e in event_list[0]:
#        delta_times = timestamp_csv[0]-e
#        nearest.append(np.argmin(np.abs(delta_times)))
#    return nearest   




#movie_file = 'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/Video.avi'
#
#target_dir= 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test_frames'
#
#
#def frame_before_trials(target_dir,filename,cleaned_idx):
#    video=cv2.VideoCapture(filename)
#    success, image=video.read()
#    success=True
#    count = 0
#    for i in cleaned_idx:
#        video.set(cv2.CAP_PROP_POS_FRAMES, i)
#        success, image = video.read()
#        if count < 10:
#            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
#        else:
#            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
#        count += 1
#    return image
#
#










