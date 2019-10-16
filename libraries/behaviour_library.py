# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:31:25 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import pandas as pd 


hardrive_path = r'F:/' 



#given a session it finds the file called TrialEnd.csv which contains the timestamp of each trial together with the outcome of that trial (Food or Missed), and 
#it return the list of reward and misses trial by searching the string in column one  



def trial_outcome_index(session):
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




def find_trial_and_misses(sessions_subset):    
    success_trials = []
    missed_trials = []
    for session in sessions_subset: 
        try:
            rewards, misses = trial_outcome_index(session) #would take session instead
            success_trials.append(len(rewards))
            missed_trials.append(len(misses))
        except Exception: 
            print('error'+ session)
            continue
    return success_trials, missed_trials   


  
def calculate_trial_per_min(sessions_subset):    
    total_trials = []
    session_length = []
    try:
        for session in sessions_subset: 
            #trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            #RewardOutcome_file = np.genfromtxt(trial_end_path, usecols=[1], dtype= str)
        
            counter_csv = os.path.join(hardrive_path, session + '/Video.csv')
            counter = np.genfromtxt(counter_csv, usecols = 1)
        
            tot_frames = counter[-1] - counter[0]
            session_length_minutes = tot_frames/120/60
            
            rewards, misses = trial_outcome_index(session)
            tot_trials = len(rewards + misses)
            total_trials.append(tot_trials)
            session_length.append(session_length_minutes)
    except Exception: 
            print('error'+ session)
            pass
        
    return total_trials, session_length  



def calculate_full_trial_speed(sessions_subset):
    
    full_trials_speed_seconds = []
    
    
    try:
        for session in sessions_subset:
            
            
            trial_start_path = os.path.join(hardrive_path, session + '/events/' + 'TrialStart.csv')
            trial_start_path_time = timestamp_CSV_to_pandas(trial_start_path)
                        
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            trial_end_path_time = timestamp_CSV_to_pandas(trial_end_path)
            
            if len(trial_start_path_time)>len(trial_end_path_time):
                
                trial_start_path_time=trial_start_path_time[:-1]
                #removing the date 
                #trial_start_path_time_no_date = pd.Series([val.time() for val in trial_start_path_time])
                #trial_end_path_time_no_date= pd.Series([val.time() for val in trial_end_path_time])
                               
                trial_speed= trial_end_path_time - trial_start_path_time
                trial_speed_seconds = trial_speed.dt.total_seconds()
                
                full_trials_speed_seconds.append(trial_speed_seconds)

                
    except Exception: 
            print('error'+ session)
            pass
    return full_trials_speed_seconds



def calculate_trial_speed_from_ball_touch(sessions_subset):
    
    touch_to_reward_speed_seconds = []
    
    try:
        for session in sessions_subset:
            
         
            touch_path = os.path.join(hardrive_path, session + '/events/' + 'RatTouchBall.csv')
            touch_time = timestamp_CSV_to_pandas(touch_path)
                        
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            trial_end_path_time = timestamp_CSV_to_pandas(trial_end_path)
                     
            trial_speed= trial_end_path_time - touch_time            

            trial_speed_seconds = trial_speed.dt.total_seconds()
                                
            touch_to_reward_speed_seconds.append(trial_speed_seconds)
            
    except Exception: 
            print('error'+ session)
            pass
    return touch_to_reward_speed_seconds



def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(timestamp)
    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
    return timestamp_Series

def closest_timestamps_to_events(timestamp_list, event_list):
    nearest  = []
    for e in event_list:
        delta_times = timestamp_list-e
        nearest.append(np.argmin(np.abs(delta_times)))
    return nearest  



######################################################################
    
#create a .csv with the start-end-touch idx and save it in the rat session folder 
    


def start_touch_end_idx(sessions_subset):
            
    for session in sessions_subset:
        start_idx = []
        end_idx = []
        touch_idx = []
        
        try:
           
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(hardrive_path, session + '/events/')
            csv_path = 'Trial_idx.csv'


            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            trial_start_path = os.path.join(hardrive_path, session + '/events/' + 'TrialStart.csv')
            touch_path = os.path.join(hardrive_path, session + '/events/' + 'RatTouchBall.csv')
            
            video_csv_path = os.path.join(hardrive_path, session + '/Video.csv')
            video_path = os.path.join(hardrive_path, session + '/Video.avi')
            
            
            start = behaviour.timestamp_CSV_to_pandas(trial_start_path)
            end = behaviour.timestamp_CSV_to_pandas(trial_end_path)
            touch = behaviour.timestamp_CSV_to_pandas(touch_path)
            
            video_time = behaviour.timestamp_CSV_to_pandas(video_csv_path)
            
            
            start_closest = behaviour.closest_timestamps_to_events(video_time, start)
            end_closest = behaviour.closest_timestamps_to_events(video_time, end)
            touch_closest = behaviour.closest_timestamps_to_events(video_time, touch)
            
            if len(start_closest)>len(end_closest):
                start_closest = start_closest[:-1]
                if len(touch_closest)>len(end_closest): 
                    touch_closest = touch_closest[:-1]
                                    
            
            start_idx.append(start_closest)
            end_idx.append(end_closest)
            touch_idx.append(touch_closest)
            
            
            np.savetxt(csv_dir_path + csv_path, np.vstack((start_idx,end_idx,touch_idx)).T,delimiter=',', fmt='%s')

        except Exception: 
            print('error'+ session)
        continue




#############################################################################
            
def trial_nose_trajectory(nose_file,trial_closest_ir,end_closest_ir):    
    nose=np.genfromtxt(nose_file)
    trial_closest_array=np.array(trial_closest_ir)
    end_closest_array=np.array(end_closest_ir)
    dif=abs(end_closest_array-trial_closest_array)
    count=0
    n=len(trial_closest_ir)
    nose_trial_trajectory= [[] for _ in range(n)] 
    for i in trial_closest_ir:
        nose_trial_trajectory[count]=nose[i:i+dif[count]]
        count += 1
    return nose_trial_trajectory




def session_speed(sessions_subset):
        
    n = len(sessions_subset)
    sessions_speed = [[] for _ in range(n)]         
    for count, session in enumerate(sessions_subset):  
        try: 
            
            script_dir = os.path.join(hardrive_path + session) 
            centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
            
            trajectory_x = centroid_tracking[:,0]
            trajectory_y = centroid_tracking[:,1]
            diff_x = np.diff(trajectory_x)
            diff_y = np.diff(trajectory_y)    
            diff_x_square = diff_x**2
            diff_y_square = diff_y**2
            speed = np.sqrt(diff_x_square + diff_y_square)
            sessions_speed[count] = speed
            
  
        except Exception: 
            print('error'+ session)
        continue
                               
    return sessions_speed 

    
def speed_touch_to_reward(sessions_subset, sessions_speed):
    
    l = len(sessions_subset)
    Level_2_touch_to_reward_speed = [[] for _ in range(l)]     
    #count = 0
    
    for count in np.arange(l):
        session = sessions_subset[count]
        speed = sessions_speed[count]
        
        script_dir = os.path.join(hardrive_path + session) 
        trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        end_touch_diff = abs(trial_idx[:,1] - trial_idx[:,2])
    
        n = len(trial_idx)
        touch_idx = trial_idx[:,2]        
        touch_to_reward_speed = [[] for _ in range(n)] 
                       
        count_1 = 0
    
        for touch in touch_idx:
            touch_to_reward_speed[count_1] = speed[touch:touch + end_touch_diff[count_1]]
            count_1 += 1
            
        Level_2_touch_to_reward_speed[count] = touch_to_reward_speed

                 

    return Level_2_touch_to_reward_speed
        

def speed_start_to_touch(sessions_subset, sessions_speed):
    
    l = len(sessions_subset)
    Level_2_start_to_touch_speed = [[] for _ in range(l)]     
    
    for count in np.arange(l):
        session = sessions_subset[count]
        speed = sessions_speed[count]
        
        script_dir = os.path.join(hardrive_path + session) 
        trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        start_touch_diff = abs(trial_idx[:,0] - trial_idx[:,2])
    
        n = len(trial_idx)
        start_idx = trial_idx[:,0]        
        start_to_touch_speed = [[] for _ in range(n)] 
                       
        count_1 = 0
    
        for start in start_idx:
            start_to_touch_speed[count_1] = speed[start:start + start_touch_diff[count_1]]
            count_1 += 1
            
        Level_2_start_to_touch_speed[count] = start_to_touch_speed

                 

    return Level_2_start_to_touch_speed
       

 




def speed_around_touch(sessions_subset,sessions_speed):
    
    x=len(sessions_subset)
    speed_touch_Level_2 = [[] for _ in range(x)] 
    
    for count in np.arange(x):
        session = sessions_subset[count]
        speed = sessions_speed[count]
                
        script_dir = os.path.join(hardrive_path + session) 
        csv_dir_path = os.path.join(hardrive_path, session + '/events/')
        trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        touch_idx = trial_idx[:,2]
        offset = 360
        n = len(touch_idx)
        speed_touch = [[] for _ in range(n)]
        
        count_1 = 0
        for i in touch_idx:
            speed_touch[count_1]=speed[i-offset:i+offset]
            count_1 += 1
        speed_touch_Level_2[count] = speed_touch
    
    return speed_touch_Level_2
                
                
                        
                        
                        
                        
                
            
            














#def distance(nose_trajectory,ball_coordinates,trial_closest_ir,end_closest_ir):
#    trial_closest_array=np.array(trial_closest_ir)
#    end_closest_array=np.array(end_closest_ir)
#    dif=abs(end_closest_array-trial_closest_array)
#    count=0
#    n=len(ball_coordinates)
#    new_ball=[[] for _ in range(n)]
#    for i in ball_coordinates:
#        shape=len(range(dif[count]))
#        ball=np.repeat([i],[shape],axis=0)
#        new_ball[count]=ball[:]
#        count += 1     
#    dst= [[] for _ in range(n)]
#    for e in range(n):
#        distances = (new_ball[e]-nose_trajectory[e])**2
#        distances = distances.sum(axis=-1)
#        distances = np.sqrt(distances)
#        dst[e]=distances
#    return dst
#
#
#
#
#
#def touch(trial_closest_ir,start_closest_ir):
#    trial_closest_array=np.array(trial_closest_ir)
#    start_array=np.array(start_closest_ir)
#    #by doing the difference I find the len between the 2
#    trial_touch_value=abs(start_array-trial_closest_array) # this should be enough to be able to then subcract this value to an array of ascending number but different enght according the the trial
#    return trial_touch_value
#
#



############################################################################################################
#
#
#def idx_Event(trial_end):
#    RewardOutcome_file=np.genfromtxt(trial_end,usecols=[1], dtype= str)
#    RewardOutcome_idx=[]
#    count=0
#    for i in RewardOutcome_file:
#        count += 1
#        if i =='Missed':
#            RewardOutcome_idx.append(count-1)
#    reward=np.array(RewardOutcome_idx)
#    return RewardOutcome_idx
#
#
#
#
#
#def idx_Event(trial_end):
#    RewardOutcome_file=np.genfromtxt(trial_end,usecols=[1], dtype= str)
#    RewardOutcome_idx=[]
#    for count, in enumerate(RewardOutcome_file):
#        if e =='Missed':
#            RewardOutcome_idx.append(i)
#    reward=np.array(RewardOutcome_idx)
#    return RewardOutcome_idx
#
#
#
#
#
#def tone_frame(target_dir,video_avi_file_path,nearest):
#    video=cv2.VideoCapture(video_avi_file_path)
#    success, image=video.read()
#    success=True
#    count = 0
#    for i in nearest:
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
#def closest_value_in_array(array,value_list):
#    nearest  = []
#    for e in value_list:
#        delta = array-e
#        nearest.append(np.argmin(np.abs(delta)))
#    return nearest   
#
#

def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(timestamp)
    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
    return timestamp_Series
#
#      
#def closest_timestamps_to_events(timestamp_list, event_list):
#    nearest  = []
#    for e in event_list:
#        delta_times = timestamp_list-e
#        nearest.append(np.argmin(np.abs(delta_times)))
#    return nearest  
#
#
#
#








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










