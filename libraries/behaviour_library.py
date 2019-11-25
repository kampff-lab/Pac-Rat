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
import cv2
import math

hardrive_path = r'F:/' 

#1 timestamp_Series = timestamp_CSV_to_pandas(filename)
#2 nearest = closest_timestamps_to_events(timestamp_list, event_list)
#3 rewards, misses = trial_outcome_index(session)
#4 success_trials, missed_trials = calculate_trial_and_misses(sessions_subset)
#5 total_trials, session_length = calculate_trial_per_min(sessions_subset)
#6 full_trials_speed_seconds = calculate_full_trial_speed(sessions_subset)
#7 touch_to_reward_speed_seconds = calculate_trial_speed_from_ball_touch(sessions_subset)
#8 start_touch_end_idx(sessions_subset)
#9 start_end_idx(sessions_subset)
#10 sessions_speed = session_speed(sessions_subset)
#11 Level_2_touch_to_reward_speed = speed_touch_to_reward(sessions_subset, sessions_speed)
#12 Level_2_start_to_touch_speed = speed_start_to_touch(sessions_subset, sessions_speed)
#13 speed_around_touch_Level_2 = speed_around_touch(sessions_subset,sessions_speed, offset = 360)
#14 image = frame_before_trials(target_dir,filename,cleaned_idx)
#15 ball_coordinates = centroid_ball(frame_folder)
#16 quadrant_1,quadrant_2,quadrant_3,quadrant_4 = ball_positions_based_on_quadrant_of_appearance(session)
#17 x_nose_tracking_snippets_te, y_nose_tracking_snippets_te, x_tail_base_tracking_snippets_te, y_tail_base_trial_tracking_snippets_te = 
#   create_tracking_snippets_touch_to_end(sessions_subset,start_snippet_idx = 0,end_snippet_idx = 1,mid_snippet_idx = 2)
#18 first_values_x_nose,first_values_y_nose,first_values_x_tail_base,first_values_y_tail_base = x_y_at_touch(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2)
#19 sessions_degrees_at_touch = nose_butt_angle_touch(first_values_x_nose,first_values_y_nose,first_values_x_tail_base,first_values_y_tail_base)
#20
            

####################################   1   ##########################################


#takes a filename containing timestamps and convert them in datetime series to be used to find closest timestamps

def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(timestamp)
    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
    return timestamp_Series


####################################   2   ##########################################

#find where the closest timestamps of an event of interest timestamp is and it return the idx 
def closest_timestamps_to_events(timestamp_list, event_list):
    nearest  = []
    for e in event_list:
        delta_times = timestamp_list-e
        nearest.append(np.argmin(np.abs(delta_times)))
    return nearest  


####################################   3   ##########################################

#given a session it finds the file called TrialEnd.csv which contains the timestamp
#of each trial together with the outcome of that trial (Food or Missed), and 
#it returns the list of reward and misses trial by searching the string in column one  

def trial_outcome_index(session):
    trial_end_path = os.path.join(hardrive_path, session +'/events/'+'TrialEnd.csv')
    reward_outcome_file = np.genfromtxt(trial_end_path, usecols = [1], dtype = str)
    rewards = []
    misses = []
    for count, i in enumerate(reward_outcome_file):
        if i =='Food':
            rewards.append(count)
        else:
            misses.append(count)
    return rewards, misses


####################################   4   ##########################################

#uses def 3 to calculate the successul and missed trial for each session of a 
#given session subset, used to plot histogram of misses and reward trials
    
def calculate_trial_and_misses(sessions_subset):    
    success_trials = []
    missed_trials = []
    for session in sessions_subset: 
        try:           
            rewards, misses = trial_outcome_index(session)
            success_trials.append(len(rewards))
            missed_trials.append(len(misses))
        except Exception: 
            print('error'+ session)
            pass
    return success_trials, missed_trials   

####################################   5   ##########################################

#calculate the trials per minutes for each session of a given session subset, this def is used
#to generate the trial/min plot 

def calculate_trial_per_min(sessions_subset):    
    total_trials = []
    session_length = []
    for session in sessions_subset: 
        try:  
            #opening the video csv file column 1 which contains the counter
            counter_csv = os.path.join(hardrive_path, session + '/Video.csv')
            counter = np.genfromtxt(counter_csv, usecols = 1)

            #subctract thefirst counter value from the last one to obtain the total lenght of the session
            #in frames and then convert it to minutes taking into account the video is recorded at 120fps
            tot_frames = counter[-1] - counter[0]
            #minutes conversion
            session_length_minutes = tot_frames/120/60
            
            #use def 3 to calcuate the tot amount of trials per session
            rewards, misses = trial_outcome_index(session)
            tot_trials = len(rewards + misses)
            total_trials.append(tot_trials)
            session_length.append(session_length_minutes)
        except Exception: 
            print('error'+ session)
        continue

    return total_trials, session_length


####################################   6   ##########################################

#calculate the time taken by the rat to collect the reward or miss a given trial from the start of the trial
#for session_subsets = Level 1 corresponds to the tone starting while for sessions_subset = Level 2 corresponds to the ball appearance
    
def calculate_full_trial_speed(sessions_subset):   
    full_trials_speed_seconds = []
        
    for session in sessions_subset:
        try:
            #file opening          
            trial_start_path = os.path.join(hardrive_path, session + '/events/' + 'TrialStart.csv')
            #convert the timestamps in datetime which can be subtracted 
            trial_start_path_time = timestamp_CSV_to_pandas(trial_start_path)
                        
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            #convert the timestamps in datetime which can be subtracted 
            trial_end_path_time = timestamp_CSV_to_pandas(trial_end_path)
            
            #in case the last trial started without ending (lenght of the 2 files differs by 1)
            #it removes the last value from the start list
            
            if len(trial_start_path_time)>len(trial_end_path_time):                
                trial_start_path_time = trial_start_path_time[:-1]
                
                #removing the date 
                #trial_start_path_time_no_date = pd.Series([val.time() for val in trial_start_path_time])
                #trial_end_path_time_no_date= pd.Series([val.time() for val in trial_end_path_time])
                               
                trial_speed = trial_end_path_time - trial_start_path_time
                #pandas fx to convert in seconds 
                trial_speed_seconds = trial_speed.dt.total_seconds()
                
                full_trials_speed_seconds.append(trial_speed_seconds)
                
        except Exception: 
            print('error'+ session)
        continue
    return full_trials_speed_seconds


####################################   7   ##########################################

#this def can be used only on sessions_subset = Level 2

def calculate_trial_speed_from_ball_touch(sessions_subset):
    
    touch_to_reward_speed_seconds = []
    
    for session in sessions_subset:
        try:
            #files opening and imestamp convesion      
            touch_path = os.path.join(hardrive_path, session + '/events/' + 'RatTouchBall.csv')
            touch_time = timestamp_CSV_to_pandas(touch_path)
                        
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            trial_end_path_time = timestamp_CSV_to_pandas(trial_end_path)
             
            trial_speed = trial_end_path_time - touch_time  
            #pandas fx to convert in seconds
            trial_speed_seconds = trial_speed.dt.total_seconds()
                                
            touch_to_reward_speed_seconds.append(trial_speed_seconds)
            
        except Exception: 
            print('error'+ session)
        continue
    return touch_to_reward_speed_seconds


#
#def timestamp_CSV_to_pandas(filename):
#    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
#    timestamp = timestamp_csv[0]
#    timestamp_Series= pd.to_datetime(timestamp)
#    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
#    return timestamp_Series
#
#def closest_timestamps_to_events(timestamp_list, event_list):
#    nearest  = []
#    for e in event_list:
#        delta_times = timestamp_list-e
#        nearest.append(np.argmin(np.abs(delta_times)))
#    return nearest  


####################################   8   ##########################################
    
#create a .csv with the start-end-touch idx and saves it in the rat session folder 
#sessions_subset needs to be Level 2
#column 0 = start idx   
#column 1 = end idx
#column 2 = touch idx

def start_touch_end_idx(sessions_subset):
            
    for session in sessions_subset:
        start_idx = []
        end_idx = []
        touch_idx = []
        
        try:
           
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(script_dir + '/events/')
            #name of the .csv fileto create
            csv_name = 'Trial_idx.csv'


            trial_end_path = os.path.join(csv_dir_path + 'TrialEnd.csv')
            trial_start_path = os.path.join(csv_dir_path + 'TrialStart.csv')
            touch_path = os.path.join(csv_dir_path + 'RatTouchBall.csv')
            
            video_csv_path = os.path.join(script_dir + '/Video.csv')
            
            #this def works for all the 4 files given the timestamps are located in column 0
            start = timestamp_CSV_to_pandas(trial_start_path)
            end = timestamp_CSV_to_pandas(trial_end_path)
            touch = timestamp_CSV_to_pandas(touch_path)
            #using colum 0 of the file contaning the timestamps
            video_time = timestamp_CSV_to_pandas(video_csv_path)
            
            #find closest event timestamps in the video timestamps file 
            start_closest = closest_timestamps_to_events(video_time, start)
            end_closest = closest_timestamps_to_events(video_time, end)
            touch_closest = closest_timestamps_to_events(video_time, touch)
            
            #if the lenght of the files differs by 1 value it removes the last timestamp 
            #from both start and touch file 
            if len(start_closest)>len(end_closest):
                start_closest = start_closest[:-1]
                if len(touch_closest)>len(end_closest): 
                    touch_closest = touch_closest[:-1]
                                    
            
            start_idx.append(start_closest)
            end_idx.append(end_closest)
            touch_idx.append(touch_closest)
            
            #saving a csv file containing the idx of start, end, touch
            np.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx,touch_idx)).T, delimiter=',', fmt='%s')

        except Exception: 
            print('error'+ session)
        continue



####################################   9   ##########################################
    
#create a .csv with the start-end idx and saves it in the rat session folder 
#sessions_subset needs to be Level 1
#column 0 = start idx   
#column 1 = end idx


def start_end_idx(sessions_subset):
            
    for session in sessions_subset:
        start_idx = []
        end_idx = []
        
        try:
            
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(script_dir + '/events/')
            csv_name = 'Trial_idx.csv'


            trial_end_path = os.path.join(csv_dir_path + 'TrialEnd.csv')
            trial_start_path = os.path.join(csv_dir_path + 'TrialStart.csv')
            
            video_csv_path = os.path.join(script_dir + '/Video.csv')
            
            #timestamp conversion
            start = timestamp_CSV_to_pandas(trial_start_path)
            end = timestamp_CSV_to_pandas(trial_end_path)
            
            video_time = timestamp_CSV_to_pandas(video_csv_path)
                        
            start_closest = closest_timestamps_to_events(video_time, start)
            end_closest = closest_timestamps_to_events(video_time, end)
            
            #if the lenght of the trial start is longer remove the last value 
            if len(start_closest)>len(end_closest):
                start_closest = start_closest[:-1]


            start_idx.append(start_closest)
            end_idx.append(end_closest)

            #sae a .csv file with start and end idx             
            np.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx)).T, delimiter=',', fmt='%s')

        except Exception: 
            print('error'+ session)
        continue


#############################################################################
#            
#def trial_nose_trajectory(nose_file,trial_closest_ir,end_closest_ir):    
#    nose=np.genfromtxt(nose_file)
#    trial_closest_array=np.array(trial_closest_ir)
#    end_closest_array=np.array(end_closest_ir)
#    dif=abs(end_closest_array-trial_closest_array)
#    count=0
#    n=len(trial_closest_ir)
#    nose_trial_trajectory= [[] for _ in range(n)] 
#    for i in trial_closest_ir:
#        nose_trial_trajectory[count]=nose[i:i+dif[count]]
#        count += 1
#    return nose_trial_trajectory


####################################   10   ##########################################
#calculate the speed for each session of a given sessions subset given the dlc corrected coordinates of the nose
        
def session_speed(sessions_subset):       
    n = len(sessions_subset)   
    sessions_speed = [[] for _ in range(n)] 
        
    for count, session in enumerate(sessions_subset):  
        
        try: 
            
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            corrected_coordinate_path = os.path.join(script_dir + '/DLC_corrected_coordinates')
            nose_path = os.path.join(corrected_coordinate_path + '/nose_corrected_coordinates.csv')
            nose_dlc = np.genfromtxt(nose_path, delimiter = ',', dtype = float)
            
            #select x and y positions from the dlc file
            trajectory_x = nose_dlc[:,0]
            trajectory_y = nose_dlc[:,1]
            #prepend a 0 in front of the file to maintan the same file lenght after diff
            diff_x = np.diff(trajectory_x, prepend = 0)
            diff_y = np.diff(trajectory_y, prepend = 0)    
            diff_x_square = diff_x**2
            diff_y_square = diff_y**2
            speed = np.sqrt(diff_x_square + diff_y_square)
            sessions_speed[count] = speed
              
        except Exception: 
            print('error'+ session)
        continue
                               
    return sessions_speed 


####################################   11   ##########################################

#used for Level 2 andcalculate the speed from touch to reward collection, in order to use this fx
#the trail_idx file needs to be created first or already existing 
   
def speed_touch_to_reward(sessions_subset, sessions_speed):
    
    l = len(sessions_subset)
    Level_2_touch_to_reward_speed = [[] for _ in range(l)]     
    #count = 0
    
    for count in np.arange(l):
        #selecting one session a the time and its speed 
        session = sessions_subset[count]
        speed = sessions_speed[count]
        
        #opening trial idx file
        script_dir = os.path.join(hardrive_path + session) 
        trial_idx_path = os.path.join(script_dir + '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        
        #selecting the column of touch and end, calculate the abs diff in order to calculate the 
        #how long it took to ebd the trial after the touch
        end_touch_diff = abs(trial_idx[:,1] - trial_idx[:,2])
    
        n = len(trial_idx)
        touch_idx = trial_idx[:,2]        
        touch_to_reward_speed = [[] for _ in range(n)] 
                       
        count_1 = 0
        
        #create and empty list of list where to allocate the speed values
        #from touch to end for every trial in the selected session
        for touch in touch_idx:
            touch_to_reward_speed[count_1] = speed[touch:touch + end_touch_diff[count_1]]
            count_1 += 1
            
        Level_2_touch_to_reward_speed[count] = touch_to_reward_speed
        
    return Level_2_touch_to_reward_speed
        

####################################   12   ##########################################

#used for Level 2 and calculate the speed from start to touch, in order to use this fx
#the trail_idx file needs to be created first or already existing 
   
def speed_start_to_touch(sessions_subset, sessions_speed):
    
    l = len(sessions_subset)
    Level_2_start_to_touch_speed = [[] for _ in range(l)]     
    
    for count in np.arange(l):
        session = sessions_subset[count]
        speed = sessions_speed[count]
        
        script_dir = os.path.join(hardrive_path + session) 
        trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        
        #selecting the column of touch and start, calculate the abs diff in order to calculate the 
        #how long it took to touch the ball from the start of the trial
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
       

####################################   13   ##########################################
#calculate speed around touch with a specific offset 

def speed_around_touch(sessions_subset,sessions_speed, offset = 360):
    
    x=len(sessions_subset)
    speed_around_touch_Level_2 = [[] for _ in range(x)] 
    
    for count in np.arange(x):
        session = sessions_subset[count]
        speed = sessions_speed[count]
                
        script_dir = os.path.join(hardrive_path + session) 
        csv_dir_path = os.path.join(script_dir + '/events/')
        trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        
        touch_idx = trial_idx[:,2]
        n = len(touch_idx)
        speed_touch = [[] for _ in range(n)]
        
        count_1 = 0
        for i in touch_idx:
            speed_touch[count_1]=speed[i-offset:i+offset]
            count_1 += 1
        speed_around_touch_Level_2[count] = speed_touch
    
    return speed_around_touch_Level_2
                
                
                        
                        
                        
#                        
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

          

####################################   14   ##########################################

# having the idx of the evet of interest and the video file name, it saves the frames 
# in a folder , it is used by the ball coordinates script whre it is calculated the idx of interest 

def frame_before_trials(target_dir,filename,cleaned_idx):
    
    video=cv2.VideoCapture(filename)
    success, image=video.read()
    success=True
    count = 0
    for i in cleaned_idx:
        video.set(cv2.CAP_PROP_POS_FRAMES, i-100)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image 



####################################   15   ##########################################

#takes the frames saved with fx 14 and find cebtroid of the ball by finding the colour yellow
#and the moments of the left ball which is the only yellow object detected 


def centroid_ball(frame_folder):

    listoffiles = os.listdir(frame_folder)
    ball_coordinates = np.zeros((len(listoffiles),2), dtype = float)
    count=0
    for frame in listoffiles:
        
        image = cv2.imread(os.path.join(frame_folder,frame))
       
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower = np.array([22, 93, 0])
        upper = np.array([45, 255, 255])
        
        #find yellow ball and the rest is black
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(image, image, mask = mask)    

        #cv2.imshow('frame',image)
        #cv2.imshow('mask',mask)
        #cv2.imshow('res',res)
        
        #gray scale 
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(gray,127,255,0)
        
        # calculate moments of binary image
        M = cv2.moments(thresh)
        
        if M['m00'] > 0:
            
                                
            ball_x = (M['m10']/M['m00'])
            ball_y = (M['m01']/M['m00'])
            ball_coordinates[count,0]= ball_x
            ball_coordinates[count,1]= ball_y
        else:
        
            ball_coordinates[count,0]= np.nan
            ball_coordinates[count,1]= np.nan
            
            
             
        # put text and highlight the center
        #cv2.circle(image, (np.int(ball_x),np.int(ball_y)), 5, (70, 70, 70), -1)
        #cv2.putText(image, "centroid", (np.int(ball_x) - 25, np.int(ball_y) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 2)
 
        # display the image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        count +=1
    return ball_coordinates


####################################   16   ##########################################

#save the ball idx based on where it appears in the set up

def ball_positions_based_on_quadrant_of_appearance(session):
      
    ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
    ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
    
    quadrant_1 = []
    quadrant_2 = []
    quadrant_3 = []
    quadrant_4 = []
       
    for n, row in enumerate(ball_coordinates):
        try:
            if row[0] <= 800 and row[1]>=600:
                quadrant_1.append(n)
            elif row[0] >= 800 and row[1]>=600:
                quadrant_2.append(n)
            elif row[0] <= 800 and row[1]<=600:
                quadrant_3.append(n)
            else:
                quadrant_4.append(n)
                         
        except Exception: 
            print (session + '/error')
        continue 
        
    return quadrant_1,quadrant_2,quadrant_3,quadrant_4
    

####################################   17   ##########################################
    
# create nose and tail base tracking snippets from touch to end using the trial idx file 
    
def create_tracking_snippets_touch_to_end(sessions_subset,start_snippet_idx = 0,end_snippet_idx = 1,mid_snippet_idx = 2):
    
    x = len(sessions_subset)
    
    x_nose_tracking_snippets_te = [[] for _ in range(x)] 
    y_nose_tracking_snippets_te = [[] for _ in range(x)] 
    x_tail_base_tracking_snippets_te = [[] for _ in range(x)]
    y_tail_base_trial_tracking_snippets_te = [[] for _ in range(x)]
           
    
    for count in np.arange(x):
        try:
        
            session = sessions_subset[count]                
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(script_dir + '/events/')
            trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
           
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)    
            
            
            x_nan_nose, y_nan_nose = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
            x_nan_tail_base, y_nan_tail_base = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
            
            trial_lenght_touch_to_end = abs(trial_idx[:,mid_snippet_idx] - trial_idx[:,end_snippet_idx])
            touch_idx = trial_idx[:,mid_snippet_idx]        
            
            l=len(touch_idx)
    
            x_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            x_tail_base_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)]
            y_tail_base_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)]
            

            
            for c, touch in enumerate(touch_idx):
                x_nose_trial_tracking_snippets_touch_to_end[c] = x_nan_nose[touch:touch + trial_lenght_touch_to_end[c]]
                y_nose_trial_tracking_snippets_touch_to_end[c] = y_nan_nose[touch:touch + trial_lenght_touch_to_end[c]]
                x_tail_base_trial_tracking_snippets_touch_to_end[c] = x_nan_tail_base[touch:touch + trial_lenght_touch_to_end[c]]
                y_tail_base_trial_tracking_snippets_touch_to_end[c] = y_nan_tail_base[touch:touch + trial_lenght_touch_to_end[c]]
        
            
            x_nose_tracking_snippets_te[count]= x_nose_trial_tracking_snippets_touch_to_end
            y_nose_tracking_snippets_te[count]= y_nose_trial_tracking_snippets_touch_to_end
            x_tail_base_tracking_snippets_te[count]= x_tail_base_trial_tracking_snippets_touch_to_end
            y_tail_base_trial_tracking_snippets_te[count]= y_tail_base_trial_tracking_snippets_touch_to_end
            
            print(count)        
        
        except Exception: 
            print (session + '/error')
            continue
     
    return x_nose_tracking_snippets_te, y_nose_tracking_snippets_te, x_tail_base_tracking_snippets_te, y_tail_base_trial_tracking_snippets_te


####################################   18   ##########################################
    
# create list of list containing the first x and y values for tail at nose at the moment of touch (taken from touch to end snippets), 
#if nan the fx looks for the first value which is not nan
        
def x_y_at_touch(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2):
    
    x = len(sessions_subset)
    first_values_x_nose=[[] for _ in range(x)]
    first_values_y_nose=[[] for _ in range(x)]
    first_values_x_tail_base=[[] for _ in range(x)]
    first_values_y_tail_base=[[] for _ in range(x)]
    
    
    
    for count in np.arange(x):
        try:
        
            session = sessions_subset[count]                
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(script_dir + '/events/')
            trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')


            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)    

            
            x_nan_nose, y_nan_nose = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
            x_nan_tail_base, y_nan_tail_base = DLC.DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
            
            trial_lenght_touch_to_end = abs(trial_idx[:,mid_snippet_idx] - trial_idx[:,end_snippet_idx])
            touch_idx = trial_idx[:,mid_snippet_idx]        
            
            l=len(touch_idx)
    
            x_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            x_tail_base_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)]
            y_tail_base_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)]
            

            
            for c, touch in enumerate(touch_idx):
                x_nose_trial_tracking_snippets_touch_to_end[c] = x_nan_nose[touch:touch + trial_lenght_touch_to_end[c]]
                y_nose_trial_tracking_snippets_touch_to_end[c] = y_nan_nose[touch:touch + trial_lenght_touch_to_end[c]]
                x_tail_base_trial_tracking_snippets_touch_to_end[c] = x_nan_tail_base[touch:touch + trial_lenght_touch_to_end[c]]
                y_tail_base_trial_tracking_snippets_touch_to_end[c] = y_nan_tail_base[touch:touch + trial_lenght_touch_to_end[c]]



            first_x_value_nose = np.zeros((len(x_nose_trial_tracking_snippets_touch_to_end),),dtype=float)
            first_y_value_nose = np.zeros((len(x_nose_trial_tracking_snippets_touch_to_end),),dtype=float)
            first_x_value_tail_base = np.zeros((len(x_nose_trial_tracking_snippets_touch_to_end),),dtype=float)
            first_y_value_tail_base = np.zeros((len(x_nose_trial_tracking_snippets_touch_to_end),),dtype=float)
            
            #it takes the first vale of the snippets corresponding to the touch moment and if nan the fx next will look for the first number and take that one instead
            for value in np.arange(len(x_nose_trial_tracking_snippets_touch_to_end)):
                
                first_x_value_nose[value] = next((x for x in x_nose_trial_tracking_snippets_touch_to_end[value] if not np.isnan(x)),0)
                first_y_value_nose[value] = next((x for x in y_nose_trial_tracking_snippets_touch_to_end[value] if not np.isnan(x)),0)
                first_x_value_tail_base[value] = next((x for x in x_tail_base_trial_tracking_snippets_touch_to_end[value] if not np.isnan(x)),0)
                first_y_value_tail_base[value] = next((x for x in y_tail_base_trial_tracking_snippets_touch_to_end[value] if not np.isnan(x)),0)
 
               
                first_values_x_nose[count] = first_x_value_nose
                first_values_y_nose[count] = first_y_value_nose
                first_values_x_tail_base[count] = first_x_value_tail_base
                first_values_y_tail_base[count] = first_y_value_tail_base
            
            print(count)        
        
        except Exception: 
            print (session + '/error')
            continue
     
    return first_values_x_nose,first_values_y_nose,first_values_x_tail_base,first_values_y_tail_base



####################################   19   ##########################################

#first_values_x_nose,first_values_y_nose,first_values_x_tail_base,first_values_y_tail_base are calculated with fx 18 
#it calculates the angle between tail and nose at the moment in which the rats touch the ball

def nose_butt_angle_touch(first_values_x_nose,first_values_y_nose,first_values_x_tail_base,first_values_y_tail_base):
    
    l= len(first_values_x_nose)
    
    sessions_degrees_at_touch = [[] for _ in range(l)]
    
    for count in np.arange(l):
        try:
                
            deltax = np.array(first_values_x_nose)[count] - np.array(first_values_x_tail_base)[count]
            deltay = np.array(first_values_y_nose)[count] - np.array(first_values_y_tail_base)[count]
        
            shape = len(deltax)
            degrees = np.zeros((shape,),dtype=float)
            
            for i in np.arange(shape):
                #degree correctection so that 0 degree is localted on the right (where the poke is)  and 90 degree is at the top
                degrees_temp = math.degrees(math.atan2(-deltay[i], deltax[i])) #/math.pi*180 or math.degrees to change from radians to degrees
                #add 360 is the results is negative
                if degrees_temp < 0:
                    degrees_final = 360 + degrees_temp
                    degrees[i]= degrees_final
                else:
                    degrees_final = degrees_temp
                    degrees[i]= degrees_final
                    
            sessions_degrees_at_touch[count] = degrees
        except Exception: 
            print (count)
            continue 
        
    return sessions_degrees_at_touch


#
#ball_coordinates='F:/Videogame_Assay/AK_33.2/2018_04_08-10_55/events/Ball_coordinates.csv'
##vector from poke to ball centroid
#def orientation_poke_ball(ball_coordinates):
#    ball_position  = np.genfromtxt(ball_coordinates, delimiter = ',', dtype = float)
#    #count=0
#    poke_ball_orientation = []
#    shape = len(ball_position)
#    poke = np.array([[1400,959]])    # poke position in a 664-350 frames
#    poke_coordinates = np.repeat(poke, [shape], axis=0)
#    diff = ball_position - poke_coordinates
#    for i in diff:
#        poke_ball_orientation.append(math.degrees(math.atan2(i[0], -i[1])))
#    #count += 1
#    return poke_ball_orientation




#def furthest_estremes(body_file,tail_file):
#    extremes_list_A = np.genfromtxt(body_file, dtype=float)
#    extremes_list_B = np.genfromtxt(tail_file, dtype=float)
#    count=0
#    shape=len(extremes_list_A)
#    nose=np.zeros((shape,2),dtype=float)
#    back=np.zeros((shape,2),dtype=float)
#    for i,e in list(zip(extremes_list_A,extremes_list_B)):
#        if any(np.isnan(i)) or any(np.isnan(e)):
#            nose[count,:]=np.nan
#            back[count,:]=np.nan
#        else:
#            dist1 = distance.euclidean(i[:2],e[:2]) + distance.euclidean(i[:2],e[2:])
#            dist2 = distance.euclidean(i[2:],e[:2]) + distance.euclidean(i[2:],e[2:])
#            if dist1> dist2:
#                furthest_extreme = i[:2]
#                back_extreme =i[2:]  
#            else:
#                furthest_extreme = i[2:]
#                back_extreme=i[:2]
#            nose[count,:]=furthest_extreme
#            back[count,:]=back_extreme
#        count += 1
#    return nose, back
#



        
            
            
            






#frame_before_trials('C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test',r'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Video.avi',cleaned_idx)

##vector from back to nose of the rat
#def orientation_body(nose,back):
#    #count=0
#    body_orientation=[]
#    body_diff=nose-back
#    for i in body_diff:
#        body_orientation.append(math.degrees(math.atan2(i[0], -i[1])))
#    #count += 1
#    return body_orientation
#
#
#
#
##dor product  
#def dot(vA, vB):
#    return vA[0]*vB[0]+vA[1]*vB[1]
#
#
##it doesnt matter if I use diff or nose_trajectory as input of the fx
#    
#def angle(nose_trajectories,shape,ball_coordinates):
#    trajectory_angle=[]
#    trajectory_angle_radians=[]
#    #nose_trajectory_int=nose_trajectory.astype(int)
#    for i in range(shape):
#        #select the xy position to get the angle: 99 correspond to the 
#        #start_closest while 0 the start of the trajectory and -1 the last 
#        #point of the trajectory(these parameters will need to change according to 
#        #the lenght of the trajectory and number of frames before and after the start_closest
#        #lineA=((ball[i]/2),(nose_trajectories[i,69,:]))
#        #lineB=((ball[i]/2),([332, 350]))
#        #129 corresponds to 1sec before the touch of the ball which is at frame 249 (tot lenght 499, 250before touch and 250
#        #after, 332 and 350 it is the position of the poke port in IR space which is half of the color one, in fact the ball coordinates
#        #need to be divioded by 2 so that all the parameteres will be referred to the Ir movie)
#        lineA=((nose_trajectories[i,129,:]),(ball_coordinates[i]))
#        lineB=(([332, 340]),(ball_coordinates[i]))
#        #lineA=((diff[i,129,:]), ball       
#        #lineB=(ball,([332, 350])
#        # Get nicer vector form
#        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
#        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
#        # Get dot prod
#        dot_prod = dot(vA, vB)
#        # Get magnitudes
#        magA = dot(vA, vA)**0.5
#        magB = dot(vB, vB)**0.5
#        # Get cosine value
#        # cos_ = dot_prod/magA/magB
#        # Get angle in radians and then convert to degrees
#        angle = math.acos(dot_prod/magB/magA)
#        trajectory_angle_radians.append(angle)
#        # Basically doing angle <- angle mod 360
#        ang_deg = math.degrees(angle)%360
#        if ang_deg-180>=0:
#            trajectory_angle.append(360 - ang_deg)  
#        else: 
#            trajectory_angle.append(ang_deg)
#    return trajectory_angle,trajectory_angle_radians
#


















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

#def timestamp_CSV_to_pandas(filename):
#    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
#    timestamp = timestamp_csv[0]
#    timestamp_Series= pd.to_datetime(timestamp)
#    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
#    return timestamp_Series
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










