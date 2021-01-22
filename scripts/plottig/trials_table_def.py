# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:25:27 2020

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
from matplotlib.colors import PowerNorm  
from matplotlib.colors import LogNorm 
from pylab import *
from matplotlib.ticker import LogFormatterExponent
import seaborn as sns 
from scipy.spatial import distance
from scipy import stats
from scipy.stats import *
import matplotlib.colors
import cv2

import importlib
importlib.reload(prs)
importlib.reload(behaviour)

hardrive_path = r'F:/' 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']






#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)



#parse ball tracking file 

def moving_light_and_joystick_parser(filename):
    
    split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
    timestamps = split_data[:,0]
    positions_strings = split_data[:,1]
    for index, s in enumerate(positions_strings):
        tmp = s.replace('(', '')
        tmp = tmp.replace(')', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp.replace(' ', '')
        positions_strings[index] = tmp
    tracking = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
    
    return timestamps, tracking





def reaction_time(sessions_subset,trial_file = 'Trial_idx.csv',tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600]):
    
    #sessions_subset=sessions_subset[:4]
    poke= poke_coordinates
    l = len(sessions_subset)
    
        
    rat_reaction_time = [[] for _ in range(l)]
    avg_reaction_time =[]
    std_reaction_time =[] 
    
    for count in np.arange(l):
        
        session = sessions_subset[count]    

        script_dir = os.path.join(hardrive_path + session) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        crop_tracking_path = os.path.join(script_dir + tracking_file)
        #parse ball tracking file         

        crop = np.genfromtxt(crop_tracking_path, delimiter= tracking_delimiter)
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        success,misses = behaviour.trial_outcome_index(script_dir)
        
        onset = trial_idx[:,0]
        ttr = abs(onset-trial_idx[:,1])
        rat_position_at_start = crop[onset]


        rat_position_at_start_success= rat_position_at_start[success]
        ttr_success = ttr[success]

                      
        bad_idx = []
        
        for p in range(len(rat_position_at_start_success)):
            if  rat_position_at_start_success[p,0]>1250.0 and 450.0 <rat_position_at_start_success[p,1] < 750.0:
                bad_idx.append(p)
            

        #print(len(bad_idx))
        #print(len(trial_idx))
        good_idx =[ele for ele in range(len(rat_position_at_start_success)) if ele not in bad_idx]
                #poke_coord=[poke]*len(onset)
        #print(len(good_idx))
        #print(len(rat_position_at_start_success))

        rat_position_at_start_good= rat_position_at_start_success[good_idx]
        ttr_good = ttr_success[good_idx]

       
        session_rat_poke_dist = [] 
        
        for e in range(len(rat_position_at_start_good)):        
        
            dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_start_good[e]-poke)**2)))
            #euclidian = distance.euclidean(rat_position_at_start[e], poke_coord[e])  
        
            session_rat_poke_dist.append(dist_rat_poke)
       
        
        session_reaction_time = session_rat_poke_dist/ttr_good
        #rewarded_trial_rt = session_reaction_time[good_idx]
        avg_rt = np.mean(session_reaction_time)
        std_rt = np.std(session_reaction_time)
        
        avg_reaction_time.append(avg_rt)
        std_reaction_time.append(std_rt)
        
        rat_reaction_time[count]=session_reaction_time 
        print(count)
    return rat_reaction_time, avg_reaction_time,std_reaction_time



#####def to make the trail table


#calculates 4 euclidian distances using the crop.csv :
    
#    rat position at start - ball position
#    rat position at touch - poke position
#    rat position 120 prior to touch - rat position at touch
#    rat position 120- after touch - rat position at touch

   
def distance_events(sessions_subset,frames=120, trial_file = 'Trial_idx.csv',ball_file = 'BallPosition.csv', 
                    tracking_file = '/events/Tracking.csv',tracking_delimiter=None, poke_coordinates = [1,0] ): #',' for '/crop.csv', None for shaders
    
    #poke = [1400,600]
    poke = poke_coordinates
    l = len(sessions_subset)
    
    rat_ball = [[] for _ in range(l)]
    rat_poke = [[] for _ in range(l)]
    before_touch = [[] for _ in range(l)]
    after_touch = [[] for _ in range(l)]
    
   
    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            crop_tracking_path = os.path.join(script_dir + tracking_file)
            #parse ball tracking file         

            crop = np.genfromtxt(crop_tracking_path, delimiter= tracking_delimiter)
            trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)                
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + ball_file)   
            if ball_file == 'BallPosition.csv':
           
                timestamps, ball_coordinates = moving_light_and_joystick_parser(ball_coordinates_path)
                
            else:
            
                ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
                    
 
            
            start = trial_idx[:,0]
            rat_position_at_start = crop[start]
            touch = trial_idx[:,2]
            rat_position_at_touch = crop[touch]
            rat_before_ball = crop[touch - frames]
            rat_after_ball = crop[touch + frames]
            
            
            
            session_rat_ball_dist = []
            session_rat_poke_dist = []
            session_rat_before_touch=[]
            session_rat_after_touch=[]
            
            for e in range(len(start)):
                
                #dist = distance.euclidean(rat_position_at_start[e], ball_coordinates[e])
                
                dist_rat_ball = (np.sqrt(np.nansum((rat_position_at_start[e]-ball_coordinates[e])**2)))            
                dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_touch[e]-poke)**2)))
                dist_before_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_before_ball[e])**2)))
                dist_after_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_after_ball[e])**2)))
                
                
                session_rat_ball_dist.append(dist_rat_ball)
                session_rat_poke_dist.append(dist_rat_poke)
                session_rat_before_touch.append(dist_before_touch)
                session_rat_after_touch.append(dist_after_touch)
                
                
                #.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx,touch_idx,ball_on_idx)).T, delimiter=',', fmt='%s')

                
            rat_ball[count]=session_rat_ball_dist
            rat_poke[count]=session_rat_poke_dist
            before_touch[count]=session_rat_before_touch
            after_touch[count]=session_rat_after_touch
            
           
            
        except Exception: 
            print('error'+ session)
        continue
    
    return rat_ball, rat_poke, before_touch, after_touch




################################################
    



def rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv',tracking_delimiter = ','):
    
    l = len(sessions_subset)
    event_rat_coordinates = [[] for _ in range(l)]
    rat_coordinates_before = [[] for _ in range(l)]
    rat_coordinates_after = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        crop_tracking_path = os.path.join(script_dir + tracking_file)
        crop = np.genfromtxt(crop_tracking_path,delimiter= tracking_delimiter)
                       
        rat_event = trial_idx[:,event]
        rat_pos = crop[rat_event]
            
        event_rat_coordinates[count]= rat_pos
 
        rat_before = crop[rat_event-offset]
        rat_after = crop[rat_event+offset]
        
        
        rat_coordinates_before[count] = rat_before
        rat_coordinates_after[count] = rat_after
    
    return event_rat_coordinates, rat_coordinates_before,rat_coordinates_after




###############################################
    
# start to touch idx diff and touch to end idx diff

def time_to_events(sessions_subset, trial_file = 'Trial_idx_cleaned.csv'):
    
    l = len(sessions_subset)
    st_time = [[] for _ in range(l)]
    te_time = [[] for _ in range(l)]
    se_time = [[] for _ in range(l)]

    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
          
            trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
    
            #selecting the column of touch and start, calculate the abs diff in order to calculate the 
            #how long it took to touch the ball from the start of the trial
            start_touch_diff = abs(trial_idx[:,0] - trial_idx[:,2])
            touch_end_diff = abs(trial_idx[:,1] - trial_idx[:,2])
            start_to_end_diff = abs(trial_idx[:,0] - trial_idx[:,1])
            
            st_time[count] = start_touch_diff
            te_time[count] = touch_end_diff
            se_time[count] = start_to_end_diff
            
            
            #csv_dir_path = os.path.join(script_dir + '/events/')
            
            #csv_name = 'Start_touch_idx_diff.csv'
            #np.savetxt(csv_dir_path + csv_name,start_touch_diff, fmt='%s')
            #csv_name = 'Touch_reward_idx_diff.csv'
            #np.savetxt(csv_dir_path + csv_name,touch_end_diff, fmt='%s')
            
            print(len(start_touch_diff))
            print(len(touch_end_diff))
            print(len(te_time))
            print('saving DONE')


        except Exception: 
            print('error'+ session)
        continue
                            
    return st_time, te_time, se_time 
################################
    

def time_to_events_moving_and_joystick(sessions_subset, trial_file = 'Trial_idx_cleaned.csv'):
    
    l = len(sessions_subset)
    st_time = [[] for _ in range(l)]
    te_time = [[] for _ in range(l)]
    se_time = [[] for _ in range(l)]
    tt_time =  [[] for _ in range(l)]
    bot_time =  [[] for _ in range(l)]
    
    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            
            trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
    
            #selecting the column of touch and start, calculate the abs diff in order to calculate the 
            #how long it took to touch the ball from the start of the trial
            start_touch_diff = abs(trial_idx[:,0] - trial_idx[:,2])
            touch_end_diff = abs(trial_idx[:,1] - trial_idx[:,2])
            start_to_end_diff = abs(trial_idx[:,0] - trial_idx[:,1])
            trigger_to_touch =  abs(trial_idx[:,2] - trial_idx[:,4])
            ball_ob_to_trigger = abs(trial_idx[:,3] - trial_idx[:,4])
            
            st_time[count] = start_touch_diff
            te_time[count] = touch_end_diff
            se_time[count] = start_to_end_diff
            tt_time[count] = trigger_to_touch
            bot_time[count] = ball_ob_to_trigger
            
            
            
            #csv_dir_path = os.path.join(script_dir + '/events/')
            
            #csv_name = 'Start_touch_idx_diff.csv'
            #np.savetxt(csv_dir_path + csv_name,start_touch_diff, fmt='%s')
            #csv_name = 'Touch_reward_idx_diff.csv'
            #np.savetxt(csv_dir_path + csv_name,touch_end_diff, fmt='%s')
            
            print(len(start_touch_diff))
            print(len(touch_end_diff))
            print(len(te_time))
            print('saving DONE')


        except Exception: 
            print('error'+ session)
        continue
                            
    return st_time, te_time, se_time, tt_time,bot_time




########################
    

def trial_count(event):
    
    rat_tot_trial_count = []

    for i in range(len(event)):
        rat_tot_trial_count.append(len(event[i]))


    return sum(rat_tot_trial_count)

############################


def trial_count_each_session(event):
    
    rat_tot_trial_count = []

    for i in range(len(event)):
        rat_tot_trial_count.append(i)


    return rat_tot_trial_count

#################
    
def trial_counter(sessions_subset, trial_file= 'Trial_idx_cleaned.csv'):
      
    l = len(sessions_subset)
    session_trials = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        trial_count = trial_count_each_session(trial_idx)
        
        session_trials[count] = trial_count
        
    return session_trials



#sesssion_trials = trial_counter(sessions_subset)


#####tot outcome
 
    

def outcome_over_sessions(sessions_subset, trial_file = '/TrialEnd.csv'):
    
    l = len(sessions_subset)
    session_outcome = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        trial_end_path = os.path.join(script_dir+ '/events/' + trial_file)
        outcome = np.genfromtxt(trial_end_path, usecols = [1], dtype = str)
        
        session_outcome[count] = outcome
        
    return session_outcome

    
    

#######moving light outcome
    
def cleaned_outcome_sessions(sessions_subset, trial_file = 'Trial_outcome_cleaned.csv'):


    l = len(sessions_subset)
    cleaned_outcome = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        outcome_cleaned_path = os.path.join(script_dir+ '/events/' + trial_file)
        outcome = np.genfromtxt(outcome_cleaned_path, dtype = str)
        
        cleaned_outcome[count] = outcome
        
    return cleaned_outcome











###################
    



def ball_positions_finder(sessions_subset, ball_file ='Ball_positions_cleaned.csv' ):
      
    l = len(sessions_subset)
    ball_rat = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        
        ball_coordinates_path = os.path.join(script_dir+ '/events/' + ball_file)    
        ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
        
        ball_rat[count] = ball_coordinates
        
    return ball_rat

##########################shader ball
    

def ball_positions_shaders(sessions_subset, ball_file ='BallPosition.csv', trial_file='Trial_idx.csv' ):
      
    l = len(sessions_subset)
    ball_rat = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        
        ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + ball_file)    
        timestamps, ball_coordinates = moving_light_and_joystick_parser(ball_coordinates_path)
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        
        
        if len(ball_coordinates)>len(trial_idx):
            ball_coordinates = ball_coordinates[:-1]
        

        
        ball_rat[count] = ball_coordinates
        
    return ball_rat








##########################
    
def ball_positions_finder_moving_and_joystick(sessions_subset):
      
    l = len(sessions_subset)
    ball_rat = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        
        ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_positions_cleaned.csv')    
        ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
        
        ball_rat[count] = ball_coordinates
        
    return ball_rat


############################
    

#save the ball idx based on where it appears in the set up

def ball_positions_based_on_quadrant_of_appearance(sessions_subset):
    
    l = len(sessions_subset)
    quadrants = [[] for _ in range(l)]
    
    for count in np.arange(l):
        
        try:
            
            session = sessions_subset[count]       
                     
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
            ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
            
            q = []
              
            for n, row in enumerate(ball_coordinates):
               
                if row[0] <= 800 and row[1]>=600:
                    q.append(1)
                elif row[0] >= 800 and row[1]>=600:
                    q.append(2)
                elif row[0] <= 800 and row[1]<=600:
                    q.append(3)
                    
                else:
                    q.append(4)
                         
            quadrants[count]= q
                    
        except Exception: 
            print (session + '/error')
        continue 
        
    return quadrants 





###################################
    


def rat_position_around_event(sessions_subset, event=2, offset=120):
    
    l = len(sessions_subset)
    nose_before = [[] for _ in range(l)]
    nose_after = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        corrected_coordinate_path = os.path.join(script_dir + '/DLC_corrected_coordinates')
        nose_path = os.path.join(corrected_coordinate_path + '/nose_corrected_coordinates.csv')
        nose_dlc = np.genfromtxt(nose_path, delimiter = ',', dtype = float)
    
        
       
        rat_event = trial_idx[:,event]
        rat_pos_before = nose_dlc[rat_event-offset]
        rat_pos_after = nose_dlc[rat_event+offset]
            
        nose_before[count] = rat_pos_before
        nose_after[count]= rat_pos_after
    
    return nose_before, nose_after



#####################################


def rat_position_around_event_snippets(sessions_subset, event=2, offset=240, folder = 'Trial_idx_cleaned.csv'):
    
    l = len(sessions_subset)
    x_crop_snippet = []
    y_crop_snippet = []
    x_shader_snippet = []
    y_shader_snippet = []
    
    
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + folder)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        shader_tracking_path = os.path.join(script_dir + '/events/' +'Tracking.csv')
        shader_tracking = np.genfromtxt(shader_tracking_path)
        
        crop_tracking_path = os.path.join(script_dir + '/crop.csv')
        crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
    
        
       
        rat_event = trial_idx[:,event]
        offset_before = rat_event-offset
        offset_after = rat_event+offset
        
        x_crop_snippet_session = np.zeros((len(rat_event),offset*2))
        y_crop_snippet_session = np.zeros((len(rat_event),offset*2))
        
        x_shader_snippet_session = np.zeros((len(rat_event),offset*2))          
        y_shader_snippet_session = np.zeros((len(rat_event),offset*2))
        
        for i in range(len(rat_event)):
            
            rat_around_event_shaders = shader_tracking[offset_before[i]:offset_after[i]]
            rat_around_event_crop = crop[offset_before[i]:offset_after[i]]
            
            x_crop_snippet_session[i] = rat_around_event_crop[:,0]
            y_crop_snippet_session[i] = rat_around_event_crop[:,1]
            x_shader_snippet_session[i] =  rat_around_event_shaders[:,0]
            y_shader_snippet_session[i]=  rat_around_event_shaders[:,1]
        
        x_crop_snippet.extend(x_crop_snippet_session)
        y_crop_snippet.extend(y_crop_snippet_session)
        x_shader_snippet.extend(x_shader_snippet_session)
        y_shader_snippet.extend(y_shader_snippet_session)
            
        print(count)
                
        
    return x_crop_snippet, y_crop_snippet,x_shader_snippet,y_shader_snippet


def rat_position_around_event_snippets_ephys(sessions_subset, event=4, offset=240, folder = 'Trial_idx_cleaned.csv'): # 4 for miving light and joytick (when ball move away)
    
    l = len(sessions_subset)

    x_shader_snippet = []
    y_shader_snippet = []
    
    
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + folder)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        shader_tracking_path = os.path.join(script_dir + '/events/' +'Tracking.csv')
        shader_tracking = np.genfromtxt(shader_tracking_path)
        

        
       
        rat_event = trial_idx[:,event]
        offset_before = rat_event-offset
        offset_after = rat_event+offset
        

        
        x_shader_snippet_session = np.zeros((len(rat_event),offset*2))          
        y_shader_snippet_session = np.zeros((len(rat_event),offset*2))
        
        for i in range(len(rat_event)):
            
            rat_around_event_shaders = shader_tracking[offset_before[i]:offset_after[i]]

            

            x_shader_snippet_session[i] =  rat_around_event_shaders[:,0]
            y_shader_snippet_session[i]=  rat_around_event_shaders[:,1]
        

        x_shader_snippet.extend(x_shader_snippet_session)
        y_shader_snippet.extend(y_shader_snippet_session)
            
        print(count)
                
        
    return x_shader_snippet,y_shader_snippet






#
#
#
## having the idx of the evet of interest and the video file name, it saves the frames 
## in a folder , it is used by the ball coordinates script whre it is calculated the idx of interest 
#
#hardrive_path = r'F:/' 
#rat_summary_table_path = 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv'
#
#Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
#
#sessions_subset = Level_2_pre
#
#
#target_dir = 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test_frames_after_120'
#
#
#session = sessions_subset[0]
#        
#    
#script_dir = os.path.join(hardrive_path + session) 
#
#  
#video_path = os.path.join(hardrive_path, session + '/Video.avi')
#
#
#frames = frame_before_trials(target_dir,video_path,event= 2, offset=240)
#
#
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
#
#
#
#ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
#ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
#            
#
#len(ball_coordinates)
#
#
#
#
#def rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120):
#    
#    l = len(sessions_subset)
#    event_rat_coordinates = [[] for _ in range(l)]
#    rat_coordinates_before = [[] for _ in range(l)]
#    rat_coordinates_after = [[] for _ in range(l)]
#  
#    for count in np.arange(l):
#    
#        
#        session = sessions_subset[count]
#        
#    
#        script_dir = os.path.join(hardrive_path + session) 
#
#        trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
#        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
#        crop_tracking_path = os.path.join(script_dir + '/crop.csv')
#        crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
#                       
#        rat_event = trial_idx[:,event]
#        rat_pos = crop[rat_event]
#            
#        event_rat_coordinates[count]= rat_pos
# 
#        rat_before = crop[rat_event-offset]
#        rat_after = crop[rat_event+offset]
#        
#        
#        rat_coordinates_before[count] = rat_before
#        rat_coordinates_after[count] = rat_after
#    
#    return event_rat_coordinates, rat_coordinates_before,rat_coordinates_after
#
#
#
#
#
#
#
#
#
image = 'F:/Videogame_Assay/AK_49.2/2019_09_18-12_27/Ball_frames/frame02.jpg'
im= cv2.imread(image)

#count = 15
#fig = plt.figure(figsize=(15,9))
#
#plt.subplot(1,2,1)
plt.imshow(im)
#plt.plot(ball_coordinates[count,0],ball_coordinates[count,1],'d','y')
#
#
#plt.plot(np.array(rat_coordinates_before)[0,count][0],np.array(rat_coordinates_before)[0,count][1],'o','r')
#
#
#
#plt.plot([np.array(rat_coordinates_before)[0,count][0],ball_coordinates[count,0]],[np.array(rat_coordinates_before)[0,count][1],ball_coordinates[count,1]],'r','-')
#
#plt.plot([np.array(rat_coordinates_after)[0,count][0],ball_coordinates[count,0]],[np.array(rat_coordinates_after)[0,count][1],ball_coordinates[count,1]],'g','-')
#
#
#
#
#
#image_after = 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test_frames_after_120/frame15.jpg'
#im_after= cv2.imread(image_after)
#
#
#plt.subplot(1,2,2)
#
#
#
#plt.imshow(im_after)
#
#
#plt.plot(ball_coordinates[count,0],ball_coordinates[count,1],'d','y')
#
#
#plt.plot(np.array(rat_coordinates_after)[0,count][0],np.array(rat_coordinates_after)[0,count][1],'o','r')
#
#
#
#plt.plot([np.array(rat_coordinates_before)[0,count][0],ball_coordinates[count,0]],[np.array(rat_coordinates_before)[0,count][1],ball_coordinates[count,1]],'r')
#
#plt.plot([np.array(rat_coordinates_after)[0,count][0],ball_coordinates[count,0]],[np.array(rat_coordinates_after)[0,count][1],ball_coordinates[count,1]],'g')
#
#
#
#
#plt.plot(np.array(event_rat_coordinates)[0,count][0],np.array(event_rat_coordinates)[0,count][1],'s','w')
#
#
