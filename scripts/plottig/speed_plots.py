# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:59:46 2019

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


#Level_0 = prs.Level_0_paths(rat_summary_table_path)
#Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
#Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
#Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
# =============================================================================
# Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
# 
# 
# #saving a Trial_idx_csv containing the idx of start-end-touch 0-1-2
# sessions_subset = Level_2_pre
# behaviour.start_end_touch_ball_idx(sessions_subset)
# =============================================================================

#calcute speedtracking diff

s = len(rat_summary_table_path)

Level_2_start_to_touch_speed_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         session_speed = behaviour.session_speed_crop_tracking(sessions_subset)
         Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, session_speed)
         
         Level_2_start_to_touch_speed_all_rats[r] = Level_2_start_to_touch_speed
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
   
#####################################################################


s = len(rat_summary_table_path)

st_all_rats = [[] for _ in range(s)]
te_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         st_time, te_time = time_to_events(sessions_subset)
         
         st_all_rats[r] = st_time
         te_all_rats[r] = te_time
         
    
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  





# =============================================================================
#  
 
# rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']
# rat = rat_summary_table_path[0]
# Level_2_pre = prs.Level_2_pre_paths(rat)
# sessions_subset = Level_2_pre[0]
 
# =============================================================================
#


 
#calculating distance rat at start- ball position + dist rat at touch and poke   

s = len(rat_summary_table_path)

rat_ball_all_rats = [[] for _ in range(s)]
rat_poke_all_rats = [[] for _ in range(s)]
before_touch_all_rats= [[] for _ in range(s)]
after_touch_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         rat_ball, rat_poke, before_touch, after_touch = distance_events(sessions_subset)
         
         rat_ball_all_rats[r] = rat_ball
         rat_poke_all_rats[r] = rat_poke
         before_touch_all_rats[r] = before_touch
         after_touch_all_rats[r] = after_touch
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    



##########################################
        
    
    
    

f,ax = plt.subplots(figsize=(15,5))

for rat in arange(len(rat_summary_table_path)):
    
      
    before  = before_touch_all_rats[rat]
    after = after_touch_all_rats [rat]


    flattened_before = [val for sublist in before for val in sublist]
    flattened_after = [val for sublist in after for val in sublist]
    
    delta = np.array(flattened_after) - np.array(flattened_before)
    print(rat)
    print(len(flattened_before))
    print(len(flattened_after))
    
   # plt.figure()
    plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  

    #plt.bar(range(len(delta)),delta, width= 0.05)
    #plt.boxplot(flattened_before,flattened_after)
    
    ax.hlines(0,0,800,linewidth=0.5)
    





for r in 
test = before_touch_all_rats[0]
test2 = after_touch_all_rats[0]

delta_touch =[]

for i in range(len(test)):
    
    before  = test[i]
    after = test2[i]
    delta = np.array(before)-np.array(after)
    delta_touch.append(delta)

plt.plot(delta_touch[1],'.')


###########################################
rat_poke_all_rats
rat_ball_all_rats
st_all_rats 

#test plots



plt.figure()
for rat in arange(len(rat_summary_table_path)):
    
      
    rat_ball_selection  = rat_ball_all_rats[rat]
    st_selection = st_all_rats [rat]


    flattened_rat_ball = [val for sublist in rat_ball_selection for val in sublist]
    flattened_st = [val for sublist in st_selection for val in sublist]
    print(rat)
    print(len(flattened_rat_ball))
    print(len(flattened_st))
    
    plt.plot(flattened_rat_ball,flattened_st, '.', alpha=.7, markersize=.5, color= 'k')

################################################################

rat_poke_all_rats
te_all_rats 

plt.figure()
for rat in arange(len(rat_summary_table_path)):
    
      
    rat_poke_selection  = rat_poke_all_rats[rat]
    te_selection = te_all_rats [rat]


    flattened_rat_poke = [val for sublist in rat_poke_selection for val in sublist]
    flattened_te = [val for sublist in te_selection for val in sublist]
    print(rat)
    print(len(flattened_rat_poke))
    print(len(flattened_te))
    
    plt.plot(flattened_rat_poke, flattened_te, '.', alpha=.7, markersize=.5, color= 'k')

#############################################################






plt.figure()
for rat in arange(len(rat_summary_table_path)):
    
      
    rat_ball_selection  = rat_ball_all_rats[rat]
    st_speed = Level_2_start_to_touch_speed_all_rats[rat]


    flattened_rat_ball = [val for sublist in rat_ball_selection for val in sublist]
    flattened_st_speed = [val for sublist in st_speed for val in sublist]
    print(rat)
    print(len(flattened_rat_ball))
    print(len(flattened_st_speed))
    
    plt.plot(flattened_rat_ball,flattened_st_speed, '.', alpha=.7, markersize=.5, color= 'k')
    
    
    







   
def distance_events(sessions_subset,frames=120):
    
    poke = [1400,600]
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
            crop_tracking_path = os.path.join(script_dir + '/crop.csv')
            crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
            trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
                
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
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
                dist_before_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_before_ball)**2)))
                dist_after_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_after_ball)**2)))
                
                
                session_rat_ball_dist.append(dist_rat_ball)
                session_rat_poke_dist.append(dist_rat_poke)
                session_rat_before_touch.append(dist_before_touch)
                session_rat_after_touch.append(dist_after_touch)
                
            rat_ball[count]=session_rat_ball_dist
            rat_poke[count]=session_rat_poke_dist
            before_touch[count]=session_rat_before_touch
            after_touch[count]=session_rat_after_touch
            
            
        except Exception: 
            print('error'+ session)
        continue
    
    return rat_ball, rat_poke, before_touch, after_touch








def time_to_events(sessions_subset):
    
    l = len(sessions_subset)
    st_time = [[] for _ in range(l)]
    te_time = [[] for _ in range(l)]
    
    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            crop_tracking_path = os.path.join(script_dir + '/crop.csv')
            crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
            trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
    
            #selecting the column of touch and start, calculate the abs diff in order to calculate the 
            #how long it took to touch the ball from the start of the trial
            start_touch_diff = abs(trial_idx[:,0] - trial_idx[:,2])
            touch_end_diff = abs(trial_idx[:,0] - trial_idx[:,1])
            st_time[count] = start_touch_diff
            te_time[count] = touch_end_diff
              
        except Exception: 
            print('error'+ session)
        continue
                            
    return st_time, te_time
























#calculate speed for all the session (adapted to use nose corrected coordinates instead of the crop that we used originally)
sessions_speed = behaviour.session_speed(sessions_subset)

#calculate the speed from start of the trial to touch of the ball using the trial idx csv file 
Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, sessions_speed)

#calculate the speed from ball touch to reward using the csv file saved wit the idx 
Level_2_touch_to_reward_speed = behaviour.speed_touch_to_reward(sessions_subset, sessions_speed)

#from the speed of the session extract chunck of 6 seconds around the touch idx (360 frames before and 360 frames after)
speed_touch_Level_2 = behaviour.speed_around_touch(sessions_subset,sessions_speed)


#############################################################################################################
#plotting the speed around touch by doing the mean of each session and plotting the sessions 
#check because in some cases the speed touch is empty array thats why i used try
#remove the diff greater than 20 before calculating the mean

means = []
    
for row in speed_touch_Level_2:
    try:
        session_array = np.array(row)
        session_array[session_array>=20] = np.NaN
        mean_session = np.nanmean(session_array,axis=0)
        means.append(mean_session)        
    except Exception: 
        continue     

             

figure_name0 = figure_name = 'RAT_' + RAT_ID + '_speed_around_touch2.png'
plot_main_title = 'RAT ' + RAT_ID + 'speed around touch +/- 3 sec'
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


cmap = sns.color_palette("hls", len(means))
for count,i in enumerate(means):
    plt.plot(i, color=cmap[count], label='session %d'%count)

plt.axvline(360,color='k')
plt.ylim(0,5)
plt.legend()
f0.tight_layout()
f0.subplots_adjust(top = 0.87)

    
script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + RAT_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f0.savefig(results_dir + figure_name, transparent=True)
#########################################################################################################



l = len(sessions_speed)

mean_touch_to_reward = []
std_touch_to_reward = [] 
 
for count in np.arange(l):
    session_touch_to_reward = Level_2_touch_to_reward_speed[count]
    concat_speed = [item for sublist in session_touch_to_reward for item in sublist]
    touch_reward_mean = np.nanmean(concat_speed)
    mean_touch_to_reward.append(touch_reward_mean)
    touch_reward_std = np.nanstd(concat_speed)
    std_touch_to_reward.append(touch_reward_std)
    

mean_start_to_touch = []
std_start_to_touch = []

    
for count in np.arange(l):
    session_start_to_touch = Level_2_start_to_touch_speed[count]
    concat_start_to_touch_speed = [item for sublist in session_start_to_touch for item in sublist]
    start_touch_mean = np.nanmean(concat_start_to_touch_speed)
    mean_start_to_touch.append(start_touch_mean)
    start_touch_std = np.nanstd(concat_start_to_touch_speed)
    std_start_to_touch.append(start_touch_std)
    
    
test = 

    
stack_mean = np.vstack((mean_start_to_touch,mean_touch_to_reward)).T  
stack_std = np.vstack((std_start_to_touch,std_touch_to_reward)).T 


plt.scatter(mean_start_to_touch,range(l),marker = 'o',color= 'steelblue',alpha = .8)
plt.scatter(mean_touch_to_reward,range(l),marker = 'o',color= 'g',alpha = .8)

    



####################################################################


ball = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/events/Ball_coordinates.csv'
ball_pos = np.genfromtxt(ball, delimiter = ',', dtype = float)
centroid_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/crop.csv'
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)




figure_name = figure_name = 'RAT_' + RAT_ID + '_Speed_Touch_to_reward_Level2.pdf'
plot_main_title = 'RAT ' + RAT_ID + ' Speed_Touch_to_reward_Level2' + 'Level_2'

 
for i, session in enumerate(sessions_subset): 
    try:
        behaviour.full_trial_idx(sessions_subset)
        
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f0.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)

f0.tight_layout()
f0.subplots_adjust(top = 0.87)

