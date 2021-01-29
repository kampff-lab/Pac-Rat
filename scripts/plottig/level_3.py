# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:06:49 2020

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


import importlib
importlib.reload(prs)
importlib.reload(behaviour)

hardrive_path = r'F:/' 

rat_summary_table_path = ['F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']






colours = ['#32CD32', '#ADFF2F','#7FFFD4','#6495ED']
RAT_ID = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'


results_dir =os.path.join(main_folder + figure_folder)


s = len(rat_summary_table_path)



trial_per_min_all_rats = [[] for _ in range(s)]
missed = [[] for _ in range(s)]
rewarded = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    
     Level_3_pre = prs.Level_3_moving_light_paths(rat)
     sessions_subset = Level_3_pre
     
     session_reward = []
     session_missed =[]
     trial_per_min=[]
     
     for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
        csv_dir_path = os.path.join(session_path + '/events/')

        trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx_cleaned.csv')
        trial_outcome_path= os.path.join(csv_dir_path + 'Trial_outcome_cleaned.csv') 
        counter_csv = os.path.join(session_path + '/Video.csv')
        counter = np.genfromtxt(counter_csv, usecols = 1)

        #subctract thefirst counter value from the last one to obtain the total lenght of the session
        #in frames and then convert it to minutes taking into account the video is recorded at 120fps
        tot_frames = counter[-1] - counter[0]
        #minutes conversion
        session_length_minutes = tot_frames/120/60

        
        trial_idx =np.genfromtxt(trial_idx_path, delimiter=',')
        outcome =np.genfromtxt(trial_outcome_path,dtype=str)
        #print(len(trial_idx))
        #print(len(outcome))
        

        food=[]
        
        for o in np.arange(len(outcome)):
            if outcome[o]=='Food':
                
                food.append(o)
                
        trial_per_min.append(len(food)/session_length_minutes)
        session_reward.append(len(food))
        session_missed.append(len(trial_idx)-len(food))
        #print(len(food))
        #print(len(trial_idx)-len(food))

     rewarded[r]=session_reward
     missed[r]=session_missed
     trial_per_min_all_rats[r]=trial_per_min
     print(r)

missed_array = np.array([[1,0,1,0,1],
                  [1,0,0,1,0],
                  [0,1,2,nan,nan],
                  [0,0,0,0,nan]])


rewarded_array =np.array([[65,67,97,106,76],
                          [38,90,105,130,87],
                          [24,54,48,nan,nan],
                          [52,104,73,78,nan]])


trial_per_min_array = np.array([[1.137426936116951, 1.4158710218984527,1.6977128035840603, 1.79913013755142, 1.456244793898249],
                                 [1.1255692640603594,1.4294317347080585, 1.7132950334840398, 2.0140770212317283, 1.5081874062499248],
                                 [0.47772945804799966, 1.3398072304103161, 0.893612069968584,nan,nan],
                                 [1.050142627544366, 2.01639940218928, 1.5108441007804303, 1.5167760945716375,nan]])
   
    
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for count, row in enumerate(rewarded_array):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3)
   
    plt.title('Level 3 rewarded trial', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    plt.ylim(0,140,20)
   
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.ylim(0,3)
    #plt.legend()
    f.tight_layout()


#f.savefig(results_dir + figure_name, transparent=True)       
    
mean_trial = np.nanmean(rewarded_array, axis=0)

sem = stats.sem(rewarded_array, nan_policy='omit', axis=0)


plt.plot(mean_trial,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2) 


figure_name='level 3 rewarded trial count mean and sem.pdf'

#SAVING
f.savefig(results_dir + figure_name, transparent=True)    

#t test 


t_test = stats.ttest_rel(rewarded_array[:,0],rewarded_array[:,-1],nan_policy='omit')


target = open(main_folder +"stats_level 3 rewarded trial count mean and sem.txt", 'w')
target.writelines(str(mean_trial) +str(sem) + str(t_test)+' LEVEL 3: day 1 Vs day 5, PLOT: mean trial rewarded   +- SEM, Level 3.py')

target.close()



######################trial/ min level 3




    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for count, row in enumerate(trial_per_min_array):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3)
   
    plt.title('Level 3 trial/min', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 3 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    #plt.ylim(0,140,20)
   
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylim(0,3,.2)
    #plt.legend()
    f.tight_layout()


#f.savefig(results_dir + figure_name, transparent=True)       
    
mean_trial = np.nanmean(trial_per_min_array, axis=0)

sem = stats.sem(trial_per_min_array, nan_policy='omit', axis=0)


plt.plot(mean_trial,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2) 


figure_name='level 3 rewarded trial per min mean and sem.pdf'

#SAVING
f.savefig(results_dir + figure_name, transparent=True)    

#t test 


t_test = stats.ttest_rel(trial_per_min_array[:,0],trial_per_min_array[:,-1],nan_policy='omit')


target = open(main_folder +"stats_level 3 rewarded trial count mean and sem.txt", 'w')
target.writelines(str(mean_trial) +str(sem) + str(t_test)+' LEVEL 3: day 1 Vs day 5, PLOT: mean trial rewarded /min  +- SEM, Level 3.py')

target.close()























###################







trial_table_moving_light_path = 'F:/Videogame_Assay/Trial_table_final_level_3_moving_light.csv'
trial_table_joystick_path = 'F:/Videogame_Assay/Trial_table_final_level_3_joystick.csv'


#selectin first moving light day and tracking 


    
s = len(rat_summary_table_path)


ball_pos_all_rats = [[] for _ in range(s)]
snippet_before = [[] for _ in range(s)]
snippet_after = [[] for _ in range(s)]
snippet_centred_before = [[] for _ in range(s)]
snippet_centred_after = [[] for _ in range(s)]



for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_3_pre = prs.Level_3_moving_light_paths(rat)
         sessions_subset = Level_3_pre[0]#[3:6]
         
         rat_coordinates_before,ball_coordinates,rat_coordinates_after,centre_po_before, centre_po_after = rat_event_idx_and_pos_finder(sessions_subset, event=2, offset = 360, trial_number=11)
         
         snippet_before[r] =  rat_coordinates_before
         snippet_after[r] =  rat_coordinates_after
         ball_pos_all_rats[r] = ball_coordinates
         snippet_centred_before[r] =centre_po_before
         snippet_centred_after[r] =centre_po_after
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  


len(snippet_before)
len(ball_pos_all_rats)



plt.plot(snippet_centred_before[0][:][:,0],snippet_centred_before[0][:][:,1])








def rat_event_idx_and_pos_finder(sessions_subset, event=2, offset = 360, trial_number=11): #start_event=0, end_event= 1
    
    ball_coordinates = []
    rat_coordinates_before = [[] for _ in range(trial_number)]
    rat_coordinates_after = [[] for _ in range(trial_number)]
    centre_po_before =  [[] for _ in range(trial_number)]
    centre_po_after =  [[] for _ in range(trial_number)]
    
    
    session = sessions_subset
    

    script_dir = os.path.join(hardrive_path + session) 

    trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx_cleaned.csv')
    trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
    #crop_tracking_path = os.path.join(script_dir + '/crop.csv')
    #crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
    shader_tracking_path = os.path.join(script_dir + '/events/' +'Tracking.csv')
    shader_tracking = np.genfromtxt(shader_tracking_path)
    ball_shaders_path= os.path.join(script_dir+ '/events/' + 'BallPosition.csv')
    split_data = np.genfromtxt(ball_shaders_path, delimiter=[33,100], dtype='unicode')
    timestamps = split_data[:,0]
    positions_strings = split_data[:,1]
    for index, s in enumerate(positions_strings):
        tmp = s.replace('(', '')
        tmp = tmp.replace(')', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp.replace(' ', '')
        positions_strings[index] = tmp
    ball_shaders = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
    
    start  = trial_idx[:11,event]       
    ball = ball_shaders[:11]

               
    for count in np.arange(trial_number):

        
        rat_pos_before = shader_tracking[start[count]-offset:start[count]:]
        rat_pos_after = shader_tracking[start[count]:start[count]+offset]
        centre_before = rat_pos_before - ball[count]
        centre_after = rat_pos_after- ball[count]
    
        
        rat_coordinates_before[count] = rat_pos_before
        ball_coordinates.append(ball)
        rat_coordinates_after[count] = rat_pos_after
        centre_po_before[count] = centre_before
        centre_po_after[count] = centre_after

    return rat_coordinates_before,ball_coordinates,rat_coordinates_after,centre_po_before, centre_po_after













plt.plot(test[0][:,0],test[0][:,1])
plt.plot(test[1][:,0],test[1][:,1])
plt.plot(test[2][:,0],test[2][:,1])
plt.plot(test[3][:,0],test[3][:,1])

test  = [val for sublist in snippet for val in sublist]

diff_test =  np.diff(test[1],axis=0)
plt.plot(diff_test[:,0],diff_test[:,1])










'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/BallPosition.csv'


tracking = 'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/BallTracking.csv'
tracking_open = np.genfromtxt(tracking, delimiter = ',', usecols= [1,2])
track_diff = np.diff(tracking_open[:,0])


binary = []
for v in range(len(track_diff)):
    if track_diff[v] ==0:
        binary.append(0)
    else:
        binary.append(1)


trans = np.argwhere(np.diff(binary)).squeeze()


trans_list = []
for i in range(len(binary)):
    
    sub = binary[i+1] - binary[i]
    if sub ==1:
        trans_list.append(i)
        

len(trans_list)

save= np.savetxt(test_open, delimiter=',')



'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/BallTracking.csv'
'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/BallPosition.csv'
'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/BallOn.csv'


def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(filename)
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



filename = 'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/BallTracking.csv'
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamps = split_data[:,0]
positions_strings = split_data[:,1]
for index, s in enumerate(positions_strings):
    tmp = s.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('\n', '')
    tmp = tmp.replace(' ', '')
    positions_strings[index] = tmp
positions = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
len(positions)


time_ball = timestamps
time_tracking = timestamps


ball_timestamps = pd.to_datetime(time_ball)
tracking_timestamp = pd.to_datetime(time_tracking)


nearest= closest_timestamps_to_events(tracking_timestamp, ball_timestamps)

len(nearest)



idx = positions[nearest]





#finding trigger events

trial_idx_path='F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/events/Trial_idx.csv'

trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 

'F:/Videogame_Assay/AK_49.1/2019_09_19-11_14/Video.csv'



start = trial_idx[:,0]
touch  = trial_idx[:,2]

diff_touch_start = np.abs(start-touch)


good_trials = diff_touch_start>10

good_ball = positions[:-1][good_trials]



times_good =timestamps[:-1][good_trials]



timestamps, positions


timestamps_good = [[] for _ in range(len(good_ball))]


for ball in range(len(good_ball)):
    
    ball_times = []
    
    for pos in range(len(positions)):
    
        if round(positions[pos,0],6) - round(good_ball[ball,0],6) == 0  and round(positions[pos,1],6) -  round(good_ball[ball,1],6) == 0:
            
            ball_times.append(pos)
        
            
        timestamps_good[ball]= ball_times

    print(ball)



            
time_good_ball =  pd.to_datetime(times_good) 
time_tracking = pd.to_datetime(timestamps) 
        
nearest_good_ball= closest_timestamps_to_events(time_tracking, time_good_ball)


trigger=[]

for i in range(len(timestamps_good)):
    
    if timestamps_good[i] ==[]:
        trigger.append([])
        
    else:
        
    
        trigger.append(timestamps_good[i][-1])
    















