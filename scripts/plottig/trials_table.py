# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:22:29 2020

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
import trials_table_def as trial


import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(trial)


hardrive_path = r'F:/' 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']





rat_summary_table_path_moving_light = ['F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']



rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']

RAT_ID_moving = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

RAT_ID = RAT_ID_ephys


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys

#rat_ball_all_rats = [[] for _ in range(s)]
#rat_poke_all_rats = [[] for _ in range(s)]
#before_touch_all_rats= [[] for _ in range(s)]
#after_touch_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path): 
    
    
    try:  
         #rat = rat_summary_table_path
         Level_2_pre= prs.Level_2_post_paths(rat)
         sessions_subset = Level_2_pre
         
         distance_rat_at_start_ball, distance_rat_at_touch_poke, distance_rat_ball_before_touch, distance_rat_ball_after_touch = trial.distance_events(sessions_subset,frames=120, trial_file = 'Trial_idx.csv',
                         ball_file = 'BallPosition.csv', tracking_file = '/events/Tracking.csv',
                         tracking_delimiter=None, poke_coordinates = [1,0]) # crop : '/crop.csv', delimiter  ',',  ball:  Ball_coordinates , poke :[1400,600/ shaders: '/events/Tracking.csv'
                                                                             #delimiter : None, ball:  'BallPosition.csv', poke =[1,0]
         
         
         dist_rat_at_start_ball_flat = [val for sublist in distance_rat_at_start_ball for val in sublist]
         dist_rat_at_touch_poke_flat = [val for sublist in distance_rat_at_touch_poke for val in sublist]
         dist_rat_ball_before_touch_flat = [val for sublist in distance_rat_ball_before_touch for val in sublist]
         dist_rat_ball_after_touch_flat =  [val for sublist in distance_rat_ball_after_touch for val in sublist]
         
         print(len(dist_rat_at_start_ball_flat))
         print(len(dist_rat_at_touch_poke_flat))
         print(len(dist_rat_ball_before_touch_flat))
         print(len(dist_rat_ball_after_touch_flat))
         
         rat_pos_at_start, before_start, after_start = trial.rat_event_crop_pos_finder(sessions_subset, event=0, offset = 120,trial_file = 'Trial_idx.csv',
                                                                                       tracking_file = '/events/Tracking.csv',tracking_delimiter = None)
         
         rat_pos_at_touch, before_touch, after_touch = trial.rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120,trial_file = 'Trial_idx.csv',
                                                                                       tracking_file = '/events/Tracking.csv',tracking_delimiter = None)
         
         
         rat_pos_at_start_flat =  [val for sublist in rat_pos_at_start for val in sublist]
         rat_pos_at_touch_flat = [val for sublist in rat_pos_at_touch for val in sublist]
         rat_pos_before_touch =  [val for sublist in before_touch for val in sublist]
         rat_pos_after_touch =  [val for sublist in after_touch for val in sublist]
         
         print(len(rat_pos_at_start_flat))
         print(len(rat_pos_at_touch_flat))
         print(len(rat_pos_before_touch))
         print(len(rat_pos_after_touch))
         
         
         st_idx_diff, te_idx_diff, se_idx_diff = trial.time_to_events(sessions_subset,trial_file = 'Trial_idx.csv')
         
         st_idx_diff_flat = [val for sublist in st_idx_diff for val in sublist]
         te_idx_diff_flat = [val for sublist in te_idx_diff for val in sublist]
         se_idx_diff_flat = [val for sublist in se_idx_diff for val in sublist]
         
         print(len(st_idx_diff_flat))
         print(len(te_idx_diff_flat))
         print(len(se_idx_diff_flat))
         
         
         
         
         ball_pos = trial.ball_positions_shaders(sessions_subset, ball_file ='BallPosition.csv', trial_file= 'Trial_idx.csv' )
         
         ball_flat = [val for sublist in ball_pos for val in sublist]


         print(len(ball_flat))
         #quadrants = ball_positions_based_on_quadrant_of_appearance(sessions_subset)
        
         #q_flat = [val for sublist in quadrants for val in sublist]
         rat_id = [r]*len(ball_flat)

         print(len(rat_id))
         
         
         session_trials = trial.trial_counter(sessions_subset, trial_file= 'Trial_idx.csv')
         
         trial_count_flat = [val for sublist in session_trials for val in sublist]

         
         print(len(trial_count_flat))
         
                 
         outcome_sessions = outcome_over_sessions(sessions_subset, trial_file = '/TrialEnd.csv')
         outcome = [val for sublist in outcome_sessions for val in sublist]
         print(len(outcome))
         
         csv_name = RAT_ID[r] +'_Trial_table_level_3_pre_SHADERS.csv'
         #csv_name = RAT_ID[8:][r] +'_Trial_table.csv'
         
         #moving_str = ['moving_light'  for x in range(len(ball_flat) - 6)]
         
         #trial_type_array = ['touhing_light', 'touching_light','touching_light','touching_light', 'touching light', 'first_moving_light']
         

         #final_trial_type = trial_type_array + moving_str
         
         #print(len(final_trial_type))
         
         np.savetxt(results_dir + csv_name, np.vstack((rat_id,
                                                       st_idx_diff_flat,
                                                       te_idx_diff_flat,
                                                       se_idx_diff_flat,
                                                       np.array(rat_pos_at_start_flat)[:,0],
                                                       np.array(rat_pos_at_start_flat)[:,1],
                                                       np.array(rat_pos_at_touch_flat)[:,0],
                                                       np.array(rat_pos_at_touch_flat)[:,1],
                                                       np.array(rat_pos_before_touch)[:,0],
                                                       np.array(rat_pos_before_touch)[:,1],
                                                       np.array(rat_pos_after_touch)[:,0],
                                                       np.array(rat_pos_after_touch)[:,1],
                                                       dist_rat_at_start_ball_flat,
                                                       dist_rat_at_touch_poke_flat,
                                                       dist_rat_ball_before_touch_flat,
                                                       dist_rat_ball_after_touch_flat,
                                                       np.array(ball_flat)[:,0],
                                                       np.array(ball_flat)[:,1],
                                                       trial_count_flat,
                                                       outcome
                                                       
                                                       )).T, delimiter=',', fmt='%s') #final_trial_type (only for moving light)
                                                       
                                                                                                            
                                                    

        
         print(rat+'DONE')
         print(r)
         
         
    except Exception: 
        print (rat + '/error')
        continue    
       
    


        

trial_table_folder = 'E:/thesis_figures/Tracking_figures/Trial_table_moving_light_ephys_shaders/'

csv_in_folder = os.listdir(trial_table_folder)


all_rats_trial_table = []

for i in csv_in_folder:
    
 
    file = np.genfromtxt(trial_table_folder + i, delimiter=',',dtype=str)
    all_rats_trial_table.append(file)
    
    


flat_table = [val for sublist in all_rats_trial_table for val in sublist]

final_table = np.array(flat_table)

csv_name = '/Trial_table_final_level_3_moving_light_ephys_shaders.csv'


np.savetxt(results_dir + csv_name, final_table,delimiter=',', fmt='%s')


#################################################################################                                                      
trial_table = 'F:/Videogame_Assay/Trial_table_final_level_2_touching_light.csv'
table_open =np.genfromtxt(trial_table, delimiter = ',')


for r, rat in enumerate(rat_summary_table_path):
    
    
    try:    
         Level_2_pre= prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre
         

         rat_pos_at_touch, before_touch, after_touch = trial.rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120)
         
         rat_pos_at_start_flat =  [val for sublist in rat_pos_at_start for val in sublist]
         rat_pos_at_touch_flat = [val for sublist in rat_pos_at_touch for val in sublist]
         rat_pos_before_touch =  [val for sublist in before_touch for val in sublist]
         rat_pos_after_touch =  [val for sublist in after_touch for val in sublist]
         
         print(len(rat_pos_at_start_flat))
         print(len(rat_pos_at_touch_flat))
         print(len(rat_pos_before_touch))
         print(len(rat_pos_after_touch))




###################################
# snippets around_touch

session_type ='moving_light_ephys_catch'

#x_crop_all_rats = []
#y_crop_all_rats =[]
x_shaders_all_rats= []
y_shaders_all_rats = []

for r, rat in enumerate(rat_summary_table_path):
    
    
    try:    
         Level_3_pre= Level_3_moving_post_paths(rat)
         sessions_subset = Level_3_pre
         
         #x_crop_snippet, y_crop_snippet,x_shader_snippet,y_shader_snippet = rat_position_around_event_snippets(sessions_subset, event=2, offset=360, folder = 'Trial_idx_cleaned.csv')
         x_shader_snippet,y_shader_snippet = rat_position_around_event_snippets_ephys(sessions_subset, event=2, offset=360, folder = 'Trial_idx_cleaned.csv') # 4 for moving light and joytick (when ball move away)
             
         #x_crop_all_rats.extend(x_crop_snippet)
         #y_crop_all_rats.extend(y_crop_snippet)
         x_shaders_all_rats.extend(x_shader_snippet)
         y_shaders_all_rats.extend(y_shader_snippet)

         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
    
    
#print(len(x_crop_all_rats))      
#print(len(y_crop_all_rats))         
print(len(x_shaders_all_rats))    
print(len(y_shaders_all_rats))



#csv_name = 'x_crop_snippets_around_touch_' + session_type +'.csv'

#np.savetxt(results_dir + csv_name,x_crop_all_rats, delimiter=',', fmt='%s')


#csv_name = 'y_crop_snippets_around_touch_'  + session_type +'.csv'
#np.savetxt(results_dir + csv_name,y_crop_all_rats, delimiter=',', fmt='%s')


csv_name = 'x_shaders_snippets_around_touch_all_rats_'  + session_type +'.csv'
np.savetxt(results_dir + csv_name,x_shaders_all_rats, delimiter=',', fmt='%s')


csv_name = 'y_shaders_snippets_around_touch_all_rats_'  + session_type +'.csv'
np.savetxt(results_dir + csv_name,y_shaders_all_rats, delimiter=',', fmt='%s')


crop_diff_x = np.diff(x_crop_all_rats, prepend = 0,axis=0)
crop_diff_y = np.diff(y_crop_all_rats, prepend = 0,axis=0)    
crop_diff_x_square = crop_diff_x**2
crop_diff_y_square = crop_diff_y**2
crop_speed = np.sqrt(crop_diff_x_square + crop_diff_y_square)

crop_median = np.nanmedian(crop_speed,axis=0)

len(crop_median)

plt.figure()
plt.plot(range(len(crop_median)),crop_median)
plt.title('crop')

avg_speed = np.nanmean(np.sqrt(dx*dx + dy*dy), axis=0)

shader_diff_x = np.diff(x_shaders_all_rats, prepend = 0,axis=1)
shader_diff_y = np.diff(y_shaders_all_rats, prepend = 0,axis=1)    
shader_diff_x_square = shader_diff_x**2
shader_diff_y_square = shader_diff_y**2
shader_speed = np.sqrt(shader_diff_x_square + shader_diff_y_square)


shader_median = np.nanmedian(shader_speed,axis=0)

len(shader_median)

f,ax = plt.subplots(figsize=(9,6))

plt.plot(range(len(shader_median)),shader_median)
plt.title('shaders')

ax.vlines(359,min(shader_median)-.01,max(shader_median)+.01,linewidth=0.9,color= 'k')




bounds =  [0,648, 1247, 1710, 2599, 2841, 3418, 3614, 4272, 4643, 4921, 5225]


####################################trial table moving light


rat_summary_table_path_moving_light = ['F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']

RAT_ID_moving = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

RAT_ID = RAT_ID_ephys



#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys




#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path_moving_light)

#rat_ball_all_rats = [[] for _ in range(s)]
#rat_poke_all_rats = [[] for _ in range(s)]
#before_touch_all_rats= [[] for _ in range(s)]
#after_touch_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    
    try:    
         Level_3_pre=   Level_3_moving_post_paths(rat) #joystick_post   Level_3_moving_post_paths  Level_3_moving_light_paths #joystick_post_paths
         sessions_subset = Level_3_pre
         
         distance_rat_at_start_ball, distance_rat_at_touch_poke, distance_rat_ball_before_touch, distance_rat_ball_after_touch = trial.distance_events(sessions_subset,frames=120, trial_file = 'Trial_idx_cleaned.csv',
                                                                                                                                                ball_file = 'Ball_positions_cleaned.csv', tracking_file = '/events/Tracking.csv',
                                                                                                                                                tracking_delimiter=None, poke_coordinates = [1,0])
         
                  
         
         
         dist_rat_at_start_ball_flat = [val for sublist in distance_rat_at_start_ball for val in sublist]
         dist_rat_at_touch_poke_flat = [val for sublist in distance_rat_at_touch_poke for val in sublist]
         dist_rat_ball_before_touch_flat = [val for sublist in distance_rat_ball_before_touch for val in sublist]
         dist_rat_ball_after_touch_flat =  [val for sublist in distance_rat_ball_after_touch for val in sublist]
         
         print(len(dist_rat_at_start_ball_flat))
         print(len(dist_rat_at_touch_poke_flat))
         print(len(dist_rat_ball_before_touch_flat))
         print(len(dist_rat_ball_after_touch_flat))
         
         rat_pos_at_start, before_start, after_start = trial.rat_event_crop_pos_finder(sessions_subset, event=0, offset = 0,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv',tracking_delimiter = None)
         
         rat_pos_at_touch, before_touch, after_touch = trial.rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv',tracking_delimiter = None)
         
         rat_pos_at_trigger, before_trigger, after_trigger = trial.rat_event_crop_pos_finder(sessions_subset, event=4, offset = 120,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv',tracking_delimiter = None)
         
         

         
         rat_pos_at_start_flat =  [val for sublist in rat_pos_at_start for val in sublist]
         rat_pos_at_touch_flat = [val for sublist in rat_pos_at_touch for val in sublist]
         rat_pos_before_touch =  [val for sublist in before_touch for val in sublist]
         rat_pos_after_touch =  [val for sublist in after_touch for val in sublist]
         rat_pos_at_trigger_flat =  [val for sublist in rat_pos_at_trigger for val in sublist]
         rat_pos_before_trigger = [val for sublist in before_trigger for val in sublist]
         rat_pos_after_trigger =  [val for sublist in after_trigger for val in sublist]
         
         print(len(rat_pos_at_start_flat))
         print(len(rat_pos_at_touch_flat))
         print(len(rat_pos_before_touch))
         print(len(rat_pos_after_touch))
         print(len(rat_pos_at_trigger_flat))
         print(len(rat_pos_before_trigger))
         print(len(rat_pos_after_trigger))
         
         
         st_idx_diff, te_idx_diff, se_idx_diff, tt_idx_diff, bot_idx_diff = trial.time_to_events_moving_and_joystick(sessions_subset, trial_file = 'Trial_idx_cleaned.csv')
         
                                                                                                               
         st_idx_diff_flat = [val for sublist in st_idx_diff for val in sublist]
         te_idx_diff_flat = [val for sublist in te_idx_diff for val in sublist]
         se_idx_diff_flat = [val for sublist in se_idx_diff for val in sublist]
         tt_idx_diff_flat = [val for sublist in tt_idx_diff for val in sublist]
         bot_idx_diff_flat =  [val for sublist in bot_idx_diff for val in sublist]
         
         
         print(len(st_idx_diff_flat))
         print(len(te_idx_diff_flat))
         print(len(se_idx_diff_flat))
         print(len(tt_idx_diff_flat))
         print(len(bot_idx_diff_flat))
         
         
         
         ball_pos =  trial.ball_positions_finder(sessions_subset, ball_file ='Ball_positions_cleaned.csv' )
         
         ball_flat = [val for sublist in ball_pos for val in sublist]


         print(len(ball_flat))
         #quadrants = ball_positions_based_on_quadrant_of_appearance(sessions_subset)
        
         #q_flat = [val for sublist in quadrants for val in sublist]
         rat_id = [r]*len(ball_flat)

         print(len(rat_id))
         
         
         session_trials = trial.trial_counter(sessions_subset, trial_file= 'Trial_idx_cleaned.csv')
         
         trial_count_flat = [val for sublist in session_trials for val in sublist]
         
         
         claned_outcome = cleaned_outcome_sessions(sessions_subset, trial_file = 'Trial_outcome_cleaned.csv')

         outcome = [val for sublist in claned_outcome for val in sublist]
         
         print(len(outcome))
         print(len(trial_count_flat))
         
         csv_name = RAT_ID[r] +'_Trial_table_moving_light_ephys_SHADERS.csv'
         #csv_name = RAT_ID[8:][r] +'_Trial_table.csv'
         
         #moving_str = ['moving_light'  for x in range(len(ball_flat) - 6)]
         
         #trial_type_array = ['touhing_light', 'touching_light','touching_light','touching_light', 'touching light', 'first_moving_light']
         

         #final_trial_type = trial_type_array + moving_str
         
         #print(len(final_trial_type))
         
         np.savetxt(results_dir + csv_name, np.vstack((rat_id,
                                                       st_idx_diff_flat,
                                                       te_idx_diff_flat,
                                                       se_idx_diff_flat,
                                                       tt_idx_diff_flat,
                                                       bot_idx_diff_flat,
                                                       np.array(rat_pos_at_start_flat)[:,0],
                                                       np.array(rat_pos_at_start_flat)[:,1],
                                                       np.array(rat_pos_at_touch_flat)[:,0],
                                                       np.array(rat_pos_at_touch_flat)[:,1],
                                                       np.array(rat_pos_before_touch)[:,0],
                                                       np.array(rat_pos_before_touch)[:,1],
                                                       np.array(rat_pos_after_touch)[:,0],
                                                       np.array(rat_pos_after_touch)[:,1],
                                                       np.array(rat_pos_at_trigger_flat)[:,0],
                                                       np.array(rat_pos_at_trigger_flat)[:,1],
                                                       np.array(rat_pos_before_trigger)[:,0],
                                                       np.array(rat_pos_before_trigger)[:,1],
                                                       np.array(rat_pos_after_trigger)[:,0],
                                                       np.array(rat_pos_after_trigger)[:,1],                                                      
                                                       dist_rat_at_start_ball_flat,
                                                       dist_rat_at_touch_poke_flat,
                                                       dist_rat_ball_before_touch_flat,
                                                       dist_rat_ball_after_touch_flat,
                                                       np.array(ball_flat)[:,0],
                                                       np.array(ball_flat)[:,1],
                                                       trial_count_flat,
                                                       outcome
                                                       
                                                       )).T, delimiter=',', fmt='%s') #final_trial_type (only for moving light)
                                                       
                                                                                                            
                                                    

        
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
       


trial_table_folder = 'E:/thesis_figures/Tracking_figures/Trial_Table_ephys_moving_light/'


csv_in_folder = os.listdir(trial_table_folder)


all_rats_trial_table = []

for i in csv_in_folder:
    
 
    file = np.genfromtxt(trial_table_folder + i, delimiter=',',dtype=str)
    all_rats_trial_table.append(file)
    
    


flat_table = [val for sublist in all_rats_trial_table for val in sublist]

final_table = np.array(flat_table)

csv_name = '/Trial_table_final_level_3_moving_light_ephys.csv'


np.savetxt(results_dir + csv_name, final_table,delimiter=',', fmt='%s')

##########################reaction time 


s = len(rat_summary_table_path)

avg_reaction_time_all_rats = [[] for _ in range(s)]
std_reaction_time =[[] for _ in range(s)]
rt_all_rats = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path): 
    

     #rat = rat_summary_table_path[0]
     Level_1= Level_1_paths(rat)
     sessions_subset = Level_1
     
     rt,avg,std = reaction_time(sessions_subset,trial_file = 'Trial_idx.csv',
                                           tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600])
     
     avg_reaction_time_all_rats[r]=avg
     std_reaction_time[r]=std
     rt_all_rats[r]=rt
     print(rat)

#plot test

select_6000=np.zeros((s,4))    
select_10000_20000 =  np.zeros((s,2)) 

for i in range(len(rat_summary_table_path)):
    #plt.figure()
    sel=avg_reaction_time_all_rats[i][:4]
    sel2=avg_reaction_time_all_rats[i][-2:]
    select_6000[i,:]=sel
    select_10000_20000[i,:]=sel2
        
    
final=np.hstack((select_6000,select_10000_20000))*120


for l in range(len(final)):
    plt.plot(final[l])


colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name = 'mean_reaction_time_level_1.pdf'

for count, row in enumerate(final):    
    
  
    plt.plot(row,color=colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    
    
    
    plt.title('reaction speed level 1',fontsize = 16)
    plt.ylabel('dst to poke / time to reward in secons (frame*120)', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    ax.axes.get_xaxis().set_visible(True) 
    #ax.set_ylim(ymin= -10 ,ymax= 260)
    #plt.yticks((np.arange(0, 350, 50)))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(-10,300)


mean= np.nanmean(final, axis=0)

sem = stats.sem(final, nan_policy='omit', axis=0)


plt.plot(mean,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(len(mean)), mean, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  
plt.yticks((np.arange(0, 140, 20)))
#plt.legend()
f.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)

#t test level 2

t_test = stats.ttest_rel(final[:,0],final[:,3],nan_policy='omit')
t_test_2 = stats.ttest_rel(final[:,3],final[:,4],nan_policy='omit')
t_test_3 = stats.ttest_rel(final[:,4],final[:,5],nan_policy='omit')
#Ttest_relResult(statistic=-3.292444121706258, pvalue=0.007173538082732699)




target = open(main_folder +"level_1_dst_speed.txt", 'w')
target.writelines(str(mean) +str(sem)+str(t_test)+ str(t_test_2)+str(t_test_3)+ ' LEVEL 1: dst to poke / time to reward in secons (frame*120) mean +- sem, trials_table.py')

target.close()

#####reaction time level 2  at ball on and also 



s = len(rat_summary_table_path)

avg_reaction_time_all_rats = [[] for _ in range(s)]
std_reaction_time =[[] for _ in range(s)]
rt_all_rats = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path): 
    

     #rat = rat_summary_table_path[0]
     Level_2= prs.Level_3_moving_light_paths(rat)
     sessions_subset = Level_2
     
     rt, avg,std =  reaction_time_level_2_and_3(sessions_subset,trial_file = 'Trial_idx_cleaned.csv',outcome='/Trial_outcome_cleaned.csv',tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600])


     
     avg_reaction_time_all_rats[r]=avg
     std_reaction_time[r]=std
     rt_all_rats[r]=rt
     print(rat)



for l in range(len(avg_reaction_time_all_rats)):
    plt.plot(avg_reaction_time_all_rats[l])



select_level_2=np.zeros((s,4))    
 

for i in range(len(rat_summary_table_path)):
    #plt.figure()
    sel=avg_reaction_time_all_rats[i][:4]

    select_level_2[i,:]=sel
   
final=select_level_2*120

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name = 'mean_reaction_time_level_2.pdf'

for count, row in enumerate(final):    
    
  
    plt.plot(row,color=colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    
    
    
    plt.title('reaction speed level 2',fontsize = 16)
    plt.ylabel('dst to poke / time to reward in secons (frame*120)', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    ax.axes.get_xaxis().set_visible(True) 
    #ax.set_ylim(ymin= -10 ,ymax= 260)
    #plt.yticks((np.arange(0, 350, 50)))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(-10,300)


mean= np.nanmean(np.array(final), axis=0)

sem = stats.sem(final, nan_policy='omit', axis=0)


plt.plot(mean,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(len(mean)), mean, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  
plt.yticks((np.arange(0, 180, 20)))
#plt.legend()
f.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)

