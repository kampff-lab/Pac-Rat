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



#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

s = len(rat_summary_table_path)

#rat_ball_all_rats = [[] for _ in range(s)]
#rat_poke_all_rats = [[] for _ in range(s)]
#before_touch_all_rats= [[] for _ in range(s)]
#after_touch_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path[8:]):
    
    
    try:    
         Level_2_pre= prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre
         
         distance_rat_at_start_ball, distance_rat_at_touch_poke, distance_rat_ball_before_touch, distance_rat_ball_after_touch = distance_events(sessions_subset,frames=120, trial_folder = 'Trial_idx.csv',ball_folder = 'Ball_coordinates.csv' )
         
         dist_rat_at_start_ball_flat = [val for sublist in distance_rat_at_start_ball for val in sublist]
         dist_rat_at_touch_poke_flat = [val for sublist in distance_rat_at_touch_poke for val in sublist]
         dist_rat_ball_before_touch_flat = [val for sublist in distance_rat_ball_before_touch for val in sublist]
         dist_rat_ball_after_touch_flat =  [val for sublist in distance_rat_ball_after_touch for val in sublist]
         
         print(len(dist_rat_at_start_ball_flat))
         print(len(dist_rat_at_touch_poke_flat))
         print(len(dist_rat_ball_before_touch_flat))
         print(len(dist_rat_ball_after_touch_flat))
         
         rat_pos_at_start, before_start, after_start = trial.rat_event_crop_pos_finder(sessions_subset, event=0, offset = 0, trial_folder = 'Trial_idx.csv')
         
         rat_pos_at_touch, before_touch, after_touch = trial.rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120,trial_folder = 'Trial_idx.csv')
         
         rat_pos_at_start_flat =  [val for sublist in rat_pos_at_start for val in sublist]
         rat_pos_at_touch_flat = [val for sublist in rat_pos_at_touch for val in sublist]
         rat_pos_before_touch =  [val for sublist in before_touch for val in sublist]
         rat_pos_after_touch =  [val for sublist in after_touch for val in sublist]
         
         print(len(rat_pos_at_start_flat))
         print(len(rat_pos_at_touch_flat))
         print(len(rat_pos_before_touch))
         print(len(rat_pos_after_touch))
         
         
         st_idx_diff, te_idx_diff, se_idx_diff = trial.time_to_events(sessions_subset)
         
         st_idx_diff_flat = [val for sublist in st_idx_diff for val in sublist]
         te_idx_diff_flat = [val for sublist in te_idx_diff for val in sublist]
         se_idx_diff_flat = [val for sublist in se_idx_diff for val in sublist]
         
         print(len(st_idx_diff_flat))
         print(len(te_idx_diff_flat))
         print(len(se_idx_diff_flat))
         
         
         
         
         ball_pos = ball_positions_finder(sessions_subset)
         
         ball_flat = [val for sublist in ball_pos for val in sublist]


         print(len(ball_flat))
         #quadrants = ball_positions_based_on_quadrant_of_appearance(sessions_subset)
        
         #q_flat = [val for sublist in quadrants for val in sublist]
         rat_id = [r]*len(ball_flat)

         print(len(rat_id))
         
         
         session_trials = trial_counter(sessions_subset)
         
         trial_count_flat = [val for sublist in session_trials for val in sublist]

         
         print(len(trial_count_flat))
         
         csv_name = RAT_ID[r] +'_Trial_table.csv'
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
                                                       trial_count_flat
                                                       
                                                       )).T, delimiter=',', fmt='%s') #final_trial_type (only for moving light)
                                                       
                                                                                                            
                                                    

        
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
       
    


        

trial_table_folder = r'E:\thesis_figures\Tracking_figures\Trial_Table_level_3_moving_light/'


csv_in_folder = os.listdir(trial_table_folder)


all_rats_trial_table = []

for i in csv_in_folder:
    
 
    file = np.genfromtxt(trial_table_folder + i, delimiter=',',dtype=str)
    all_rats_trial_table.append(file)
    
    


flat_table = [val for sublist in all_rats_trial_table for val in sublist]

final_table = np.array(flat_table)

csv_name = '/Trial_table_final_level_3_moving_light.csv'


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

session_type ='joystick'

x_crop_all_rats = []
y_crop_all_rats =[]
x_shaders_all_rats= []
y_shaders_all_rats = []

for r, rat in enumerate(rat_summary_table_path[8:]):
    
    
    try:    
         Level_3_pre= prs.Level_3_joystick_paths(rat)
         sessions_subset = Level_3_pre
         
         x_crop_snippet, y_crop_snippet,x_shader_snippet,y_shader_snippet = rat_position_around_event_snippets(sessions_subset, event=4, offset=360, folder = 'Trial_idx_cleaned.csv')
         
         x_crop_all_rats.extend(x_crop_snippet)
         y_crop_all_rats.extend(y_crop_snippet)
         x_shaders_all_rats.extend(x_shader_snippet)
         y_shaders_all_rats.extend(y_shader_snippet)

         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
    
    
print(len(x_crop_all_rats))      
print(len(y_crop_all_rats))         
print(len(x_shaders_all_rats))    
print(len(y_shaders_all_rats))



csv_name = 'x_crop_snippets_around_touch_' + session_type +'.csv'

np.savetxt(results_dir + csv_name,x_crop_all_rats, delimiter=',', fmt='%s')


csv_name = 'y_crop_snippets_around_touch_'  + session_type +'.csv'
np.savetxt(results_dir + csv_name,y_crop_all_rats, delimiter=',', fmt='%s')


csv_name = 'x_shaders_snippets_around_touch_'  + session_type +'.csv'
np.savetxt(results_dir + csv_name,x_shaders_all_rats, delimiter=',', fmt='%s')


csv_name = 'y_shaders_snippets_around_touch_'  + session_type +'.csv'
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



#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

s = len(rat_summary_table_path_moving_light)

#rat_ball_all_rats = [[] for _ in range(s)]
#rat_poke_all_rats = [[] for _ in range(s)]
#before_touch_all_rats= [[] for _ in range(s)]
#after_touch_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path_moving_light):
    
    
    try:    
         Level_3_pre= prs.Level_3_joystick_paths(rat)
         sessions_subset = Level_3_pre
         
         distance_rat_at_start_ball, distance_rat_at_touch_poke, distance_rat_ball_before_touch, distance_rat_ball_after_touch = distance_events(sessions_subset,frames=120, trial_file = 'Trial_idx_cleaned.csv',
                                                                                                                                                ball_file = 'Ball_positions_cleaned.csv', tracking_file = '/events/Tracking.csv' )
         
         dist_rat_at_start_ball_flat = [val for sublist in distance_rat_at_start_ball for val in sublist]
         dist_rat_at_touch_poke_flat = [val for sublist in distance_rat_at_touch_poke for val in sublist]
         dist_rat_ball_before_touch_flat = [val for sublist in distance_rat_ball_before_touch for val in sublist]
         dist_rat_ball_after_touch_flat =  [val for sublist in distance_rat_ball_after_touch for val in sublist]
         
         print(len(dist_rat_at_start_ball_flat))
         print(len(dist_rat_at_touch_poke_flat))
         print(len(dist_rat_ball_before_touch_flat))
         print(len(dist_rat_ball_after_touch_flat))
         
         rat_pos_at_start, before_start, after_start = rat_event_crop_pos_finder(sessions_subset, event=0, offset = 0,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv')
         
         rat_pos_at_touch, before_touch, after_touch = rat_event_crop_pos_finder(sessions_subset, event=2, offset = 120,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv')
         
         rat_pos_at_trigger, before_trigger, after_trigger = rat_event_crop_pos_finder(sessions_subset, event=4, offset = 120,trial_file = 'Trial_idx_cleaned.csv',tracking_file = '/events/Tracking.csv')
         
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
         
         
         st_idx_diff, te_idx_diff, se_idx_diff, tt_idx_diff, bot_idx_diff = time_to_events_moving_and_joystick(sessions_subset, trial_file = 'Trial_idx_cleaned.csv')
         
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
         
         
         
         ball_pos =  ball_positions_finder(sessions_subset, ball_file ='Ball_positions_cleaned.csv' )
         
         ball_flat = [val for sublist in ball_pos for val in sublist]


         print(len(ball_flat))
         #quadrants = ball_positions_based_on_quadrant_of_appearance(sessions_subset)
        
         #q_flat = [val for sublist in quadrants for val in sublist]
         rat_id = [r]*len(ball_flat)

         print(len(rat_id))
         
         
         session_trials = trial_counter(sessions_subset, trial_file= 'Trial_idx_cleaned.csv')
         
         trial_count_flat = [val for sublist in session_trials for val in sublist]

         
         print(len(trial_count_flat))
         
         csv_name = RAT_ID[r] +'_Trial_table_joystick.csv'
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
                                                       trial_count_flat
                                                       
                                                       )).T, delimiter=',', fmt='%s') #final_trial_type (only for moving light)
                                                       
                                                                                                            
                                                    

        
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
       


trial_table_folder = r'E:\thesis_figures\Tracking_figures\Trial_Table_level_3_joystick/'


csv_in_folder = os.listdir(trial_table_folder)


all_rats_trial_table = []

for i in csv_in_folder:
    
 
    file = np.genfromtxt(trial_table_folder + i, delimiter=',',dtype=str)
    all_rats_trial_table.append(file)
    
    


flat_table = [val for sublist in all_rats_trial_table for val in sublist]

final_table = np.array(flat_table)

csv_name = '/Trial_table_final_level_3_joystick.csv'


np.savetxt(results_dir + csv_name, final_table,delimiter=',', fmt='%s')




