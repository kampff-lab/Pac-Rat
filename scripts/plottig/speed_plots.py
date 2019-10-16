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




rat_summary_table_path = r'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_40.2'


Level_0 = prs.Level_0_paths(rat_summary_table_path)
Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)

sessions_subset = Level_2_pre
behaviour.start_touch_end_idx(sessions_subset)
sessions_speed = behaviour.session_speed(sessions_subset)



rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_40.2'

Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
sessions_subset = Level_2_pre

sessions_speed = behaviour.session_speed(sessions_subset)

Level_2_touch_to_reward_speed = behaviour.speed_touch_to_reward(sessions_subset, sessions_speed)

speed_touch_Level_2 = behaviour.speed_around_touch(sessions_subset,sessions_speed)
Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, sessions_speed)




speed_touch_Level_2 = speed_around_touch(sessions_subset,sessions_speed)


means= []

for row in speed_touch_Level_2:
    mean_session = np.mean(row,axis=0)
    means.append(mean_session)
    

figure_name0 = figure_name = 'RAT_' + rat_ID + '_speed_around_touch.pdf'
plot_main_title = 'RAT ' + rat_ID + 'speed around touch +/- 3 sec'
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()



cmap=sns.color_palette("hls", len(means))
for count,i in enumerate(means):
    plt.plot(i, color=cmap[count], label='%d'%count)

plt.axvline(360,color='k')
plt.legend()
f0.tight_layout()
f0.subplots_adjust(top = 0.87)






















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
    
    
    


    
stack_mean = np.vstack((mean_start_to_touch,mean_touch_to_reward)).T  
stack_std = np.vstack((std_start_to_touch,std_touch_to_reward)).T 


plt.scatter(mean_start_to_touch,range(l),marker = 'o',color= 'steelblue',alpha = .8)
plt.scatter(mean_touch_to_reward,range(l),marker = 'o',color= 'g',alpha = .8)

    





















figure_name = figure_name = 'RAT_' + rat_ID + '_Speed_Touch_to_reward_Level2.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Speed_Touch_to_reward_Level2' + 'Level_2'

 
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

