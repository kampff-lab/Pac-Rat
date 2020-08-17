# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:01:27 2020

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import behaviour_library as behaviour
import parser_library as prs
from scipy import stats
import pandas as pd
from scipy import stats



import importlib
importlib.reload(prs)
importlib.reload(behaviour)

hardrive_path = r'F:/' 


#main folder rat ID
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)



hardrive_path = r'F:/' 
rat_ID = 'AK_50.2'
rat_summary_table_path = r'F:/Videogame_Assay/AK_50.2_behaviour_only.csv'



figure_name = 'RAT_' + rat_ID + '_Trial_per_Session.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Trial/Session'


Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
Level_3_pre = prs.Level_3_pre_paths(rat_summary_table_path)






rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']









RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']


rat_summary_table_path = rat_summary_ephys
RAT_ID = RAT_ID_ephys



s  = len(rat_summary_table_path)

tot_trial_per_session =  [[] for _ in range(s)]
rat_session_length = [[] for _ in range(s)]

rat_success = [[] for _ in range((s))]
rat_miss = [[] for _ in range(s)]

trial_per_min = [[] for _ in range(s)]

sessions = [[] for _ in range(s)]

dates = [[] for _ in range(s)]


for count, rat in enumerate(rat_summary_table_path):
       
    #Level_2_post =  prs.Level_2_pre_paths(rat)
    #Level_2_post  = Level_2_post[:6]
    #Level_2_dates = Level_2_session_dates(rat_summary_table_path)



    
    Level_2_post = prs.Level_2_post_paths(rat)
    #Level_2_post = Level_2_pre

    
    sessions[count]= Level_2_post
    #Level_2_post = Level_2_post[:sessions_to_consider]
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_post)
    tot_trial_per_session[count]=total_trials
    rat_session_length[count]=session_length
    
    success_trials, missed_trials = behaviour.calculate_trial_and_misses(Level_2_post)

    
    rat_success[count]=success_trials
    rat_miss[count]=missed_trials
    #trials_per_minutes_L_1 = np.array(total_trials)/np.array(session_length)
    trials_per_minutes_L_2 = np.array(success_trials)/np.array(session_length)
    trial_per_min[count]=trials_per_minutes_L_2    
    
    
    
    print(count)



AK_33_2 = [0.94552989, 1.92502992, 1.77768211, 2.25678119, 1.9813374 ,1.65885298]



#position 0 
array_33_2 = [0.94552989, 1.92502992, 1.77768211, 2.12159905 , 1.65885298]
#position 10




rat_ok_sessions = [[] for _ in range(s)]
rat_trial_per_min_good =  [[] for _ in range(s)]
success_ok = [[] for _ in range(s)]

for d in range(len(rat_success)):
    
    good = np.array(rat_success[d])>15
    
    ok_sessions = np.array(sessions[d])[good]
    trial_per_min_ok = np.array(trial_per_min)[d][good]
    success = np.array(rat_success[d])[good]
    rat_ok_sessions[d] = ok_sessions
    rat_trial_per_min_good[d]=trial_per_min_ok
    success_ok[d] = success

    
    
###### ad hoc trials calculation
    
    
    
def Level_2_session_dates(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_2_dates = []
    for row in range(len(rat_summary)):
         if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 2' and rat_summary[row][3]== 'post':
            Level_2_dates.append(rat_summary[row][1])
         else:
            continue
    return Level_2_dates


Level_2_dates = Level_2_session_dates(rat_summary_table_path)




sessions_to_consider = 4

rat_means = []
rat_sem = []

for d in range(len(RAT_ID)):
    
    sel = rat_trial_per_min_good[0][:4]
    mean = np.mean(sel)
    sem = stats.sem(sel, nan_policy='omit', axis=0)
    rat_means.append(mean)
    rat_sem.append(sem)
    




figure_name =  '_avg_trial_per_min_level2_ephys.pdf'
    
f,ax = plt.subplots(figsize=(8,7))


sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

width = .35 # the width of the bars



ax.bar(np.arange(len(rat_means))-width/2,rat_means_pre,yerr = rat_sem_pre,width = width,align='center',color ='r', edgecolor ='white',alpha=0.7)

ax.bar(np.arange(len(rat_means))+width/2,rat_means,yerr = rat_sem,width = width, align='center', color ='g',edgecolor ='white',alpha=0.7)

ax.set_xticks(range(len(rat_means)))
ax.set_xticklabels(RAT_ID)


ax.axes.get_xaxis().set_visible(True) 

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('mean trial/min with sem(4 sessions over 15 trials)')
plt.xlabel('rats')
plt.ylim(ymax=3.0)
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)




pre_success =[[ 124, 77, 72, 65],
 [ 37, 89, 34, 66],
 [ 37, 54, 96, 77],
 [ 27, 30, 101, 113],
 [ 38, 63, 64, 106],
 [ 68, 48, 43, 101, ]]


pre_trial_lenght = [[64.41458333333334,43.877361111111114,33.93666666666667,39.78652777777778],
 [41.40763888888889,58.49875,52.24180555555555,63.66152777777778],
 [50.01208333333333,32.84986111111111,56.97097222222222,58.53125],
 [44.68013888888889,31.65041666666667,65.69013888888888,59.931805555555556],
 [47.96736111111111,60.89236111111111,50.817083333333336,51.51958333333334],
 [55.159305555555555,30.60875,41.569583333333334, 49.69625]]


trials_per_minutes_L_2_pre=np.array(pre_success)/np.array(pre_trial_lenght)


rat_means_pre = []
rat_sem_pre = []

for d in range(len(RAT_ID)):
    
    sel = trials_per_minutes_L_2_pre[d]
    mean = np.mean(sel)
    sem = stats.sem(sel, nan_policy='omit', axis=0)
    rat_means_pre.append(mean)
    rat_sem_pre.append(sem)
    

pre_misses = rat_miss


event_means = []
event_sem= []

for d in range(len(RAT_ID)):
    
    sel = success_ok[d]#[:4]
    mean = np.mean(sel)
    sem = stats.sem(sel, nan_policy='omit', axis=0)
    event_means.append(mean)
    event_sem.append(sem)




figure_name =  '_avg_trial_count_level2_ephys.pdf'
    
f,ax = plt.subplots(figsize=(8,7))


sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

width = .35 # the width of the bars



ax.bar(np.arange(len(rat_means))-width/2,event_means_pre,yerr = event_sem_pre, width = width,align='center',color ='r', edgecolor ='white',alpha=0.7)

ax.bar(np.arange(len(rat_means))+width/2,event_means,yerr = event_sem,width = width, align='center', color ='g',edgecolor ='white',alpha=0.7)

ax.set_xticks(range(len(rat_means)))
ax.set_xticklabels(RAT_ID)


ax.axes.get_xaxis().set_visible(True) 

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('mean trial count with sem(4 sessions over 15 trials)')
plt.xlabel('rats')
plt.ylim(ymax=100.0)
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)

percent = []

for d in range(len(RAT_ID)):
    
    initial = event_means_pre[d]
    final  =event_means[d]

    percentage_change = (initial-final)/initial *100
    percent.append(percentage_change)



figure_name =  '_%_of_change_level2_ephys.pdf'
    
f,ax = plt.subplots(figsize=(8,7))


sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.plot(percent,)

ax.set_xticks(range(len(rat_means)))
ax.set_xticklabels(RAT_ID)


ax.axes.get_xaxis().set_visible(True) 

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('pre mean -post mean/pre mean *100)')
plt.xlabel('rats')
plt.ylim(ymax=80.0)
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)






