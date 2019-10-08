# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:52:12 2019

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


hardrive_path = r'F:/' 
rat_ID = 'AK_48.4'
rat_summary_table_path = r'F:/Videogame_Assay/AK_48.4_IrO2.csv'



figure_name = 'RAT_' + rat_ID + '_Trial_per_Session.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Trial/Session'


Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
Level_3_moving = prs.Level_3_moving_light_paths(rat_summary_table_path)



f,ax = plt.subplots(2,2,figsize=(10,7))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)



#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

success_trials_L_1, missed_trials_L_1 = behaviour.PLOT_trial_and_misses(Level_1)

x = np.array(range(len((Level_1))))

ax[0,0].bar(x, success_trials_L_1, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,0].bar(x, missed_trials_L_1, bottom = success_trials_L_1, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
ax[0,0].legend(loc ='best', frameon=False , fontsize = 'x-small') #ncol=2
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / Session', fontsize = 10)
#ax[0,0].set_xlabel('Sessions')




success_trials_L_2_pre, missed_trials_L_2_pre = behaviour.PLOT_trial_and_misses(Level_2_pre)

x = np.array(range(len((Level_2_pre))))
ax[0,1].bar(x, success_trials_L_2_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,1].bar(x, missed_trials_L_2_pre, bottom = success_trials_L_2_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
#ax[0,1].set_ylabel('Trials / Session')
#ax[0,0].set_xlabel('Sessions')




success_trials_L_2_post, missed_trials_L_2_post = behaviour.PLOT_trial_and_misses(Level_2_post)

x = np.array(range(len((Level_2_post))))
ax[1,0].bar(x, success_trials_L_2_post, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[1,0].bar(x, missed_trials_L_2_post, bottom = success_trials_L_2_post, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
ax[1,0].set_ylabel('Trials / Session', fontsize = 10)
ax[1,0].set_xlabel('Sessions', fontsize = 10)



#success_trials_L_3_pre ,missed_trials_L_3_pre = behaviour.PLOT_trial_and_misses(Level_3_pre)

#x = np.array(range(len((Level_3_pre))))
#ax[1,0].bar(x, success_trials_L_3_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,0].bar(x, missed_trials_L_3_pre, bottom = success_trials_L_3_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 3 pre surgery')
#ax[1,0].set_ylabel('Trials / Session')
#ax[1,0].set_xlabel('Sessions')



success_trials_L_3_post, missed_trials_L_3_post = behaviour.PLOT_trial_and_misses(Level_3_post)

x = np.array(range(len((Level_3_post))))
ax[1,1].bar(x, success_trials_L_3_post, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[1,1].bar(x, missed_trials_L_3_post, bottom = success_trials_L_3_post, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,1].set_title('Level 3 post surgery', fontsize = 13)
ax[1,1].set_ylabel('Trials / Session', fontsize = 10)
ax[1,1].set_xlabel('Sessions', fontsize = 10)
f.tight_layout()
f.subplots_adjust(top = 0.87)



#CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .tiff



#main folder rat ID
script_dir = os.path.join(hardrive_path +'Videogame_Assay/' + rat_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    
        

################################################################################################################################



rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv','F:/Videogame_Assay/AK_31.1_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2','AK 31.1', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3']


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_1 = np.zeros((len(RAT_ID),5),dtype=float)

for count, rat in enumerate(rat_summary_table_path):
       
    Level_1_6000 = prs.Level_1_paths_6000_3000(rat)
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_1_6000)
    trials_per_minutes_L_1 = np.array(total_trials)/np.array(session_length)
    
    if len(trials_per_minutes_L_1) == 5:
        rat_trial_min_Level_1[count,]=trials_per_minutes_L_1
    else:
        npad = 5 - len(trials_per_minutes_L_1)
        trials_per_minutes_L_1_padded = np.pad(trials_per_minutes_L_1, pad_width=(0, npad), mode='constant')
        rat_trial_min_Level_1[count,] = trials_per_minutes_L_1_padded


rat_trial_min_Level_1[rat_trial_min_Level_1 == 0] = np.nan


f = plt.figure(figsize=(20,10))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()

for count, row in enumerate(rat_trial_min_Level_1):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .8, label = RAT_ID[count])
    plt.title('Level 1 Trial/Min',fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    plt.legend()
    f.tight_layout()



    
f2 = plt.figure(figsize=(20,10))    

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


mean_trial_speed = np.nanmean(rat_trial_min_Level_1, axis=0)
stderr = stats.sem(rat_trial_min_Level_1, nan_policy='omit')

plt.plot(mean_trial_speed,marker = 'o',color= 'k',alpha = .8)
#plt.fill_between(range(5),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial_speed, yerr=stderr, fmt='o', ecolor='orangered',color='steelblue', capsize=2)  
plt.title('Level 1 AVG Trial/Min',fontsize = 16)
plt.ylabel('AVG Trial/Min', fontsize = 13)
plt.xlabel('Level 1 Sessions', fontsize = 13)
plt.legend()
f2.tight_layout()
                
####################################################################################################                
 

       
rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv','F:/Videogame_Assay/AK_31.1_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2','AK 31.1', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3']


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_2_pre = []

for count, rat in enumerate(rat_summary_table_path):
       
    Level_2_pre = prs.Level_2_pre_paths(rat)
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    trials_per_minutes_L_2_pre = np.array(total_trials)/np.array(session_length)
    rat_trial_min_Level_2_pre.append(trials_per_minutes_L_2_pre.tolist())

    

def find_max_list(list):
    list_len = [len(i) for i in list]
    print(max(list_len))


max_sessions = find_max_list(rat_trial_min_Level_2_pre)



#rat_trial_min[rat_trial_min==0] = np.nan


f3 = plt.figure(figsize=(20,10))
#f2 = plt.figure(figsize=(20,10))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()

for count, row in enumerate(rat_trial_min_Level_2_pre):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .8, label = RAT_ID[count])
    plt.title('Level 2 Trial/Min', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    plt.legend()
    f3.tight_layout()
    plt.legend()
    
    
f4 = plt.figure(figsize=(20,10))    

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()





    

mean_trial_speed_Level_2_pre = np.mean(np.array(rat_trial_min_Level_2_pre), axis=0)
stderr_Level_2_pre = stats.sem(rat_trial_min_Level_2_pre, nan_policy='omit')

plt.plot(mean_trial_speed_Level_2_pre, marker = 'o',color= 'k',alpha = .8)
#plt.fill_between(range(5),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(max_sessions, mean_trial_speed_Level_2_pre, yerr=stderr_Level_2_pre, fmt='o', ecolor='orangered',color='steelblue', capsize=2)
plt.title('Level 2 AVG Trial/Min',fontsize = 16)
plt.ylabel('AVG Trial/Min', fontsize = 13)
plt.xlabel('Level 2 Sessions', fontsize = 13)
plt.legend()
f4.tight_layout()
                
 












####################################################################################################

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv','F:/Videogame_Assay/AK_31.1_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2','AK 31.1', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3']



rat_summary_table_path=['F:/Videogame_Assay/AK_41.1_Pt.csv']

tot_trials = []

for count, rat in enumerate(rat_summary_table_path):
       
    Level_2_pre = prs.Level_2_pre_paths(rat)
    total_trials, session_length = calculate_trial_per_min(Level_2_pre)
    tot_trials.append(total_trials)
    touch_to_reward_speed_seconds= calculate_trial_speed_from_ball_touch(Level_2_pre)

f6 = plt.figure(figsize=(20,10))    

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()           
flat_list = [item for sublist in touch_to_reward_speed_seconds for item in sublist]
plt.vlines(range(len(flat_list)),flat_list, ymax=40)

positions=46
plt.axvline(x=positions)









####################################################################################################

figure_name = 'RAT_' + rat_ID + '_Trial_per_Minute.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Trial/Min'

f,ax = plt.subplots(2,2,figsize=(10,7))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()



#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

total_trials_L_1, session_length_L_1 = behaviour.PLOT_trial_per_min(Level_1)


trials_per_minutes_L_1 = np.array(total_trials_L_1)/np.array(session_length_L_1)
x = np.array(range(len((Level_1))))
ax[0,0].plot(x, trials_per_minutes_L_1, color ='r', marker = 'o', alpha = .8)
# Create green bars (middle), on top of the firs ones
#ax[0,0].bar(x, trials_per_minutes,  color ='r', edgecolor ='white', width = 1, alpha = .5)
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / min', fontsize = 10)
#ax[0,0].set_xlabel('Sessions')




total_trials_L_2_pre, session_length_L_2_pre = behaviour.PLOT_trial_per_min(Level_2_pre)

trials_per_minutes_L_2_pre = np.array(total_trials_L_2_pre)/np.array(session_length_L_2_pre)
x = np.array(range(len((Level_2_pre))))
ax[0,1].plot(x, trials_per_minutes_L_2_pre, color ='b', marker = 'o', alpha = .8)
#ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
#ax[0,1].set_ylabel('Trials / Session')
#ax[0,0].set_xlabel('Sessions')




total_trials_L_2_post, session_length_L_2_post = behaviour.PLOT_trial_per_min(Level_2_post)

trials_per_minutes_L_2_post = np.array(total_trials_L_2_post)/np.array(session_length_L_2_post)
x = np.array(range(len((Level_2_post))))
ax[1,0].plot(x, trials_per_minutes_L_2_post, color ='g', marker = 'o', alpha = .8)
# Create green bars (middle), on top of the firs ones
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
ax[1,0].set_ylabel('Trials / min', fontsize = 10)
ax[1,0].set_xlabel('Sessions', fontsize = 10)



#total_trials_L_3_pre ,session_length_L_3_pre = behaviour.PLOT_trial_per_min(Level_3_pre)

#x = np.array(range(len((Level_3_pre))))
#ax[1,0].bar(x, success_trials_L_3_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,0].bar(x, missed_trials_L_3_pre, bottom = success_trials_L_3_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 3 pre surgery')
#ax[1,0].set_ylabel('Trials / Session')
#ax[1,0].set_xlabel('Sessions')



total_trials_L_3_post, session_length_L_3_post = behaviour.PLOT_trial_per_min(Level_3_post)

trials_per_minutes_L_3_post = np.array(total_trials_L_3_post)/np.array(session_length_L_3_post)
x = np.array(range(len((Level_3_post))))

ax[1,1].plot(x, trials_per_minutes_L_3_post, color ='c', marker = 'o', alpha = .8)

#ax[1,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,1].set_title('Level 3 post surgery', fontsize = 13)
ax[1,1].set_ylabel('Trials / Session', fontsize = 10)
ax[1,1].set_xlabel('Sessions', fontsize = 10)
f.tight_layout()
f.subplots_adjust(top = 0.87)



#CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .tiff



#main folder rat ID
script_dir = os.path.join(hardrive_path +'Videogame_Assay/' + rat_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    