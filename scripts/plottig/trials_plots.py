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
rat_ID = 'AK_33.2'
rat_summary_table_path = r'F:/Videogame_Assay/AK_33.2_Pt.csv'



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

#plot trial time per each rat in a different colour

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv'
                          ,'F:/Videogame_Assay/AK_46.2_IrO2.csv','F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9','#C0C0C0','#B0C4DE']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3','AK 46.2','AK 50.1','AK 50.2']

sessions_to_consider = 4


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_1 = np.zeros((len(RAT_ID),sessions_to_consider),dtype=float)

for count, rat in enumerate(rat_summary_table_path):
       
    Level_1_6000 = prs.Level_1_paths_6000_3000(rat)
    Level_1_6000 = Level_1_6000[:sessions_to_consider]
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_1_6000)
    trials_per_minutes_L_1 = np.array(total_trials)/np.array(session_length)
    rat_trial_min_Level_1[count,]=trials_per_minutes_L_1
    print(count)



   
#    if len(trials_per_minutes_L_1) == 5:
#        rat_trial_min_Level_1[count,]=trials_per_minutes_L_1
#    else:
#        npad = 5 - len(trials_per_minutes_L_1)
#        trials_per_minutes_L_1_padded = np.pad(trials_per_minutes_L_1, pad_width=(0, npad), mode='constant')
#        rat_trial_min_Level_1[count,] = trials_per_minutes_L_1_padded


rat_trial_min_Level_1[rat_trial_min_Level_1 == 0] = np.nan


# PLOT AND SAVE SUMMARY FIGURE OF TRIAL/MIN LEVEL 1

hardrive_path = r'F:/' 

#figure_name = 'Summary_Trial_per_Min_Level_1_6000.pdf'
figure_name = 'Summary_Trial_per_Min_Level_1_6000.png'
plot_main_title = 'Trial_per_Min_Level_1_6000'


f = plt.figure(figsize=(20,10))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


for count, row in enumerate(rat_trial_min_Level_1):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 1 Trial/Min',fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    plt.xticks((np.arange(0, 5, 1)))
    plt.xlim(-0.1,3.5)
    plt.legend()
    f.tight_layout()


#SAVING


script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    


#PLOT AND SAVE SUMMARY PLOT OF AVG TRIAL/MIN LEVEL 1


hardrive_path = r'F:/' 

#figure_name = 'Summary_Trial_per_Min_Level_1_6000.pdf'
figure_name = 'Summary_Trial_per_Min_Level_1_6000_with_SEM.png'
plot_main_title = 'Trial_per_Min_Level_1_6000'




f = plt.figure(figsize=(20,10))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()
colours = ['#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC', '#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC','#DCDCDC']
for count, row in enumerate(rat_trial_min_Level_1):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .4, label = RAT_ID[count])
    plt.title('Level 1 Trial/Min',fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    plt.xticks((np.arange(0, 5, 1)))
    plt.xlim(-0.1,3.5)
    plt.legend()
    f.tight_layout()





#plot only mean and standard error 


#figure_name = 'Summary_AVG_Trial_per_Min_Level_1_6000.pdf'
figure_name = 'Summary_AVG_Trial_per_Min_Level_1_6000.png'
plot_main_title_f = 'AVG_Trial_per_Min_Level_1_6000'

    
f2 = plt.figure(figsize=(20,10))    

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


mean_trial_speed = np.nanmean(rat_trial_min_Level_1, axis=0)

sem = stats.sem(rat_trial_min_Level_1, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k',alpha = .8)
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(4), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  
 
plt.title('Level 1 AVG Trial/Min',fontsize = 16)
plt.ylabel('AVG Trial/Min', fontsize = 13)
plt.xlabel('Level 1 Sessions', fontsize = 13)
plt.xticks((np.arange(0, 5, 1)))
plt.legend()
f2.tight_layout()




#CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .pdf

hardrive_path = r'F:/' 
#main folder rats
script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .pdf
f2.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    
                
####################################################################################################                
 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv'
                          ,'F:/Videogame_Assay/AK_46.2_IrO2.csv','F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9','#C0C0C0','#B0C4DE']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3','AK 46.2','AK 50.1','AK 50.2']


sessions_to_consider =5
#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_2_pre = np.zeros((len(RAT_ID),sessions_to_consider),dtype=float)

for count, rat in enumerate(rat_summary_table_path):
       
    Level_2_pre = prs.Level_2_pre_paths(rat)
    Level_2_pre = Level_2_pre[:sessions_to_consider]
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    trials_per_minutes_L_2_pre = np.array(total_trials)/np.array(session_length)
    rat_trial_min_Level_2_pre[count,]=trials_per_minutes_L_2_pre
    print(count)
    

def find_max_list(list):
    list_len = [len(i) for i in list]
    print(max(list_len))




#PLOT AND SAVE TRIAL/MIN LEVEL 2


#figure_name = 'Summary_Trial_per_Min_Level_2.pdf'

#figure_name = 'Summary_Trial_per_Min_Level_2.png'

figure_name = 'Summary_Trial_per_Min_Level_2_SEM.png'
plot_main_title_ = 'Trial_per_Min_Level_2'

f3 = plt.figure(figsize=(20,10))


sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()

for count, row in enumerate(rat_trial_min_Level_2_pre):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 2 Trial/Min', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    plt.xticks((np.arange(0, 5, 1)))
    plt.xlim(-0.1,4.5)
    plt.legend()
    f3.tight_layout()

 
    
 
    
    
    
mean_trial_speed = np.nanmean(rat_trial_min_Level_2_pre, axis=0)

sem = stats.sem(rat_trial_min_Level_2_pre, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k',alpha = .8)
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2) 
hardrive_path = r'F:/' 
#main folder rats
script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .pdf
f3.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)   
    
    
#PLOT AVG TRIAL/MIN LEVEL 2   
#get list of list ready and padded with nan   
        
def boolean_indexing(v, fillval=np.nan):
   lens = np.array([len(item) for item in v])
   mask = lens[:,None] > np.arange(lens.max())
   out = np.full(mask.shape,fillval)
   out[mask] = np.concatenate(v)
   return out

rat_trial_min_Level_2_pre_array= boolean_indexing(rat_trial_min_Level_2_pre, fillval=np.nan)

mean_trial_speed_Level_2_pre =  np.nanmean(rat_trial_min_Level_2_pre_array, axis=0) 
stderr_Level_2_pre = stats.sem(rat_trial_min_Level_2_pre_array, nan_policy='omit')



#plot avg trial/min level 2 issues with sem


f4 = plt.figure(figsize=(20,10))    


figure_name = 'Summary_AVG_Trial_per_Min_Level_2.pdf'
plot_main_title_f = 'AVG_Trial_per_Min_Level_2'


sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


plt.plot(mean_trial_speed_Level_2_pre, marker = 'o',color= 'steelblue',alpha = .8)
#plt.fill_between(range(5),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(len(np.max(rat_trial_min_Level_2_pre))), mean_trial_speed_Level_2_pre, yerr=stderr_Level_2_pre, fmt='o', ecolor='orangered',color='steelblue', capsize=2)
plt.title('Level 2 AVG Trial/Min',fontsize = 16)
plt.ylabel('AVG Trial/Min', fontsize = 13)
plt.xlabel('Level 2 Sessions', fontsize = 13)
plt.legend()
f4.tight_layout()



hardrive_path = r'F:/' 
#main folder rats
script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .pdf
f4.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)    










####################################################################################################

#PLOT SPEED FROM TOUCH TO REWARD per rat all the sessions

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv'
                          ,'F:/Videogame_Assay/AK_46.2_IrO2.csv','F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9','#C0C0C0','#B0C4DE']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3','AK 46.2','AK 50.1','AK 50.2']



for count, rat in enumerate(rat_summary_table_path):
    
    tot_trials = []
    Level_2_pre = prs.Level_2_pre_paths(rat)
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    tot_trials.append(total_trials)
    touch_to_reward_speed_seconds = behaviour.calculate_trial_speed_from_ball_touch(Level_2_pre)

    total_trials_array = np.array(tot_trials)
    flat_list = [item for sublist in touch_to_reward_speed_seconds for item in sublist]
    vertical_lines =  np.cumsum(total_trials_array) + .5

    figure_name = 'RAT_'+ RAT_ID[count] + '_Touch_to_Reward_Speed.pdf'
    plot_main_title =  RAT_ID[count] + 'Touch_to_Reward_Speed'
    
    fig = plt.figure(figsize=(20,5))    

    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine()           

    plt.plot(range(len(flat_list)), flat_list, 'o' , color = '#1E90FF', alpha = .4, markersize = 3)
    plt.xlim(0,len(flat_list))
    plt.ylim(0,50)
    plt.xticks((np.arange(0, len(flat_list), 50)))
    plt.ylabel('Time (s)', fontsize = 13)
    plt.xlabel('Trials/Session', fontsize = 13) 
    plt.suptitle('Level 2 Touch_to_Reward_Speed',fontsize = 16)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)

    for i in vertical_lines:
    
        plt.axvline(x = i , color='k', linestyle='--',linewidth =.5)
        #plt.text(i-25,45,'Session%d' %count, ha='right',va='center',fontsize=10)
        #axvspan

    hardrive_path = r'F:/' 
    script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + RAT_ID[count])
    #create a folder where to store the plots 
    main_folder = os.path.join(script_dir +'/Summary')
    #create a folder where to save the plots
    results_dir = os.path.join(main_folder + '/Behaviour/')


    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #save the fig in .tiff
    fig.savefig(results_dir + figure_name, transparent=True)
    

####Level 2_speed touch to end and std with sliding window 1 step 10 samples

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv'
                          ,'F:/Videogame_Assay/AK_46.2_IrO2.csv','F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9','#C0C0C0','#B0C4DE']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3','AK 46.2','AK 50.1','AK 50.2']

RAT_ID =  ['AK 50.2']   
rat_summary_table_path =['F:/Videogame_Assay/AK_50.2_behaviour_only.csv']
for count, rat in enumerate(rat_summary_table_path):
    
    tot_trials = []
    session_std_40_included = []
    session_std_40_excluded  = []
    window=10

    Level_2_pre = prs.Level_2_pre_paths(rat)
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    tot_trials.append(total_trials)
    touch_to_reward_speed_seconds = behaviour.calculate_trial_speed_from_ball_touch(Level_2_pre)

    total_trials_array = np.array(tot_trials)
    flat_list = [item for sublist in touch_to_reward_speed_seconds for item in sublist]
    flat_array = np.array(flat_list)
    

    for i in np.arange(len(flat_list)-window):
        create_slice = flat_array[i:i+window]        
        slice_std = np.nanstd(create_slice)
        session_std_40_included.append(slice_std)
        
    for i in np.arange(len(flat_list)-window):
        flat_array[flat_array>=40] = np.NaN
        create_slice_wo_40 = flat_array[i:i+window]      
        slice_std_wo_40 = np.nanstd(create_slice_wo_40)
        session_std_40_excluded.append(slice_std_wo_40)

    vertical_lines =  np.cumsum(total_trials_array) + .5

    figure_name = 'RAT_'+ RAT_ID[count] + '_Touch_to_Reward_Speed_STD.png'
    plot_main_title =  RAT_ID[count] + 'Touch_to_Reward_Speed_STD'
    
    fig = plt.figure(figsize=(20,5))    

    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine()           
    
    
    
    #plt.plot(range(len(flat_list)), flat_list, 'o' , color = '#1E90FF', alpha = .4, markersize = 3)
    coorected_x = np.zeros(window//2,)
    coorected_x.fill(np.NaN)
    stack = np.hstack((coorected_x,session_std_40_excluded))

    plt.plot(stack,'-',color='#228B22',alpha =.5)     
    plt.plot(range(len(flat_list)), flat_list, 'o' , color = '#1E90FF', alpha = .4, markersize = 3)
    plt.xlim(0,len(flat_list))
    plt.ylim(0,50)
    plt.xticks((np.arange(0, len(flat_list), 50)))
    plt.ylabel('Time (s)', fontsize = 13)
    plt.xlabel('Trials/Session', fontsize = 13) 
    plt.suptitle('Level 2 Touch_to_Reward_Speed_STD',fontsize = 16)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)

    for i in vertical_lines:
    
        plt.axvline(x = i , color='k', linestyle='--',linewidth =.5)
        #plt.text(i-25,45,'Session%d' %count, ha='right',va='center',fontsize=10)
        #axvspan




    hardrive_path = r'F:/' 
    script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + RAT_ID[count])
    #create a folder where to store the plots 
    main_folder = os.path.join(script_dir +'/Summary')
    #create a folder where to save the plots
    results_dir = os.path.join(main_folder + '/Behaviour/')


    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #save the fig in .tiff
    fig.savefig(results_dir + figure_name, transparent=True)


########level 1



for count, rat in enumerate(rat_summary_table_path):
    
    tot_trials = []
    Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat)
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_1_6000_3000)
    tot_trials.append(total_trials)
    full_trial_speed = behaviour.calculate_full_trial_speed(Level_1_6000_3000)

    total_trials_array = np.array(tot_trials)
    flat_list = [item for sublist in full_trial_speed for item in sublist]
    vertical_lines =  np.cumsum(total_trials_array) + .5

    figure_name = 'RAT_'+ RAT_ID[count] + '_full_trial_speed_Level_1.pdf'
    plot_main_title =  RAT_ID[count] + 'full_trial_speed'
    
    fig = plt.figure(figsize=(20,5))    

    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine()           

    plt.plot(range(len(flat_list)), flat_list, 'o' , color = '#32CD32', alpha = .4, markersize = 3)
    plt.xlim(0,len(flat_list))
    plt.ylim(0,50)
    plt.xticks((np.arange(0, len(flat_list), 50)))
    plt.ylabel('Time (s)', fontsize = 13)
    plt.xlabel('Trials/Session', fontsize = 13) 
    plt.suptitle('Level 1 full_trial_speed',fontsize = 16)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)

    for i in vertical_lines:
    
        plt.axvline(x = i , color='k', linestyle='--',linewidth =.5)
        #plt.text(i-25,45,'Session%d' %count, ha='right',va='center',fontsize=10)
        #axvspan

    hardrive_path = r'F:/' 
    script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + RAT_ID[count])
    #create a folder where to store the plots 
    main_folder = os.path.join(script_dir +'/Summary')
    #create a folder where to save the plots
    results_dir = os.path.join(main_folder + '/Behaviour/')


    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #save the fig in .tiff
    fig.savefig(results_dir + figure_name, transparent=True)
























#####################################################################################################
#
#figure_name = 'RAT_' + rat_ID + '_Trial_per_Minute.pdf'
#plot_main_title = 'RAT ' + rat_ID + ' Trial/Min'
#
#f,ax = plt.subplots(2,2,figsize=(10,7))
#f.suptitle(plot_main_title)
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine()
#
#
#
##CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG
#
#total_trials_L_1, session_length_L_1 = behaviour.PLOT_trial_per_min(Level_1)
#
#
#trials_per_minutes_L_1 = np.array(total_trials_L_1)/np.array(session_length_L_1)
#x = np.array(range(len((Level_1))))
#ax[0,0].plot(x, trials_per_minutes_L_1, color ='r', marker = 'o', alpha = .8)
## Create green bars (middle), on top of the firs ones
##ax[0,0].bar(x, trials_per_minutes,  color ='r', edgecolor ='white', width = 1, alpha = .5)
#ax[0,0].set_title('Level 1', fontsize = 13)
#ax[0,0].set_ylabel('Trials / min', fontsize = 10)
##ax[0,0].set_xlabel('Sessions')
#
#
#
#
#total_trials_L_2_pre, session_length_L_2_pre = behaviour.PLOT_trial_per_min(Level_2_pre)
#
#trials_per_minutes_L_2_pre = np.array(total_trials_L_2_pre)/np.array(session_length_L_2_pre)
#x = np.array(range(len((Level_2_pre))))
#ax[0,1].plot(x, trials_per_minutes_L_2_pre, color ='b', marker = 'o', alpha = .8)
##ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
##ax[0,1].set_ylabel('Trials / Session')
##ax[0,0].set_xlabel('Sessions')
#
#
#
#
#total_trials_L_2_post, session_length_L_2_post = behaviour.PLOT_trial_per_min(Level_2_post)
#
#trials_per_minutes_L_2_post = np.array(total_trials_L_2_post)/np.array(session_length_L_2_post)
#x = np.array(range(len((Level_2_post))))
#ax[1,0].plot(x, trials_per_minutes_L_2_post, color ='g', marker = 'o', alpha = .8)
## Create green bars (middle), on top of the firs ones
##ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
#ax[1,0].set_ylabel('Trials / min', fontsize = 10)
#ax[1,0].set_xlabel('Sessions', fontsize = 10)
#
#
#
##total_trials_L_3_pre ,session_length_L_3_pre = behaviour.PLOT_trial_per_min(Level_3_pre)
#
##x = np.array(range(len((Level_3_pre))))
##ax[1,0].bar(x, success_trials_L_3_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
### Create green bars (middle), on top of the firs ones
##ax[1,0].bar(x, missed_trials_L_3_pre, bottom = success_trials_L_3_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
##ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
##ax[1,0].set_title('Level 3 pre surgery')
##ax[1,0].set_ylabel('Trials / Session')
##ax[1,0].set_xlabel('Sessions')
#
#
#
#total_trials_L_3_post, session_length_L_3_post = behaviour.PLOT_trial_per_min(Level_3_post)
#
#trials_per_minutes_L_3_post = np.array(total_trials_L_3_post)/np.array(session_length_L_3_post)
#x = np.array(range(len((Level_3_post))))
#
#ax[1,1].plot(x, trials_per_minutes_L_3_post, color ='c', marker = 'o', alpha = .8)
#
##ax[1,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,1].set_title('Level 3 post surgery', fontsize = 13)
#ax[1,1].set_ylabel('Trials / Session', fontsize = 10)
#ax[1,1].set_xlabel('Sessions', fontsize = 10)
#f.tight_layout()
#f.subplots_adjust(top = 0.87)
#
#
#
##CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .tiff
#
#
#
##main folder rat ID
#script_dir = os.path.join(hardrive_path +'Videogame_Assay/' + rat_ID)
##create a folder where to store the plots 
#main_folder = os.path.join(script_dir +'/Summary')
##create a folder where to save the plots
#results_dir = os.path.join(main_folder + '/Behaviour/')
#
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
##save the fig in .tiff
#f.savefig(results_dir + figure_name, transparent=True)
##f.savefig(results_dir + figure_name)      
    