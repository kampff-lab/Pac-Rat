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
from scipy import stats

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



f,ax = plt.subplots(2,2,figsize=(10,7))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name = 'RAT_' + rat_ID + '_' +'_trail_count_level_1_and_2.pdf'

#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

success_trials_L_1, missed_trials_L_1 = behaviour.calculate_trial_and_misses(Level_1)

x = np.array(range(len((Level_1))))

ax[0,0].bar(x, success_trials_L_1, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,0].bar(x, missed_trials_L_1, bottom = success_trials_L_1, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
ax[0,0].legend(loc ='best', frameon=False , fontsize = 'x-small') #ncol=2
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / Session', fontsize = 10)
ax[0,0].set_ylim(ymin= 0,ymax= 300)
plt.xticks((np.arange(0, 300, 50)))
#ax[0,0].tick_params(axis='x',which='both',bottom=False)
#ax[0,0].set_xlabel('Sessions')
# Hide the right and top spines
#ax[0,0].spines['right'].set_visible(False)
#ax[0,0].spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax[0,0].yaxis.set_ticks_position('left')
ax[0,0].xaxis.set_ticks_position('bottom')



success_trials_L_2_pre, missed_trials_L_2_pre = behaviour.calculate_trial_and_misses(Level_2_pre)
#33.2 only due to double session 9/4/2018
#39+33 = 72 trials and no misses

#success_trials_L_2_pre = [38,124,77,72,65,87,154,118]
#missed_trials_L_2_pre = [7,0,1,0,1,3,4,2]


x = np.array(range(len((success_trials_L_2_pre)))) 
ax[0,1].bar(x, success_trials_L_2_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,1].bar(x, missed_trials_L_2_pre, bottom = success_trials_L_2_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
ax[0,1].set_ylim(ymin= 0,ymax= 150)
plt.xticks((np.arange(0, 150, 50)))
#ax[0,1].set_ylabel('Trials / Session')
#ax[0,0].set_xlabel('Sessions')
ax[0,1].yaxis.set_ticks_position('left')
ax[0,1].xaxis.set_ticks_position('bottom')

f.savefig(results_dir + figure_name, transparent=True)
          
          
          
#
#
#success_trials_L_2_post, missed_trials_L_2_post = behaviour.calculate_trial_and_misses(Level_2_post)
#
#x = np.array(range(len((Level_2_post))))
#ax[1,0].bar(x, success_trials_L_2_post, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,0].bar(x, missed_trials_L_2_post, bottom = success_trials_L_2_post, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
##ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
#ax[1,0].set_ylabel('Trials / Session', fontsize = 10)
#ax[1,0].set_xlabel('Sessions', fontsize = 10)
#
#
#
##success_trials_L_3_pre ,missed_trials_L_3_pre = behaviour.PLOT_trial_and_misses(Level_3_pre)
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
#success_trials_L_3_post, missed_trials_L_3_post = behaviour.calculate_trial_and_misses(Level_3_moving)
#
#x = np.array(range(len((Level_3_moving))))
#ax[1,1].bar(x, success_trials_L_3_post, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,1].bar(x, missed_trials_L_3_post, bottom = success_trials_L_3_post, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
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
#    
#        

################################################################################################################################

#USED FR FINAL THESIS PLOTS -- TRIAL/MIN

#LEVEL 1
#plot trial time per each rat in a different colour 



rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']



colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

sessions_to_consider = 4


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_1 = np.zeros((len(RAT_ID),sessions_to_consider),dtype=float)

for count, rat in enumerate(rat_summary_table_path):
       
    Level_1_6000 = prs.Level_1_paths_6000_3000(rat)
    Level_1_6000 = Level_1_6000[:sessions_to_consider]
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_1_6000)
    success_trials, missed_trials = behaviour.calculate_trial_and_misses(Level_1_6000)
    
    #trials_per_minutes_L_1 = np.array(total_trials)/np.array(session_length)
    trials_per_minutes_L_1 = np.array(success_trials)/np.array(session_length)
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

#hardrive_path = r'F:/' 

#figure_name = 'Summary_Trial_per_Min_Level_1_6000.pdf'
#figure_name = 'Summary_Trial_per_Min_Level_1_6000.png'

#f,ax = plt.subplots(figsize=(10,7))
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#for count, row in enumerate(rat_trial_min_Level_1):
#    
#    
#    sns.lineplot(data=row, marker= 'o',palette = colours, hue=10)
#    
#    

figure_name =  '_Trial_rewarded_per_Session_colour_sem_level1.pdf'
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

for count, row in enumerate(rat_trial_min_Level_1):    
    
  
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 1 Trial/Min',fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    plt.xticks((np.arange(0, 5, 1)))
    ax.axes.get_xaxis().set_visible(True) 
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim(-0.1,3.5)
    plt.ylim(-0.2,6)


mean_trial_speed = np.nanmean(rat_trial_min_Level_1, axis=0)

sem = stats.sem(rat_trial_min_Level_1, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(4), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  

#plt.legend()
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)

        
#t test level 1



t_test = stats.ttest_rel(rat_trial_min_Level_1[:,0],rat_trial_min_Level_1[:,3])
#Ttest_relResult(statistic=-4.348986156425727, pvalue=0.0011573324303187724)

t_test_rewarded = stats.ttest_rel(rat_trial_min_Level_1[:,0],rat_trial_min_Level_1[:,3])
#Ttest_relResult(statistic=-4.1616123362819275, pvalue=0.0015850232246721154)


target = open(main_folder +"stats_REWARDED_ONLY_Level_1_trial_per_min.txt", 'w')
target.writelines(str(t_test_rewarded)+' LEVEL 1: day 1 Vs day 4, PLOT: trial rewarded/min mean +- SEM, trials_plot.py')
target.close()






#############################################################################################################

#script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
##create a folder where to store the plots 
#main_folder = os.path.join(script_dir +'/Summary')
##create a folder where to save the plots
#results_dir = os.path.join(main_folder + '/Behaviour/')
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
##save the fig in .tiff
#f.savefig(results_dir + figure_name, transparent=True)
##f.savefig(results_dir + figure_name)      
#    


#PLOT AND SAVE SUMMARY PLOT OF AVG TRIAL/MIN LEVEL 1



#figure_name = 'Summary_Trial_per_Min_Level_1_6000.pdf'
figure_name = 'Summary_Trial_per_Min_Level_1_6000_with_SEM.pdf'
#plot_main_title = 'Trial_per_Min_Level_1_6000'




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
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim(-0.1,3.5)
    plt.legend()
    f.tight_layout()





#plot only mean and standard error 


#figure_name = 'Summary_AVG_Trial_per_Min_Level_1_6000.pdf'
figure_name = 'Summary_AVG_Trial_per_Min_Level_1_6000.png'
#plot_main_title_f = 'AVG_Trial_per_Min_Level_1_6000'

    
f2 = plt.figure(figsize=(20,10))    

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


mean_trial_speed = np.nanmean(rat_trial_min_Level_1, axis=0)

sem = stats.sem(rat_trial_min_Level_1, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(4), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  
 
plt.title('Level 1 AVG Trial/Min',fontsize = 16)
plt.ylabel('AVG Trial/Min', fontsize = 13)
plt.xlabel('Level 1 Sessions', fontsize = 13)
plt.xticks((np.arange(0, 5, 1)))
plt.legend()
f2.tight_layout()




#CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .pdf

#hardrive_path = r'F:/' 
##main folder rats
#script_dir = os.path.join(hardrive_path +'Videogame_Assay/')
##create a folder where to store the plots 
#main_folder = os.path.join(script_dir +'/Summary')
##create a folder where to save the plots
#results_dir = os.path.join(main_folder + '/Behaviour/')
#
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
##save the fig in .pdf
#f2.savefig(results_dir + figure_name, transparent=True)
##f.savefig(results_dir + figure_name)      
    
                
####################################################################################################                



#USED FOR THESIS PLOT TRIAL/MIN LEVEL 2

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


##33.2 and 50.1 double sessions
#33.2  9/4/18 day  4  2.1215990570670855 trial/min = 72/33.93666666666667 //// real session 5 = 1.65885
#50.1 5/11/19 day  4   0.9246395423501256 trial/min     ///////real session 5 =  1.16385
sessions_to_consider = 6 

double_sessions = [rat_summary_table_path[0], rat_summary_table_path[10]]
rat_trial_min_double = np.zeros((len(double_sessions),sessions_to_consider),dtype=float)


for d, double in enumerate(double_sessions):
    
    Level_2_pre = prs.Level_2_pre_paths(double)
    Level_2_pre = Level_2_pre[:sessions_to_consider]
    total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    #success_trials, missed_trials = behaviour.calculate_trial_and_misses(Level_1_6000)
    

    trials_per_minutes_double = np.array(total_trials)/np.array(session_length)
    rat_trial_min_double[d,]=trials_per_minutes_double
    print(d)









    
Level_2_pre = prs.Level_2_pre_paths(double_sessions[0])
Level_2_pre = Level_2_pre[:sessions_to_consider]
total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
    

double_trials = total_trials[3] + total_trials[4]
double_length =  session_length[3] + session_length[4]
trial_per_min =   double_trials/double_length  


AK_33_2 = [0.94552989, 1.92502992, 1.77768211, 2.25678119, 1.9813374 ,1.65885298]
AK_50_1= [0.2848259 , 0.5090041 , 1.03106944, 1.05609455, 0.83027912,1.16384894]




#33.2, 50.1
#position 0 
array_33_2 = [0.94552989, 1.92502992, 1.77768211, 2.12159905 , 1.65885298]
#position 10
array_50_1 = [0.2848259 , 0.5090041 , 1.03106944, 0.92463954 , 1.16384894]


#exclude 33.2 and 50.1 which have double days


rat_summary_selection = [ 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv',
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                         'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

sessions_to_consider =5
#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
rat_trial_min_Level_2_pre = np.zeros((len(rat_summary_selection),sessions_to_consider),dtype=float)



for count, rat in enumerate(rat_summary_selection):
    try:    
        Level_2_pre = prs.Level_2_pre_paths(rat)
        Level_2_pre = Level_2_pre[:sessions_to_consider]
        total_trials, session_length = behaviour.calculate_trial_per_min(Level_2_pre)
        success_trials, missed_trials = behaviour.calculate_trial_and_misses(Level_2_pre)
    
        trials_per_minutes_L_2_pre = np.array(success_trials)/np.array(session_length)
        #trials_per_minutes_L_2_pre = np.array(total_trials)/np.array(session_length)
        rat_trial_min_Level_2_pre[count,]=trials_per_minutes_L_2_pre
        print(count)
    except Exception: 
        continue    


final_array = np.insert(rat_trial_min_Level_2_pre, 0, array_33_2, 0) 


rat_trial_min_Level_2_pre_final = np.insert(final_array, 10, array_50_1, 0) 





#t test 


t_test_trial_per_min_Level_2 = stats.ttest_rel(rat_trial_min_Level_2_pre_final[:,0],rat_trial_min_Level_2_pre_final[:,4])
#Ttest_relResult(statistic=-7.4017486831911725, pvalue=1.3568982038567815e-05)

t_test_trial_per_min_Level_2_rewarded = stats.ttest_rel(rat_trial_min_Level_2_pre_final[:,0],rat_trial_min_Level_2_pre_final[:,4])
#Ttest_relResult(statistic=-7.345068197447017, pvalue=1.4573202538736475e-05)


target = open(main_folder +"stats_level_2_trial_rewarded_per_minutes.txt", 'w')
target.writelines(str(t_test_trial_per_min_Level_2_rewarded) +' LEVEL 2: day 1 Vs day 5, PLOT: trial rewarded /min  +- SEM, trials_plot.py')

target.close()




def find_max_list(list):
    list_len = [len(i) for i in list]
    print(max(list_len))




#PLOT AND SAVE TRIAL/MIN LEVEL 2


#figure_name = 'Summary_Trial_per_Min_Level_2.pdf'

#figure_name = 'Summary_Trial_per_Min_Level_2.png'

figure_name = 'Summary_Trial_rewarded_per_Min_Level_2_with_sem.pdf'
plot_main_title_ = 'Trial_per_Min_Level_2'

    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for count, row in enumerate(rat_trial_min_Level_2_pre_final):
    
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 2 Trial/Min', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    plt.xticks((np.arange(0, 5, 1)))
    plt.xlim(-0.1,4.5)
    plt.yticks((np.arange(0, 4, .5)))
   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylim(0,3)
    #plt.legend()
    f.tight_layout()


#f.savefig(results_dir + figure_name, transparent=True)       
    
mean_trial_speed = np.nanmean(rat_trial_min_Level_2_pre_final, axis=0)

sem = stats.sem(rat_trial_min_Level_2_pre_final, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2) 




#SAVING
f.savefig(results_dir + figure_name, transparent=True)    





#######################################################################################################################################
#trial count success VS misses


rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']



colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']

sessions_to_consider = 4


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
success_L_1 = np.zeros((len(RAT_ID),sessions_to_consider),dtype=float)
miss_L_1 = np.zeros((len(RAT_ID),sessions_to_consider),dtype=float)


for count, rat in enumerate(rat_summary_table_path):
       
    Level_1_6000 = prs.Level_1_paths_6000_3000(rat)
    Level_1_6000 = Level_1_6000[:sessions_to_consider]
    success_trials_L_1, missed_trials_L_1 = behaviour.calculate_trial_and_misses(Level_1_6000)
    
    success_L_1[count,]=success_trials_L_1
    miss_L_1[count,]= missed_trials_L_1
    print(count)







figure_name =  '_trial_count_level1.pdf'
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

for count, row in enumerate(success_L_1):    
    
  
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 1 Trial count',fontsize = 16)
    plt.ylabel('trial number', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    ax.axes.get_xaxis().set_visible(True) 
    #ax.set_ylim(ymin= -10 ,ymax= 260)
    #plt.yticks((np.arange(0, 300, 50)))
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(0,160)


mean_trial_count = np.nanmean(success_L_1, axis=0)

sem = stats.sem(success_L_1, nan_policy='omit', axis=0)


plt.plot(mean_trial_count,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(sessions_to_consider), mean_trial_count, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  

#plt.legend()
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)

#t test level 2

t_test_success = stats.ttest_rel(success_L_1[:,0],success_L_1[:,3])
#Ttest_relResult(statistic=-3.292444121706258, pvalue=0.007173538082732699)




target = open(main_folder +"stats_level_1_success_trial.txt", 'w')
target.writelines(str(t_test_success) +' LEVEL 1: day 1 Vs day 4, PLOT: success trial mean +- SEM, trials_plot.py')

target.close()






#########missed

figure_name =  '_missed_trial_level1.pdf'
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

for count, row in enumerate(miss_L_1):    
    
  
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 1 Trial count',fontsize = 16)
    plt.ylabel('trial number', fontsize = 13)
    plt.xlabel('Level 1 Sessions', fontsize = 13)
    plt.ylim(0,80)
    plt.yticks((np.arange(0, 80, 20)))
    ax.axes.get_xaxis().set_visible(True) 
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(0,6)


mean_trial_miss = np.nanmean(miss_L_1, axis=0)

sem = stats.sem(miss_L_1, nan_policy='omit', axis=0)


plt.plot(mean_trial_miss,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(4), mean_trial_miss, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2)  

#plt.legend()
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)


t_test_miss = stats.ttest_rel(miss_L_1[:,0],miss_L_1[:,3])
#Ttest_relResult(statistic=2.4770754201107477, pvalue=0.03073292290519083)


target = open(main_folder +"stats_level_1_miss_trial.txt", 'w')
target.writelines(str(t_test_miss) +' LEVEL 1 1: day 1 Vs day 4, PLOT: miss trial mean +- SEM, trials_plot.py')

target.close()



#################level 2 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


##33.2 and 50.1 double sessions
#33.2  9/4/18 day  4  2.1215990570670855 trial/min = 72/33.93666666666667 //// real session 5 = 1.65885
#50.1 5/11/19 day  4   0.9246395423501256 trial/min     ///////real session 5 =  1.16385
sessions_to_consider = 6 

double_sessions = [rat_summary_table_path[0], rat_summary_table_path[10]]

success_double = np.zeros((len(RAT_ID[:2]),sessions_to_consider),dtype=float)
miss_double = np.zeros((len(RAT_ID[:2]),sessions_to_consider),dtype=float)


for d, double in enumerate(double_sessions):
    
    Level_2_pre = prs.Level_2_pre_paths(double)
    Level_2_pre = Level_2_pre[:sessions_to_consider]
    success_d, missed_d = behaviour.calculate_trial_and_misses(Level_2_pre)
        
    success_double[d,]=success_d
    miss_double[d,]=missed_d
    print(d)


success_double= [[ 38., 124.,  77.,  39.,  33.,  65.],
                [ 13.,  24.,  46.,  21.,  22.,  73.]]
    
    
    
success_double_correct_33_2 = [38.,124.,77.,72.,65.]  
success_double_correct_50_1 = [13.,24.,46.,43.,73.]


miss_double = ([[7., 0., 1., 0., 0., 1.],
                [2., 2., 3., 0., 1., 0.]])

    
miss_double_correct_33_2= [7.,0.,1.,0.,1.]
miss_double_correct_50_1= [2.,2.,3.,1.,0.]
    



rat_summary_selection = [ 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv',
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                         'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

sessions_to_consider = 5


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
success_L_2 = np.zeros((len(rat_summary_selection),sessions_to_consider),dtype=float)
miss_L_2 = np.zeros((len(rat_summary_selection),sessions_to_consider),dtype=float)


for count, rat in enumerate(rat_summary_selection):
       
    Level_2_pre = prs.Level_2_pre_paths(rat)
    Level_2_pre = Level_2_pre[:sessions_to_consider]
    success_trials_L_2_pre, missed_trials_L_2_pre = behaviour.calculate_trial_and_misses(Level_2_pre)
    
    success_L_2[count,]=success_trials_L_2_pre
    miss_L_2[count,]= missed_trials_L_2_pre
    print(count)




final_success = np.insert(success_L_2, 0, success_double_correct_33_2, 0) 
final_success_L_2 = np.insert(final_success, 10, success_double_correct_50_1, 0) 



final_miss = np.insert(miss_L_2, 0, miss_double_correct_33_2, 0) 
final_miss_L_2 = np.insert(final_miss, 10, miss_double_correct_50_1, 0) 




#######level 2 success plot


figure_name =  '_trial_count_level2.pdf'
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

for count, row in enumerate(final_success_L_2):    
    
  
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 2 Trial count',fontsize = 16)
    plt.ylabel('trial number', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    ax.axes.get_xaxis().set_visible(True) 
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(0,150)
    plt.xticks((np.arange(0, 150, 50)))
    ax.set_ylim(ymin= 0,ymax= 150)


mean_trial_count_L_2 = np.nanmean(final_success_L_2, axis=0)

sem_L_2 = stats.sem(final_success_L_2, nan_policy='omit', axis=0)


plt.plot(mean_trial_count_L_2,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(sessions_to_consider), mean_trial_count_L_2, yerr= sem_L_2, fmt='o', ecolor='k',color='k', capsize=2)  

#plt.legend()
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)

#t test level 2

t_test_success_L_2 = stats.ttest_rel(final_success_L_2[:,0],final_success_L_2[:,4])
#Ttest_relResult(statistic=-5.1325785753223885, pvalue=0.0003269935176528684)


target = open(main_folder +"stats_level_2_success_trial.txt", 'w')
target.writelines(str(t_test_success_L_2) +' LEVEL 2: day 1 Vs day 5, PLOT: success trial mean +- SEM, trials_plot.py')

target.close()


##################################miss level 2



figure_name =  '_missed_trial_level_2.pdf'
    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

for count, row in enumerate(final_miss_L_2):    
    
  
    plt.plot(row, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 2 Trial count',fontsize = 16)
    plt.ylabel('trial number', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    #plt.ylim(0,10)
    #plt.yticks((np.arange(0, 10, 1)))
    ax.axes.get_xaxis().set_visible(True) 
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.xlim(-0.1,3.5)
    #plt.ylim(0,6)


mean_trial_miss_L_2 = np.nanmean(final_miss_L_2, axis=0)

sem_L_2 = stats.sem(final_miss_L_2, nan_policy='omit', axis=0)


plt.plot(mean_trial_miss_L_2,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-sem,mean_trial_speed+sem, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(sessions_to_consider), mean_trial_miss_L_2, yerr= sem_L_2, fmt='o', ecolor='k',color='k', capsize=2)  

#plt.legend()
f.tight_layout()

#SAVING
f.savefig(results_dir + figure_name, transparent=True)


t_test_miss_L_2 = stats.ttest_rel(final_miss_L_2[:,0],final_miss_L_2[:,4])
#Ttest_relResult(statistic=1.6540981256071408, pvalue=0.12633120700424816)


target = open(main_folder +"stats_level_2_miss_trial.txt", 'w')
target.writelines(str(t_test_miss_L_2) +' LEVEL 2: day 1 Vs day 5, PLOT: miss trial mean +- SEM, trials_plot.py')

target.close()


























   
    
#PLOT AVG TRIAL/MIN LEVEL 2   
#get list of list ready and padded with nan   
        
def boolean_indexing(v, fillval=np.nan):
   lens = np.array([len(item) for item in v])
   mask = lens[:,None] > np.arange(lens.max())
   out = np.full(mask.shape,fillval)
   out[mask] = np.concatenate(v)
   return out

rat_trial_min_Level_2_pre_array= boolean_indexing(rat_trial_min_Level_2_pre_final, fillval=np.nan)

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




#########################################################################################


rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.4_IrO2.csv', 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                          'F:/Videogame_Assay/AK_31.2_behaviour_only.csv','F:/Videogame_Assay/AK_46.1_behaviour_only.csv','F:/Videogame_Assay/AK_48.3_behaviour_only.csv'
                          ,'F:/Videogame_Assay/AK_46.2_IrO2.csv','F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']

colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9','#C0C0C0','#B0C4DE']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 48.1','AK 48.4', 'AK 49.1', 'AK 49.2' ,'AK 31.2', 'AK 46.1', 'AK 48.3','AK 46.2','AK 50.1','AK 50.2']


x = len(rat_summary_table_path)

rat_total_videogame_lenght = [[] for _ in range(x)] 

for count, rat in enumerate(rat_summary_table_path):
       
    hardrive_path = r'F:/' 
    rat_summary = np.genfromtxt(rat, delimiter = ',', skip_header = 2 , dtype = str, usecols=0)
    l= len(rat_summary)
        
    tot_videogame_time = [[] for _ in range(l)]
           
    for count1, session in enumerate(rat_summary):
        try:
                
            counter_csv = os.path.join(hardrive_path, session + '/Video.csv')
            counter = np.genfromtxt(counter_csv, usecols = 1)

            tot_frames = counter[-1] - counter[0]
            session_length_minutes = tot_frames/120/60
            tot_videogame_time[count1] = session_length_minutes
            print(session)
            
            rat_total_videogame_lenght[count] = tot_videogame_time
            
  
        except Exception: 
            continue       
      
np.sum([np.sum([s for s in rat if type(s) is not list and s > 0]) for rat in rat_total_videogame_lenght])

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
    