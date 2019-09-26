# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:52:12 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

figure_name = 'RAT_AK_48.4_Trial_per_Session.tiff'
plot_main_title = 'RAT AK_48.4 Trial/Session'

f,ax = plt.subplots(2,2,figsize=(10,7))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)




#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

success_trials_L_1, missed_trials_L_1 = PLOT_trial_and_misses(Level_1)

x = np.array(range(len((Level_1))))

ax[0,0].bar(x, success_trials_L_1, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,0].bar(x, missed_trials_L_1, bottom = success_trials_L_1, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
ax[0,0].legend(loc ='best', frameon=False , fontsize = 'x-small') #ncol=2
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / Session', fontsize = 10)
#ax[0,0].set_xlabel('Sessions')




success_trials_L_2_pre, missed_trials_L_2_pre = PLOT_trial_and_misses(Level_2_pre)

x = np.array(range(len((Level_2_pre))))
ax[0,1].bar(x, success_trials_L_2_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,1].bar(x, missed_trials_L_2_pre, bottom = success_trials_L_2_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
#ax[0,1].set_ylabel('Trials / Session')
#ax[0,0].set_xlabel('Sessions')




success_trials_L_2_post, missed_trials_L_2_post = PLOT_trial_and_misses(Level_2_post)

x = np.array(range(len((Level_2_post))))
ax[1,0].bar(x, success_trials_L_2_post, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[1,0].bar(x, missed_trials_L_2_post, bottom = success_trials_L_2_post, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
ax[1,0].set_ylabel('Trials / Session', fontsize = 10)
ax[1,0].set_xlabel('Sessions', fontsize = 10)



#success_trials_L_3_pre ,missed_trials_L_3_pre = PLOT_trial_and_misses(Level_3_pre)

#x = np.array(range(len((Level_3_pre))))
#ax[1,0].bar(x, success_trials_L_3_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,0].bar(x, missed_trials_L_3_pre, bottom = success_trials_L_3_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 3 pre surgery')
#ax[1,0].set_ylabel('Trials / Session')
#ax[1,0].set_xlabel('Sessions')



success_trials_L_3_post, missed_trials_L_3_post = PLOT_trial_and_misses(Level_3_post)

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
script_dir = os.path.join(hardrive_path +'Videogame_Assay' + '/AK_48.1')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    
        




def PLOT_trial_per_min(sessions_subset):    
    total_trials = []
    session_length = []
    try:
        for session in sessions_subset: 
            trial_end_path = os.path.join(hardrive_path, session + '/events/' + 'TrialEnd.csv')
            RewardOutcome_file = np.genfromtxt(trial_end_path, usecols=[1], dtype= str)
        
            counter_csv = os.path.join(hardrive_path, session + '/Video.csv')
            counter = np.genfromtxt(counter_csv, usecols = 1)
        
            tot_frames = counter[-1] - counter[0]
            session_length_minutes = tot_frames/120/60
            tot_trials  =len(RewardOutcome_file)
            total_trials.append(tot_trials)
            session_length.append(session_length_minutes)
    except Exception: 
            print('error'+ session)
            pass
        
    return total_trials, session_length  






figure_name = 'RAT_AK_48.4_Trial_per_Minute.tiff'
plot_main_title = 'RAT AK_48.4 Trial/Min'

f,ax = plt.subplots(2,2,figsize=(10,7))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()




#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

total_trials_L_1, session_length_L_1 = PLOT_trial_per_min(Level_1)

trials_per_minutes_L_1 = np.array(total_trials_L_1)/np.array(session_length_L_1)

x = np.array(range(len((Level_1))))

ax[0,0].plot(x, trials_per_minutes_L_1, color ='r', marker = 'o', alpha = .8)
# Create green bars (middle), on top of the firs ones
#ax[0,0].bar(x, trials_per_minutes,  color ='r', edgecolor ='white', width = 1, alpha = .5)
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / min', fontsize = 10)
#ax[0,0].set_xlabel('Sessions')




total_trials_L_2_pre, session_length_L_2_pre = PLOT_trial_per_min(Level_2_pre)

trials_per_minutes_L_2_pre = np.array(total_trials_L_2_pre)/np.array(session_length_L_2_pre)

x = np.array(range(len((Level_2_pre))))

ax[0,1].plot(x, trials_per_minutes_L_2_pre, color ='b', marker = 'o', alpha = .8)
#ax[0,1].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[0,1].set_title('Level 2 pre surgery', fontsize = 13)
#ax[0,1].set_ylabel('Trials / Session')
#ax[0,0].set_xlabel('Sessions')




total_trials_L_2_post, session_length_L_2_post = PLOT_trial_per_min(Level_2_post)

trials_per_minutes_L_2_post = np.array(total_trials_L_2_post)/np.array(session_length_L_2_post)

x = np.array(range(len((Level_2_post))))

ax[1,0].plot(x, trials_per_minutes_L_2_post, color ='g', marker = 'o', alpha = .8)
# Create green bars (middle), on top of the firs ones

#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
ax[1,0].set_title('Level 2 post surgery', fontsize = 13)
ax[1,0].set_ylabel('Trials / min', fontsize = 10)
ax[1,0].set_xlabel('Sessions', fontsize = 10)



#total_trials_L_3_pre ,session_length_L_3_pre = PLOT_trial_per_min(Level_3_pre)

#x = np.array(range(len((Level_3_pre))))
#ax[1,0].bar(x, success_trials_L_3_pre, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
## Create green bars (middle), on top of the firs ones
#ax[1,0].bar(x, missed_trials_L_3_pre, bottom = success_trials_L_3_pre, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
#ax[1,0].legend(loc='best',frameon=False , fontsize = 'x-small') #ncol=2
#ax[1,0].set_title('Level 3 pre surgery')
#ax[1,0].set_ylabel('Trials / Session')
#ax[1,0].set_xlabel('Sessions')



total_trials_L_3_post, session_length_L_3_post = PLOT_trial_per_min(Level_3_post)

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
script_dir = os.path.join(hardrive_path +'Videogame_Assay' + '/AK_48.4')
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    