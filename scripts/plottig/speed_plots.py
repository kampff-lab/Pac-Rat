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
from scipy import stats


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


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

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
Level_2_touch_to_reward_speed_all_rats = [[] for _ in range(s)]

mean_st_all_rats =[[] for _ in range(s)]
std_st_all_rats =[[] for _ in range(s)]
mean_te_all_rats = [[] for _ in range(s)]
std_te_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         session_speed = behaviour.session_speed_crop_tracking(sessions_subset)
         Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, session_speed)
         Level_2_touch_to_reward_speed = behaviour.speed_touch_to_reward(sessions_subset, session_speed)
         
         l = len(session_speed)

         mean_start_to_touch = []
         std_start_to_touch = [] 
         mean_touch_to_reward =[]
         std_touch_to_reward = []
         
         
         for count in np.arange(l):
             
             session_start_to_touch = Level_2_start_to_touch_speed[count]
             session_touch_to_reward = Level_2_touch_to_reward_speed[count]
             concat_speed_st = [item for sublist in session_start_to_touch for item in sublist]
             concat_speed_te = [item for sublist in session_touch_to_reward for item in sublist]
             
             
             start_touch_mean = np.nanmean(concat_speed_st)
             mean_start_to_touch.append(start_touch_mean)
             start_touch_std = np.nanstd(concat_speed_st)
             std_start_to_touch.append(start_touch_std)
             
             touch_reward_mean = np.nanmean(concat_speed_te)
             mean_touch_to_reward.append(touch_reward_mean)
             touch_to_reward_std = np.nanstd(concat_speed_te)
             std_touch_to_reward.append(touch_to_reward_std)
             
             
         Level_2_start_to_touch_speed_all_rats[r] = Level_2_start_to_touch_speed
         Level_2_touch_to_reward_speed_all_rats[r]=Level_2_touch_to_reward_speed
         mean_st_all_rats[r] = mean_start_to_touch
         std_st_all_rats[r] = std_start_to_touch
         mean_te_all_rats[r] = mean_touch_to_reward
         std_te_all_rats[r] = std_touch_to_reward
         
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    


#####################################
        
f,ax = plt.subplots(figsize=(15,5))



tot_before= []
tot_after = []

for rat in arange(len(rat_summary_table_path)):
    
    
    f,ax = plt.subplots(figsize=(7,5))
      
    before  = mean_st_all_rats[rat]
    after = mean_te_all_rats [rat]


    #flattened_before = [val for sublist in before for val in sublist]
    #flattened_after = [val for sublist in after for val in sublist]
    
    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    print(rat)
    print(len(before))
    print(len(after))
    
    
    filtered_before = np.array(before)[~np.isnan(before)]
    filtered_after = np.array(after)[~np.isnan(after)]
    
    tot_before.append(filtered_before)
    tot_after.append(filtered_after)
    
    stack= np.vstack((filtered_before,filtered_after))
    to_plot = stack.tolist()
    
   # plt.figure()
    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
    
    #plt.bar(range(len(delta)),delta, width= 0.05)
    plt.boxplot(to_plot)
    plt.ylim((0,2.5))
    
    #ax.hlines(0,0,800,linewidth=0.5)
      
    
 
    
flattened_before = [val for sublist in tot_before for val in sublist]
flattened_after = [val for sublist in tot_after for val in sublist]
tot_stack= np.vstack((flattened_before,flattened_after))

to_plot = tot_stack.tolist()
    
   # plt.figure()
    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
    
    #plt.bar(range(len(delta)),delta, width= 0.05)
plt.boxplot(to_plot)
#plt.ylim((0,2.5))

plt.plot(flattened_before,flattened_after,'.')

plt.plot(np.unique(flattened_before), np.poly1d(np.polyfit(flattened_before, flattened_after, 1))(np.unique(flattened_before)))


m,b = np.polyfit(flattened_before, flattened_after, 1)

# m 0.4675482133356882
# b 0.9965590810071268


coef = np.polyfit(flattened_before,flattened_after,1)
poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

plt.plot(flattened_before,flattened_after, 'yo', flattened_before, poly1d_fn(flattened_before), '--k')





test_speed = stats.ttest_rel(flattened_before,flattened_after)

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


st_idx_tot =[]
te_idx_tot=[]


for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    st_idx  = st_all_rats[rat]
    te_idx = te_all_rats [rat]


    flattened_st= [val for sublist in st_idx for val in sublist]
    flattened_te = [val for sublist in te_idx for val in sublist]
    
    st_idx_tot.extend(flattened_st)
    te_idx_tot.extend(flattened_te)
    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    #total_delta.extend(delta)
    
print(len(st_idx_tot))
print(len(te_idx_tot))


max_st = max(st_idx_tot)
max_te = max(te_idx_tot)



#test from session retrieve st and te.csv and pool them





























####################

#plot st VS te calculated usinf IDX - MEDIAN


    
tot_stack= np.vstack((st_idx_tot,te_idx_tot))

final_to_plot = tot_stack.tolist()

plt.figure()
plt.boxplot(final_to_plot, showfliers=False)

plt.figure()
plt.boxplot(final_to_plot, showfliers=True)


st_array = np.array(st_idx_tot)

st_norm = ( st_array- st_array.min()) / (st_array.max() - st_array.min())


te_array = np.array(te_idx_tot)

te_norm = ( te_array- te_array.min()) / (te_array.max() - te_array.min())


median_st = np.median(st_idx_tot)
median_te = np.median(te_idx_tot)
sem_st = stats.sem(st_idx_tot, nan_policy='omit')
sem_te = stats.sem(te_idx_tot, nan_policy='omit')


materials = ['before_touch', 'after_touch']
x_pos = np.arange(len(materials))
medians= [median_st, median_te]
errors = [sem_st, sem_te]



f,ax = plt.subplots(figsize=(10,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)


ax.bar(materials[0], medians[0], yerr=errors[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
ax.bar(materials[1], medians[1], yerr=errors[1] ,align='center', color ='b', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)


plt.figure()
plt.plot(te_idx_tot,st_idx_tot, '.', alpha=.7, markersize=.5, color= 'k')
plt.ylim(0,20000)

# =============================================================================
#  
 
# rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']
# rat = rat_summary_table_path[0]
# Level_2_pre = prs.Level_2_pre_paths(rat)
# sessions_subset = Level_2_pre[0]
# 
# =============================================================================
#


 


############################################################################
##########################################
        
#plot DELTA distance rat before touch (120 frames) - rat at touch VS rat at touch - rat after touch (120 frames)



sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

figure_name =  'delta_speed_after_minus_before_rats_overlap_120frames.pdf' 
  

total_delta = []    
total_flat_before =[]
total_flat_after = []



f,ax = plt.subplots(figsize=(15,5))


for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    before  = before_touch_all_rats[rat]
    after = after_touch_all_rats [rat]


    flattened_before = [val for sublist in before for val in sublist]
    flattened_after = [val for sublist in after for val in sublist]
    
    total_flat_before.extend(flattened_before)
    total_flat_after.extend(flattened_after)
    
    delta = np.array(flattened_after) - np.array(flattened_before)
    total_delta.extend(delta)
    
    print(rat)
    print(len(flattened_before))
    print(len(flattened_after))
    
    
    
    stack= np.vstack((flattened_before,flattened_after))
    to_plot = stack.tolist()
    
    ax.scatter(range(len(delta)),delta,alpha=0.7, edgecolors='none', s=7,c='k')
    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
    plt.title('Level 2 delta speed around touch',fontsize = 16)
    plt.ylabel('speed (idx_count)', fontsize = 13)
    plt.xlabel('trial', fontsize = 13)
    
    #plt.bar(range(len(delta)),delta, width= 0.05)
    #plt.boxplot(to_plot)
    ax.axes.get_xaxis().set_visible(True) 
       
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.hlines(0,0,900,linewidth=0.8,color= 'r')
    
    #plt.xlim(-0.1,3.5)
    #plt.ylim(-0.2,6)
    f.tight_layout()
    


##########################################
#flattened allthe rats 

        
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

figure_name2 =  'flattemed_before_touch_VSafter_touch_120frames.pdf' 
  

f2,ax = plt.subplots(figsize=(8,7))

stack= np.vstack((total_flat_before,total_flat_after))
to_plot = stack.tolist()

plt.boxplot(to_plot)


plt.title('Level 2 before VS after 120frames',fontsize = 16)
plt.ylabel('speed idx count', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#plt.legend()
f2.tight_layout()




############################################

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

figure_name3 =  'delta_all_rats_concat.pdf' 
  

f3,ax = plt.subplots(figsize=(15,5))

ax.scatter(range(len(total_delta)),total_delta,alpha=0.7, edgecolors='none', s=7,c='k')



ax.hlines(0,0,len(total_delta),linewidth=0.8,color= 'r')
plt.title('Level 2 delta_after minus before 120frames concat',fontsize = 16)
plt.ylabel('delta speed idx count', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#plt.legend()
f3.tight_layout()

#####################bar plot



median_before_touch = np.median(total_flat_before)
median_after_touch = np.median(total_flat_after)
sem_before_touch = stats.sem(total_flat_before, nan_policy='omit')
sem_after_touch = stats.sem(total_flat_after, nan_policy='omit')


materials = ['before_touch_120frames', 'after_touch_120frames']
x_pos = np.arange(len(materials))
medians= [median_before_touch, median_after_touch]
errors = [sem_before_touch, sem_after_touch]



figure_name4 =  'dist_before_after_120_barplot.pdf' 

f4,ax = plt.subplots(figsize=(10,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)


ax.bar(materials[0], medians[0], yerr=errors[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
ax.bar(materials[1], medians[1], yerr=errors[1] ,align='center', color ='b', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)

plt.title('before and after touch 120sec',fontsize = 16)
plt.ylabel('dist', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#plt.legend()
f4.tight_layout()


###############savings



f.savefig(results_dir + figure_name, transparent=True)
f2.savefig(results_dir + figure_name2, transparent=True)
f3.savefig(results_dir + figure_name3, transparent=True)
f4.savefig(results_dir + figure_name4, transparent=True)



############################################
###############################################################



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


###########################################
rat_poke_all_rats
rat_ball_all_rats

st_idx_tot
te_idx_tot

    

f3,ax = plt.subplots(figsize=(7,5))

rat_ball_concat = []


for rat in arange(len(rat_summary_table_path)):
    
      
    rat_ball_selection  = rat_ball_all_rats[rat]


    flattened_rat_ball = [val for sublist in rat_ball_selection for val in sublist]
    
    rat_ball_concat.extend(flattened_rat_ball)

     
    print(rat)
    print(len(rat_ball_concat))
    print(len(st_idx_tot))
    
    
    
plt.figure()    
plt.plot(rat_ball_concat,st_idx_tot, '.', alpha=.8, markersize=.5, color= 'k')
plt.ylim(0,10000)

plt.hist(rat_ball_concat,bins=300)



#plt.ylim(0,10000)

################################################################

rat_poke_all_rats
te_idx_tot

plt.figure()

rat_poke_concat = []


for rat in arange(len(rat_summary_table_path)):
    
      
    rat_poke_selection  = rat_poke_all_rats[rat]

    flattened_rat_poke = [val for sublist in rat_poke_selection for val in sublist]
   
    
    rat_poke_concat.extend(flattened_rat_poke)
    
    
    print(rat)
    
    print(len(rat_poke_concat))
    print(len(te_idx_tot))
    

plt.figure()
plt.plot(rat_poke_concat, te_idx_tot, '.', alpha=.7, markersize=.5, color= 'k')





plt.hist(rat_poke_concat, bins=300)
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
    
    
   ##################################### 


##################################            

 

# rat = rat_summary_table_path[0]
# Level_2_pre = prs.Level_2_pre_paths(rat)
# sessions_subset = Level_2_pre[0]


 
st_pooled=[]
te_pooled=[]
                
for r,rat in enumerate(rat_summary_table_path):
    
    try:
               
        Level_2_pre = prs.Level_2_pre_paths(rat)
        sessions_subset = Level_2_pre#[3:6]   
        l=len(sessions_subset)
         
        rat_st = []
        rat_te = []
        
        
        for count in np.arange(l):
        

        
            session = sessions_subset[count]
    
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
    
            st_path= os.path.join(script_dir+ '/events/' + 'Start_touch_idx_diff.csv')           
            st_diff = np.genfromtxt(st_path, delimiter = ',')
            
            rat_st.extend(st_diff)
            
            te_path = os.path.join(script_dir+ '/events/' + 'Touch_reward_idx_diff.csv')
            te_diff = np.genfromtxt(te_path, delimiter = ',')
            
            rat_te.extend(te_diff)
                
        st_pooled.extend(rat_st)
        te_pooled.extend(rat_te)    
            
    except Exception: 
        print('error'+ session)
    continue    

print(len(st_pooled)) 
print(len(te_pooled))   
    
    
    ############################

   
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
                dist_before_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_before_ball[e])**2)))
                dist_after_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_after_ball[e])**2)))
                
                
                session_rat_ball_dist.append(dist_rat_ball)
                session_rat_poke_dist.append(dist_rat_poke)
                session_rat_before_touch.append(dist_before_touch)
                session_rat_after_touch.append(dist_after_touch)
                
                
                #.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx,touch_idx,ball_on_idx)).T, delimiter=',', fmt='%s')

                
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
            touch_end_diff = abs(trial_idx[:,1] - trial_idx[:,2])
            st_time[count] = start_touch_diff
            te_time[count] = touch_end_diff
            
            
            csv_dir_path = os.path.join(script_dir + '/events/')
            
            csv_name = 'Start_touch_idx_diff.csv'
            np.savetxt(csv_dir_path + csv_name,start_touch_diff, fmt='%s')
            csv_name = 'Touch_reward_idx_diff.csv'
            np.savetxt(csv_dir_path + csv_name,touch_end_diff, fmt='%s')
            
            print(len(start_touch_diff))
            print(len(touch_end_diff))
            print('saving DONE')


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

