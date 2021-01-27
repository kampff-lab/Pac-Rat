# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:43:34 2021

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
import cv2

import importlib
importlib.reload(prs)
importlib.reload(behaviour)

hardrive_path = r'F:/' 

main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#####level 2
    
    
def reaction_time_level_2(sessions_subset,trial_file = 'Trial_idx.csv',tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600]):
    
    poke= poke_coordinates
    l = len(sessions_subset)
    
        
    rat_reaction_time = [[] for _ in range(l)]
    avg_reaction_time =[]
    std_reaction_time =[]     
    
    
    for count in np.arange(l):
        
        session = sessions_subset[count]    

        script_dir = os.path.join(hardrive_path + session) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        crop_tracking_path = os.path.join(script_dir + tracking_file)
        #parse ball tracking file         

        crop = np.genfromtxt(crop_tracking_path, delimiter= tracking_delimiter)
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        success,misses = behaviour.trial_outcome_index(script_dir)

        
        
        onset = trial_idx[:,2]
        ttr = abs(onset-trial_idx[:,1])
        rat_position_at_start = crop[onset]
        

        rat_position_at_start_success= rat_position_at_start[success]
        ttr_success = ttr[success]
                     
       
        session_rat_poke_dist = [] 
        
        for e in range(len(rat_position_at_start_success)):        
        
            dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_start_success[e]-poke)**2)))
            #euclidian = distance.euclidean(rat_position_at_start[e], poke_coord[e])  
        
            session_rat_poke_dist.append(dist_rat_poke)
       
        
        session_reaction_time = session_rat_poke_dist/ttr_success
      
        avg_rt = np.mean(session_reaction_time)
        std_rt = np.std(session_reaction_time)
        
        avg_reaction_time.append(avg_rt)
        std_reaction_time.append(std_rt)
        
        rat_reaction_time[count]=session_reaction_time 
        print(count)
        
    return rat_reaction_time, avg_reaction_time, std_reaction_time

##########################level 3
    
def reaction_time_level_3(sessions_subset,onset_idx = 4,trial_file = 'Trial_idx_cleaned.csv',outcome='/Trial_outcome_cleaned.csv',tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600]):
    
    poke= poke_coordinates
    l = len(sessions_subset)
    
        
    rat_reaction_time = [[] for _ in range(l)]
    avg_reaction_time =[]
    std_reaction_time =[]     
    
    
    for count in np.arange(l):
        
        session = sessions_subset[count]    

        script_dir = os.path.join(hardrive_path + session) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        crop_tracking_path = os.path.join(script_dir + tracking_file)
        #parse ball tracking file         

        crop = np.genfromtxt(crop_tracking_path, delimiter= tracking_delimiter)
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        #uccess,misses = behaviour.trial_outcome_index(script_dir)
        outcome_path = os.path.join(script_dir+ '/events/'+outcome)
        outcome_open =np.genfromtxt(outcome_path, dtype = str) 
        
        success = []
        for o, out in enumerate(outcome_open):
            if out=='Food':
                success.append(o)
        
        
        onset = trial_idx[:,onset_idx] #4 trigger # 2 catch
        ttr = abs(onset-trial_idx[:,1]) # 1 end
        rat_position_at_start = crop[onset] # position at trigger 
        

        rat_position_at_start_success= rat_position_at_start[success]
        ttr_success = ttr[success]
                     
       
        session_rat_poke_dist = [] 
        
        for e in range(len(rat_position_at_start_success)):        
        
            dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_start_success[e]-poke)**2)))
            #euclidian = distance.euclidean(rat_position_at_start[e], poke_coord[e])  
        
            session_rat_poke_dist.append(dist_rat_poke)
       
        
        session_reaction_time = session_rat_poke_dist/ttr_success
        #rewarded_trial_rt = session_reaction_time[good_idx]
        avg_rt = np.mean(session_reaction_time)
        std_rt = np.std(session_reaction_time)
        
        avg_reaction_time.append(avg_rt)
        std_reaction_time.append(std_rt)
        
        rat_reaction_time[count]=session_reaction_time 
        print(count)
        
    return rat_reaction_time, avg_reaction_time,std_reaction_time







#########level 1 


def reaction_time(sessions_subset,trial_file = 'Trial_idx.csv',tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600]):
    
    #sessions_subset=sessions_subset[:4]
    poke= poke_coordinates
    l = len(sessions_subset)
    
        
    rat_reaction_time = [[] for _ in range(l)]
    avg_reaction_time =[]
    std_reaction_time =[] 
    
    for count in np.arange(l):
        
        session = sessions_subset[count]    

        script_dir = os.path.join(hardrive_path + session) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        crop_tracking_path = os.path.join(script_dir + tracking_file)
        #parse ball tracking file         

        crop = np.genfromtxt(crop_tracking_path, delimiter= tracking_delimiter)
        trial_idx_path = os.path.join(script_dir+ '/events/' + trial_file)
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        success,misses = behaviour.trial_outcome_index(script_dir)
        
        onset = trial_idx[:,0]
        ttr = abs(onset-trial_idx[:,1])
        rat_position_at_start = crop[onset]


        rat_position_at_start_success= rat_position_at_start[success]
        ttr_success = ttr[success]

                      
        bad_idx = []
        
        for p in range(len(rat_position_at_start_success)):
            if  rat_position_at_start_success[p,0]>1250.0 and 450.0 <rat_position_at_start_success[p,1] < 750.0:
                bad_idx.append(p)
            

        #print(len(bad_idx))
        #print(len(trial_idx))
        good_idx =[ele for ele in range(len(rat_position_at_start_success)) if ele not in bad_idx]
                #poke_coord=[poke]*len(onset)
        #print(len(good_idx))
        #print(len(rat_position_at_start_success))

        rat_position_at_start_good= rat_position_at_start_success[good_idx]
        ttr_good = ttr_success[good_idx]

       
        session_rat_poke_dist = [] 
        
        for e in range(len(rat_position_at_start_good)):        
        
            dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_start_good[e]-poke)**2)))
            #euclidian = distance.euclidean(rat_position_at_start[e], poke_coord[e])  
        
            session_rat_poke_dist.append(dist_rat_poke)
       
        
        session_reaction_time = session_rat_poke_dist/ttr_good
        #rewarded_trial_rt = session_reaction_time[good_idx]
        avg_rt = np.mean(session_reaction_time)
        std_rt = np.std(session_reaction_time)
        
        avg_reaction_time.append(avg_rt)
        std_reaction_time.append(std_rt)
        
        rat_reaction_time[count]=session_reaction_time 
        print(count)
    return rat_reaction_time, avg_reaction_time,std_reaction_time


##########################reaction time  PLOTTING 
#LEVE 1

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

#####reaction time LEVEL 2 

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']



colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']



s = len(rat_summary_table_path)

avg_reaction_time_all_rats = [[] for _ in range(s)]
std_reaction_time =[[] for _ in range(s)]
rt_all_rats = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path): 
    

     #rat = rat_summary_table_path[0]
     Level_2= prs.Level_2_pre_paths(rat)
     sessions_subset = Level_2
     
     rt, avg,std =  reaction_time_level_2(sessions_subset,trial_file = 'Trial_idx.csv',tracking_file = '/crop.csv',tracking_delimiter=',',
                                          poke_coordinates = [1400,600])

     
     avg_reaction_time_all_rats[r]=avg
     std_reaction_time[r]=std
     rt_all_rats[r]=rt
     print(rat)



for l in range(len(avg_reaction_time_all_rats)):
    plt.plot(avg_reaction_time_all_rats[l])



select_level_2=np.zeros((s,5))    
 

for i in range(len(rat_summary_table_path)):
    #plt.figure()
    sel=avg_reaction_time_all_rats[i][:5]

    select_level_2[i,:]=sel
   
final=select_level_2*120

    
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
    plt.xlabel('Level 2 Sessions', fontsize = 13)
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
plt.yticks((np.arange(0, 220, 20)))
#plt.legend()
f.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)

t_test = stats.ttest_rel(final[:,0],final[:,4],nan_policy='omit')

#Ttest_relResult(statistic=-3.292444121706258, pvalue=0.007173538082732699)




target = open(main_folder +"level_2_dst_speed.txt", 'w')
target.writelines(str(mean) +str(sem)+str(t_test)+ ' LEVEL 2: dst to poke / time to reward in secons (frame*120) mean +- sem, trials_table.py')

target.close()

###PLOTTING LEVEL 3 


rat_summary_table_path = ['F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']




colours = ['#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']



s = len(rat_summary_table_path)

avg_reaction_time_all_rats = [[] for _ in range(s)]
std_reaction_time =[[] for _ in range(s)]
rt_all_rats = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path): 
    

     #rat = rat_summary_table_path[0]
     Level_3= prs.Level_3_moving_light_paths(rat)
     sessions_subset = Level_3
     
     rt, avg,std =  reaction_time_level_3(sessions_subset,onset_idx = 2,trial_file = 'Trial_idx_cleaned.csv',outcome='/Trial_outcome_cleaned.csv',
                                          tracking_file = '/crop.csv',tracking_delimiter=',', poke_coordinates = [1400,600])
    
     avg_reaction_time_all_rats[r]=avg
     std_reaction_time[r]=std
     rt_all_rats[r]=rt
     print(rat)



for l in range(len(avg_reaction_time_all_rats)):
    plt.plot(avg_reaction_time_all_rats[l])



select_level_3=[]

for i in np.arange(len(rat_summary_table_path)):
    #plt.figure()
    sel=np.array(avg_reaction_time_all_rats[i])*120

    select_level_3.append(sel)


final=np.array([[165.11899394, 167.42154615, 192.18840315, 188.70493079,166.0677011 ],
[200.83600747, 186.27117725, 199.42551454, 217.20806414,201.86719385],
[ 95.62154468, 159.49015417, 126.02467042,nan,nan],
[194.70626691, 193.76326486, 181.54441614, 186.17321256,nan]])


    
f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name = 'mean_reaction_time_level_3.pdf'

for count, row in enumerate(select_level_3):    
    
  
    plt.plot(row,color=colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    
    
    
    plt.title('reaction speed level 3',fontsize = 16)
    plt.ylabel('dst to poke / time to reward in secons (frame*120)', fontsize = 13)
    plt.xlabel('Level 3 Sessions', fontsize = 13)
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
plt.yticks((np.arange(0, 350, 50)))
#plt.legend()
f.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)

#Ttest_relResult(statistic=-3.292444121706258, pvalue=0.007173538082732699)

t_test = stats.ttest_rel(np.array(final[:,0]),final[:,4],nan_policy='omit')


target = open(main_folder +"level_3_dst_speed.txt", 'w')
target.writelines(str(mean) +str(sem)+str(t_test)+ ' LEVEL 3: dst to poke / time to reward in secons (frame*120) mean +- sem, trials_table.py')

target.close()
