# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 0: measure recording statistics

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
#os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 
import seaborn as sns

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)
hardrive_path = r'F:/' 


rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

RAT_ID = RAT_ID_ephys 

#hardrive_path = r'F:/'
#main_folder = 'E:/thesis_figures/'
#figure_folder = 'Tracking_figures/'
#
#results_dir =os.path.join(main_folder + figure_folder)
#
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys


for r, rat in enumerate(rat_summary_table_path): 
     

    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
        
    
    for s, session in enumerate(sessions_subset):
        
        try:
            
            # Specify paths
            #session  = sessions_subset
            session_path =  os.path.join(hardrive_path,session)


            # Specify raw data path
            raw_path = os.path.join(session_path +'/Amplifier.bin')
            
            # Compute and store channel stats
            ephys.measure_raw_amplifier_stats(raw_path)
            
            # Load channel stats and display
#            stats_path = raw_path[:-4] + '_stats.csv'
#            stats = np.genfromtxt(stats_path, dtype=np.float32, delimiter=',')
#            
#            figure_name= '/amplifier_stats.pdf'
#            f,ax = plt.subplots(figsize=(7,6))
#        
#            plt.plot(stats[:,1], 'b.')
#            
#            f.savefig(session_path + figure_name, transparent=True)
#            f.close()
#            
    
        except Exception: 
            print (rat + '/error')
            continue    
           
        
#pstats plots
            
        

for r, rat in enumerate(rat_summary_table_path): 
     

    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
        
    
    for s, session in enumerate(sessions_subset):
        
        
        session_path =  os.path.join(hardrive_path,session)        
        #Load channel stats and display
        raw_path = os.path.join(session_path +'/Amplifier.bin')
        stats_path = raw_path[:-4] + '_stats.csv'
        stats = np.genfromtxt(stats_path, dtype=np.float32, delimiter=',')
        
        figure_name= '/amplifier_stats.pdf'
        f,ax = plt.subplots(figsize=(7,6))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=False)
        
        plt.plot(stats[:,1], 'k.')
        ax.axes.get_xaxis().set_visible(True) 
        
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('mean channels std')
        plt.xlabel('channels')
        
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        f.tight_layout()       
        
        
        f.savefig(session_path + figure_name, transparent=True)
        plt.close()
        
        print(session +'saved')
       




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


## Specify session folder
##session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
#session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'
##
## Specify data paths
#raw_path = os.path.join(session_path +'/Amplifier.bin')
#
## Load raw data and convert to microvolts
#mean_int, std_int, mean_uV, std_uV = ephys.measure_raw_channel_stats(raw_path, 0)
#
##FIN