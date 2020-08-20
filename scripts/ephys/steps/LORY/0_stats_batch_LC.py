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

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)



rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

RAT_ID = RAT_ID_ephys 

hardrive_path = r'F:/'
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys


for r, rat in enumerate(rat_summary_table_path): 
    
    
    try:    

        Level_2_post = prs.Level_2_post_paths(rat)
        sessions_subset = Level_2_post
        
        for s, session in enumerate(sessions_subset):
            
            # Specify paths
            session  = sessions_subset
            session_path =  os.path.join(hardrive_path,session)


            # Specify raw data path
            raw_path = os.path.join(session_path +'/Amplifier.bin')
            
            # Compute and store channel stats
            ephys.measure_raw_amplifier_stats(raw_path)
            
            # Load channel stats and display
            stats_path = raw_path[:-4] + '_stats.csv'
            stats = np.genfromtxt(stats_path, dtype=np.float32, delimiter=',')
            
            figure_name= '/amplifier_stats.pdf'
            f,ax = plt.subplots(figsize=(7,6))
        
            plt.plot(stats[:,1], 'b.')
            
            f.savefig(session_path + figure_name, transparent=True)
            f.close()
        

    except Exception: 
        print (rat + '/error')
        continue    
       
    



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