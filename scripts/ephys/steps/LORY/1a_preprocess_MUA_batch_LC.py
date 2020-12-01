# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1a: (headtsage) mean rereferencing/cleaning for MUA analysis

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
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

RAT_ID = RAT_ID_ephys[0] 

hardrive_path = r'F:/'


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)

rat_summary_table_path=rat_summary_ephys[0]
probe_map_flatten = ephys.probe_map.flatten()


for r, rat in enumerate(rat_summary_table_path): 
    
       

    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    for s, session in enumerate(sessions_subset):
        
        try:
                 
            # Specify raw data path
            raw_path = os.path.join(hardrive_path+session +'/Amplifier.bin')
            
            # Specify channels to exclude (in reference calculation)
            disconnected_channels = np.array([12, 13, 18, 19, 108, 109 ,115])
            
            bad_channels_idx = ephys.bad_channel(session, min_imp = 10000, max_imp = 6000000)
            bad_channels_path =  os.path.join(hardrive_path + session+'/bad_channels.csv')
            bad_channels = np.genfromtxt(bad_channels_path, delimiter=',',dtype=int)
          
                        
            #bad channels are idx while disconnecte are the actual channel number, for the cleaning I need the actual channel number to sepa
            ##rate between headstages
            exclude_channels = np.sort(np.hstack((disconnected_channels,probe_map_flatten[bad_channels])))
            print(len(exclude_channels))
            
            # Clean raw data and store binary file
            ephys.clean_raw_amplifier(raw_path, exclude_channels)   
    #            
    #            # Load cleaned data and display
    #            data = np.fromfile(raw_path, count=(30000*3*128), dtype=np.uint16)
    #            data = np.reshape(data, (-1, 128)).T
    #            plt.plot(data[87,:], 'r')
    #            
    #            cleaned_path = raw_path[:-4] + '_cleaned.bin'
    #            cleaned_data = np.fromfile(cleaned_path, count=(30000*3*128), dtype=np.uint16)
    #            cleaned_data = np.reshape(cleaned_data, (-1, 128)).T
    #            plt.plot(cleaned_data[87,:], 'b')
    #            plt.show()
            print(session + 'saved')

        except Exception: 
            print(session+'error')
            continue   
#FIN
