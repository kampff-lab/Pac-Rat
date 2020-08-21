# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1b: downsample to 1 kHz for LFP analysis

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

RAT_ID = RAT_ID_ephys 

hardrive_path = r'F:/'


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys


for r, rat in enumerate(rat_summary_table_path): 
    
    
    try:    

        Level_2_post = prs.Level_2_post_paths(rat)
        sessions_subset = Level_2_post
        
        for s, session in enumerate(sessions_subset):            
                    
            # Specify data path
            data_path = os.path.join(hardrive_path+session +'/Amplifier.bin')
            
            # Downsample data 
            ephys.downsample_raw_amplifier(data_path)
            
            # Load and display downsampled data
 





           
            
# Load and display downsampled data
ch = 35
cleaned = np.fromfile(data_path, count=(30000*30*128), dtype=np.uint16)
cleaned = np.reshape(cleaned, (-1, 128)).T
downsampled_path = data_path[:-4] + '_downsampled.bin'
downsampled = np.fromfile(downsampled_path, count=(1000*30*128), dtype=np.uint16)
downsampled = np.reshape(downsampled, (-1, 128)).T
plt.plot(cleaned[55,:], 'r')
plt.plot(np.arange(30000) * 30, downsampled[55,:], 'b')
plt.show()

 