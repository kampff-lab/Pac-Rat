# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:20:48 2020

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


#Level 2 saving trial idx in each session folder under events folder

for count, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_post = prs.Level_2_post_paths(rat)
         sessions_subset = Level_2_post
         
         behaviour.start_end_touch_ball_idx(sessions_subset)
         print(rat)
         print(count)
         
    except Exception: 
        print (rat + '/error')
        continue    
    
    
         
#Level 3 saving trial idx in each session folder under events folder

for count, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_3_post = prs.Level_3_post_paths(rat)
         sessions_subset = Level_3_post
         
         behaviour.start_end_touch_ball_idx(sessions_subset)
         print(rat)
         print(count)
         
    except Exception: 
        print (rat + '/error')
        continue    
    
    
         








