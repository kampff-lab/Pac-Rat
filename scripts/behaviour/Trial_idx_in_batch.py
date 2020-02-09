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


rat_summary_table_path = 'F:/Videogame_Assay/AK_48.1_IrO2.csv'
hardrive_path = r'F:/' 

Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
sessions_subset = Level_2_pre


Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post



behaviour.start_end_touch_ball_idx(sessions_subset)




