# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:32:45 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import cv2
import numpy as np
import os
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import start_to_end_trial_clips_library as clips
import behaviour_library as behaviour
import parser_library as prs



rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
rat_ID = 'AK_40.2'


Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)

sessions_subset = Level_2_pre



clips.CLIPS_start_to_end_trial(sessions_subset)



