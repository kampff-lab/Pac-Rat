# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:18:16 2019


Sessions_parser.py 

it parses a summary .csv file and returns a list of paths for a specific
videogame level of interest


@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs



# hard drive base path where the sessions are stored - My Book Duo (F:)

hardrive_path = r'F:/'

rat_summary_table_path =  hardrive_path + 'Videogame_Assay/AK_50.2_behaviour_only.csv'

#reminder of .csv summary structure

#each rat has a summary .csv file where the following infos are stored: 

#Path, Session, Level, Surgery, Name, Weight, Include, Comments

# Path[0] = path to the session which starts with the following format :'Videogame_Assay/AK_rat_name/date-time' (Example : 'Videogame_Assay/AK_33.2/2018_05_17-15_34')
# Session[1] = the date of the session without the time
# Level[2] = level of the Videogame, Level 0,1,2,3...
# Surgery[3] = indicate if t he session has been done prior or after surgery, pre = before surgery, post = after surgery
# Name[4] = name of the Level (habituation, training...)
# Weight[5] = weight of the rat in grams 
# Include[6] = inclusion criteria Y or N
# Comments[7] = comments regarding the session 

#the data starts from row number 2 ( row 0 = rat name and row 1 = title of the column)


# Reload modules
import importlib
importlib.reload(prs)



#PARSING THE MAIN EXCEL SHEET WITH RAT SUMMARY INTO LEVELS
     

Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
Level_3_pre = prs.Level_3_pre_paths(rat_summary_table_path)
Level_3_post = prs.Level_3_post_paths(rat_summary_table_path)
Level_3_moving_light = prs.Level_3_moving_light_paths(rat_summary_table_path)
Level_3_joystick = prs.Level_3_joystick_paths(rat_summary_table_path)
All_levels_post= all_post_surgery_levels_paths(rat_summary_table_path)






