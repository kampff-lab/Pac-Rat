# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:56:17 2019

parser libray.py

functions to parse files 

@author: KAMPFF-LAB-ANALYSIS3

"""
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
hardrive_path = r'F:/'



# hard drive base path where the sessions are stored - My Book Duo (F:)

#hardrive_path = r'F:/'

#rat_summary_table_path =  hardrive_path + 'Videogame_Assay/AK_48.4_IrO2.csv'

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
#each fx return a list containing the str object = all the session paths for a Level 
# list items format example = Videogame_Assay/AK_48.4/2019_06_24-17_10



def Level_0_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_0 = []
    for row in range(len(rat_summary)):
        if rat_summary[row][2] == 'Level 0' and rat_summary[row][6] == 'Y':
            Level_0.append(rat_summary[row][0])
        else:
            continue
    return Level_0




def Level_1_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_1 = []
    for row in range(len(rat_summary)):
        if rat_summary[row][2] == 'Level 1' and rat_summary[row][6] == 'Y':
            Level_1.append(rat_summary[row][0])
        else:
            continue
    return Level_1
 
    
def Level_1_paths_6000_3000(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_1_6000_3000 = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 1' and rat_summary[row][7] == '6000':
            Level_1_6000_3000.append(rat_summary[row][0])
        else:
            continue
    return Level_1_6000_3000


def Level_1_paths_10000(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_1_10000 = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 1' and rat_summary[row][7] == '10000':
            Level_1_10000.append(rat_summary[row][0])
        else:
            continue
    return Level_1_10000

def Level_1_paths_20000(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_1_20000 = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 1' and rat_summary[row][7] == '20000':
            Level_1_20000.append(rat_summary[row][0])
        else:
            continue
    return Level_1_20000






def Level_2_pre_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_2_pre = []  
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 2' and rat_summary[row][3]== 'pre':
            Level_2_pre.append(rat_summary[row][0])
        else:
            continue
    return Level_2_pre
    

def Level_2_post_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_2_post = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 2' and rat_summary[row][3]== 'post':
           Level_2_post.append(rat_summary[row][0])
        else:
            continue
    return Level_2_post



def Level_3_pre_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_3_pre = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 3' and rat_summary[row][3]== 'pre':
           Level_3_pre.append(rat_summary[row][0])
        else:
            continue
    return Level_3_pre




def Level_3_post_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_3_post = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][2] == 'Level 3' and rat_summary[row][3]== 'post':
           Level_3_post.append(rat_summary[row][0])
        else:
            continue
    return Level_3_post



def Level_3_moving_light_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_3_moving_light = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][4] == 'moving light':
           Level_3_moving_light.append(rat_summary[row][0])
        else:
            continue
    return Level_3_moving_light




def Level_3_joystick_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    Level_3_joystick = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][4] == 'joystick':
           Level_3_joystick.append(rat_summary[row][0])
        else:
            continue
    return Level_3_joystick



def all_post_surgery_levels_paths(rat_summary_table_path):
    rat_summary = np.genfromtxt(rat_summary_table_path, delimiter = ',', skip_header = 2 , dtype = str)
    All_levels_post = []
    for row in range(len(rat_summary)):
        if not rat_summary[row][6] == 'N' and rat_summary[row][3] == 'post':
           All_levels_post.append(rat_summary[row][0])
        else:
            continue
    return All_levels_post

     

    