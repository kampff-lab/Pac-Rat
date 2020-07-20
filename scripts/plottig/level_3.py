# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:06:49 2020

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


import importlib
importlib.reload(prs)
importlib.reload(behaviour)

hardrive_path = r'F:/' 

rat_summary_table_path = ['F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']






#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'




trial_table_moving_light_path = 'F:/Videogame_Assay/Trial_table_final_level_3_moving_light.csv'
trial_table_joystick_path = 'F:/Videogame_Assay/Trial_table_final_level_3_joystick.csv'


#selectin first moving light day and tracking 


    
s = len(rat_summary_table_path)


ball_pos_all_rats = [[] for _ in range(s)]
snippet_before = [[] for _ in range(s)]
snippet_after = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_3_pre = prs.Level_3_moving_light_paths(rat)
         sessions_subset = Level_3_pre[1]#[3:6]
         
         rat_coordinates_before,ball_coordinates,rat_coordinates_after = rat_event_idx_and_pos_finder(sessions_subset, event=2, offset = 360, trial_number=11)
         
         snippet_before[r] =  rat_coordinates_before
         snippet_after[r] =  rat_coordinates_after
         ball_pos_all_rats[rat] = ball_pos
         
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  


len(touch_all_rats)
len(pos_at_touch_all_rats)




def rat_event_idx_and_pos_finder(sessions_subset, event=2, offset = 360, trial_number=11): #start_event=0, end_event= 1
    
    ball_coordinates = []
    rat_coordinates_before = [[] for _ in range(trial_number)]
    rat_coordinates_after = [[] for _ in range(trial_number)]
    
    session = sessions_subset
    

    script_dir = os.path.join(hardrive_path + session) 

    trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
    trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
    #crop_tracking_path = os.path.join(script_dir + '/crop.csv')
    #crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
    shader_tracking_path = os.path.join(script_dir + '/events/' +'Tracking.csv')
    shader_tracking = np.genfromtxt(shader_tracking_path)
    ball_shaders_path= os.path.join(script_dir+ '/events/' + 'BallPosition.csv')
    split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
    timestamps = split_data[:,0]
    positions_strings = split_data[:,1]
    for index, s in enumerate(positions_strings):
        tmp = s.replace('(', '')
        tmp = tmp.replace(')', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp.replace(' ', '')
        positions_strings[index] = tmp
    ball_shaders = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
               
        for count in np.arange(trial_number):
    
               
       
        start  = trial_idx[:11,event]
        
        ball = ball_shaders[event][:11]
        
        rat_pos_before = shader_tracking[start-offset]
        rat_pos_after = shader_tracking[start+offset]
    
        
        rat_coordinates[count] = rat_pos
        ball_coordinates.append(ball)
        rat_coordinates_after[count] = rat_pos_after
    
    return rat_coordinates_before,ball_coordinates,rat_coordinates_after













plt.plot(test[0][:,0],test[0][:,1])
plt.plot(test[1][:,0],test[1][:,1])
plt.plot(test[2][:,0],test[2][:,1])
plt.plot(test[3][:,0],test[3][:,1])

test  = [val for sublist in snippet for val in sublist]

diff_test =  np.diff(test[1],axis=0)
plt.plot(diff_test[:,0],diff_test[:,1])










'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/BallPosition.csv'


tracking = 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/BallTracking.csv'
tracking_open = np.genfromtxt(tracking, delimiter = ',', usecols= [1,2])
track_diff = np.diff(tracking_open[:,0])


binary = []
for v in range(len(track_diff)):
    if track_diff[v] ==0:
        binary.append(0)
    else:
        binary.append(1)


trans = np.argwhere(np.diff(binary)).squeeze()


trans_list = []
for i in range(len(binary)):
    
    sub = binary[i+1] - binary[i]
    if sub ==1:
        trans_list.append(i)
        



save= np.savetxt(test_open, delimiter=',')


test = 'F:/Videogame_Assay/AK_49.1/2019_09_20-14_18/events/BallTracking.csv'
test_open = np.genfromtxt(test, dtype=floT


                          
                          
                          
                          
                          
sel = test_open[:,1]

s = sel.split()
non_grouped = pp.Word(pp.printables, excludeChars="_(),")

 csv.reader()
t=  re.split(r',\s*(?=[^)]*(?:\(|$))', sel) 

import re

with open(test) as f:
    for line in map(str.strip, f):
        l = re.split('_,(,_)', line)
    print(len(l), l)




df = pd.read_csv(test, delimiter='\s*(?=\()', engine='python')
df.columns = df.columns.str.replace('[()]', '')
df = df.replace('[()]', '', regex=True)



import pandas as pd
