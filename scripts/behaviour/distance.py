# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import DLC_parser_library as DLC
import scipy.signal
import seaborn as sns
from matplotlib.lines import Line2D

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
sessions_subset = Level_2_pre


x_centroid_te, y_centroid_te = create_tracking_snippets_touch_to_end_centroid(sessions_subset,end_snippet_idx = 1,mid_snippet_idx = 2)

x = len(sessions_subset)

sessions_dst_te = [[] for _ in range(x)] 

for count in np.arange(x):
        try:        
            x_snippets = np.copy(x_centroid_te[count])
            y_snippets = np.copy(y_centroid_te[count])
            
            l =len(x_snippets)
            
            final_te_units = [[] for _ in range(l)] 
            
            for trial in np.arange(l):
                
                dx = np.diff(x_snippets[trial], prepend = x_snippets[trial][0])
                dy = np.diff(y_snippets[trial], prepend = y_snippets[trial][0])


                dst = np.sqrt(dx*dx + dy*dy)

                travelled_dst = np.cumsum(dst)/np.sum(dst)
                
                replicated = np.hstack([travelled_dst, travelled_dst[::-1]])


                resampled = scipy.signal.resample_poly(replicated,2000, len(replicated))

                final = resampled[0:1000]
                
                final_te_units[trial] = final    
                
            sessions_dst_te[count] = final_te_units

            print(count)        
        
        except Exception: 
            continue
     
 
x_centroid_st, y_centroid_st = create_tracking_snippets_start_to_end_centroid(sessions_subset, start_snippet_idx = 0, mid_snippet_idx=2)
    
    
x = len(sessions_subset)

sessions_dst_st = [[] for _ in range(x)] 

for count in np.arange(x):
        try:        
            x_snippets = np.copy(x_centroid_st[count])
            y_snippets = np.copy(y_centroid_st[count])
            
            l =len(x_snippets)
            
            final_st_units = [[] for _ in range(l)] 
            
            for trial in np.arange(l):
                #it is not working if first item is a nan 
                dx = np.diff(x_snippets[trial])#prepend = x_snippets[trial][0])
                dy = np.diff(y_snippets[trial]) #prepend = y_snippets[trial][0])
                
                dx = np.hstack((0,dx))
                dy = np.hstack((0,dy))
                
                dst = np.sqrt(dx*dx + dy*dy)

                travelled_dst = np.cumsum(dst)/np.sum(dst)
                
                replicated = np.hstack([travelled_dst, travelled_dst[::-1]])


                resampled = scipy.signal.resample_poly(replicated,2000, len(replicated))

                final = resampled[0:1000]
                
                final_st_units[trial] = final    
                
            sessions_dst_st[count] = final_st_units

            print(count)        
        
        except Exception: 
            continue    
    
    
    

#x_tracking_trial_0_te = x_centroid_te[0][0]
#y_tracking_trial_0_te = x_centroid_st[0][0]
#dst_trial_0 = sessions_dst_te[0][0]
#
#x_tracking_trial_0_st = x_centroid_st[0][0]
#y_tracking_trial_0_st = y_centroid_st[0][0]
#
#clip_path = r'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Clips_annotation/Clip00.csv'
#clip = np.genfromtxt (clip_path, delimiter = ',', dtype = str, usecols=0)
#time = np.genfromtxt (clip_path, delimiter = ',', dtype = int, usecols=1)
#
#for i in time:
#    
#    plt.axvline(x = i , color='k', linestyle='--',linewidth =.5)


#number_of_subplots= len(sessions_subset)
#
#plot_main_title = 'distance_te_40.2'
#dst = sessions_dst_st
#
#f = plt.figure(figsize=(20,10))
#f.suptitle(plot_main_title)
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine()
#
#   
#for i in np.arange(number_of_subplots): 
#    try:
#        
#        trial_dst = dst[i]
#        ax = f.add_subplot(3, 4, 1+i, frameon=False)
#        for trial in trial_dst:
#            plot = plt.plot(trial,'-',color = '#DC143C', alpha=.1)
#            ax.tick_params(axis='both', which='major', labelsize=10)  
#
#    except Exception: 
#        continue       
#
#f.tight_layout()
#f.subplots_adjust(top = 0.87)




#plot based on te lenght  (max 40sec = 4800)



x = len(sessions_subset)

short_5 = []
med_20 = []
long_30 = []
missed_40 = []

for count in np.arange(x):
        try:      
            
            x_snippets = np.copy(x_centroid_te[count])
            
            l =len(x_snippets)
            
            trials_lenght = [[] for _ in range(l)] 
            
            for trial, values in enumerate(x_snippets):   
                
                t_lenght = len(x_snippets[trial])
                
                if t_lenght <= 600:
                    short_5.append(trial)                  
                elif t_lenght > 600 and t_lenght <= 1200:
                    med_20.append(trial)
                        
                elif t_lenght > 1200 and t_lenght <= 3600:
                            
                    long_30.append(trial)
                else:
                    
                    missed_40.append(trial)
                    
                    
        except Exception: 
            continue       
           
                            
                            
sub=sessions_dst_te[count]                        
array=np.array(sub)
f = plt.figure()
for i in short_5:
    plt.plot(sub[i],'-',color = '#DC143C', alpha=.2)
f = plt.figure()
for i in med_20:
    plt.plot(sub[i],'-',color = 'b', alpha=.2)
f = plt.figure()
for i in long_30:
    plt.plot(sub[i],'-',color = 'k', alpha=.2)
f = plt.figure()
for i in missed_40:
    plt.plot(sub[i],'-',color = 'g', alpha=.2)
             
session=sessions_subset[-1]             
quadrant_1,quadrant_2,quadrant_3,quadrant_4 = ball_positions_based_on_quadrant_of_appearance(session)

                       
array=np.array(sub)
f = plt.figure()
for i in quadrant_1:
    plt.plot(sub[i],'-',color = '#DC143C', alpha=.2)
f = plt.figure()
for i in quadrant_2:
    plt.plot(sub[i],'-',color = 'b', alpha=.2)
f = plt.figure()
for i in quadrant_3:
    plt.plot(sub[i],'-',color = 'k', alpha=.2)
f = plt.figure()
for i in quadrant_4:
    plt.plot(sub[i],'-',color = 'g', alpha=.2)





int1_5= list(set(quadrant_1).intersection(short_5))
int1_20= list(set(quadrant_1).intersection(med_20))
int1_30= list(set(quadrant_1).intersection(long_30))
int1_40= list(set(quadrant_1).intersection(missed_40))

int2_5= list(set(quadrant_2).intersection(short_5))
int2_20= list(set(quadrant_2).intersection(med_20))
int2_30= list(set(quadrant_2).intersection(long_30))
int2_40= list(set(quadrant_2).intersection(missed_40))

int3_5= list(set(quadrant_3).intersection(short_5))
int3_20= list(set(quadrant_3).intersection(med_20))
int3_30= list(set(quadrant_3).intersection(long_30))
int3_40= list(set(quadrant_3).intersection(missed_40))
    
int4_5= list(set(quadrant_4).intersection(short_5))
int4_20= list(set(quadrant_4).intersection(med_20))
int4_30= list(set(quadrant_4).intersection(long_30))
int4_40= list(set(quadrant_4).intersection(missed_40))

alpha = .8
sub=sessions_dst_st[count]  
#plot_main_title = 'Distance_Touch_to_End_40.2'
plot_main_title = 'Distance_Start_to_Touch_40.2'
number_of_subplots= np.arange(4)
f = plt.figure(figsize=(10,10))
f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()

labels = [ 'less than 5sec', 'between 5sec and 20sec', 'between 20sec and 30sec', 'greater than 30sec']

line_style = ['-','--',':', '-.']
color = ['k','b','r','g']

plt.subplot(221)
for i in int1_5 :
    plt.plot(sub[i], color = color[0], alpha=alpha, label = labels[0])
for i in int2_5 : 
    plt.plot(sub[i], color = color[1], alpha=alpha, label = labels[0])
for i in int3_5 : 
    plt.plot(sub[i], color = color[2], alpha=alpha, label = labels[0])
for i in int4_5 : 
    plt.plot(sub[i], color = color[3], alpha=alpha, label = labels[0])
plt.title(labels[0])
plt.ylabel('norm distance TE')

plt.subplot(222)
for i in int1_20 : 
    plt.plot(sub[i], color = color[0], alpha=alpha, label = labels[1])
for i in int2_20 : 
    plt.plot(sub[i], color = color[1], alpha=alpha, label = labels[1])
for i in int3_20 : 
    plt.plot(sub[i], color = color[2], alpha=alpha, label = labels[1])
for i in int4_20 : 
    plt.plot(sub[i], color = color[3], alpha=alpha, label = labels[1])
plt.title(labels[1])


plt.subplot(223)
for i in int1_30 : 
    plt.plot(sub[i], color = color[0], alpha=alpha,label = labels[2])
for i in int2_30 : 
    plt.plot(sub[i], color = color[1], alpha=alpha, label = labels[2])
for i in int3_30 : 
    plt.plot(sub[i], color = color[2], alpha=alpha, label = labels[2])
for i in int4_30 : 
    plt.plot(sub[i], color = color[3], alpha=alpha, label = labels[2])
plt.title(labels[2])
plt.xlabel('norm time')
plt.ylabel('norm distance TE')

plt.subplot(224)
for i in int1_40 : 
    plt.plot(sub[i], color = color[0], alpha=alpha, label = labels[3])
for i in int2_40 : 
    plt.plot(sub[i], color = color[1], alpha=alpha, label = labels[3])
for i in int3_40 : 
    plt.plot(sub[i], color = color[2], alpha=alpha, label = labels[3])
for i in int4_40 : 
    plt.plot(sub[i], color = color[3], alpha=alpha, label = labels[3])
plt.title(labels[3])
plt.xlabel('norm time')


custom_lines = [Line2D([0], [0], color=color[0], lw=4),
                Line2D([0], [0], color=color[1], lw=4),
                Line2D([0], [0], color=color[2], lw=4),
                Line2D([0], [0], color=color[3], lw=4)]

plt.legend(custom_lines, ['Q1', 'Q2', 'Q3', 'Q4'])










test = [len(t) for t in x_centroid_te[0]]
all_test = [[len(t) for t in s] for s in x_centroid_te]
 for t in all_test:
    plt.figure(); hist(t,40,range=(0,5000))
for t in all_test:
    hist(t,40,range=(0,5000),alpha=0.2)    
[382]: f = plt.figure()
for i,t in enumerate(all_test):
    ax=f.add_subplot(3,4,i+1); hist(t,40,range=(0,5000)); plt.ylim(0,41)
#bin_size = 100
#y_mean = []
#for value in np.arange(len(y)-bin_size):
#    x_values= y[value:(value + bin_size)]
#    xmedian = np.nanmedian(x_values)
#    y_med.append(xmedian)









