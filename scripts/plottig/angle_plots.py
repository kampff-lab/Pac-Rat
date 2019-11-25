# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:53:44 2019

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
import DLC_parser_library as DLC
import tracking_plots as tracking
import seaborn as sns  


rat_summary_table_path =  'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 
#rat_ID = 'AK_40.2'


Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
sessions_subset = Level_2_pre

#run trial idx saving first#

#behaviour.start_touch_end_idx(sessions_subset)


x_nose_snippets_te,y_nose_snippets_te,x_tail_base_snippets_te,y_tail_snippets_te = create_snippets_touch_to_end(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2)


first_x_nose, first_y_nose, first_x_tail_base, first_y_tail_base = first_x_y_at_touch(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2)


sessions_degrees =  nose_butt_angle_touch(first_x_nose,first_y_nose,first_x_tail_base,first_y_tail_base)

q_1_idx,q_2_idx,q_3_idx,q_4_idx  = quadrant_degrees(sessions_degrees)







######plot  all session for one quadrant in subplot ####

x_nose_snippets= x_nose_snippets_te            
y_nose_snippets = y_nose_snippets_te



quadrant = 'top'
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,5))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


number_of_subplots= [0,2,6]

   
for count,i in enumerate(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q1 = q_1_idx[i]
        q2 = q_2_idx[i]
        q3 = q_3_idx[i]
        q4 = q_4_idx[i]
        ax = f2.add_subplot(1, 3, 1+count, frameon=False)
        ax.invert_yaxis()
        for q in q1:
            plot= ax.plot(x[q],y[q],'-',color = '#DC143C', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)
        for qua in q2:
            plot= ax.plot(x[qua],y[qua],'-',color = '#DC143C', alpha=.2)
            #ax.tick_params(axis='both', which='major', labelsize=10)            
        for q in q3:
            plot= ax.plot(x[q],y[q],'-',color = '#1E90FF', alpha=.2)                          
        for q in q4:
            plot= ax.plot(x[q],y[q],'-',color = '#1E90FF', alpha=.2)

            
            
            
    except Exception: 
        continue           

f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        



#number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,5))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()



number_of_subplots= [0,2,6]

for count,i in enumerate(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q1 = q_1_idx[i]
        q2 = q_2_idx[i]
        q3 = q_3_idx[i]
        q4 = q_4_idx[i]
        ax = f2.add_subplot(1, 3, 1+count, frameon=False)
        ax.invert_yaxis()
        for q in q1:
            plot= ax.plot(x[q],y[q],'-',color = 'k', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)
        for qua in q2:
            plot= ax.plot(x[qua],y[qua],'-',color = 'k', alpha=.2)
            #ax.tick_params(axis='both', which='major', labelsize=10)            
        for q in q3:
            plot= ax.plot(x[q],y[q],'-',color = 'k', alpha=.2)                          
        for q in q4:
            plot= ax.plot(x[q],y[q],'-',color = 'k', alpha=.2)

            
            
            
    except Exception: 
        continue           

f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        


quadrant = 'bottom'
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()




   
for i in np.arange(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q3 = q_3_idx[i]
        q4 = q_4_idx[i]
        ax = f2.add_subplot(3, 4, 1+i, frameon=False)
        for q in q3:
            plot= ax.plot(x[q],y[q],'-',color = '#FF1493', alpha=.05)
            ax.tick_params(axis='both', which='major', labelsize=10)
            for qua in q4:
                plot= ax.plot(x[qua],y[qua],'-',color = '#1E90FF', alpha=.05)
                #ax.tick_params(axis='both', which='major', labelsize=10)
            
    except Exception: 
        continue           

f2.tight_layout()
f2.subplots_adjust(top = 0.87)










quadrant=1
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()

   
for i in np.arange(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q1 = q_1_idx[i]
        ax = f2.add_subplot(3, 4, 1+i, frameon=False)
        for q in q1:
            plot= ax.plot(x[q],y[q],'-',color = '#DC143C', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)

    except Exception: 
        continue           

f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        

#############################

quadrant = 2
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


            


   
for i in np.arange(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q2 = q_2_idx[i]
        ax = f2.add_subplot(3, 4, 1+i, frameon=False)
        for q in q2:
            plot= ax.plot(x[q],y[q],'-',color = '#228B22', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)

    except Exception: 
        continue       
    
f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        

#########################
        
    
quadrant = 3
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


            


   
for i in np.arange(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q3 = q_3_idx[i]
        ax = f2.add_subplot(3, 4, 1+i, frameon=False)
        for q in q3:
            plot= ax.plot(x[q],y[q],'-',color = '#6495ED', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)
    except Exception: 
        continue       

f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        
#############################

quadrant =4
number_of_subplots= len(sessions_degrees)

plot_main_title = 'quadrant ' + str(quadrant) + ' _angle_snippets_touch_to_end' 

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


            
   
for i in np.arange(number_of_subplots): 
    try:
        
        x=x_nose_snippets[i]
        y=y_nose_snippets[i]
        q4 = q_4_idx[i]
        ax = f2.add_subplot(3, 4, 1+i, frameon=False)
        for q in q4:
            plot= ax.plot(x[q],y[q],'-',color = 'k', alpha=.2)
            ax.tick_params(axis='both', which='major', labelsize=10)
    except Exception: 
        continue 

f2.tight_layout()
f2.subplots_adjust(top = 0.87)

            