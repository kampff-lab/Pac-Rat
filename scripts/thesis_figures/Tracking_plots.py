# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:16:04 2020

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
#import DLC_parser_library as DLC
import seaborn as sns
import matplotlib.ticker as ticker



#RATs for figure 1 = 33.2, 40.2 ,50.1

rat_summary_table_path = 'F:/Videogame_Assay/AK_49.1_behaviour_only.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_49.1'


Level_0 = prs.Level_0_paths(rat_summary_table_path)
Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
Level_3_moving = prs.Level_3_moving_light_paths(rat_summary_table_path)
Level_3_joystick = prs.Level_3_joystick_paths(rat_summary_table_path)


#main folder rat ID
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


########################PLOTS#####################################
#Level 0 hist2d and % at poke with ROI :  x = 1250 , 450  <y< 750

#figure_name0 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_0.pdf'
#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_0'

sessions_subset = Level_0
number_of_subplots= len(sessions_subset)

#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
#f0 =plt.figure(figsize=(7,6))
#f0 =plt.figure()
#f0.suptitle(plot_main_title)

percentage_at_poke_0  = []
 
for i, session in enumerate(sessions_subset): 
    try:
        
        f0 =plt.figure()
        figure_name0 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_0.pdf'
        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_0.append(percentage)
  
   
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))   
        ax = f0.add_subplot(1, 1, 1+i, frameon=False)
 
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4) #density = True
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")
        plt.ylim(0, None)
        plt.xlim(0, None)
        
        f0.tight_layout()
        f0.subplots_adjust(top = 0.87)

        #save the fig in .pdf
        f0.savefig(results_dir + figure_name0, transparent=True)
        plt.close()
                
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

#f0.tight_layout()
#f0.subplots_adjust(top = 0.87)
#
#
##save the fig in .tiff
#f0.savefig(results_dir + figure_name0, transparent=True)
#
#plt.close()

#test = np.linspace(0,1200,10)
#plt.figure()
#sns.kdeplot(x_centroid, y_centroid)
#x_ROI = []
#y_ROI = []
#
#for i in range(len(x_centroid)):
#    if x[i]>1250.0 and  450.0 <y[i] <750.0:
#        x_ROI.append(x[i])
#        y_ROI.append(y[i])
#    else:
#        continue

#percentage= (len(x)/len(x_centroid))*100    
#
#plt.figure()
#plt.imshow(image)    
#plt.plot(x_ROI,y_ROI, '.',color='y')

#plot = plt.plot(x_centroid,y_centroid, 'o', markersize=2.5,color = '#6495ED' ,alpha=0.03)

##########Level 0 over all the rats ########
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




##############################################################################
#Level 1 6000/3000 heatmap



#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_6000_3000'

sessions_subset = Level_1_6000_3000


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
#f2 =plt.figure(figsize=(20,10))
#f2.suptitle(plot_main_title)

percentage_at_poke_6000  = []
   
for i, session in enumerate(sessions_subset): 
    try:
        f2= plt.figure(i)
        #f2.suptitle(plot_main_title)
        figure_name2 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_1_6000_3000.pdf'

        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_6000.append(percentage)
 
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))
        ax = f2.add_subplot(1, 1, 1, frameon=False) 
        
        #plot = plt.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1200,10),np.arange(0,900,10)], norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")     
        plt.ylim(0, None)
        plt.xlim(0, None)

        
        f2.tight_layout()
        f2.subplots_adjust(top = 0.87)
        
        #f2.savefig(results_dir + figure_name2, transparent=True)
        print(i)
        #plt.close()

        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       


#f2.savefig(results_dir + figure_name2, transparent=True)



      
#####################################################################################       
#Level 1 10000/6000


#figure_name4 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_1_10000.pdf'

sessions_subset = Level_1_10000

number_of_subplots= len(sessions_subset)

#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_10000'
f4 =plt.figure()

percentage_at_poke_10000 = []
   
for i, session in enumerate(sessions_subset): 
    try:
        f4= plt.figure(i)
        #f2.suptitle(plot_main_title)
        figure_name4 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_1_10000.pdf'

        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_10000.append(percentage)
 
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))
        ax = f4.add_subplot(1, 1, 1, frameon=False) 
        
        #plot = plt.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1200,10),np.arange(0,900,10)], norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")     
        plt.ylim(0, None)
        plt.xlim(0, None)

        
        f4.tight_layout()
        f4.subplots_adjust(top = 0.87)
        
        f4.savefig(results_dir + figure_name4, transparent=True)
        plt.close()

        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       



#############################################################################################
#2000



sessions_subset = Level_1_20000

number_of_subplots= len(sessions_subset)

#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_20000'

#f5 =plt.figure(figsize=(20,10))
#f5.suptitle(plot_main_title)

percentage_at_poke_20000 = []
   
f5 =plt.figure()
#f0.suptitle(plot_main_title)


for i, session in enumerate(sessions_subset): 
    try:
        
        
        figure_name5 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_1_20000.pdf'
        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_20000.append(percentage)
 
    
    
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))   
        ax = f5.add_subplot(1, 1, 1+i, frameon=False)
 

        #plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1200,10),np.arange(0,900,10)], norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")
        plt.ylim(0, None)
        plt.xlim(0, None)
        
        
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f5.tight_layout()
f5.subplots_adjust(top = 0.87)


#save the fig in .tiff
f5.savefig(results_dir + figure_name5, transparent=True)

plt.close()



######################################################################

#Level 2 




sessions_subset = Level_2_pre[:5]
   


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
#f2 =plt.figure(figsize=(20,10))
#f2.suptitle(plot_main_title)

percentage_at_poke_level_2_pre  = []
   
for i, session in enumerate(sessions_subset): 
    try:
        f6= plt.figure(i)
        #f2.suptitle(plot_main_title)
        figure_name6 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_2.pdf'

        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_level_2_pre.append(percentage)
 
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))
        ax = f6.add_subplot(1, 1, 1, frameon=False) 
        
        #plot = plt.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1200,10),np.arange(0,900,10)], norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")     
        plt.ylim(0, None)
        plt.xlim(0, None)

        
        f6.tight_layout()
        f6.subplots_adjust(top = 0.87)
        
        f6.savefig(results_dir + figure_name6, transparent=True)
        print(i)
        plt.close()

        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue    


###################level 3 moving 
        


sessions_subset = Level_3_joystick
   


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
#f2 =plt.figure(figsize=(20,10))
#f2.suptitle(plot_main_title)

percentage_at_poke_level_3  = []
   
for i, session in enumerate(sessions_subset): 
    try:
        f6= plt.figure(i)
        #f2.suptitle(plot_main_title)
        figure_name6 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_3_joystick.pdf'

        x_ROI = []
        y_ROI = []
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x = centroid_tracking_wo_nan[:,0]
        y = centroid_tracking_wo_nan[:,1]
        for e in range(len(x)):
            if x[e]>1250.0 and  450.0 <y[e] <750.0:
                x_ROI.append(x[e])
                y_ROI.append(y[e])
            else:
                continue
        percentage = (len(x_ROI)/len(x))*100
        percentage_at_poke_level_3.append(percentage)
 
        x_centroid = (x - min(x))
        y_centroid = (y - min(y))
        ax = f6.add_subplot(1, 1, 1, frameon=False) 
        
        #plot = plt.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1200,10),np.arange(0,900,10)], norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plot = ax.hist2d(x_centroid, y_centroid, bins=[np.arange(0,1250,10),np.arange(0,950,10)],density = True, norm = LogNorm(), cmap='viridis',vmin=10e-8, vmax=10e-4)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(rat_ID +'_' +session[-16:-6], fontsize = 10)
        plt.xticks([0,600,1200])
        plt.yticks([0,450,900])
        sns.axes_style("white")     
        plt.ylim(0, None)
        plt.xlim(0, None)

        
        f6.tight_layout()
        f6.subplots_adjust(top = 0.87)
        
        #f6.savefig(results_dir + figure_name6, transparent=True)
        print(i)
        #plt.close()

        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue    
    
    

















###############################################

tot_percentages = np.concatenate((percentage_at_poke_0, percentage_at_poke_6000, percentage_at_poke_10000,percentage_at_poke_20000,percentage_at_poke_level_2_pre), axis= 0)
#plt.bar(range(len(tot_percentages)),tot_percentages)

color = ['#87CEEB','#00BFFF','#00BFFF','#00BFFF','#00BFFF','#4169E1','#4169E1','#0000FF', '#4B0082','#4B0082','#4B0082']

#87CEEB #sky blue
#00BFFF dodger blue
#4169E1 royal blue
#0000FF blue
f,ax = plt.subplots()
#f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)
figure_name = rat_ID +'%_at_poke.pdf' 


#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG


tot_sessions = np.size(Level_0) +  np.size(Level_1) + 3


ax.bar(range(tot_sessions), tot_percentages, color =color , edgecolor ='white', width = 1, label ='% at poke', alpha = .6)
# Create green bars (middle), on top of the firs ones


ax.legend(loc ='best', frameon=False , fontsize = 'small') #ncol=2
ax.set_ylabel('%', fontsize = 10)
plt.ylim(0,100)
f.savefig(results_dir + figure_name, transparent=True)
plt.close()

#ax[0,0].set_xlabel('Sessions')


################################
# save .csv with intermediate ROI % for each rat each session


rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv',
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                         'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']


csv_dir_path = 'F:/Videogame_Assay/Summary/Impedance/' 


#####level 0 

n = len(rat_summary_table_path)

percentage_at_poke_level_0 = [[] for _ in range(n)]



for rat in arange(n):
    try:
       

        Level_0 = prs.Level_0_paths(rat_summary_table_path[rat])    
        
        percentage_at_poke = []
        for i, session in enumerate(Level_0): 
           
            x_ROI = []
            y_ROI = []
            centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
            centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
            x = centroid_tracking_wo_nan[:,0]
            y = centroid_tracking_wo_nan[:,1]
            for e in range(len(x)):
                if x[e]>1250.0 and  450.0 <y[e] <750.0:
                    x_ROI.append(x[e])
                    y_ROI.append(y[e])
                else:
                    continue
            percentage = (len(x_ROI)/len(x))*100
            percentage_at_poke.append(percentage)
            print(session)
        percentage_at_poke_level_0[rat] = percentage_at_poke
          
        print(rat)
    except Exception: 
        print (session + '/error')
        continue

        
csv_name = 'summary_Level_0_percentage_at_poke.csv'
df_percentage_level_0 = pd.DataFrame(percentage_at_poke_level_0)
final_percentage_array_level_0 = np.array(df_percentage_level_0)

#save level 0 percentage in summary folder
np.savetxt(csv_dir_path + csv_name,final_percentage_array_level_0 , delimiter=',', fmt='%s')
        






#level 0 boxplot as example 

#test = np.array(flatten_probe[bad_channels_idx[5]])




main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


test= np.array((percentage_at_poke_level_0))

figure_name =   'summary_level_0_at_poke.pdf'


f,ax = plt.subplots(figsize=(8,7),frameon=True)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.figure(frameon=False)
plt.boxplot(test, showfliers=True)
sns.despine(top=True, right=True, left=False, bottom=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.yaxis.major.formatter._useMathText = True
plt.ylim(0,30)
f.savefig(results_dir + figure_name, transparent=True)











##########level 1
n = len(rat_summary_table_path)

percentage_at_poke_level_1 = [[] for _ in range(n)]



for rat in range(n):
    try:
       

        Level_1 = prs.Level_1_paths(rat_summary_table_path[rat])    
        
        percentage_at_poke = []
        for i, session in enumerate(Level_1): 
           
            x_ROI = []
            y_ROI = []
            centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
            centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
            x = centroid_tracking_wo_nan[:,0]
            y = centroid_tracking_wo_nan[:,1]
            for e in range(len(x)):
                if x[e]>1250.0 and  450.0 <y[e] <750.0:
                    x_ROI.append(x[e])
                    y_ROI.append(y[e])
                else:
                    continue
            percentage = (len(x_ROI)/len(x))*100
            percentage_at_poke.append(percentage)
            print(session)
        percentage_at_poke_level_1[rat] = percentage_at_poke
          
        print(rat)
    except Exception: 
        print (session + '/error')
        continue

      
        
csv_name = 'summary_Level_1_percentage_at_poke.csv'
df_percentage_level_1 = pd.DataFrame(percentage_at_poke_level_1)
final_percentage_array_level_1 = np.array(df_percentage_level_1)

#save summary csv file in summary folder 
np.savetxt(csv_dir_path + csv_name,final_percentage_array_level_1 , delimiter=',', fmt='%s')
        





##########level 2
        
n = len(rat_summary_table_path)

percentage_at_poke_level_2 = [[] for _ in range(n)]



for rat in range(n):
    try:
       

        Level_2 = prs.Level_2_pre_paths(rat_summary_table_path[rat])    
        
        percentage_at_poke = []
        for i, session in enumerate(Level_2): 
           
            x_ROI = []
            y_ROI = []
            centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
            centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
            x = centroid_tracking_wo_nan[:,0]
            y = centroid_tracking_wo_nan[:,1]
            for e in range(len(x)):
                if x[e]>1250.0 and  450.0 <y[e] <750.0:
                    x_ROI.append(x[e])
                    y_ROI.append(y[e])
                else:
                    continue
            percentage = (len(x_ROI)/len(x))*100
            percentage_at_poke.append(percentage)
            print(session)
        percentage_at_poke_level_2[rat] = percentage_at_poke
          
        print(rat)
    except Exception: 
        print (session + '/error')
        continue

           



csv_name = 'summary_Level_2_percentage_at_poke.csv'
df_percentage_level_2 = pd.DataFrame(percentage_at_poke_level_2)
final_percentage_array_level_2 = np.array(df_percentage_level_2)

#save summary csv file in summary folder 
np.savetxt(csv_dir_path + csv_name,final_percentage_array_level_2 , delimiter=',', fmt='%s')
        





#plotting from file 



rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv',
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                         'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']






poke_files = ['F:/Videogame_Assay/Summary/Behaviour/summary_Level_1_percentage_at_poke.csv',
              'F:/Videogame_Assay/Summary/Behaviour/summary_Level_2_percentage_at_poke.csv']



test_open = np.genfromtxt(poke_files[0],delimiter=',')




main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)



figure_name =   'summary_level_1_at_poke_with_mask.pdf'

file = poke_files[0]
at_poke = np.genfromtxt(file,delimiter=',')


####test mask nana 
# Filter data using np.isnan
mask = ~np.isnan(at_poke)
filtered_data = [d[m] for d, m in zip(at_poke.T, mask.T)]





f,ax = plt.subplots(figsize=(8,7))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.figure(frameon=False)
plt.boxplot(filtered_data, showfliers=True)
sns.despine(top=True, right=True, left=False, bottom=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.yaxis.major.formatter._useMathText = True
plt.ylim(0,100)



f.savefig(results_dir + figure_name, transparent=True)



t_test_1_4 = stats.ttest_rel(filtered_data[0],filtered_data[3])
t_test_4_6 = stats.ttest_rel(filtered_data[3],filtered_data[5])
t_test_6_8 = stats.ttest_rel(filtered_data[5],filtered_data[7])
t_test_4_8 = stats.ttest_rel(filtered_data[3],filtered_data[7])


tot_trials=[]
for t in np.arange(len(filtered_data)):
    tot_trials.append(len(filtered_data[t]))

target = open(main_folder +"stats_level_1_%_at_poke.txt", 'w')
target.writelines('_1vs4_'+ str(t_test_1_4) + '_4vs6_'+str(t_test_4_6)+'_6vs8_'+str(t_test_6_8)+'_4vs8_'+str(t_test_4_8)+' PLOT: boxplot % at poke level 1, rel t test, Tracking_plots.py')

target.close()




#######lines


colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


figure_name =   'summary_level_1_at_poke_lines.pdf'
    
f,ax = plt.subplots(figsize=(10,8))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for count, row in enumerate(filtered_data):
    
    plt.plot(row.T, color = colours[count], marker = 'o', alpha = .3, label = RAT_ID[count])
    plt.title('Level 2 Trial/Min', fontsize = 16)
    plt.ylabel('Trial/Min', fontsize = 13)
    plt.xlabel('Level 2 Sessions', fontsize = 13)
    #plt.xticks((np.arange(0, 5, 1)))
    #plt.xlim(-0.1,4.5)
    #plt.yticks((np.arange(0, 4, .5)))
   
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.ylim(0,3)
    #plt.legend()
    f.tight_layout()


#f.savefig(results_dir + figure_name, transparent=True)       
    
mean_trial_speed = np.nanmean(filtered_data, axis=0)

#sem = stats.sem(filtered_data, nan_policy='omit', axis=0)


plt.plot(mean_trial_speed,marker = 'o',color= 'k')
#plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
plt.errorbar(range(5), mean_trial_speed, yerr= sem, fmt='o', ecolor='k',color='k', capsize=2) 




























#################################level 2 


file = poke_files[1]
at_poke_L2= np.genfromtxt(file,delimiter=',')

mask = ~np.isnan(at_poke_L2)
filtered_data_L2 = [d[m] for d, m in zip(at_poke_L2.T, mask.T)]

figure_name =   'summary_level_2_at_poke_with_mask.pdf'


f,ax = plt.subplots(figsize=(8,7),frameon=True)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.figure(frameon=False)
plt.boxplot(filtered_data_L2[:5], showfliers=True) # [:,:5] for the first 5 days only
sns.despine(top=True, right=True, left=False, bottom=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.yaxis.major.formatter._useMathText = True
plt.ylim(0,100)


f.savefig(results_dir + figure_name, transparent=True)


t_test_0_4 = stats.ttest_rel(filtered_data_L2[0],filtered_data_L2[4])

tot_trials=[]
for t in np.arange(filtered_data_L2):
    tot_trials.append(len(filtered_data_L2[t]))




target = open(main_folder +"stats_level_2_%_at_poke.txt", 'w')
target.writelines('_0vs4_'+ str(t_test_0_4) + 'PLOT: boxplot % at poke level 1, rel t test, Tracking_plots.py')

target.close()

#b = np.zeros([len(percentage_at_poke_level_1),len(max(percentage_at_poke_level_1,key = lambda x: len(x)))])
#for i,j in enumerate(percentage_at_poke_level_1):
#    b[i][0:len(j)] = j






















    
np.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx)).T, delimiter=',', fmt='%s')


'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/events'



#main folder rat ID
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


########################PLOTS#####################################
#Level 0 hist2d and % at poke with ROI :  x = 1250 , 450  <y< 750

#figure_name0 =  'RAT_' + rat_ID + '_'+ session[-16:] +'_lognorm_heatmap_Level_0.pdf'
#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_0'

sessions_subset = Level_0
number_of_subplots= len(sessions_subset)


##########################################################################################

#figure_name4 = 'RAT_' + rat_ID + '_Centroid_tracking_Level_2_pre.pdf'
#sessions_subset = Level_2_pre
#
#number_of_subplots= len(sessions_subset)
#
#plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_2_pre'
#
#f4 =plt.figure(figsize=(20,10))
#f4.suptitle(plot_main_title)
#
#
#   
#for i, session in enumerate(sessions_subset): 
#    try:
#        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
#        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
#        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
#        x_centroid = centroid_tracking_wo_nan[:,0]
#        y_centroid = centroid_tracking_wo_nan[:,1]    
#        ax = f4.add_subplot(2, 4, 1+i, frameon=False)
#        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
#        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
#        ax.set_title(session[-16:-6], fontsize = 13)
#        #ax.set_ylabel('Trials / Session', fontsize = 10)
#        #ax.set_xlabel('Sessions', fontsize = 10)
#    except Exception: 
#        print (session + '/error')
#        continue       
#
#f4.tight_layout()
#f4.subplots_adjust(top = 0.87)
#
#
######SAVINGS#######
#
#
##main folder rat ID
#script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + rat_ID)
##create a folder where to store the plots 
#main_folder = os.path.join(script_dir +'/Summary')
##create a folder where to save the plots
#results_dir = os.path.join(main_folder + '/Behaviour/')
#
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
##save the fig in .tiff
#f0.savefig(results_dir + figure_name0, transparent=True)
#f1.savefig(results_dir + figure_name1, transparent=True)
#f2.savefig(results_dir + figure_name2, transparent=True)
#f3.savefig(results_dir + figure_name3, transparent=True)
#f4.savefig(results_dir + figure_name4, transparent=True)
##f.savefig(results_dir + figure_name)      
#    
