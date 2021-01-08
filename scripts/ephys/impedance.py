# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:47:05 2018

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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors
import seaborn as sns
import os
import glob
from scipy import stats
import ephys_library as ephys
hardrive_path = r'F:/' 


probe_map=np.array([[103,78,81,118,94,74,62,24,49,46,7],
                    [121,80,79,102,64,52,32,8,47,48,25],
                    [123,83,71,104,66,84,38,6,26,59,23],
                    [105,69,100,120,88,42,60,22,57,45,5],
                    [101,76,89,127,92,67,56,29,4,37,9],
                    [119,91,122,99,70,61,34,1,39,50,27],
                    [112,82,73,97,68,93,40,3,28,51,21],
                    [107,77,98,125,86,35,58,31,55,44,14],
                    [110,113,87,126,90,65,54,20,2,43,11],
                    [117,85,124,106,72,63,36,0,41,15,16],
                    [114,111,75,96,116,95,33,10,30,53,17]])


RAT_ID = '33.2'
rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)

sessions_subset = Level_2_post



flatten_probe = ephys.probe_map.flatten()

np.set_printoptions(suppress=True)

mean_impedance_Level_2_post = np.zeros((len(sessions_subset),121))
sem_impedance_Level_2_post = np.zeros((len(sessions_subset),121))



for s, session in enumerate(sessions_subset):
    try:
        #list_impedance_path = []
        impedance_path = os.path.join(hardrive_path, session)
        matching_files_daily_imp = glob.glob(impedance_path + "\*imp*") 
        #for matching_file in matching_files:
           # list_impedance_path.append(matching_file)
            
        impedance_list_array=np.array(matching_files_daily_imp)
        session_all_measurements = np.zeros((len(impedance_list_array), 121))
        
        for i, imp in enumerate(impedance_list_array):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            session_all_measurements[i,:] = imp_remapped
                
    
        mean_imp_session= np.mean(session_all_measurements,axis=0)
        sem_imp= stats.sem(session_all_measurements,axis=0)
        
        mean_impedance_Level_2_post[s,:]=mean_imp_session
        sem_impedance_Level_2_post[s,:]=sem_imp
        print(session)

    except Exception: 
        continue       


#remove the double sessions which do not contain the impedance measures 

mask = (np.nan_to_num(mean_impedance_Level_2_post) != 0).any(axis=1)

final_mean_impedance_Level_2_post = mean_impedance_Level_2_post[mask]
final_sem_impedance_Level_2_post = sem_impedance_Level_2_post[mask]






#find outlier channels

       
bad_channels_idx = [[] for _ in range(len(final_mean_impedance_Level_2_post))] 


for count in range(len(final_mean_impedance_Level_2_post)):

    idx_bad_imp = [idx for idx, val in enumerate(final_mean_impedance_Level_2_post[count]) if val > 6000000 or val < 10000 ] #6000000 
    print (min(final_mean_impedance_Level_2_post[count]))
    print (max(final_mean_impedance_Level_2_post[count]))
    if idx_bad_imp == 0 :
        
        bad_channels_idx[count] = []
    else:
       bad_channels_idx[count] = idx_bad_imp 
    



#test = np.array(flatten_probe[bad_channels_idx[5]])
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)




figure_name =  RAT_ID + '_impedance_level2_without.pdf'


f,ax = plt.subplots(figsize=(15,11),frameon=False)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

mean_imp = final_mean_impedance_Level_2_post.tolist()


#plt.figure(frameon=False)
plt.boxplot(mean_imp, showfliers=False)
sns.despine(top=True, right=True, left=False, bottom=False)

ax.yaxis.major.formatter._useMathText = True

f.savefig(results_dir + figure_name, transparent=True)


# plt.figure(frameon=False)
# sns.set(style ="white")
# sns.despine(left=True)
# sns.boxplot(data=mean_imp , linewidth=1)
# 
# sns.despine(top=True, right=True, left=False, bottom=False)
# plt.xlabel('days')
# plt.ylabel('impedance_(ohms)')
# 

















#################################################################################

  


#saline and surgery and day 1
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


surgery_rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                           'F:/Videogame_Assay/AK_48.1_IrO2.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv']
                      
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

n = len(surgery_rat_summary_table_path)
hardrive_path = r'F:/' 



for rat in range(n):
    try:
    
        Level_2_post = prs.Level_2_post_paths(surgery_rat_summary_table_path[rat])
        sessions_subset = Level_2_post
         
        first_session = sessions_subset[0]      
        impedance_path_first_session = os.path.join(hardrive_path, first_session)    
    
        matching_files_test_probe = glob.glob(impedance_path_first_session + "\*saline*") 
        matching_files_surgery_day = glob.glob(impedance_path_first_session + "\*surgery*") 
        matching_files_daily_imp = glob.glob(impedance_path_first_session + "\*imp*") 
        
        #saline
        
        saline_all_measurements = np.zeros((len(matching_files_test_probe), 121))
        
        for i, imp in enumerate(matching_files_test_probe):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            saline_all_measurements[i,:] = imp_remapped
                
        mean_saline = np.mean(saline_all_measurements,axis=0)
        sem_imp_saline = stats.sem(saline_all_measurements,axis=0)

        figure_name =  RAT_ID[rat] + '_impedance_saline_without_outlier_15000.pdf'

        f,ax = plt.subplots(figsize=(15,11),frameon=False)
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=True)
        
        saline_to_plot = mean_saline.tolist()
        flierprops = dict(marker='o', markerfacecolor='k', markersize=2, linestyle='none')
        #plt.figure(frameon=False)
        plt.boxplot(saline_to_plot, showfliers=True,flierprops=flierprops )
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        #ax.yaxis.major.formatter._useMathText = True
        plt.title('saline')
        plt.ylim(15000,70000)
        
        f.savefig(results_dir + figure_name, transparent=True)
        plt.close()

        
        
        
        #surgery 
        
        
        surgery_all_measurements = np.zeros((len(matching_files_surgery_day), 121))
        
        for i, imp in enumerate(matching_files_surgery_day):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            surgery_all_measurements[i,:] = imp_remapped
                
        mean_surgery = np.mean(surgery_all_measurements,axis=0)
        sem_imp_surgery = stats.sem(surgery_all_measurements,axis=0)
    

       #day 1

        day_1_all_measurements = np.zeros((len(matching_files_daily_imp), 121))
        
        for i, imp in enumerate(matching_files_daily_imp):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            day_1_all_measurements[i,:] = imp_remapped
                
    
        mean_imp_day_1= np.mean(day_1_all_measurements,axis=0)
        sem_imp_day_1 = stats.sem(day_1_all_measurements,axis=0)
        
        
        
        data_stack = np.vstack((mean_saline,mean_surgery,mean_imp_day_1))
        data_to_plot = data_stack.tolist()
        
        figure_name =  RAT_ID[rat] + '_impedance_saline_surgery_day1_without_outlier.pdf'

        f,ax = plt.subplots(figsize=(15,11),frameon=False)
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=True)
        
        
        
        #plt.figure(frameon=False)
        plt.boxplot(data_to_plot, showfliers=True, flierprops=flierprops)
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        #ax.yaxis.major.formatter._useMathText = True
        plt.title('saline'+'_surgery'+ '_day_1')
        #plt.ylim(0,2000000)
        
        f.savefig(results_dir + figure_name, transparent=True)
        plt.close()
        print(rat)

    except Exception: 
        continue       






#################################################

#data_stack = np.vstack((mean_saline,mean_surgery,mean_imp_day_1))
#
#        data_to_plot = data_stack.tolist()
#
#
#figure_name =  RAT_ID + '_impedance_saline_surgery_without_outlier.pdf'
#
#
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#
#
##plt.figure(frameon=False)
#plt.boxplot(data_to_plot, showfliers=False)
#sns.despine(top=True, right=True, left=False, bottom=False)
#
#ax.yaxis.major.formatter._useMathText = True
#plt.title('saline'+'_surgery'+ '_day_1')
##plt.ylim(0,2000000)
#
#f.savefig(results_dir + figure_name, transparent=True)
#
#
##heatmap saline  VS surgey VS day 1
#
#N=121
##map the impedance
#probe_remap=np.reshape(probe_map.T, newshape=N)
#
##test saline
#
#
#saline_impedance_map=np.reshape(mean_saline,newshape=probe_map.shape)
#
#threshold = 100000
#
#
#figure_name = RAT_ID + 'saline_test_impedance_heatmap.pdf'
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#
#ax = sns.heatmap(saline_impedance_map,annot=True,  cmap="YlGnBu", vmin=0, vmax=threshold, annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)
#
#
#f.savefig(results_dir + figure_name, transparent=True)
#
#
#
##test surgery
#
#
#surgery_impedance_map=np.reshape(mean_surgery,newshape=probe_map.shape)
#
#threshold = 100000
#figure_name = RAT_ID + 'surgery_test_impedance_heatmap.pdf'
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#
#ax = sns.heatmap(surgery_impedance_map,annot=True,  cmap="YlGnBu",vmin=0, vmax = 500000,annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)
#
#
#f.savefig(results_dir + figure_name, transparent=True)
#
#












#######################################################################################

# omnetics examples before and after pedot 
#folder located at day 1 of recording together with the saline test and surgery impedances

def avg_sem_omnetics_imp(omnetics_list, match = "\*sal*" ):

    probe_mean = np.zeros((len(omnetics_list), 32))
    probe_sem =  np.zeros((len(omnetics_list), 32))
    
    for o, omnetics in enumerate(omnetics_list):
    
        search_file = glob.glob(omnetics + match) 
        avg_impedance =  avg_imp(search_file)
        mean = np.mean(avg_impedance,axis =0)
        probe_mean[o,:] = mean
        sem = stats.sem(avg_impedance,axis =0)
        probe_sem[o,:] = sem    
        
    return probe_mean, probe_sem
    
   
    
def avg_imp(impedance_file):
    
    impedance = np.zeros((3,32))
    
    for f, file in enumerate(impedance_file):
        open_file = np.genfromtxt(file, dtype=float, skip_header =2, usecols =1 )
        final_file = open_file[:32]
        impedance[f] = final_file
    
    return impedance


###################################
    

main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


surgery_rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                           'F:/Videogame_Assay/AK_48.1_IrO2.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv']
                      
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

folder_list = ['/Probe_61_rat_33.2' , '/Probe_62_rat_40.2', '/Probe_14_rat_41.1', '/Probe_15_rat_41.2','/probe_222_rat_48.1','/Probe_220_rat_48.4']       

n = len(surgery_rat_summary_table_path)
hardrive_path = r'F:/' 




for rat in range(n):
    
    try:
        
        Level_2_post = prs.Level_2_post_paths(surgery_rat_summary_table_path[rat])
        sessions_subset = Level_2_post
         
        first_session = sessions_subset[0]      
        probe_path_first_session = os.path.join(hardrive_path, first_session) 


        probe_path =  os.path.join(probe_path_first_session + folder_list[rat])

        omnetics_list = glob.glob(probe_path + "\*omnetics*") 




        saline_probe_mean, saline_probe_sem =  avg_sem_omnetics_imp(omnetics_list, match = "\*sal*" )

        PEDOT_probe_mean, PEDOT_probe_sem =  avg_sem_omnetics_imp(omnetics_list, match = "\*post*" )

    

        probe_stack = np.vstack((saline_probe_mean,PEDOT_probe_mean))
        
        
        #allows alternating omnetics
        #test = np.hstack([saline_probe_mean, PEDOT_probe_mean]).reshape(8, 32)

        probe_to_plot = probe_stack.tolist()



        figure_name =  RAT_ID[rat] + '_impedance_saline_VS_PEDOT.pdf'

        f,ax = plt.subplots(figsize=(15,11),frameon=False)
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=True)
        
        
        
        #plt.figure(frameon=False)
        plt.boxplot(probe_to_plot, showfliers=False)
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        #ax.yaxis.major.formatter._useMathText = True
        plt.title('saline'+'pedot')
        #plt.ylim(0,2000000)
        
        f.savefig(results_dir + figure_name, transparent=True)
        plt.close()
        
        
        figure_name =  RAT_ID[rat] + '_impedance_PEDOT.pdf'

        f,ax = plt.subplots(figsize=(15,11),frameon=False)
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=True)
        
        
        
        #plt.figure(frameon=False)
        plt.boxplot(PEDOT_probe_mean.tolist(), showfliers=False)
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        #ax.yaxis.major.formatter._useMathText = True
        plt.title('saline'+'pedot')
        #plt.ylim(0,0.06)
        
        
        
        #f.savefig(results_dir + figure_name, transparent=True)
   
        #plt.close()
        print(rat)

    except Exception: 
        continue       









##############################################################################    
###platinum         
        
surgery_rat_summary_table_path_Pt = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv']
                           
                      
RAT_ID_Pt = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2'] 

folder_list_Pt = ['/Probe_61_rat_33.2' , '/Probe_62_rat_40.2', '/Probe_14_rat_41.1', '/Probe_15_rat_41.2']       
folder_list = folder_list_Pt

# omnetics examples before and after pedot 
#folder located at day 1 of recording together with the saline test and surgery impedances
n =len(surgery_rat_summary_table_path_Pt)


saline_Pt = []
saline_sem = []
pedot_Pt = []
pedot_sem = []

for rat in range(n):
    
    try:
        
        Level_2_post = prs.Level_2_post_paths(surgery_rat_summary_table_path_Pt[rat])
        sessions_subset = Level_2_post
         
        first_session = sessions_subset[0]      
        probe_path_first_session = os.path.join(hardrive_path, first_session) 


        probe_path =  os.path.join(probe_path_first_session + folder_list[rat])

        omnetics_list = glob.glob(probe_path + "\*omnetics*") 


        match = "\*sal*"  



        for o, omnetics in enumerate(omnetics_list):
        
            search_file = glob.glob(omnetics + match) 
            avg_impedance =  avg_imp(search_file)
            mean = np.mean(avg_impedance,axis =0)
            saline_Pt.append(mean)
            #saline_sem = stats.sem(avg_impedance,axis =0)
            
            

        match = "\*pos*"  



        for o, omnetics in enumerate(omnetics_list):
        
            search_file = glob.glob(omnetics + match) 
            avg_impedance =  avg_imp(search_file)
            mean = np.mean(avg_impedance,axis =0)
            pedot_Pt.append(mean)
            #pedot_sem = stats.sem(avg_impedance,axis =0)
            

        print(rat)

    except Exception: 
        continue       
       




flat_saline_Pt = np.array(saline_Pt).flatten()

flat_pedot_Pt = np.array(pedot_Pt).flatten()



#THESIS
#scatter of all the data point connected 


from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.ticker as ticker


figure_name = 'ticks_log_'+'platinum_saline_VS_pedot_all_data_full_dots.pdf'

f,ax = plt.subplots(figsize=(5,7))


x_axis_saline = np.array([0] * len(flat_saline_Pt))
x_axis_pedot = np.array([2] *len(flat_pedot_Pt))


plt.scatter(x_axis_saline, flat_saline_Pt,s=20, facecolors='r', edgecolors='r')

plt.scatter(x_axis_pedot, flat_pedot_Pt,s=20, facecolors='g', edgecolors='g')

plt.plot((x_axis_saline,x_axis_pedot),(flat_saline_Pt,flat_pedot_Pt), color='k', alpha = .3 , linewidth= .2)
plt.yscale('log')
ax.set_xlim(-.5,2.5)
ax.set_ylim(0.001,100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

# Save the figure and show
plt.tight_layout()


        
f.savefig(results_dir + figure_name, transparent=True)
plt.close()



#plt.minorticks_off() 
######################################data cleaning
#THESIS


th_max = 6
th_min = 0.01
 

Pt_ok = [i for i,v in enumerate(flat_saline_Pt) if  th_min< v < th_max]

flat_saline_Pt_cleaned = flat_saline_Pt[Pt_ok] 
flat_pedot_Pt_cleaned = flat_pedot_Pt[Pt_ok]
print(len(flat_saline_Pt_cleaned))

    
mean_saline_Pt = np.mean(flat_saline_Pt_cleaned) #1.35574609375
sem_saline_Pt =  stats.sem(flat_saline_Pt_cleaned)



mean_pedot_Pt = np.mean(flat_pedot_Pt_cleaned) #0.8658209635416667
sem_pedot_Pt = stats.sem(flat_pedot_Pt_cleaned)


test_Pt = stats.ttest_rel(flat_saline_Pt_cleaned,flat_pedot_Pt_cleaned)



target = open(main_folder +"stats_Pt_saline_VS_PEDOT_thmax_6_thmin_001_mean_and_sem.txt", 'w')
target.writelines(str(test_Pt) + 'saline:'+ str(mean_saline_Pt) +'_'+ str(sem_saline_Pt) +'pedot:' + str(mean_pedot_Pt) +'_'+str(sem_pedot_Pt)+
                  'Pt_saline_VS_PEDOT, PLOT: barplot mean+- SEM, n = 481, thmax=6 thmin = 0.01, impedance.py')

target.close()


#th min 0.01 and t  max 6 =  Ttest_relResult(statistic=16.935291422777482, pvalue=8.846732106309465e-51) n=481
#th 5 =  Ttest_relResult(statistic=19.698943557103416, pvalue=1.1115078141741984e-63) n = 479
#th 6 =  Ttest_relResult(statistic=16.924951583851268, pvalue=9.417706940703045e-51) n=482
#th 8 =  Ttest_relResult(statistic=15.086107783897232, pvalue=2.052201403381959e-42)  n = 484
#th 10 =   Ttest_relResult(statistic=14.825585811695863, pvalue=2.547211031857156e-41)
#th 1 = statistic=25.58967226582703, pvalue=8.682862522544012e-91 n=466
##############################

#THESIS

materials = ['Saline', 'PEDOT']
x_pos = np.arange(len(materials))
means_Pt = [mean_saline_Pt, sem_saline_Pt]
errors_Pt = [sem_saline_Pt, sem_pedot_Pt]


figure_name = 'th_'+ str(th_max) +str(th_min) +'_platinum_saline_VS_pedot.pdf'

f,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(5,7))

#f,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(5,7))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)


#ax1.bar(x_pos[0], means_Pt[0], yerr=errors_Pt[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)




ax1.bar(x_pos[0], means_Pt[0], yerr=errors_Pt[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
ax2.bar(x_pos[0], means_Pt[0], yerr=errors_Pt[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)

ax2.bar(x_pos[1], means_Pt[1], yerr=errors_Pt[1], align='center', color ='g', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
#ax.set_ylabel('impedance')
#ax.set_xticks(x_pos)
#ax.set_xticklabels(materials)
#ax.set_title('saline_VS_pedot_iridium_probe')
plt.yticks(fontsize=15)
ax2.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')

ax1.tick_params(bottom='off', left='off', labelleft='on')
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax2.spines['left'].set_visible(True)
plt.xlim(-.5,1.5)

ax2.set_ylim(0,.1)
#ax1.set_ylim(0,.6)
ax1.set_ylim(0.4,0.7)
#ax1.set_yticks(np.arange(1000,1501,100))
ax1.tick_params(axis='both', which='major', pad=15)
ax2.tick_params(axis='both', which='major', pad=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)


for tick in ax2.get_xticklabels():
    tick.set_rotation(0)
d = .015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)      
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
plt.yticks(fontsize=15)
plt.show()

# Save the figure and show
plt.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)
plt.close()
















            
#####IrOx THESIS

surgery_rat_summary_table_path_IrOx = ['F:/Videogame_Assay/AK_48.1_IrO2.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv']

['F:/Videogame_Assay/AK_48.1_IrO2.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv']

RAT_ID_IrO2 = ['AK 48.1','AK 48.4']

folder_list_IrO2 = ['/probe_222_rat_48.1','/Probe_220_rat_48.4']

folder_list = folder_list_IrO2

n= len(surgery_rat_summary_table_path_IrOx)


saline_Ir = []
saline_sem = []
pedot_Ir = []
pedot_sem = []

for rat in range(n):
    
    try:
        
        Level_2_post = prs.Level_2_post_paths(surgery_rat_summary_table_path_IrOx[rat])
        sessions_subset = Level_2_post
         
        first_session = sessions_subset[0]      
        probe_path_first_session = os.path.join(hardrive_path, first_session) 


        probe_path =  os.path.join(probe_path_first_session + folder_list[rat])

        omnetics_list = glob.glob(probe_path + "\*omnetics*") 


        match = "\*sal*"  



        for o, omnetics in enumerate(omnetics_list):
        
            search_file = glob.glob(omnetics + match) 
            avg_impedance =  avg_imp(search_file)
            mean = np.mean(avg_impedance,axis =0)
            saline_Ir.append(mean)
            #saline_sem = stats.sem(avg_impedance,axis =0)
            
            

        match = "\*pos*"  



        for o, omnetics in enumerate(omnetics_list):
        
            search_file = glob.glob(omnetics + match) 
            avg_impedance =  avg_imp(search_file)
            mean = np.mean(avg_impedance,axis =0)
            pedot_Ir.append(mean)
            #pedot_sem = stats.sem(avg_impedance,axis =0)
            

        print(rat)

    except Exception: 
        continue       



flat_saline_Ir = np.array(saline_Ir).flatten()

flat_pedot_Ir = np.array(pedot_Ir).flatten()



figure_name = 'tickslog_'+'iridium_saline_VS_pedot_all_data_full_dots.pdf'

f,ax = plt.subplots(figsize=(5,7))

x_axis_saline = np.array([0] * len(flat_saline_Ir))
x_axis_pedot = np.array([2] *len(flat_pedot_Ir))


plt.scatter(x_axis_saline, flat_saline_Ir,s=20, facecolors='r', edgecolors='r')

plt.scatter(x_axis_pedot, flat_pedot_Ir,s=20, facecolors='g', edgecolors='g')

plt.plot((x_axis_saline,x_axis_pedot),(flat_saline_Ir,flat_pedot_Ir), color='k', alpha = .3 , linewidth= .2)


plt.yscale('log')
ax.set_xlim(-.5,2.5)
ax.set_ylim(0.001,100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

# Save the figure and show

plt.tight_layout()


        
f.savefig(results_dir + figure_name, transparent=True)
plt.close()

__


#####################cleaning con th 5 e 10
#THESIS

#th = 6 # as well as for ephys pre processing

th_max = 6
th_min = 0.01
 

Ir_ok = [i for i,v in enumerate(flat_saline_Ir) if th_min< v < th_max]

flat_saline_Ir_cleaned = flat_saline_Ir[Ir_ok]   
flat_pedot_Ir_cleaned = flat_pedot_Ir[Ir_ok]
print(len(flat_saline_Ir_cleaned))


  
mean_saline_Ir = np.mean(flat_saline_Ir_cleaned) 
sem_saline_Ir =  stats.sem(flat_saline_Ir_cleaned)


mean_pedot_Ir = np.mean(flat_pedot_Ir_cleaned) 
sem_pedot_Ir = stats.sem(flat_pedot_Ir_cleaned)


test_ir = stats.ttest_rel(flat_saline_Ir_cleaned,flat_pedot_Ir_cleaned)

#th min 0.01 th max 6  Ttest_relResult(statistic=26.75295406609847, pvalue=5.99192743175084e-71) n=220
#th 5 = statistic=28.456691199228985, pvalue=2.387479347949912e-75  n = 219
#th 6 = Ttest_relResult(statistic=26.75295406609847, pvalue=5.99192743175084e-71) n =220
#th 8 = Ttest_relResult(statistic=26.75295406609847, pvalue=5.99192743175084e-71) n = 220
#th 10 = Ttest_relResult(statistic=26.75295406609847, pvalue=5.99192743175084e-71) n = 220
# th 1 =  Ttest_relResult(statistic=20.991170460032713, pvalue=6.457309854275329e-31n = 67




#Ttest_relResult(statistic=-5.1325785753223885, pvalue=0.0003269935176528684)


target = open(main_folder +"stats_IrOx_saline_VS_PEDOT_th_6_tmin_001_mean_and_sem.txt", 'w')
target.writelines(str(test_ir) + 'saline:'+ str( mean_saline_Ir)+'_'+str(sem_saline_Ir)+ '_'+str(mean_pedot_Ir) + '_'+ str(sem_pedot_Ir)+
                  ' IrOx_saline_VS_PEDOT, PLOT: barplot mean +- SEM, n=220, th max =6,th min =0.01, impedance.py')

target.close()




#####THESIS




materials = ['Saline', 'PEDOT']
x_pos = np.arange(len(materials))
means_Ir = [mean_saline_Ir, mean_pedot_Ir]
errors_Ir = [sem_saline_Ir, sem_pedot_Ir]


figure_name = 'th_' +str(th_min) +str(th_max)+'iridium_saline_VS_pedot.pdf'

f,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(5,7))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)


ax1.bar(x_pos[0], means_Ir[0], yerr=errors_Ir[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
ax2.bar(x_pos[0], means_Ir[0], yerr=errors_Ir[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)

ax2.bar(x_pos[1], means_Ir[1], yerr=errors_Ir[1], align='center', color ='g', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
#ax.set_ylabel('impedance')
#ax.set_xticks(x_pos)
#ax.set_xticklabels(materials)
#ax.set_title('saline_VS_pedot_iridium_probe')
ax2.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')

ax1.tick_params(bottom='off', left='off', labelleft='on')

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax2.spines['left'].set_visible(True)
plt.xlim(-.5,1.5)

ax2.set_ylim(0,.1)
#ax1.set_ylim(0,.6)
ax1.set_ylim(.4,1.6)
#ax1.set_yticks(np.arange(1000,1501,100))
ax1.tick_params(axis='both', which='major', pad=15)
ax2.tick_params(axis='both', which='major', pad=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)



for tick in ax2.get_xticklabels():
    tick.set_rotation(0)
d = .015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)      
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
plt.show()

# Save the figure and show
plt.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)
plt.close()





#final_Ir = (np.vstack((np.array(saline_Ir).flatten(),np.array(pedot_Ir).flatten()))).T
#
#plt.boxplot(final_Ir, showfliers=False)
#
#














###################################################################################



surgery_rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                                'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                                'F:/Videogame_Assay/AK_48.1_IrO2.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv']
                      
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']



flatten_probe = ephys.probe_map.flatten()

#np.set_printoptions(suppress=True)


all_rats_daily_impedance = [[] for _ in range(len(surgery_rat_summary_table_path))]
all_rats_saline = [[] for _ in range(len(surgery_rat_summary_table_path))]
all_rats_surgery = [[] for _ in range(len(surgery_rat_summary_table_path))]



for rat in range(len(surgery_rat_summary_table_path)):
    
    try:
        
        csv_dir_path = 'F:/Videogame_Assay/Summary/Impedance/' 
        csv_name = RAT_ID[rat]+'_impedance_summary.csv'
        All_levels_post = prs.all_post_surgery_levels_paths(surgery_rat_summary_table_path[rat])
        sessions_subset = All_levels_post
        rat_daily_impedance = [[] for _ in range(len(All_levels_post))]
        rat_saline = []
        rat_surgery = []
        
        first_session = sessions_subset[0]      
        impedance_path_first_session = os.path.join(hardrive_path, first_session) 
        
        #saline
               
        matching_files_test_probe = glob.glob(impedance_path_first_session + "\*saline*") 
        
        saline_all_measurements = np.zeros((len(matching_files_test_probe), 121))
        
        for i, imp in enumerate(matching_files_test_probe):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            saline_all_measurements[i,:] = imp_remapped
                
        mean_saline = np.mean(saline_all_measurements,axis=0)
        rat_saline.append(mean_saline)
        
        print('saline_done')
        #surgery
        
        matching_files_surgery_day = glob.glob(impedance_path_first_session + "\*surgery*")
        
        surgery_all_measurements = np.zeros((len(matching_files_surgery_day), 121))
        
        for i, imp in enumerate(matching_files_surgery_day):
            read_file = pd.read_csv(imp)
            impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
            imp_remapped= impedance[flatten_probe]
            surgery_all_measurements[i,:] = imp_remapped
                
        mean_surgery = np.mean(surgery_all_measurements,axis=0)
        
        rat_surgery.append(mean_surgery)
        print('surgery_done')
    
    
    
        
        #daily measure post surgery 
        
        
        
        for s, session in enumerate(sessions_subset):
         
            #list_impedance_path = []
            impedance_path = os.path.join(hardrive_path, session)
            matching_files_daily_imp = glob.glob(impedance_path + "\*imp*") 
            #for matching_file in matching_files:
               # list_impedance_path.append(matching_file)
                
            impedance_list_array=np.array(matching_files_daily_imp)
            
            
            session_all_measurements = np.zeros((len(impedance_list_array), 121))
            
            for i, imp in enumerate(impedance_list_array):
                read_file = pd.read_csv(imp)
                impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
                imp_remapped= impedance[flatten_probe]
                session_all_measurements[i,:] = imp_remapped
                    
            
            mean_imp_session= np.mean(session_all_measurements,axis=0)
                #sem_imp= stats.sem(session_all_measurements,axis=0)
                
            rat_daily_impedance[s]=mean_imp_session
               
                #sem_impedance_Level_2_post[s,:]=sem_imp
            print(session) 
            
        all_rats_daily_impedance[rat] = rat_daily_impedance
        all_rats_saline[rat] = rat_saline
        all_rats_surgery[rat] = rat_surgery
        
        np.savetxt(csv_dir_path + csv_name, np.vstack((rat_saline,rat_surgery,rat_daily_impedance)).T, delimiter=',')
        
        
        print(rat) 
        print(len(sessions_subset))

    except Exception: 
        print(session+ 'error')
        continue




############################################
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)




##########


#test = np.array([ elem for singleList in all_rats_daily_impedance for elem in singleList]).T


base_folder = 'F:/Videogame_Assay/Summary/Impedance/'

summary_list = ['AK 33.2_impedance_summary.csv','AK 40.2_impedance_summary.csv',
                 'AK 41.1_impedance_summary.csv','AK 41.2_impedance_summary.csv',
                 'AK 48.1_impedance_summary.csv','AK 48.4_impedance_summary.csv']




n =len(summary_list)








#
#for s in range(n):
#      
#    
#    figure_name = summary_list[s][:7] +'_impedance_summary_lineplot.pdf'
#   
#    f,ax= plt.subplots(figsize=(7,5))
# 
#    sns.set()
#    sns.set_style('white')
#    sns.axes_style('white')
#    sns.despine(left=False)
#    
#    summary_path = os.path.join(base_folder, summary_list[s])
#    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
#    #th = 5000000 
#    #imp_ok = [i for i,v in enumerate(summary[:,0]) if v < th]        
#    #slicing =  summary[imp_ok]   
#    slice_median = np.nanmedian(summary,axis =0)
#    #plt.bar(range(len(slice_median)),slice_median)
#    error = stats.sem(summary, axis =0)
#    
#       
#    plt.plot(slice_median,marker = 'o',color= '#6495ED')
#    #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
#    plt.errorbar(range(len(slice_median)), slice_median, yerr= error, ecolor='k',color='#6495ED', capsize=0,elinewidth = 1)  #fmt='.'
#    ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
#    #plt.ticklabel_format(axis="y", style="sci")
#    ax.set_ylim(ymax=9000000)
#    ax.set_xlim(xmax=30)
#    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
#    plt.yticks(np.arange(0,9000000 , 500000))
#    plt.title(summary_list[s][:7])
#    
#    plt.tight_layout()
#    
#    f.savefig(results_dir + figure_name, transparent=True)
#    plt.close()
#

#USED all in one plot THESIS
    
 

    
figure_name = '_impedance_summary_lineplot.pdf'
   
f,ax= plt.subplots(figsize=(10,7))
 
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for s in range(n):
      
    
    
    summary_path = os.path.join(base_folder, summary_list[s])
    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
    #th = 5000000 
    #imp_ok = [i for i,v in enumerate(summary[:,0]) if v < th]        
    #slicing =  summary[imp_ok]   
    slice_median = np.nanmedian(summary,axis =0)
    #plt.bar(range(len(slice_median)),slice_median)
    error = stats.sem(summary, axis =0)
    
    if s < 4:
       
        plt.plot(slice_median,marker = 'o',color= '#6495ED')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        plt.errorbar(range(len(slice_median)), slice_median, yerr= error, ecolor='k',color='#6495ED', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        ax.set_ylim(ymin=-1000000,ymax=10000000)
        ax.set_xlim(xmax=30)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        #ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
    else:
        
        plt.plot(slice_median,marker = 'o',color= 'r')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        plt.errorbar(range(len(slice_median)), slice_median, yerr= error, ecolor='k',color='r', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        ax.set_ylim(ymin= -1000000,ymax=10000000)
        ax.set_xlim(xmax=30)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        #ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
    #plt.yticks(np.arange(0,9000000 , 500000))
    #plt.title(summary_list[s][:7])
    #plt.close()
    
plt.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)
plt.close()



#used for saline surgery and day one comparison

figure_name = '_impedance_saline_surgery_day1_lineplot.pdf'
   
f,ax= plt.subplots(figsize=(8,7))
 
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

t_tests = np.zeros((n,3))



#target = open(main_folder +"stats_IrOx_saline_VS_PEDOT.txt", 'w')
#target.writelines(str(test_ir) +' IrOx_saline_VS_PEDOT, PLOT: barplot +- SEM, n=219,, impedance.py')
#
#target.close()
#





for s in range(n):
      
    
    
    summary_path = os.path.join(base_folder, summary_list[s])
    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
    #th = 5000000 
    #imp_ok = [i for i,v in enumerate(summary[:,0]) if v < th]        
    #slicing =  summary[imp_ok]   
    slice_median = np.nanmedian(summary,axis =0)
    #plt.bar(range(len(slice_median)),slice_median)
    error = stats.sem(summary, axis =0)
    slice_median_to_use = slice_median[:3]
    error_to_use = error[:3]
    t_tests[s]=slice_median_to_use

    if s < 4:
       
        plt.plot(slice_median_to_use,marker = 'o',color= '#6495ED')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        plt.errorbar(range(len(slice_median_to_use)), slice_median_to_use, yerr= error_to_use, ecolor='k',color='#6495ED', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        ax.set_ylim(ymin= -600000,ymax=1500000)
        #ax.set_xlim(xmax=30)
        #ax.set_yscale('log')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
    else:
        
        plt.plot(slice_median_to_use,marker = 'o',color= 'r')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        plt.errorbar(range(len(slice_median_to_use)), slice_median_to_use, yerr= error_to_use, ecolor='k',color='r', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        ax.set_ylim(ymin=-600000,ymax=1500000)
        #ax.set_xlim(xmax=4)
        #ax.set_yscale('log')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
        
    #plt.yticks(np.arange(0,9000000 , 500000))
    #plt.title(summary_list[s][:7])
    #plt.close()
    
plt.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)
plt.close()

#stats

comparison_saline_day1_Pt = stats.ttest_rel(t_tests[:4,0],t_tests[:4,2])
#Ttest_relResult(statistic=-3.3139623662137554, pvalue=0.04525781899003773)

comparison_saline_day1_Ir = stats.ttest_rel(t_tests[4:,0],t_tests[4:,2])



target = open(main_folder +"daily_impedance_linelot_saline_to_first_day_median_pt_ir.txt", 'w')
target.writelines('saline_day_1_Pt:'+ str(comparison_saline_day1_Pt)+'_'+'saline_day_1_Ir:'+ str(comparison_saline_day1_Ir)
                        +' PLOT: median +- SEM, n=121, no th bur excluded 7 not connected ch, impedance.py')

target.close()


#target = open(main_folder +"stats_IrOx_saline_VS_PEDOT.txt", 'w')
#target.writelines(str(test_ir) +' IrOx_saline_VS_PEDOT, PLOT: barplot +- SEM, n=219,, impedance.py')
#
#target.close()
#
























#USED plot diff over days 


figure_name = '_impedance_summary_absdiff.pdf'
   
f,ax= plt.subplots(figsize=(10,7))
 
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


for s in range(n):
      
    
    
    summary_path = os.path.join(base_folder, summary_list[s])
    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
    #summary =summary_open[:,2:]
 
    test_diff= np.diff(summary)
    diff_avg = np.abs(np.nanmedian(test_diff,axis=0))
    error_diff = stats.sem(test_diff, axis =0)
    
    #plt.figure()
    
    if s < 4:
       
        plt.plot(diff_avg,marker = 'o',color= '#6495ED')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        #plt.errorbar(range(len(diff_avg)), diff_avg, yerr= error_diff, ecolor='k',color='#6495ED', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        #ax.set_ylim(ymax=9000000)
        ax.set_xlim(xmax=30)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
    else:
        
        plt.plot(diff_avg,marker = 'o',color= 'r')
        #plt.fill_between(range(4),mean_trial_speed-stderr,mean_trial_speed+stderr, alpha = 0.5, edgecolor ='#808080', facecolor ='#DCDCDC')
        #plt.errorbar(range(len(diff_avg)), diff_avg, yerr= error_diff, ecolor='k',color='r', capsize=0,elinewidth = 1)  #fmt='.'
        ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
        #plt.ticklabel_format(axis="y", style="sci")
        #ax.set_ylim(ymax=9000000)
        ax.set_xlim(xmax=30)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
    #plt.yticks(np.arange(0,9000000 , 500000))
    #plt.title(summary_list[s][:7])
    #plt.close()
    
plt.tight_layout()
f.savefig(results_dir + figure_name, transparent=True)




#########################################
from scipy.ndimage.filters import gaussian_filter


base_folder = 'F:/Videogame_Assay/Summary/Impedance/'

summary_list = ['AK 33.2_impedance_summary.csv','AK 40.2_impedance_summary.csv',
                 'AK 41.1_impedance_summary.csv','AK 41.2_impedance_summary.csv',
                 'AK 48.1_impedance_summary.csv','AK 48.4_impedance_summary.csv']

RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

N =121

n =len(summary_list)

diff=[]

for s in range(n):
      
    
    
    summary_path = os.path.join(base_folder, summary_list[s])
    results_dir = os.path.join(base_folder + 'Impedance_daily_heatmap')
    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
    summary_shape = np.shape(summary)
    days = summary_shape[1]

    
    for d in range(days):
        
        imp = np.array(summary[:,d])
        impedance_map = np.reshape(imp,newshape=probe_map.shape)
        
        
          
        figure_name = RAT_ID[s] +'_' + str(d) +'.png'
        
        f,ax= plt.subplots(figsize=(7,5))
 
        
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=True)
        
        ax = sns.heatmap(gaussian_filter(impedance_map, sigma=2),  cmap="YlGnBu",vmin=250000, vmax = 3000000,annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
        
        bottom, top = ax.get_ylim()
        
        ax.set_ylim(bottom + 0.5, top - 0.5)       
       

        f.savefig(results_dir + figure_name, transparent=True)
        plt.close()
        print(d)

#####diff 
from matplotlib.colors import LogNorm

sns.heatmap( df,  norm=LogNorm())       
n =len(summary_list)

diff=[]

for s in range(n):
      
    
    
    summary_path = os.path.join(base_folder, summary_list[s])
    results_dir = os.path.join(base_folder + 'Impedance_daily_heatmap')
    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
    summary_shape = np.shape(summary)
    days = summary_shape[1]
    
    
    imp_diff = abs(summary[:,1]-summary[:,6])
    diff_map = np.reshape(imp_diff,newshape=probe_map.shape)
    

    f,ax= plt.subplots(figsize=(12,7))
 
    
    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine(left=True)    
    
    sns.heatmap(diff_map,annot=True,  cmap="GnBu",norm=LogNorm(vmin=diff_map.min(), vmax=diff_map.max()),annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
    
    bottom, top = ax.get_ylim()
    
    ax.set_ylim(bottom + 0.5, top - 0.5)










#  testing example
#'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/saline_1.csv'
#'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/saline_2.csv'
#'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/saline_3.csv'
#
#read_file = pd.read_csv('F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/saline_3.csv')
#
#impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
#imp_remapped3= impedance[flatten_probe]
#
#
#stack = np.vstack((imp_remapped1,imp_remapped2,imp_remapped3))
#mean = np.mean(stack, axis=0)

#
#surgery_impedance_map=np.reshape(mean_surgery,newshape=probe_map.shape)
#
#threshold = 100000
#figure_name = RAT_ID + 'surgery_test_impedance_heatmap.pdf'
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#
#ax = sns.heatmap(surgery_impedance_map,annot=True,  cmap="YlGnBu",vmin=0, vmax = 500000,annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)
#        
#N=121
##map the impedance
#probe_remap=np.reshape(probe_map.T, newshape=N)
#
##test saline
#
#
#saline_impedance_map=np.reshape(mean_saline,newshape=probe_map.shape)
#
#threshold = 100000
#
#
#figure_name = RAT_ID + 'saline_test_impedance_heatmap.pdf'
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#
#ax = sns.heatmap(saline_impedance_map,annot=True,  cmap="YlGnBu", vmin=0, vmax=threshold, annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)
#
#
#f.savefig(results_dir + figure_name, transparent=True)
#
#






#
#N=121
##map the impedance
#probe_remap=np.reshape(probe_map.T, newshape=N)
#imp_map=impedance[probe_remap]
#impedance_map=np.reshape(imp_map,newshape=probe_map.shape)
#
#
#
##main folder rat ID
#script_dir = os.path.dirname(day)
##create a folder where to store the plots inside the daily sessions
#session=os.path.join(script_dir,'Analysis')
##create a folder where to save the plots
#results_dir = os.path.join(session, 'Daily_Health_Check/')
##plot name
#sample_file_name_heatmap = "Impedance_heatmap"
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
#plt.figure(1)
#ax = sns.heatmap(impedance_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
#plt.title("IMPEDANCE")
#plt.xlabel('shanks_PA')
#plt.ylabel('shanks_DV')
#plt.savefig(results_dir + sample_file_name_heatmap)
#







#for s in range(n):
#    
#    f,ax= plt.subplots(figsize=(7,5))
#    figure_name = summary_list[s][:7] +'_impedance_summary_barplot.pdf'
#    sns.set()
#    sns.set_style('white')
#    sns.axes_style('white')
#    sns.despine(left=False)
#    
#    summary_path = os.path.join(base_folder, summary_list[s])
#    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)
#    #th = 5000000 
#    #imp_ok = [i for i,v in enumerate(summary[:,0]) if v < th]        
#    #slicing =  summary[imp_ok]   
#    slice_median = np.nanmedian(summary,axis =0)
#    #plt.bar(range(len(slice_median)),slice_median)
#    error = stats.sem(summary, axis =0)
#
#    ax.bar(np.arange(len(slice_median)), slice_median, yerr=error, align='center', color ='#808080', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
#    
#    ax.tick_params(bottom='off', left='off', labelleft='on', labelbottom='off')
#    ax.set_ylim(ymax=9000000)
#    #ax.set_xlim(xmax=30)
#    plt.title(summary_list[s][:7])
#    
#    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
#    plt.yticks(np.arange(0,9000000 , 500000))
#    
#    plt.tight_layout()
#           
#    f.savefig(results_dir + figure_name, transparent=True)
#    plt.close()
#
#



#
#for s in range(n):
#    
#    f,ax= plt.subplots(figsize=(10,7))
#    figure_name = summary_list[s][:7] +'_impedance_summary_barplot.pdf'
#    
#       
#    sns.set()
#    sns.set_style('white')
#    sns.axes_style('white')
#    sns.despine(left=True)
#    
#    summary_path = os.path.join(base_folder, summary_list[s])
#    summary = np.genfromtxt(summary_path,delimiter = ',', dtype=int)  
#    summary_T = summary.T
#    summary_to_plot = summary_T.tolist()
#    
#    plt.boxplot(summary_to_plot, showfliers=False)
#    sns.despine(top=True, right=True, left=False, bottom=False)
#    plt.title(summary_list[s][:7])
#    ax.yaxis.major.formatter._useMathText = True
#    plt.tight_layout()
#    
#    f.savefig(results_dir + figure_name, transparent=True)
#    plt.close()
#
#    
#    
#
#
#test = summary.T
#test_list  = test.tolist()
#
#
#
#
#
#f,ax = plt.subplots(figsize=(15,11),frameon=False)
#sns.set()
#sns.set_style('white')
#sns.axes_style('white')
#sns.despine(left=True)
#
#mean_imp = final_mean_impedance_Level_2_post.tolist()
#
#
##plt.figure(frameon=False)
#plt.boxplot(test_list, showfliers=False)
#sns.despine(top=True, right=True, left=False, bottom=False)
#
#ax.yaxis.major.formatter._useMathText = True
#
#
#
#
#
#
##rate of change
#test_ir = stats.ttest_rel(summary[:,3],summary[:,4])
#print(test_ir)
#
#
#import pandas as pd
#prices = [30.4, 32.5, 31.7, 31.2, 32.7, 34.1, 35.8, 37.8, 36.3, 36.3, 35.6]
#
#price_series = pd.Series(summary[0,:])
#price_series.pct_change()
#
#
#
#
#
#
#
#impedance_post_surgery = np.zeros([len(test),len(max(test,key = lambda x: len(x)))])
#impedance_post_surgery[:] = np.NaN
#
#
#    
#for i,j in enumerate(all_rats_daily_impedance):
#    
#    sns.heatmap(test)
# test =  
#
#
#test1= np.vstack(all_rats_daily_impedance[0])
#mean = np.mean(test,axis=0)
#plt.plot(mean)
#













#
#N=121
##map the impedance
#probe_remap=np.reshape(probe_map.T, newshape=N)
#imp_map=impedance[probe_remap]
#impedance_map=np.reshape(imp_map,newshape=probe_map.shape)
#
#
#
##main folder rat ID
#script_dir = os.path.dirname(day)
##create a folder where to store the plots inside the daily sessions
#session=os.path.join(script_dir,'Analysis')
##create a folder where to save the plots
#results_dir = os.path.join(session, 'Daily_Health_Check/')
##plot name
#sample_file_name_heatmap = "Impedance_heatmap"
#
#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)
#
#plt.figure(1)
#ax = sns.heatmap(impedance_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
#plt.title("IMPEDANCE")
#plt.xlabel('shanks_PA')
#plt.ylabel('shanks_DV')
#plt.savefig(results_dir + sample_file_name_heatmap)
#
#
#
#
#sample_file_name_bar = "Impedance_barplot"
#plt.figure(2)
##plot impedance 
#ind = probe_remap
#th= 1.00e6
#fig, ax = plt.subplots()
##norm = matplotlib.colors.Normalize(30e3, 60e3)
##y = np.array([34,40,38,50])*1e3
#
#plt.plot(ind)
#
#
#plt.bar(ind,imp_map)
#ax.bar(ind,imp_map)
##ax.bar(ind,test,color=plt.cm.plasma_r(norm(y)))
#width = 0.7 
## add some text for labels, title and axes ticks
##ax.axhline(th, color="gray")
#ax.plot([0,N], [th, th], "k--")
#ax.set_ylabel('Impedance (Omhs)')
#ax.set_xlabel('channels')
#ax.set_title('IMPEDANCE')
##ax.set_xticks(probe_remap)
#ax.set_xticklabels(probe_remap,rotation='vertical',fontsize=7)
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
##ax.ticklabel_format(useOffset=False,style='plain')
#plt.savefig(results_dir + sample_file_name_bar,facecolor=fig.get_facecolor(), edgecolor='none')
#
#
#
##ax.patch.set_facecolor('red')
##ax.patch.set_alpha(0.5)