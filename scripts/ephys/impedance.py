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
from matplotlib.ticker import LogFormatterExponent
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors
import seaborn as sns
import os
import glob
from scipy import stats
import ephys_library as ephys



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


rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)

sessions_subset = Level_2_post



flatten_probe = ephys.probe_map.flatten()


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

mean_impedance_Level_2_post_list = mean_impedance_Level_2_post.tolist()

plt.boxplot(mean_impedance_Level_2_post_list)
plt.title('impedance_rat_48.4')
plt.xlabel('days')
plt.ylabel('impedance_(ohms)')








#saline and first session 


first_session = sessions_subset[0]      
impedance_path_first_session = os.path.join(hardrive_path, first_session)    
    
matching_files_test_probe = glob.glob(impedance_path_first_session + "\*saline*") 
matching_files_surgery_day = glob.glob(impedance_path_first_session + "\*surgery*") 


#saline

saline_all_measurements = np.zeros((len(matching_files_test_probe), 121))
for i, imp in enumerate(matching_files_test_probe):
    read_file = pd.read_csv(imp)
    impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
    imp_remapped= impedance[flatten_probe]
    saline_all_measurements[i,:] = imp_remapped
        
mean_saline = np.mean(saline_all_measurements,axis=0)
sem_imp_saline = stats.sem(saline_all_measurements,axis=0)

   
#surgery 


surgery_all_measurements = np.zeros((len(matching_files_surgery_day), 121))
for i, imp in enumerate(matching_files_surgery_day):
    read_file = pd.read_csv(imp)
    impedance = np.array(read_file['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)
    imp_remapped= impedance[flatten_probe]
    surgery_all_measurements[i,:] = imp_remapped
        
mean_surgery = np.mean(surgery_all_measurements,axis=0)
sem_imp_surgery = stats.sem(surgery_all_measurements,axis=0)







data_stack = np.vstack((mean_saline,mean_surgery,mean_impedance_Level_2_post))


data_to_plot = data_stack.tolist()





plt.figure()
plt.boxplot(data_to_plot, showfliers=True)
plt.title('impedance_rat_48.4')
plt.xlabel('days')
plt.ylabel('impedance_(ohms)')

plt.yscale('log')




# omnetics examples before and after pedot 
#folder located at day 1 of recording together with the saline test and surgery impedances



folder_list = '/Probe_61_rat_33.2' , '/Probe_62_rat_40.2', 

first_session = sessions_subset[0]      
probe_path_first_session = os.path.join(hardrive_path, first_session)  
folder_name = ('/Probe_62_rat_40.2')


probe_path =  os.path.join(probe_path_first_session + folder_name)

omnetics_list = glob.glob(probe_path + "\*omnetics*") 




saline_probe_mean, saline_probe_sem =  avg_sem_omnetics_imp(omnetics_list, match = "\*sal*" )

PEDOT_probe_mean, PEDOT_probe_sem =  avg_sem_omnetics_imp(omnetics_list, match = "\*post*" )


probe_stack = np.vstack((saline_probe_mean,PEDOT_probe_mean))


probe_to_plot = probe_stack.tolist()


plt.figure()
plt.boxplot(probe_to_plot, showfliers=False)
plt.title('impedance_rat_48.4')
plt.xlabel('days')
plt.ylabel('impedance_(ohms)')


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
    
 
    i
    
    
    
    
    
    
    
def avg_imp(impedance_file):
    
    impedance = np.zeros((3,32))
    
    for f, file in enumerate(impedance_file):
        open_file = np.genfromtxt(file, dtype=float, skip_header =2, usecols =1 )
        final_file = open_file[:32]
        impedance[f] = final_file
    
    return impedance
    










test = np.genfromtxt(saline[0],dtype=float,skip_header =2,usecols =1 )
sub = test[:32]

from scipy import stats






imp = pd.read_csv(impedance_path)
impedance = np.array(imp['Impedance Magnitude at 1000 Hz (ohms)']).astype(dtype=int)




























N=121
#map the impedance
probe_remap=np.reshape(probe_map.T, newshape=N)
imp_map=impedance[probe_remap]
impedance_map=np.reshape(imp_map,newshape=probe_map.shape)



#main folder rat ID
script_dir = os.path.dirname(day)
#create a folder where to store the plots inside the daily sessions
session=os.path.join(script_dir,'Analysis')
#create a folder where to save the plots
results_dir = os.path.join(session, 'Daily_Health_Check/')
#plot name
sample_file_name_heatmap = "Impedance_heatmap"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

plt.figure(1)
ax = sns.heatmap(impedance_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
plt.title("IMPEDANCE")
plt.xlabel('shanks_PA')
plt.ylabel('shanks_DV')
plt.savefig(results_dir + sample_file_name_heatmap)




sample_file_name_bar = "Impedance_barplot"
plt.figure(2)
#plot impedance 
ind = probe_remap
th= 1.00e6
fig, ax = plt.subplots()
#norm = matplotlib.colors.Normalize(30e3, 60e3)
#y = np.array([34,40,38,50])*1e3

plt.plot(ind)


plt.bar(ind,imp_map)
ax.bar(ind,imp_map)
#ax.bar(ind,test,color=plt.cm.plasma_r(norm(y)))
width = 0.7 
# add some text for labels, title and axes ticks
#ax.axhline(th, color="gray")
ax.plot([0,N], [th, th], "k--")
ax.set_ylabel('Impedance (Omhs)')
ax.set_xlabel('channels')
ax.set_title('IMPEDANCE')
#ax.set_xticks(probe_remap)
ax.set_xticklabels(probe_remap,rotation='vertical',fontsize=7)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#ax.ticklabel_format(useOffset=False,style='plain')
plt.savefig(results_dir + sample_file_name_bar,facecolor=fig.get_facecolor(), edgecolor='none')



#ax.patch.set_facecolor('red')
#ax.patch.set_alpha(0.5)