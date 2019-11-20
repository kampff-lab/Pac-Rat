# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:47:05 2018

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors
import seaborn as sns
import os




base_directory = r'F:'
rat_ID = r'/AK_41.1/'
rat_folder = base_directory + rat_ID
day = rat_folder + '2019_02_04-14_48/'
filename = day + 'impedance1.csv'


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



imp=pd.read_csv(filename)
imp2=imp['Impedance Magnitude at 1000 Hz (ohms)']
imp_array=np.array(imp2)
impedance=imp_array.astype(dtype=int)

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