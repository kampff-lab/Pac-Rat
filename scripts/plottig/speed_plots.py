# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:59:46 2019

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

rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                          'F:/Videogame_Assay/AK_46.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.1_IrO2.csv',
                          'F:/Videogame_Assay/AK_48.3_behaviour_only.csv', 'F:/Videogame_Assay/AK_48.4_IrO2.csv', 
                          'F:/Videogame_Assay/AK_49.1_behaviour_only.csv','F:/Videogame_Assay/AK_49.2_behaviour_only.csv',
                        'F:/Videogame_Assay/AK_50.1_behaviour_only.csv', 'F:/Videogame_Assay/AK_50.2_behaviour_only.csv']






#colours = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#FFDAB9']
RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']


main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)





#Level_0 = prs.Level_0_paths(rat_summary_table_path)
#Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
#Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
#Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
# =============================================================================
# Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
# 
# 
# #saving a Trial_idx_csv containing the idx of start-end-touch 0-1-2
# sessions_subset = Level_2_pre
# behaviour.start_end_touch_ball_idx(sessions_subset)
# =============================================================================
#######################################################################
    
    
#functions to use 
    
    
    ############################
#####euclidian distances 
   
def distance_events(sessions_subset,frames=120):
    
    poke = [1400,600]
    l = len(sessions_subset)
    
    rat_ball = [[] for _ in range(l)]
    rat_poke = [[] for _ in range(l)]
    before_touch = [[] for _ in range(l)]
    after_touch = [[] for _ in range(l)]
    after_start= [[] for _ in range(l)]
    before_end = [[] for _ in range(l)]
     
   
    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            crop_tracking_path = os.path.join(script_dir + '/crop.csv')
            crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
            trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)                
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
            ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
            
            
            
            
            start = trial_idx[:,0]
            rat_position_at_start = crop[start]
            touch = trial_idx[:,2]
            rat_position_at_touch = crop[touch]
            end = trial_idx[:,1]
            rat_position_at_end = crop[end]
            
            rat_before_ball = crop[touch - frames]
            rat_after_ball = crop[touch + frames]
            
            rat_after_start = crop[start + frames]
            rat_before_end =  crop[end - frames]
            
            
            session_rat_ball_dist = []
            session_rat_poke_dist = []
            session_rat_before_touch=[]
            session_rat_after_touch=[]
            session_rat_after_start = []
            session_rat_before_end = []
            
            
            for e in range(len(start)):
                
                #dist = distance.euclidean(rat_position_at_start[e], ball_coordinates[e])
                
                dist_rat_ball = (np.sqrt(np.nansum((rat_position_at_start[e]-ball_coordinates[e])**2)))
                dist_rat_poke = (np.sqrt(np.nansum((rat_position_at_touch[e]-poke)**2)))
                dist_before_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_before_ball[e])**2)))
                dist_after_touch = (np.sqrt(np.nansum((rat_position_at_touch[e]-rat_after_ball[e])**2)))
                
                dist_after_start = (np.sqrt(np.nansum((rat_position_at_start[e]-rat_after_start[e])**2)))
                dist_before_end = (np.sqrt(np.nansum((rat_position_at_end[e]-rat_before_end[e])**2)))
                
                
                
                session_rat_ball_dist.append(dist_rat_ball)
                session_rat_poke_dist.append(dist_rat_poke)
                session_rat_before_touch.append(dist_before_touch)
                session_rat_after_touch.append(dist_after_touch)
                
                session_rat_after_start.append(dist_after_start)
                session_rat_before_end.append(dist_before_end)
                
                #.savetxt(csv_dir_path + csv_name, np.vstack((start_idx,end_idx,touch_idx,ball_on_idx)).T, delimiter=',', fmt='%s')

                
            rat_ball[count]=session_rat_ball_dist
            rat_poke[count]=session_rat_poke_dist
            before_touch[count]=session_rat_before_touch
            after_touch[count]=session_rat_after_touch
            
            after_start[count] = session_rat_after_start
            before_end[count] = session_rat_before_end
        
        
           
            
        except Exception: 
            print('error'+ session)
        continue
    
    return rat_ball, rat_poke, before_touch, after_touch, after_start, before_end

##############################################
#find all the idx of an event in the trail idx file  


def rat_event_idx_and_pos_finder(sessions_subset, event=2):
    
    l = len(sessions_subset)
    event_rat_coordinates = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        crop_tracking_path = os.path.join(script_dir + '/crop.csv')
        crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
        
        
       
        rat_event = trial_idx[:,event]
        rat_pos = crop[rat_event]
            
        event_rat_coordinates[count]= rat_pos
    
    return event_rat_coordinates




    
########################find touch idx for all the rats 
    
    
s = len(rat_summary_table_path)

touch_all_rats = [[] for _ in range(s)]
pos_at_touch_all_rats =  [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         touch_rat, rat_pos_at_touch = rat_event_idx_and_pos_finder(sessions_subset, event=2)
         
         touch_all_rats[r] = touch_rat
         pos_at_touch_all_rats[r] = rat_pos_at_touch
         
    
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  

len(touch_all_rats)
len(pos_at_touch_all_rats)

#####flatten touch


touch_idx_tot =[]
touch_coordinates = []



for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    touch_idx  = touch_all_rats[rat]
    positions = pos_at_touch_all_rats[rat]



    flattened_idx= [val for sublist in touch_idx for val in sublist]
    flattened_pos_at_touch = [val for sublist in positions for val in sublist]

    
    touch_idx_tot.extend(flattened_idx)
    touch_coordinates.extend(flattened_pos_at_touch)

    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    #total_delta.extend(delta)
    
print(len(touch_idx_tot))
print(len(touch_coordinates))


#####################find seconds to touch

max(touch_idx_tot)

minutes = np.array(touch_idx_tot)/(120*60)

max(minutes)


################################################

###before and after and savings

def time_to_events(sessions_subset):
    
    l = len(sessions_subset)
    st_time = [[] for _ in range(l)]
    te_time = [[] for _ in range(l)]
    
    for count in np.arange(l):
        
        session = sessions_subset[count]
        
        try:
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            crop_tracking_path = os.path.join(script_dir + '/crop.csv')
            crop = np.genfromtxt(crop_tracking_path, delimiter = ',', dtype = float)
            trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
    
            #selecting the column of touch and start, calculate the abs diff in order to calculate the 
            #how long it took to touch the ball from the start of the trial
            start_touch_diff = abs(trial_idx[:,0] - trial_idx[:,2])
            touch_end_diff = abs(trial_idx[:,1] - trial_idx[:,2])
            st_time[count] = start_touch_diff
            te_time[count] = touch_end_diff
            
            
            csv_dir_path = os.path.join(script_dir + '/events/')
            
            csv_name = 'Start_touch_idx_diff.csv'
            np.savetxt(csv_dir_path + csv_name,start_touch_diff, fmt='%s')
            csv_name = 'Touch_reward_idx_diff.csv'
            np.savetxt(csv_dir_path + csv_name,touch_end_diff, fmt='%s')
            
            print(len(start_touch_diff))
            print(len(touch_end_diff))
            print('saving DONE')


        except Exception: 
            print('error'+ session)
        continue
                            
    return st_time, te_time






#####################################################################

#same than before but no retriving data from rat folders


s = len(rat_summary_table_path)

st_all_rats = [[] for _ in range(s)]
te_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path[8:]):
    
    try:    
         #Level_2_pre = prs.Level_2_pre_paths(rat)
         #sessions_subset = Level_2_pre#[3:6]
         Level_3_moving_light = prs.Level_3_moving_light_paths(rat)
         sessions_subset = Level_3_moving_light#[3:6]         
         st_time, te_time = time_to_events(sessions_subset)
         
         st_all_rats[r] = st_time
         te_all_rats[r] = te_time
         
    
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  


st_idx_tot =[]
te_idx_tot=[]


for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    st_idx  = st_all_rats[rat]
    te_idx = te_all_rats [rat]


    flattened_st= [val for sublist in st_idx for val in sublist]
    flattened_te = [val for sublist in te_idx for val in sublist]
    
    st_idx_tot.extend(flattened_st)
    te_idx_tot.extend(flattened_te)
    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    #total_delta.extend(delta)
    
print(len(st_idx_tot))
print(len(te_idx_tot))


max_st = max(st_idx_tot)
max_te = max(te_idx_tot)



#test from session retrieve st and te.csv and pool them



 rat = rat_summary_table_path[0]
 Level_2_pre = prs.Level_2_pre_paths(rat)
 sessions_subset = Level_2_pre[0]


 
st_pooled=[]
te_pooled=[]
                
for r,rat in enumerate(rat_summary_table_path[8:]):
    
    try:
               
        Level_3_moving_light = prs.Level_3_moving_light_paths(rat)
        sessions_subset = Level_3_moving_light#[3:6]   
        l=len(sessions_subset)
         
        rat_st = []
        rat_te = []
        
        
        for count in np.arange(l):
        

        
            session = sessions_subset[count]
    
            script_dir = os.path.join(hardrive_path + session) 
            #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
    
            st_path= os.path.join(script_dir+ '/events/' + 'Start_touch_idx_diff.csv')           
            st_diff = np.genfromtxt(st_path, delimiter = ',')
            
            rat_st.extend(st_diff)
            
            te_path = os.path.join(script_dir+ '/events/' + 'Touch_reward_idx_diff.csv')
            te_diff = np.genfromtxt(te_path, delimiter = ',')
            
            rat_te.extend(te_diff)
                
        st_pooled.extend(rat_st)
        te_pooled.extend(rat_te)    
            
    except Exception: 
        print('error'+ session)
    continue    

print(len(st_pooled)) 
print(len(te_pooled))   
    






############################plotting
#plot ball positions at touch color coded by the time to go home ranging 0-1sec
#te_idx_tot
#touch_coordinates


#selection 1s trials


te_selection = [i for i,v in enumerate(te_pooled) if 1000 >v]


print(len(te_selection))

te_sel = np.array(te_pooled)[te_selection]
touch_pos_selection = np.array(touch_coordinates)[te_selection]

plt.scatter(touch_pos_selection[:,0],touch_pos_selection[:,1],c=color_map,cmap='jet',s=3)
plt.colorbar()

color_map = np.array(te_selection)/1000
max(color_map)
min(color_map)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cm = plt.cm.get_cmap('bwr')


# Plot the data
fig=plt.figure()
fig.patch.set_facecolor('white')
ax=fig.add_subplot(111)
s = ax.scatter(touch_pos_selection[:,0],touch_pos_selection[:,1],c=color_map,edgecolor='')
norm = mpl.colors.Normalize(vmin=0, vmax=1)
ax1 = fig.add_axes([0.95, 0.1, 0.01, 0.8])
cb = mpl.colorbar.ColorbarBase(ax1,norm=norm,cmap=cm,orientation='vertical')
cb.set_clim(vmin = 0, vmax = 1)
cb.set_label('Value of \'vel\'')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
t = np.random.rand(100)
w = np.random.rand(100)

max(te_sel)
points = len(touch_pos_selection[:,0])

fig, ax = plt.subplots(1, 1)
cmap = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=max(te_selection))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for i in range(points):
    ax.scatter(touch_pos_selection[:,0][i],touch_pos_selection[:,1][i], c=scalarMap.to_rgba(te_selection[i]), cmap=cmx.plasma, edgecolor='none')
scalarMap.set_array([])
fig.colorbar(scalarMap,ax=ax)

for a in [0.1, 0.5, 0.9]:
    ax.scatter([], [], c='k', alpha=0.5, s=a*points, label=str(a), edgecolors='none')
l1 = ax.legend(scatterpoints=1, frameon=True, loc='lower left' ,markerscale=1)
for b in [0.25, 0.5, 0.75]:
    ax.scatter([], [], c='k', alpha=b, s=50, label=str(b), edgecolors='none')
ax.legend(scatterpoints=1, frameon=True, loc='lower right' ,markerscale=1)
fig.show()

























####################

#plot st VS te calculated using IDX - MEDIAN


    
tot_stack= np.vstack((st_idx_tot,te_idx_tot))

final_to_plot = tot_stack.tolist()

plt.figure()
plt.boxplot(final_to_plot, showfliers=False)

plt.figure()
plt.boxplot(final_to_plot, showfliers=True)


#st_array = np.array(st_idx_tot)
#
#st_norm = ( st_array- st_array.min()) / (st_array.max() - st_array.min())
#
#
#te_array = np.array(te_idx_tot)
#
#te_norm = ( te_array- te_array.min()) / (te_array.max() - te_array.min())


st_seconds = np.array(st_pooled)/120
te_seconds = np.array(te_pooled)/120


median_st = np.median(st_seconds)
median_te = np.median(te_seconds)
sem_st = stats.sem(st_seconds, nan_policy='omit')
sem_te = stats.sem(te_seconds, nan_policy='omit')


materials = ['before_touch', 'after_touch']
x_pos = np.arange(len(materials))
medians= [median_st, median_te]
errors = [sem_st, sem_te]


figure_name =  'idx_diff_st_te_barplot_N1517_final_level3.pdf' 
  
f,ax = plt.subplots(figsize=(5,7))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


ax.bar(materials[0], medians[0], yerr=errors[0], align='center', color ='#6495ED', edgecolor ='white', width = .5,alpha=0.5, ecolor='black', capsize=0)
ax.bar(materials[1], medians[1], yerr=errors[1] ,align='center', color ='#228B22', edgecolor ='white', width = .5,alpha=0.5, ecolor='black', capsize=0)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.ylim(0,16)
plt.yticks(range(16))
plt.yticks(fontsize=15)
plt.ylabel('median seconds (idx diff/120)')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)

stat = mannwhitneyu(st_seconds, te_seconds)
#(9791217.0, 2.019645541580334e-296)
#level 3 MannwhitneyuResult(statistic=447433.0, pvalue=4.3838420315944956e-187)

target = open(main_folder +"MannWhutney_median_idx_diff_st_te_level3.txt", 'w')
target.writelines(str(stat)+' LEVEL 2: median te and st +- SEM, speed_plot.py')
target.close()


###########hist 

th = 40     #40sec*120 or 4800 frames
selection_st = [i for i,v in enumerate(st_seconds) if v < th]
selection_te = [i for i,v in enumerate(te_seconds) if v < th]


lower_than_40_st = np.array(st_seconds)[selection_st]
lower_than_40_te = np.array(te_seconds)[selection_te]

print(len(lower_than_40_st))
print(len(lower_than_40_te))


normalised_st = lower_than_40_st/len(lower_than_40_st)
normalised_te =lower_than_40_te/len(lower_than_40_te)


############plot distribution st and te

figure_name =  'idx_diff_st_te_hist_40_sec_overlap_final_level3.pdf' 
  
f,ax = plt.subplots(figsize=(10,4))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#to overlap the 2 hist use bins=bins       
        
#_, bins, _ = plt.hist(normalised_st, bins=100, range=[0, 0.01])
#_ = plt.hist(bar, bins=bins, alpha=0.5, density=True)        
ax.hist(normalised_st,bins = 100, color ='#6495ED',alpha = .3,range=[0, 0.01])  
        
ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('st seconds, idx/120/N')
plt.xlabel('frames max 40s_normalised by N (st = 1311,te = 1517 )')
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)

#########################

figure_name =  'idx_diff_te_hist_40_sec_N5595.pdf' 
  
f,ax = plt.subplots(figsize=(10,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

ax.hist(normalised_te,bins=100, color ='#228B22',alpha=.3,range=[0, 0.01]) #bins=bins

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('te idx diff to 40s')
plt.xlabel('frames max 4800')
plt.ylim(ymax=1000)
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)


########################st vs te


figure_name =  'scatter_seconds_te_VS_st_N1517.pdf' 
  
f,ax = plt.subplots(figsize=(6,9))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

plt.plot(st_seconds,te_seconds, '.', alpha=.7, markersize=.5, color= 'k')
#plt.plot(te_seconds, '.', alpha=.7, markersize=.5, color= 'r')

#plt.plot(np.unique(te_pooled), np.poly1d(np.polyfit(te_pooled, st_pooled, 1))(np.unique(te_pooled)))


ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('te seconds (idx/120)')
plt.xlabel('st seconds (idx/120)')
#plt.ylim(ymax=800)
#plt.xlim(xmax=20000)
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)



#remove missed trials

th = 39     #40sec*120 or 4800 frames
selection_st_wo_missed = [i for i,v in enumerate(st_seconds) if v < th]
selection_te_wo_missed = [i for i,v in enumerate(te_seconds) if v < th]


print(len(selection_st_wo_missed))
print(len(selection_te_wo_missed))


st_sel_seconds = np.array(selection_st_wo_missed)/120
te_sel_seconds = np.array(selection_te_wo_missed)/120


st_sel_seconds_nan = np.insert(st_sel_seconds, len(te_sel_seconds)-len(st_sel_seconds), np.nan)

print(len(st_sel_seconds_nan))
#plt.plot(np.unique(flattened_before), np.poly1d(np.polyfit(flattened_before, flattened_after, 1))(np.unique(flattened_before)))


#m,b = np.polyfit(flattened_before, flattened_after, 1)

# m 0.4675482133356882
# b 0.9965590810071268


#coef = np.polyfit(flattened_before,flattened_after,1)
#poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

#plt.plot(flattened_before,flattened_after, 'yo', flattened_before, poly1d_fn(flattened_before), '--k')




# =============================================================================
#  
 
# rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']
# rat = rat_summary_table_path[0]
# Level_2_pre = prs.Level_2_pre_paths(rat)
# sessions_subset = Level_2_pre[0]
# 
# =============================================================================
#



############################################################################
##########################################

#finf delta speed around touch event 

#calculating distance rat at start- ball position + dist rat at touch and poke   

s = len(rat_summary_table_path)

rat_ball_all_rats = [[] for _ in range(s)]
rat_poke_all_rats = [[] for _ in range(s)]
before_touch_all_rats= [[] for _ in range(s)]
after_touch_all_rats = [[] for _ in range(s)]
after_start_all_rats = [[] for _ in range(s)]
before_end_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         rat_ball, rat_poke, before_touch, after_touch, after_start, before_end = distance_events(sessions_subset, frames = 120)
         
         rat_ball_all_rats[r] = rat_ball
         rat_poke_all_rats[r] = rat_poke
         before_touch_all_rats[r] = before_touch
         after_touch_all_rats[r] = after_touch
         after_start_all_rats[r] = after_start
         before_end_all_rats[r] = before_end
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    
    
###############################################
       
#plot DELTA distance rat before touch (120 frames) - rat at touch VS rat at touch - rat after touch (120 frames)

def trial_count(event):
    
    rat_tot_trial_count = []

    for i in range(len(event)):
        rat_tot_trial_count.append(len(event[i]))


    return sum(rat_tot_trial_count)





total_delta = []    
total_flat_before =[]
total_flat_after = []

sum_trial_count_before=[]

total_flat_after_start=[]
total_flat_before_end=[]
tot_delta_start_end=[]
#f,ax = plt.subplots(figsize=(15,5))



for rat in arange(len(rat_summary_table_path)):
       

      
    before  = before_touch_all_rats[rat]
    count_before= trial_count(before)
    sum_trial_count_before.append(count_before)
    
    after = after_touch_all_rats [rat]


    flattened_before = [val for sublist in before for val in sublist]
    flattened_after = [val for sublist in after for val in sublist]
    
    
    aft_start = after_start_all_rats[rat]
    bef_end = before_end_all_rats[rat]
    
    flattened_after_start = [val for sublist in aft_start for val in sublist]
    flattened_before_end = [val for sublist in bef_end for val in sublist]
    
    
    total_flat_before.extend(flattened_before)
    total_flat_after.extend(flattened_after)
    
    total_flat_after_start.extend(flattened_after_start)
    total_flat_before_end.extend(flattened_before_end)
    
    
    
    
    delta = np.array(flattened_after) - np.array(flattened_before)
    total_delta.extend(delta)
    
    
    delta_start_end = np.array(flattened_before_end) - np.array(flattened_after_start)
                              
    tot_delta_start_end.extend(delta_start_end)
    
    print(rat)
    print(len(flattened_before))
    print(len(flattened_after))
    print(len(total_delta))
    print(len(tot_delta_start_end))



#########################################
    
    
    
















  
#####################

f,ax = plt.subplots(figsize=(15,6))
       
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name =  'delta_speed_120_after_start_VS120_before_reward_120frames_color_coded_final.pdf' 

#test = cumsum(sum_trial_count_before)


bounds =  [0,648, 1247, 1710, 2599, 2841, 3418, 3614, 4272, 4643, 4921, 5225]
colors = ['#FF0000','#FF8C00','#FF69B4','#BA55D3','#4B0082','#0000FF','#00BFFF','#2E8B57','#32CD32', '#ADFF2F','#7FFFD4','#D2691E']
#RAT_ID = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2', 'AK 46.1', 'AK 48.1','AK 48.3','AK 48.4', 'AK 49.1', 'AK 49.2','AK 50.1','AK 50.2']



cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))

rect = plt.scatter(range(len(total_delta)), tot_delta_start_end, s= 1,c=range(len(total_delta)),cmap=cmap, norm=norm)

cbar = plt.colorbar(rect, spacing="proportional")
cbar.set_label('rats', rotation=270, labelpad=10)
ax.hlines(0,0,len(total_delta),linewidth=0.9,color= 'k')


ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title('0,648, 1247, 1710, 2599, 2841, 3418, 3614, 4272, 4643, 4921, 5225,5702')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('delta speed pixel/second')
plt.xlabel('trials')
plt.ylim(ymax=600)
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)

#######################plt delta versus secons took to touch the ball


f,ax = plt.subplots(figsize=(15,6))
       
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)

figure_name =  'delta_speed_touch_VS_minutes_to_touch(minutes)_final.pdf' 

plt.plot(minutes,total_delta,'.',markersize=1,c='k')

ax.hlines(0,0,max(minutes),linewidth=0.9,color= 'r')


ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title('delta vs min (touch/120*60)')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('delta speed pixel/second')
plt.xlabel('minutets')
plt.ylim(ymax=600)
f.tight_layout()


f.savefig(results_dir + figure_name, transparent=True)
























#    
#    
#    stack= np.vstack((flattened_before,flattened_after))
#    to_plot = stack.tolist()
#    
#    ax.scatter(range(len(delta)),delta,alpha=0.7, edgecolors='none', s=7,c='k')
#    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
#    plt.title('Level 2 delta speed around touch',fontsize = 16)
#    plt.ylabel('speed (idx_count)', fontsize = 13)
#    plt.xlabel('trial', fontsize = 13)
#    
#    #plt.bar(range(len(delta)),delta, width= 0.05)
#    #plt.boxplot(to_plot)
#    ax.axes.get_xaxis().set_visible(True) 
#       
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    ax.hlines(0,0,900,linewidth=0.8,color= 'r')
#    
#    #plt.xlim(-0.1,3.5)
#    #plt.ylim(-0.2,6)
#    f.tight_layout()
#    


##########################################
#flattened allthe rats 

        
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

figure_name2 =  'flattened_before_touch_VSafter_touch_120frames.pdf' 
  

f2,ax = plt.subplots(figsize=(8,7))

stack= np.vstack((total_flat_before,total_flat_after))
to_plot = stack.tolist()

plt.boxplot(to_plot)


plt.title('Level 2 before VS after 120frames',fontsize = 16)
plt.ylabel('speed idx count', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#plt.legend()
f2.tight_layout()




############################################

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)

figure_name3 =  'delta_all_rats_concat.pdf' 
  

f3,ax = plt.subplots(figsize=(15,5))

ax.scatter(range(len(total_delta)),total_delta,alpha=0.7, edgecolors='none', s=7,c='k')



ax.hlines(0,0,len(total_delta),linewidth=0.8,color= 'r')
plt.title('Level 2 delta_after minus before 120frames concat',fontsize = 16)
plt.ylabel('delta speed idx count', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#plt.legend()
f3.tight_layout()

#####################bar plot



median_before_touch = np.median(total_flat_before)
median_after_touch = np.median(total_flat_after)
sem_before_touch = stats.sem(total_flat_before, nan_policy='omit')
sem_after_touch = stats.sem(total_flat_after, nan_policy='omit')


materials = ['before_touch_120frames', 'after_touch_120frames']
x_pos = np.arange(len(materials))
medians= [median_before_touch, median_after_touch]
errors = [sem_before_touch, sem_after_touch]



figure_name4 =  'dist_before_after_120_barplot_final.pdf' 

f4,ax = plt.subplots(figsize=(6,8))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


ax.bar(materials[0], medians[0], yerr=errors[0], align='center', color ='r', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)
ax.bar(materials[1], medians[1], yerr=errors[1] ,align='center', color ='b', edgecolor ='white', width = .6,alpha=0.7, ecolor='black', capsize=0)

plt.title('before and after touch 120sec',fontsize = 16)
plt.ylabel('median euclidian distance', fontsize = 13)
plt.xlabel('before_after', fontsize = 13)

ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylim(ymax=250)

#plt.legend()
f4.tight_layout()

#stat = mannwhitneyu(total_flat_before, total_flat_after)
##(statistic=8919411.0, pvalue=0.0)
#
#target = open(main_folder +"MannWhutney_median_idx_diff_st_te.txt", 'w')
#target.writelines(str(stat)+' LEVEL 2: median te and st +- SEM, speed_plot.py')
#target.close()




###############savings



f.savefig(results_dir + figure_name, transparent=True)
f2.savefig(results_dir + figure_name2, transparent=True)
f3.savefig(results_dir + figure_name3, transparent=True)
f4.savefig(results_dir + figure_name4, transparent=True)



############################################
###############################################################





###########################################
rat_poke_all_rats
rat_ball_all_rats

st_idx_tot
te_idx_tot

    

f3,ax = plt.subplots(figsize=(7,5))

rat_ball_concat = []


for rat in arange(len(rat_summary_table_path)):
    
      
    rat_ball_selection  = rat_ball_all_rats[rat]


    flattened_rat_ball = [val for sublist in rat_ball_selection for val in sublist]
    
    rat_ball_concat.extend(flattened_rat_ball)

     
    print(rat)
    print(len(rat_ball_concat))
    print(len(st_idx_tot))
    
    
    
plt.figure()    
plt.plot(rat_ball_concat,st_idx_tot, '.', alpha=.8, markersize=.5, color= 'k')


#plt.hist(rat_ball_concat,bins=300)


figure_name =  'scatter_rat_ball_dstance_VS_st_seconds_final_zoom.pdf' 

f,ax = plt.subplots(figsize=(8,8))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.plot(rat_ball_concat,st_seconds, '.', alpha=.8, markersize=.5, color= 'k')



plt.ylabel('st seconds', fontsize = 13)
plt.xlabel('rat ball euclidian distance', fontsize = 13)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylim(ymax=250)

#plt.legend()
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)
#plt.ylim(0,10000)

################################################################
ball_array
rat_poke_all_rats
te_idx_tot

plt.figure()

rat_poke_concat = []


for rat in arange(len(rat_summary_table_path)):
    
      
    rat_poke_selection  = rat_poke_all_rats[rat]

    flattened_rat_poke = [val for sublist in rat_poke_selection for val in sublist]
   
    
    rat_poke_concat.extend(flattened_rat_poke)
    
    
    print(rat)
    
    print(len(rat_poke_concat))
    print(len(te_idx_tot))
    


figure_name =  'scatter_rat_ball_dstance_VS_te_seconds_final.pdf' 

f,ax = plt.subplots(figsize=(8,8))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.plot(rat_ball_concat,te_seconds, '.', alpha=.8, markersize=.5, color= 'k')


plt.title('before and after touch 120sec',fontsize = 16)
plt.ylabel('te seconds', fontsize = 13)
plt.xlabel('rat ball euclidian distance', fontsize = 13)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
ax.axes.get_xaxis().set_visible(True) 
   
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.ylim(ymax=250)

#plt.legend()
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)







##################################################
#plot ba

te_idx_selection = [i for i,v in enumerate(te_seconds) if v>=6] #3<v <6]
print(len(te_idx_selection))

te_range = np.array(te_idx_tot)[te_idx_selection] #1433
ball_range = ball_array[te_idx_selection]

#remove zeros

remove_zero_ball = [i for i,v in enumerate(ball_range[:,0]) if v > 250]

final_ball=ball_range[remove_zero_ball]

te_sel =te_range[remove_zero_ball]
print(len(final_ball))
print(len(te_sel))

>=3 640 blue

3-6 2115 orange

>=6 2524 

>=6 2524


max(ball_array[:,0])
max(ball_array[:,1])
min(final_ball[:,1])
min(ball_array[:,1])

########################## try to plot all the range with hist#

from matplotlib.gridspec import GridSpec



fig = plt.figure(figsize=(7,7))

figure_name ='ball_positions_color_coded_based_on_touch_to_reward_distance.pdf'

gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])

ax_joint.scatter(final_ball[:,0],final_ball[:,1],s=1)

#bins=np.linspace(250,1400,50)

ax_marg_x.hist(final_ball[:,0],range=[250,1300],bins=50,alpha=.5,histtype='step', stacked=True, fill=False)
ax_marg_y.hist(final_ball[:,1],bins=50,range=[0,1100],alpha=.5,orientation="horizontal",histtype='step', stacked=True, fill=False)



plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)


ax_marg_y.set_xlim([0, 100])
ax_marg_x.set_ylim([0, 100])


# Set labels on joint
ax_joint.set_xlabel('ball x 250-1300 ')
ax_joint.set_ylabel('ball y 0-1100 ')

# Set labels on marginals
ax_marg_y.set_xlabel('ball positions')
ax_marg_x.set_ylabel('ball positions')

fig.savefig(results_dir + figure_name, transparent=True)



#############################################################



plt.figure()
for rat in arange(len(rat_summary_table_path)):
    
      
    rat_ball_selection  = rat_ball_all_rats[rat]
    st_speed = Level_2_start_to_touch_speed_all_rats[rat]


    flattened_rat_ball = [val for sublist in rat_ball_selection for val in sublist]
    flattened_st_speed = [val for sublist in st_speed for val in sublist]
    
    
    print(rat)
    print(len(flattened_rat_ball))
    print(len(flattened_st_speed))
    
    plt.plot(flattened_rat_ball,flattened_st_speed, '.', alpha=.7, markersize=.5, color= 'k')
    
    
   ##################################### 

###ball positions
##################################            


def ball_positions_finder(sessions_subset):
      
    l = len(sessions_subset)
    ball_rat = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]       
    
        script_dir = os.path.join(hardrive_path + session)     
        
        ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
        ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
        
        ball_rat[count] = ball_coordinates
        
    return ball_rat


s = len(rat_summary_table_path)

ball_all_rats = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         ball_rat = ball_positions_finder(sessions_subset)
         
         ball_all_rats[r] = ball_rat
         
            
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  

len(ball_all_rats)

#####flatten ball positions


ball_tot =[]


for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    ball_pos  = ball_all_rats[rat]



    ball_coordinates = [val for sublist in ball_pos for val in sublist]

    
    ball_tot.extend(np.array(ball_coordinates))

    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    #total_delta.extend(delta)
    
print(len(ball_tot))


#remove nans
ball_array = np.stack(ball_tot, axis =0)

ball_wo_nan = ball_array[~np.isnan(ball_array).any(axis=1)]
print(len(ball_wo_nan)) #5463

#remove x <250
remove_wrong_ball = [i for i,v in enumerate(ball_wo_nan[:,0]) if v > 250]


x= ball_wo_nan[:,0][remove_wrong_ball]
y= ball_wo_nan[:,1][remove_wrong_ball]



#plot ball positions






f,ax = plt.subplots(figsize=(9,7))


figure_name= 'heatmap_ball_positions_final.pdf'
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


ax= plt.hist2d(x,y,bins=50, norm = LogNorm(), cmap='RdBu_r')
plt.colorbar(shrink=0.8)



plt.title('ball_coordinates  excluding values less than 250_N5463',fontsize = 16)
plt.ylabel('ball y', fontsize = 13)
plt.xlabel('ball x', fontsize = 13)

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

f.savefig(results_dir + figure_name, transparent=True)



#####################################


from matplotlib.gridspec import GridSpec



fig = plt.figure(figsize=(7,7))

figure_name ='ball_position_heatmap_with_marginal_hist.pdf'

gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])

ax_joint.scatter(x,y,s=1)
ax_marg_x.hist(x,bins=50)
ax_marg_y.hist(y,bins=50,orientation="horizontal")
ax_marg_y.set_xlim([0, 200])
ax_marg_x.set_ylim([0, 200])

# Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

# Set labels on joint
ax_joint.set_xlabel('ball x ')
ax_joint.set_ylabel('ball y ')

# Set labels on marginals
ax_marg_y.set_xlabel('ball positions')
ax_marg_x.set_ylabel('ball positions')

fig.savefig(results_dir + figure_name, transparent=True)



#sns.jointplot(x,y, kind="hex", color="#4CB391",marginal_kws=dict(bins=50, rug=False))


             
              
              
              
              
#save the ball idx based on where it appears in the set up

def ball_positions_based_on_quadrant_of_appearance(session):
      
    ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')    
    ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float) 
    
    quadrant_1 = []
    quadrant_2 = []
    quadrant_3 = []
    quadrant_4 = []
       
    for n, row in enumerate(ball_coordinates):
        try:
            if row[0] <= 800 and row[1]>=600:
                quadrant_1.append(n)
            elif row[0] >= 800 and row[1]>=600:
                quadrant_2.append(n)
            elif row[0] <= 800 and row[1]<=600:
                quadrant_3.append(n)
            else:
                quadrant_4.append(n)
                         
        except Exception: 
            print (session + '/error')
        continue 
        
    return quadrant_1,quadrant_2,quadrant_3,quadrant_4




#########################################################

















#find rat nose x,y position at and around event 

def rat_nose_position_at_and_around(sessions_subset, event=2, offset=120):
    
    l = len(sessions_subset)
    nose_before = [[] for _ in range(l)]
    nose_after = [[] for _ in range(l)]
  
    for count in np.arange(l):
    
        
        session = sessions_subset[count]
        
    
        script_dir = os.path.join(hardrive_path + session) 

        trial_idx_path = os.path.join(script_dir+ '/events/' + 'Trial_idx.csv')
        trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 
        #centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        corrected_coordinate_path = os.path.join(script_dir + '/DLC_corrected_coordinates')
        nose_path = os.path.join(corrected_coordinate_path + '/nose_corrected_coordinates.csv')
        nose_dlc = np.genfromtxt(nose_path, delimiter = ',', dtype = float)
    
        
       
        rat_event = trial_idx[:,event]
        rat_pos_before = nose_dlc[rat_event-offset]
        rat_pos_after = nose_dlc[rat_event+offset]
            
        nose_before[count] = rat_pos_before
        nose_after[count]= rat_pos_after
    
    return nose_before, nose_after




#############################################################################################

s = len(rat_summary_table_path)

position_before = [[] for _ in range(s)]

position_after = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         nose_before, nose_after = rat_nose_position_at_and_around(sessions_subset, event=2, offset=120)
         
         position_before[r] = nose_before
         position_after[r] = nose_after
            
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  



#####flatten rat position 


before_tot =[]
after_tot=[]

for rat in arange(len(rat_summary_table_path)):
       
    #f,ax = plt.subplots(figsize=(7,5))
      
    before  = position_before[rat]
    after = position_after[rat]


    before_coordinates = [val for sublist in before for val in sublist]
    after_coordinates = [val for sublist in after for val in sublist]

    
    before_tot.extend(np.array(before_coordinates))
    after_tot.extend(np.array(after_coordinates))
    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    #total_delta.extend(delta)
    
print(len(before_tot))
print(len(after_tot))
#########################################################################################
    

                                                       
trial_table_path = 'F:/Videogame_Assay/Trial_table_final.csv'
trial_table = np.genfromtxt(trial_table_path, delimiter =',')


x_before_touch = trial_table[:,8]
y_before_touch =  trial_table[:,9]
x_after_touch =  trial_table[:,10]
y_after_touch =  trial_table[:,11]
ball_x =  trial_table[:,16]
ball_y =  trial_table[:,17]




ball_array = np.array(ball_tot)
before_array = np.array(before_tot)
after_array = np.array(after_tot)



#remove nans
ball_array = np.stack((ball_x,ball_y), axis=1)

ball_wo_nan = ball_array[~np.isnan(ball_array).any(axis=1)]
#print(len(ball_wo_nan)) #5463


nan_ball = ~np.isnan(ball_array).any(axis=1)


x_before_cleaned = x_before_touch[nan_ball]
y_before_cleaned = y_before_touch[nan_ball]
x_after_cleaned = x_after_touch[nan_ball]
y_after_cleaned = y_after_touch[nan_ball]

len(x_before_cleaned)



#remove x <250
cleaned_ball = [i for i,v in enumerate(ball_wo_nan[:,0]) if v > 250]

ball_final = ball_wo_nan[cleaned_ball]

len(x_ball)

x_before_final = x_before_cleaned[cleaned_ball]
y_before_final = y_before_cleaned[cleaned_ball]

x_after_final = x_after_cleaned[cleaned_ball]
y_after_final = y_after_cleaned[cleaned_ball]

len(after_final)




#fig = plt.figure(figsize=(15,9))
#
#plt.plot([before_final[:,0],ball_final[:,0]],[before_final[:,1],ball_final[:,1]],'r',alpha=.1)
#plt.plot([ball_final[:,0],after_final[:,0]],[ball_final[:,1],after_final[:,1]],'g',alpha=.1)
#


fig = plt.figure(figsize=(15,9))

plt.plot([x_before_final[:],ball_final[:,0]],[y_before_final[:],ball_final[:,1]],'r',alpha=.1)


plt.plot([ball_final[:,0],x_after_final[:]],[ball_final[:,1],y_after_final[:]],'g',alpha=.1)



#centered


fig = plt.figure(figsize=(15,9))

for i in range(len(ball_final)):
     
    
    #fig = plt.figure(figsize=(15,9))
    plt.plot([(x_before_final[i]-ball_final[i,0]),0],[(y_before_final[i]-ball_final[i,1]),0],'red',alpha=.05)
    
    plt.plot([0,(x_after_final[i]-ball_final[i,0])],[0,(y_after_final[i]-ball_final[i,1])],'b',alpha=.05)


#len(centre)

###############################
contact_below =[]
contact_top = []


for e, each in enumerate(y_before_final):
    
    if each < ball_final[e,1]:
        contact_below.append(e)
    else:
        contact_top.append(e)
        


centre= []


for i in range(len(y_before_final)):
    
    if  400  <ball_final[i,1]<800 and  600 <ball_final[i,0]<1000:       
        centre.append(i)
        
        
plt.plot(ball_final[centre][:,0],ball_final[centre][:,1], '.')


ball_centre = ball_final[centre]


x_centre_before=x_before_final[centre]
y_centre_before=y_before_final[centre]

x_centre_after=x_after_final[centre]
y_centre_after=y_after_final[centre]







centre_contact_below =[]
centre_contact_top = []


for e, each in enumerate(y_centre_before):
    
    if each < ball_centre[e,1]:
        centre_contact_below.append(e)
    else:
        centre_contact_top.append(e)

len(y_centre_before)
len(centre_contact_below)
len(centre_contact_top)



rat_poke_distance_before= trial_table[:,14]
rat_poke_distance_after=  trial_table[:,15]


distance_centre_before_below = rat_poke_distance_before[contact_below]
distance_centre_after_below =rat_poke_distance_after[contact_below]

delta_below = distance_centre_after_below-distance_centre_before_below


distance_centre_before_top =rat_poke_distance_before[contact_top]
distance_centre_after_top = rat_poke_distance_after[contact_top]



delta_top=distance_centre_after_top-distance_centre_before_top

####################################################


distance_centre_before_below = np.array(total_flat_before)[contact_below]
distance_centre_after_below =np.array(total_flat_after)[contact_below]

delta_below = distance_centre_after_below-distance_centre_before_below


distance_centre_before_top =np.array(total_flat_before)[contact_top]
distance_centre_after_top = np.array(total_flat_after)[contact_top]



delta_top=distance_centre_after_top-distance_centre_before_top

len(delta_below)


plt.boxplot((delta_below,delta_top))
plt.title('delta_below VS delta top (120)')






fig = plt.figure(figsize=(15,9))

plt.plot([x_centre_before[:],ball_centre[:,0]],[y_centre_before[:],ball_centre[:,1]],'r',alpha=.1)

fig = plt.figure(figsize=(15,9))

plt.plot([ball_centre[:,0],x_centre_after[:]],[ball_centre[:,1],y_centre_after[:]],'g',alpha=.1)









len(contact_below) #2080
len(contact_top) #3199



x_below = x_before_final[contact_below]
y_below = y_before_final[contact_below]

x_top = x_before_final[contact_top]
y_top = y_before_final[contact_top]



ball_top = ball_final[contact_top]
ball_below = ball_final[contact_below]


plt.scatter(ball_top[:,0],ball_top[:,1],color='g', s = 2)
plt.scatter(ball_below[:,0],ball_below[:,1],color= 'k', s= 2)




fig = plt.figure(figsize=(15,9))

plt.plot([x_below[:],ball_below[:,0]],[y_below[:],ball_below[:,1]],'r',alpha=.1)


plt.plot([ball_top[:,0],x_top[:]],[ball_top[:,1],y_top[:]],'g',alpha=.1)
















































#############################trial table####################


s = len(rat_summary_table_path)

position_before = [[] for _ in range(s)]

position_after = [[] for _ in range(s)]


for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         nose_before, nose_after = rat_nose_position_at_and_around(sessions_subset, event=2, offset=120)
         
         touch_rat, rat_pos_at_touch = rat_event_idx_and_pos_finder(sessions_subset, event=2)
         
         
         
         position_before[r] = nose_before
         position_after[r] = nose_after
            
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue  






#speed using tracking 





#calcute speedtracking diff

s = len(rat_summary_table_path)

Level_2_start_to_touch_speed_all_rats = [[] for _ in range(s)]
Level_2_touch_to_reward_speed_all_rats = [[] for _ in range(s)]

mean_st_all_rats =[[] for _ in range(s)]
std_st_all_rats =[[] for _ in range(s)]
mean_te_all_rats = [[] for _ in range(s)]
std_te_all_rats = [[] for _ in range(s)]

for r, rat in enumerate(rat_summary_table_path):
    
    try:    
         Level_2_pre = prs.Level_2_pre_paths(rat)
         sessions_subset = Level_2_pre#[3:6]
         
         session_speed = behaviour.session_speed_crop_tracking(sessions_subset)
         Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, session_speed)
         Level_2_touch_to_reward_speed = behaviour.speed_touch_to_reward(sessions_subset, session_speed)
         
         l = len(session_speed)

         mean_start_to_touch = []
         std_start_to_touch = [] 
         mean_touch_to_reward =[]
         std_touch_to_reward = []
         
         
         for count in np.arange(l):
             
             session_start_to_touch = Level_2_start_to_touch_speed[count]
             session_touch_to_reward = Level_2_touch_to_reward_speed[count]
             concat_speed_st = [item for sublist in session_start_to_touch for item in sublist]
             concat_speed_te = [item for sublist in session_touch_to_reward for item in sublist]
             
             
             start_touch_mean = np.nanmean(concat_speed_st)
             mean_start_to_touch.append(start_touch_mean)
             start_touch_std = np.nanstd(concat_speed_st)
             std_start_to_touch.append(start_touch_std)
             
             touch_reward_mean = np.nanmean(concat_speed_te)
             mean_touch_to_reward.append(touch_reward_mean)
             touch_to_reward_std = np.nanstd(concat_speed_te)
             std_touch_to_reward.append(touch_to_reward_std)
             
             
         Level_2_start_to_touch_speed_all_rats[r] = Level_2_start_to_touch_speed
         Level_2_touch_to_reward_speed_all_rats[r]=Level_2_touch_to_reward_speed
         mean_st_all_rats[r] = mean_start_to_touch
         std_st_all_rats[r] = std_start_to_touch
         mean_te_all_rats[r] = mean_touch_to_reward
         std_te_all_rats[r] = std_touch_to_reward
         
         
         print(rat)
         print(r)
         
    except Exception: 
        print (rat + '/error')
        continue    


#####################################
        
f,ax = plt.subplots(figsize=(15,5))



tot_before= []
tot_after = []

for rat in arange(len(rat_summary_table_path)):
    
    
    f,ax = plt.subplots(figsize=(7,5))
      
    before  = mean_st_all_rats[rat]
    after = mean_te_all_rats [rat]


    #flattened_before = [val for sublist in before for val in sublist]
    #flattened_after = [val for sublist in after for val in sublist]
    
    
    #delta = np.array(flattened_after) - np.array(flattened_before)
    print(rat)
    print(len(before))
    print(len(after))
    
    
    filtered_before = np.array(before)[~np.isnan(before)]
    filtered_after = np.array(after)[~np.isnan(after)]
    
    tot_before.append(filtered_before)
    tot_after.append(filtered_after)
    
    stack= np.vstack((filtered_before,filtered_after))
    to_plot = stack.tolist()
    
   # plt.figure()
    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
    
    #plt.bar(range(len(delta)),delta, width= 0.05)
    plt.boxplot(to_plot)
    plt.ylim((0,2.5))
    
    #ax.hlines(0,0,800,linewidth=0.5)
      
    
 
    
flattened_before = [val for sublist in tot_before for val in sublist]
flattened_after = [val for sublist in tot_after for val in sublist]
tot_stack= np.vstack((flattened_before,flattened_after))

to_plot = tot_stack.tolist()
    
   # plt.figure()
    #plt.plot(delta, 'o', alpha=1, markersize=.7,color='k')  
    
    #plt.bar(range(len(delta)),delta, width= 0.05)
plt.boxplot(to_plot)
#plt.ylim((0,2.5))

plt.plot(flattened_before,flattened_after,'.')

plt.plot(np.unique(flattened_before), np.poly1d(np.polyfit(flattened_before, flattened_after, 1))(np.unique(flattened_before)))


m,b = np.polyfit(flattened_before, flattened_after, 1)

# m 0.4675482133356882
# b 0.9965590810071268


coef = np.polyfit(flattened_before,flattened_after,1)
poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

plt.plot(flattened_before,flattened_after, 'yo', flattened_before, poly1d_fn(flattened_before), '--k')





test_speed = stats.ttest_rel(flattened_before,flattened_after)






#calculate speed for all the session (adapted to use nose corrected coordinates instead of the crop that we used originally)
sessions_speed = behaviour.session_speed(sessions_subset)

#calculate the speed from start of the trial to touch of the ball using the trial idx csv file 
Level_2_start_to_touch_speed = behaviour.speed_start_to_touch(sessions_subset, sessions_speed)

#calculate the speed from ball touch to reward using the csv file saved wit the idx 
Level_2_touch_to_reward_speed = behaviour.speed_touch_to_reward(sessions_subset, sessions_speed)

#from the speed of the session extract chunck of 6 seconds around the touch idx (360 frames before and 360 frames after)
speed_touch_Level_2 = behaviour.speed_around_touch(sessions_subset,sessions_speed)


#############################################################################################################
#plotting the speed around touch by doing the mean of each session and plotting the sessions 
#check because in some cases the speed touch is empty array thats why i used try
#remove the diff greater than 20 before calculating the mean

means = []
    
for row in speed_touch_Level_2:
    try:
        session_array = np.array(row)
        session_array[session_array>=20] = np.NaN
        mean_session = np.nanmean(session_array,axis=0)
        means.append(mean_session)        
    except Exception: 
        continue     

             

figure_name0 = figure_name = 'RAT_' + RAT_ID + '_speed_around_touch2.png'
plot_main_title = 'RAT ' + RAT_ID + 'speed around touch +/- 3 sec'
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine()


cmap = sns.color_palette("hls", len(means))
for count,i in enumerate(means):
    plt.plot(i, color=cmap[count], label='session %d'%count)

plt.axvline(360,color='k')
plt.ylim(0,5)
plt.legend()
f0.tight_layout()
f0.subplots_adjust(top = 0.87)

    
script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + RAT_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f0.savefig(results_dir + figure_name, transparent=True)
#########################################################################################################



l = len(sessions_speed)

mean_touch_to_reward = []
std_touch_to_reward = [] 
 
for count in np.arange(l):
    session_touch_to_reward = Level_2_touch_to_reward_speed[count]
    concat_speed = [item for sublist in session_touch_to_reward for item in sublist]
    touch_reward_mean = np.nanmean(concat_speed)
    mean_touch_to_reward.append(touch_reward_mean)
    touch_reward_std = np.nanstd(concat_speed)
    std_touch_to_reward.append(touch_reward_std)
    

mean_start_to_touch = []
std_start_to_touch = []

    
for count in np.arange(l):
    session_start_to_touch = Level_2_start_to_touch_speed[count]
    concat_start_to_touch_speed = [item for sublist in session_start_to_touch for item in sublist]
    start_touch_mean = np.nanmean(concat_start_to_touch_speed)
    mean_start_to_touch.append(start_touch_mean)
    start_touch_std = np.nanstd(concat_start_to_touch_speed)
    std_start_to_touch.append(start_touch_std)
    
    
test = 

    
stack_mean = np.vstack((mean_start_to_touch,mean_touch_to_reward)).T  
stack_std = np.vstack((std_start_to_touch,std_touch_to_reward)).T 


plt.scatter(mean_start_to_touch,range(l),marker = 'o',color= 'steelblue',alpha = .8)
plt.scatter(mean_touch_to_reward,range(l),marker = 'o',color= 'g',alpha = .8)

    



####################################################################


ball = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/events/Ball_coordinates.csv'
ball_pos = np.genfromtxt(ball, delimiter = ',', dtype = float)
centroid_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/crop.csv'
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)




figure_name = figure_name = 'RAT_' + RAT_ID + '_Speed_Touch_to_reward_Level2.pdf'
plot_main_title = 'RAT ' + RAT_ID + ' Speed_Touch_to_reward_Level2' + 'Level_2'

 
for i, session in enumerate(sessions_subset): 
    try:
        behaviour.full_trial_idx(sessions_subset)
        
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f0.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)

f0.tight_layout()
f0.subplots_adjust(top = 0.87)

