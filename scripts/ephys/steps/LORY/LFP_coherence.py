# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:25:23 2020

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
import mne
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from scipy import stats 
from mne import time_frequency
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 
#import plotting_probe_layout as layout
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import glob
from statsmodels.sandbox.stats.multicomp import multipletests
# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)


#testing folder 
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

hardrive_path = r'F:/'



#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']



#LEFT = BACK / RIGHT = FRONT / TOP = TOP /BOTTOM = BOTTOM)
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


#####FREQ BANDS OF INTEREST

#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz
#gamma = 30-45 and 55-100

# create raw downsampld chunks around event for each channel all the trial and save a cube ch x samples x trials

RAT_ID = RAT_ID_ephys[0] #[0]
rat_summary_table_path=rat_summary_ephys[0]#[0]

probe_map_flatten = ephys.probe_map.flatten()
len(probe_map_flatten)

for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)
    offset = 1500
    final_array = np.zeros((121,offset*2,))
    
    tot_trial=[]
    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)

        
        #filed needed path 
        csv_dir_path = os.path.join(session_path + '/events/')
        touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
        ball_on = os.path.join(hardrive_path, session +'/events/'+'BallON.csv')
        #trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
        trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
        #trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)       
        video_csv = os.path.join(session_path + '/Video.csv')       
        samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
        
        
        #files opening
        samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)
        
        trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
        touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
        ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
    
    
        downsampled_touch = np.uint32(np.array(touching_light)/30)
        downsampled_ball = np.uint32(np.array(ball)/30)
        downsampled_end= np.uint32(np.array(trial_end)/30)

        freq = 30000
        offset = 1500
        num_raw_channels = 128
        

        data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
        down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
        num_samples = int(int(len(down))/num_raw_channels)
        reshaped_down=  np.reshape(down,(num_samples,num_raw_channels))  
        down=None
        down_T = reshaped_down.T
        
        #baseline_idx = np.arange(120000,num_samples-120000,6000) 
        #remove the first early trials
        downsampled_event_idx = downsampled_end[1:]
        
        tot_trial.append(len(downsampled_event_idx))
        
        #cube ch (121)x time(3000) x trials 
        matrix_around_event = np.zeros((121,offset*2,len(downsampled_event_idx)))

            
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten                                   
                
            ch_downsampled = down_T[channel,:]#should use channel
            #down_T=None
    
            chunk = np.zeros((len(downsampled_event_idx),offset*2))
                         
    
            for e, event in enumerate(downsampled_event_idx):
                 
                chunk[e,:] = ch_downsampled[event-offset : event+offset]
                                
                #print(e)
            matrix_around_event[ch,:,:]  = chunk.T            
        
        filename = 'all_ch_1500_around_'+'reward'
        saving_folder = os.path.join(session_path + '/LFP/' + filename )
        np.save(saving_folder,matrix_around_event) 
        if s==0:
            
            final_array = matrix_around_event
            
        else:
            
            final_array= np.concatenate([final_array,matrix_around_event],axis=-1)



    #create big array with all trials and save in the summary LFP     
    final_filename= RAT_ID[r] + '_raw_snippets_1500_all_trials_' + 'reward'    
    summary_folder = 'F:/Videogame_Assay/LFP_summary/'
    saving_folder = os.path.join(summary_folder + final_filename)
    np.save(saving_folder,final_array)
              
    print(sum(tot_trial))
    print(np.shape(final_array))    
    print(r)


####coherence code 

summary_folder =  'F:/Videogame_Assay/LFP_summary/raw_snippets_around_event/'
events=['touch','ball','reward']


for e in arange(len(events)):
    
    event = events[e]
    
    for rat in arange(len(RAT_ID_ephys)):

        matching_files = glob.glob(summary_folder +"*"+RAT_ID_ephys[rat]+"*" +"*"+event+"*")
        array = np.load(matching_files[0])
    
        cluster_avg = avg_9_clusters(array,offset)
        
        dt = 0.001                                # Define the sampling interval.
        K = np.shape(cluster_avg)[2]               # Define the number of trials.
        N = np.shape(cluster_avg)[1]             # Define number of points in each trial.
        C = np.shape(cluster_avg)[0]
        T = N*dt 
        
        f = rfftfreq(N, dt) 
                
        
        #Visualizing the Phase Difference across Trials
        delta = [i for i,v in enumerate(f) if 1<= v <4 ]
        theta = [i for i,v in enumerate(f) if 4<= v <8 ]
        alpha = [i for i,v in enumerate(f) if 8<= v <12 ]
        beta = [i for i,v in enumerate(f) if 12<= v <30 ]
        gamma =  [i for i,v in enumerate(f) if  30<= v <45 or 55< v <=100 ] 
        high_gamma = [i for i, v in enumerate(f) if 100<v<300]
        
        #array to fill with all the combo 9*9
        delta_angles = zeros((C**2,len(delta), K))            # Variables to hold phase differences.
        theta_angles = zeros((C**2,len(theta), K))
        alpha_angles =  zeros((C**2,len(alpha), K))
        beta_angles =  zeros((C**2,len(beta), K))
        gamma_angles =  zeros((C**2,len(gamma), K))
        high_gamma_angles = zeros((C**2,len(high_gamma), K))
        #ch_1 =100
        #ch_2 = 15
                                 # ... and the total time of the recording.
        
       
        for k in arange(K):    
            
            x = cluster_avg[:].T[k] - mean(cluster_avg[:].T[k], axis=0)  
        
            xf = rfft((x.T))   # ... compute the Fourier transform,
           
            Sxy = []
            for i in np.arange(C):
                for j in np.arange(C):
                    Sxy.append(2 * dt**2 / T * (xf[i] * conj(xf[j])))
                    
                    #print(j)
                #print(i)
                
               
            outcome = np.array(Sxy)
            #mean of the values within a freq band
            
            delta_angles[:, :,k] = angle(outcome[:,delta])
            theta_angles[:, :,k] = angle(outcome[:,theta])
            alpha_angles[:, :,k] = angle(outcome[:,alpha])
            beta_angles[:,:, k] = angle(outcome[:,beta])
            gamma_angles[:, :,k] = angle(outcome[:,gamma])
            high_gamma_angles[:,:, k] = angle(outcome[:,high_gamma])
        
        
        #test = math.radians(delta_mean[:])
        gamma_mean = np.mean(gamma_angles.reshape((gamma_angles.shape[0], gamma_angles.shape[1]*gamma_angles.shape[2])), axis=1)
        delta_mean = np.mean(delta_angles.reshape((delta_angles.shape[0], delta_angles.shape[1]*delta_angles.shape[2])), axis=1)
        beta_mean = np.mean(beta_angles.reshape((beta_angles.shape[0],beta_angles.shape[1]*beta_angles.shape[2])), axis=1)
        alpha_mean = np.mean(alpha_angles.reshape((alpha_angles.shape[0], alpha_angles.shape[1]*alpha_angles.shape[2])), axis=1)
        theta_mean = np.mean(theta_angles.reshape((theta_angles.shape[0], theta_angles.shape[1]*theta_angles.shape[2])), axis=1)
        high_gamma_mean = np.mean(high_gamma_angles.reshape((high_gamma_angles.shape[0], high_gamma_angles.shape[1]*high_gamma_angles.shape[2])), axis=1)
        
        
        
        
        
        gamma_std = np.std(gamma_angles.reshape((gamma_angles.shape[0], gamma_angles.shape[1]*gamma_angles.shape[2])), axis=1)
        delta_std = np.std(delta_angles.reshape((delta_angles.shape[0], delta_angles.shape[1]*delta_angles.shape[2])), axis=1)
        beta_std = np.std(beta_angles.reshape((beta_angles.shape[0],beta_angles.shape[1]*beta_angles.shape[2])), axis=1)
        alpha_std = np.std(alpha_angles.reshape((alpha_angles.shape[0], alpha_angles.shape[1]*alpha_angles.shape[2])), axis=1)
        theta_std= np.std(theta_angles.reshape((theta_angles.shape[0], theta_angles.shape[1]*theta_angles.shape[2])), axis=1)
        high_gamma_std = np.std(high_gamma_angles.reshape((high_gamma_angles.shape[0], high_gamma_angles.shape[1]*high_gamma_angles.shape[2])), axis=1)
        
        
        
        
        
        
        #automatised plot for each freq
        means= [delta_mean,theta_mean,alpha_mean,beta_mean,gamma_mean, high_gamma_mean]
        stds  = [delta_std,theta_std,alpha_std,beta_std,gamma_std, high_gamma_std]
        freq_range =['delta','theta','alpha','beta','gamma', 'high_gamma']
        
        
        for freq in range(len(means)):
            
            
            reshape_mean = np.reshape(means[freq],newshape=(9,9))
            reshape_std = np.reshape(stds[freq],newshape=(9,9))
             
            #to show only half of the matrix
            mask = np.triu(np.ones_like(reshape_mean, dtype=bool))
            
            fig,ax = plt.subplots(1,2,figsize=(15,7),sharey=True)
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine() 
                
            plt.subplot(121)     
            ax= sns.heatmap(reshape_mean,annot=True,  cmap="bwr",mask=mask,vmin=-1,vmax=+1)#,annot=reshape_std
            bottom, top = ax.get_ylim()#,norm=LogNorm() # "YlGnBu" RdBu
            ax.set_ylim(bottom + 0.5, top - 0.5)

#                
            plt.title(RAT_ID_ephys[rat]+'_avg ' + freq_range[freq]+ '_'+ event)   
#                fig_name = RAT_ID_ephys[rat]+'_avg_' + freq_range[freq]+'png'
#                f.savefig('F:/Videogame_Assay/LFP_summary_plots/'+ fig_name +'.png') 
#                plt.close()
##                ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````` xh\z,l.;'
            cdxzrtyuiop ui=-09
#                f1 =plt.figure(figsize=(8,8))
#                sns.set()
#                sns.set_style('white')
#                sns.axes_style('white')
#                sns.despine() 
               
            plt.subplot(122)      
            ax = sns.heatmap(reshape_std,annot=True,  cmap="bwr",mask=mask, vmin=0,vmax=2)
            bottom, top = ax.get_ylim()#,norm=LogNorm() # "YlGnBu" RdBu
            ax.set_ylim(bottom + 0.5, top - 0.5)

            
            plt.title(RAT_ID_ephys[rat]+ '_std '+ freq_range[freq]+ '_'+ event) 
            
            
            
            fig_name = RAT_ID_ephys[rat]+'_9_clusters_avg_and_std_' + freq_range[freq] + '_'+ event
            fig.savefig('F:/Videogame_Assay/LFP_summary_plots/'+ fig_name +'.png')
            plt.close()
            
            print(fig_name+'_SAVED')
        
     




#
##test single session
#
#test = np.load('F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/LFP/all_ch_1500_around_touch.npy')       
# 
#
#
#dt = 0.001                                # Define the sampling interval.
#K = np.shape(test)[2]               # Define the number of trials.
#N = np.shape(test)[1]             # Define number of points in each trial.
#C = np.shape(test)[0]
#
#f = rfftfreq(N, dt)   
#
##Visualizing the Phase Difference across Trials
#
#j8 = where(f==30)[0][0]            # Determine index j for frequency 10 Hz.
#j24 = where(f==10)[0][0]           # Determine index j for frequency 50 Hz.
#
#phi8 = zeros((C**2, K))            # Variables to hold phase differences.
#phi24 = zeros((C**2, K))
#
##ch_1 =100
##ch_2 = 15
#T = N*dt                          # ... and the total time of the recording.
#
#ch deve  essere trialsxlenght
for k in range(K):           # For each trial, compute the cross spectrum. 
    x = test[:].T[k] - mean(test[:].T[k])  # Get the data from each electrode,
    #y = test[ch_2].T[k].T - mean(test[ch_2].T[k,:])
    xf = rfft((x.T))   # ... compute the Fourier transform,
    #yf = rfft(y - mean(y))
    #Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    Sxy = []
    for i in np.arange(C):
        for j in np.arange(C):
            Sxy.append(2 * dt**2 / T * (xf[i] * conj(xf[j])))
            outcome = np.array(Sxy)
            print(j)
        print(i)
    phi8[:, k] = angle(outcome[:,j8]) # ... and the phases.
    phi24[:, k] = angle(outcome[:, j24])
#    
# 
#outcome = np.array(Sxy)
#                            # []Plot the distributions of phases.
#_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
#a1.hist(phi8, bins=20, range=[-pi, pi])
#a2.hist(phi24, bins=20, range=[-pi, pi])
#
##ylim([0, 40])                # Set y-axis and label axes.
#a1.set_ylabel('Counts')
#a1.set_xlabel('Phase');
#a1.set_title('Angles at 20 Hz')
#
#a2.set_title('Angles at 50 Hz')
#a2.set_xlabel('Phase');

############################################################################
#avg ch of each layer


layer_avg= np.zeros((11,offset*2,K))

for trial in range(K):
    
    t=test[:,:,trial]
    
    means=[]
    count=0
    for c in range(11):
        t2=np.mean(t[0+count:11+count],axis=0)
        means.append(t2)
        count+=11
    stack = np.vstack(means)
    layer_avg[:,:,trial]=means



ch_1 =layer_avg[0,:].T
ch_2 = layer_avg[1,:].T
T = 3 



j8 = where(f==20)[0][0]       # Determine index j for frequency 10 Hz.
j24 = where(f==30)[0][0]     # Determine index j for frequency 50 Hz.

phi8 = zeros(K)              # Variables to hold phase differences.
phi24 = zeros(K)

for k in range(K):           # For each trial, compute the cross spectrum. 
    x = ch_1[k] - mean(ch_1[k])  # Get the data from each electrode,
    y = ch_2[k] - mean(ch_2[k,:])
    xf = rfft(x - mean(x))   # ... compute the Fourier transform,
    yf = rfft(y - mean(y))
    Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    phi8[k] = angle(Sxy[j8]) # ... and the phases.
    phi24[k] = angle(Sxy[j24])
                             # Plot the distributions of phases.
_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
a1.hist(phi8, bins=20, range=[-pi, pi])
a2.hist(phi24, bins=20, range=[-pi, pi])

#ylim([0, 40])                # Set y-axis and label axes.
a1.set_ylabel('Counts')
a1.set_xlabel('Phase');
a1.set_title('Angles at 30 Hz')

a2.set_title('Angles at 80 Hz')
a2.set_xlabel('Phase');

###############################################################################
#avg shanks


shank_avg= np.zeros((11,offset*2,K))

for trial in range(K):
    
    t=test[:,:,trial]
    
    means=[]
    start_array = np.arange(0,121,11)
    for c in range(11):
        
        t2=np.mean(t[start_array],axis=0)
        means.append(t2)
        start_array+=1

    shank_avg[:,:,trial]=means



ch_1 =shank_avg[0,:].T
ch_2 = shank_avg[5,:].T
T = 3 



j8 = where(f==25)[0][0]       # Determine index j for frequency 10 Hz.
j24 = where(f==35)[0][0]     # Determine index j for frequency 50 Hz.

phi8 = zeros(K)              # Variables to hold phase differences.
phi24 = zeros(K)

for k in range(K):           # For each trial, compute the cross spectrum. 
    x = ch_1[k] - mean(ch_1[k])  # Get the data from each electrode,
    y = ch_2[k] - mean(ch_2[k,:])
    xf = rfft(x - mean(x))   # ... compute the Fourier transform,
    yf = rfft(y - mean(y))
    Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    phi8[k] = angle(Sxy[j8]) # ... and the phases.
    phi24[k] = angle(Sxy[j24])
                             # Plot the distributions of phases.
_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
a1.hist(phi8, bins=20, range=[-pi, pi])
a2.hist(phi24, bins=20, range=[-pi, pi])

#ylim([0, 40])                # Set y-axis and label axes.
a1.set_ylabel('Counts')
a1.set_xlabel('Phase');
a1.set_title('Angles at 30 Hz')

a2.set_title('Angles at 80 Hz')
a2.set_xlabel('Phase');

####################################################################

#create 9 regional channel groups


cluster_avg = avg_9_clusters(test,offset)
event_array=test

def avg_9_clusters(event_array, offset):

    
    start_array = np.arange(0,121,11)
    start = [start_array[:4],start_array[4:7],start_array[7:11]]
    K= np.shape(event_array)[-1]
    cluster_avg= np.zeros((9,offset*2,K))
    #create from left to right (0 to 11)
    
    for st in np.arange(3):
        
        starting = start[st]
        
        for trial in range(K):
            
            t = event_array[:,:,trial]
            
            
            #collect all the ch belonging to each group
            group_1=[]
            group_2=[]
            group_3=[]
            
    
            count=0
            for s in np.arange(11): # 11 channels in a row
                new = starting + count
                if count <= 3:
                    #select column withon a group
                    sel = t[new]
                    group_1.append(sel)
                    count+=1
                elif   3<count<=6:
                    sel = t[new]
                    group_2.append(sel)
                    count+=1
                else:
                    sel = t[new]
                    group_3.append(sel)
                    count+=1
                   
            #depending of which idx I start with I fill the final array with the groups 
            if st==0:
                
                cluster_avg[0,:,trial] = np.mean(np.vstack(group_1),axis=0)
                cluster_avg[1,:,trial] = np.mean(np.vstack(group_2),axis=0)
                cluster_avg[2,:,trial] = np.mean(np.vstack(group_3),axis=0)
                
            elif st==1:
                
                cluster_avg[3,:,trial] = np.mean(np.vstack(group_1),axis=0)
                cluster_avg[4,:,trial] = np.mean(np.vstack(group_2),axis=0)
                cluster_avg[5,:,trial] = np.mean(np.vstack(group_3),axis=0)     
                
            else:
                
                cluster_avg[6,:,trial] = np.mean(np.vstack(group_1),axis=0)
                cluster_avg[7,:,trial] = np.mean(np.vstack(group_2),axis=0)
                cluster_avg[8,:,trial] = np.mean(np.vstack(group_3),axis=0)              
            
    return cluster_avg
 
    


event_array=test

#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz     

dt = 0.001                                # Define the sampling interval.
K = np.shape(cluster_avg)[2]               # Define the number of trials.
N = np.shape(cluster_avg)[1]             # Define number of points in each trial.
C = np.shape(cluster_avg)[0]
T=3
f = rfftfreq(N, dt)   

#Visualizing the Phase Difference across Trials
delta = [i for i,v in enumerate(f) if 1<= v <4 ]
theta = [i for i,v in enumerate(f) if 4<= v <8 ]
alpha = [i for i,v in enumerate(f) if 8<= v <12 ]
beta = [i for i,v in enumerate(f) if 12<= v <30 ]
gamma =  [i for i,v in enumerate(f) if  30<= v <45 or 55< v <=100 ]




delta_angles = zeros((C**2, K))            # Variables to hold phase differences.
theta_angles = zeros((C**2, K))
alpha_angles =  zeros((C**2, K))
beta_angles =  zeros((C**2, K))
gamma_angles =  zeros((C**2, K))
#ch_1 =100
#ch_2 = 15
T = N*dt                          # ... and the total time of the recording.

#ch deve  essere trialsxlenght
for k in range(K):           # For each trial, compute the cross spectrum. 
    x = test[:].T[k] - mean(test[:].T[k])  # Get the data from each electrode,
    #y = test[ch_2].T[k].T - mean(test[ch_2].T[k,:])
    xf = rfft((x.T))   # ... compute the Fourier transform,
    #yf = rfft(y - mean(y))
    #Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    Sxy = []
    for i in np.arange(C):
        for j in np.arange(C):
            Sxy.append(2 * dt**2 / T * (xf[i] * conj(xf[j])))
            
            print(j)
        print(i)
    outcome = np.array(Sxy)
    
    delta_angles[:, k] = angle(np.mean(outcome[:,delta],axis=1)) # ... and the phases.
    theta_angles[:, k] = angle(np.mean(outcome[:,theta],axis=1))
    alpha_angles[:, k] = angle(np.mean(outcome[:,alpha],axis=1))
    beta_angles[:, k] = angle(np.mean(outcome[:,beta],axis=1))
    gamma_angles[:, k] = angle(np.mean(outcome[:,gamma],axis=1))









delta_mean=np.mean(delta_angles,axis=1)
delta_std=np.std(delta_angles,axis=1)

theta_mean=np.mean(theta_angles,axis=1)
theta_std=np.std(theta_angles,axis=1)

alpha_mean=np.mean(alpha_angles,axis=1)
alpha_std=np.std(alpha_angles,axis=1)

beta_mean=np.mean(beta_angles,axis=1)
beta_std=np.std(beta_angles,axis=1)

gamma_mean=np.mean(gamma_angles,axis=1)
gamma_std=np.std(gamma_angles,axis=1)


means= [delta_mean,theta_mean,alpha_mean,beta_mean,gamma_mean]
stds  = [delta_std,theta_std,alpha_std,beta_std,gamma_std]
freq_range =['delta','theta','alpha','beta','gamma']


for freq in range(len(means)):
    
    
    reshape_mean = np.reshape(means[freq],newshape=(9,9))
    reshape_std = np.reshape(stds[freq],newshape=(9,9))
     
    
    mask = np.triu(np.ones_like(reshape_mean, dtype=bool))
    
    f =plt.figure(figsize=(10,10))
    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine() 
        
         
    ax = sns.heatmap(reshape_mean,annot=True,  cmap="bwr")#,annot=reshape_std
    bottom, top = ax.get_ylim()#,norm=LogNorm() # "YlGnBu" RdBu
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.title('avg ' + freq_range[freq])   
    fig_name = 'avg_' + freq_range[freq]+'png'
    f.savefig('F:/Videogame_Assay/LFP_summary_plots/'+ fig_name +'.png')             
    
    f1 =plt.figure(figsize=(10,10))
    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine() 
       
         
    ax = sns.heatmap(reshape_std,annot=True,  cmap="bwr")
    bottom, top = ax.get_ylim()#,norm=LogNorm() # "YlGnBu" RdBu
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.title('std '+ freq_range[freq]) 
    fig_name = 'std_' + freq_range[freq]+'png'
    f.savefig('F:/Videogame_Assay/LFP_summary_plots/'+ fig_name +'.png')     