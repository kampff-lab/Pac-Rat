# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: downsample to 1 kHz

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

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)


#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 


rat_summary_table_path = 'F:/Videogame_Assay/AK_41.1_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[5]
session_path =  os.path.join(hardrive_path,session)

#recording data path
raw_recording = os.path.join(session_path +'/Amplifier.bin')
downsampled_recording = os.path.join(session_path +'/Amplifier_cleaned_downsampled.bin')
cleaned_recording =  os.path.join(session_path +'/Amplifier_cleaned.bin')

#idx ro identify the start and the end of the clip of interest both in ephys samples and frames   
csv_dir_path = os.path.join(session_path + '/events/')
touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
ball_on = os.path.join(hardrive_path, session +'/events/'+'BallON.csv')
#trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
#trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)

video_csv = os.path.join(session_path + '/Video.csv')

samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)






trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
#trial prior end to current trial end based on ephys samples tp use with raw and cleaned recordings
touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
#generate random idx for baseline freq spectrum 
downsampled_touch = np.uint32(np.array(touching_light)/30)
downsampled_ball = np.uint32(np.array(ball)/30)
downsampled_end= np.uint32(np.array(trial_end)/30)
#end_samples = event_finder(trial_end_idx,video_csv,samples_for_frames_file_path)
#samples_lenght_end_to_end = np.diff(np.hstack((0, end_samples)))
#sample_start_clip = end_samples[21]
#clip_sample_lenght = samples_lenght_end_to_end[22]

freq = 30000
offset = 3000
num_raw_channels = 128


data = np.memmap(cleaned_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
down_sample_lenght = num_samples/30

# 60000 so it starts 1 min into the task

start = 60000
stop = down_sample_lenght-offset*2
idx = len(touching_light)




baseline_random = randint(start,stop,idx)
baseline_idx = np.sort(baseline_random)
test_baseline = downsampled_touch - baseline_idx
min_distance = np.min(abs(test_baseline))
max_distance = np.max(abs(test_baseline))
print(min_distance)
print(max_distance)
#plt.figure()
#plt.hist(baseline_random,bins=20)


baseline_idx = downsampled_touch + 6000




probe_map_flatten = ephys.probe_map.flatten()
new_probe_flatten_test =probe_map_flatten[[103,7,21,90,75,30]] #[103,7,21,90,75,30]

downsampled_event_idx =downsampled_touch

for ch, channel in enumerate(new_probe_flatten_test):
    try:
                
        #data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        print('converted_in_uv')
        
        ch_lowpass = ephys.butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
        ch_raw_uV = None
        
        #plt.figure()
        #plt.plot(ch_lowpass[30000:45000])
        
        ch_downsampled = ch_lowpass[::30]        
        print('lowpassed_and_downsampled')
        
        
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        


        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        baseline_chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)



        for b, base in enumerate(baseline_idx):
   
            baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            print(b)
            
            
        ch_downsampled = None
        
        chunk_lenght = offset*2
            
        p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

        p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)


        p_ch_avg = np.mean(p_ch, axis =0)
        p_ch_sem = stats.sem(p_ch, axis = 0)

        p_base_avg = np.mean(p_base, axis =0)
        p_base_sem = stats.sem(p_base)

        sns.set()
        fig = plt.figure()
        

        plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)    
        plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')
        
        plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
        plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                         alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')
       
        
        plt.title('ch_'+ str(channel))
        plt.legend(loc='best') 
        
        results_dir = 'F:/Videogame_Assay/test_plots/'
        figure_name = 'Freq_spec_around_touch'+ str(channel) + '.png'
        fig.savefig(results_dir + figure_name)
        plt.close()    
            
    except Exception:
        continue 
       




        
    
    
for i, idx in enumerate(chunk_around_event):
    plt.figure()
    plt.plot(idx)
    plt.title('i')
    
for i, idx in enumerate(baseline_chunk_around_event):
    plt.figure()
    plt.plot(idx)
    plt.title('i')
     
    
#create epochs 
test_epochs= None
        
data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)

# Reshape data to have 128 rows
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        


test_epochs = np.zeros((len(downsampled_event_idx), len(probe_map_flatten),offset*2))  
#new_probe_flatten_test = [103,7,21,90,75,30]    
for ch, channel in enumerate(probe_map_flatten):
    try:
        
        
        #data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        print('converted_in_uv')
        
        #ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
        #ch_raw_uV = None
        
        #plt.figure()
        #plt.plot(ch_lowpass[30000:45000])
        
        #ch_downsampled = ch_lowpass[::30]        
        #print('lowpassed_and_downsampled')
        
        
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        
        ch_downsampled=ch_raw_uV 

        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)

        test_epochs[:,ch,:] = chunk_around_event
        print(ch)
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))


        #for b, base in enumerate(baseline_idx):
   
            #baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            #print(b)
            

            
    except Exception:
        continue     
    
# old data 




data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)

# Reshape data to have 128 rows
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        


test_epochs = np.zeros((len(downsampled_event_idx), len(probe_map_flatten),offset*2))  
#new_probe_flatten_test = [103,7,21,90,75,30]    
for ch, channel in enumerate(probe_map_flatten):
    try:
        
        
        #data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        print('converted_in_uv')
        
        ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
        ch_raw_uV = None
        
        #plt.figure()
        #plt.plot(ch_lowpass[30000:45000])
        
        ch_downsampled = ch_lowpass[::30]        
        print('lowpassed_and_downsampled')
        
        
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        
        #ch_downsampled=ch_raw_uV 

        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)

        test_epochs[:,ch,:] = chunk_around_event
        print(ch)
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))


        #for b, base in enumerate(baseline_idx):
   
            #baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            #print(b)
            

            
    except Exception:
        continue     
#test = test_epochs[:,:,ch]    

test= None
    
freqs = np.arange(3.0, 100.0, 2.0)    
    
test_tfr2 = time_frequency.tfr_array_multitaper(test_epochs,sfreq= 1000,freqs = freqs, output= 'avg_power',n_jobs=8)   

norm = np.mean(test_tfr[94,:20,:1000],axis=1)

norm_expanded = np.repeat([norm], offset*2, axis=0).T

ch_test_norm = test_tfr[94,:20,:]/norm_expanded


ch_test = np.log(test_tfr[94,:20,:])
plt.figure()
plt.imshow(np.flipud(ch_test_norm),aspect='auto', cmap='jet')#,vmin=0.4, vmax =1.9)
#plt.axvline(6000,20,color='k')
plt.colorbar()

   
f0 =plt.figure(figsize=(20,20))
#outer_grid = gridspec.GridSpec(11, 11, wspace=0.0, hspace=0.0)

for i, ch in enumerate(probe_map_flatten):
    #inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1,
     #       subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

    norm = np.mean(test_tfr2[i,:20,:1000],axis=1)
    norm_expanded = np.repeat([norm], offset*2, axis=0).T
    ch_test_norm = test_tfr2[i,:20,:]/norm_expanded
    ch_test = np.log(test_tfr2[i,:20,:])
       
    ax = f0.add_subplot(11, 11, 1+i, frameon=False)

    plot = ax.imshow(np.flipud(ch_test_norm),aspect='auto', cmap='jet')#,vmin=0.4, vmax =1.9)
    cbar=plt.colorbar(plot,fraction=0.04, pad=0.04, aspect=10, orientation='horizontal')
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
   
f0.subplots_adjust(wspace=.02, hspace=.02)
plt.title('raw')

#
#test_plot 
#
#event = downsampled_end[20]
#
#
#chunk_around_event_raw = np.zeros((len(probe_map_flatten),offset*2))
#        
#       
#    for c, ch in enumerate(probe_map_flatten):
#             
#        chunk_around_event_raw[c,:] = ch_downsampled[event-offset : event+offset]
#            print(e)
#
#
#
#


#test plot
channel = 103


raw_data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(raw_data))/num_raw_channels)

# Reshape data to have 128 rows
reshaped_raw_data = np.reshape(raw_data,(num_samples,num_raw_channels)).T
raw = reshaped_raw_data[channel, :]
       
ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
raw = None
       
        
ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
        
        
plt.figure()
plt.plot(ch_lowpass[30000:45000])
plt.title('lowpass')
        
ch_downsampled = ch_lowpass[::30]        
 


plt.figure()
plt.plot(ch_downsampled[1000:1500])
plt.title('downsampled')
    



data_downsampled = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data_downsampled))/num_raw_channels)

# Reshape data to have 128 rows
reshaped_downsampled_data = np.reshape(data_downsampled,(num_samples,num_raw_channels)).T
downsampled = reshaped_downsampled_data[channel, :]
       
ch_down_uV = (downsampled.astype(np.float32) - 32768) * 0.195

plt.figure()
plt.plot(ch_down_uV[1000:1500])
plt.title('downsampled_cleaned')




#test different trial lenghts



freq = 30000
num_raw_channels = 128
final_trial_length = 8192 #(power of 2)
intermediate_trial_length = 9000

ball=ball[:-1]

data = np.memmap(cleaned_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
#down_sample_lenght = num_samples/30

probe_map_flatten = ephys.probe_map.flatten()
channel = 103
raw = reshaped_data[channel, :]
      
# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
raw=None
ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')


n = len(touching_light)   
ball_to_touch_chunks = [[] for _ in range(n)] 
touch_to_reward =  [[] for _ in range(n)] 



for trial in np.arange(n):
    b = ball[trial]
    t = touching_light[trial]
    e = trial_end[trial]
    ball_to_touch_chunks[trial]= ch_lowpass[b:t]
    touch_to_reward[trial]=ch_lowpass[t:e]
    

test= ball_to_touch_chunks.pop(2)
test2=ball_to_touch_chunks.pop(14)
test3=touch_to_reward.pop(2)
test4 = touch_to_reward.pop(14)


downsampled_ball_to_touch_chunks =  [[] for _ in range(len(ball_to_touch_chunks))] 
downsampling_factor_bt = []

for t, trial in enumerate(ball_to_touch_chunks):
    trial_length = len(trial)
    down = np.uint32((trial_length/intermediate_trial_length))
    downsampling_factor_bt.append(down)
    t_down = trial[::down]
    length_down = len(t_down)
    diff_from_final = length_down - final_trial_length
    downsampled_ball_to_touch_chunks[t]=t_down[:-diff_from_final]






downsampled_touch_to_reward_chunks =  [[] for _ in range(len(touch_to_reward))] 
downsampling_factor_tr = []


for t, trial in enumerate(touch_to_reward):
    trial_length = len(trial)
    down = np.uint32((trial_length/intermediate_trial_length))
    downsampling_factor_tr.append(down)
    t_down = trial[::down]
    length_down = len(t_down)
    diff_from_final = length_down - final_trial_length
    downsampled_touch_to_reward_chunks[t]=t_down[:-diff_from_final]




downsampled_1000_touch_to_reward_chunks =  [[] for _ in range(len(touch_to_reward))] 


for t, trial in enumerate(touch_to_reward):

    t_down = trial[::30]
    downsampled_1000_touch_to_reward_chunks[t]=t_down




#test_trial =downsampled_touch_to_reward_chunks[24]
#len(test_trial)
#
#
#
#
#long_trial = []
#
#
#for f, factor in enumerate(downsampling_factor):
#
#    long = [idx for idx, val in factor) if val > 9000000 ]#or val < 150000] 
#    print (min(mean_impedance_Level_2_post[s]))
#    print (max(mean_impedance_Level_2_post[s]))
#    if idx_bad_imp == 0 :
#        
#        bad_channels_idx[s] = []
#    else:
#       bad_channels_idx[s] = idx_bad_imp 
#
p_all = []
f_all = []

for t,test_trial in enumerate(downsampled_1000_touch_to_reward_chunks):

    band = 5000/len(test_trial)
    print(band)

    p_ch, f_ch = time_frequency.psd_array_multitaper(test_trial, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = band, n_jobs = 8)
    p_all.append(p_ch)
    f_all.append(f_ch)


for p in np.arange(len(downsampled_touch_to_reward_chunks)):
    plt.figure()
    plt.plot(f_all[1],p_all[1])
    plt.plot(f_all[13],p_all[13])
    plt.plot(f_all[40],p_all[40])
    plt.plot(f_all[7],p_all[7])


avg = np.mean(p_all)


#6000 2.5
#10551 1.5
# 26586 1.5


plt.figure()
plt.plot(f_ch,p_ch)



p_all = []
f_all = []

for t in downsampled_ball_to_touch_chunks:
    
    
    p_ch, f_ch = time_frequency.psd_array_multitaper(t, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
    p_all.append(p_ch)
    f_all.append(f_ch)




plt.figure()
plt.plot(f_ch,p_ch)



#11x11 freq spectra plot around events

freq = 30000
offset = 3000
num_raw_channels = 128


data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
down_sample_lenght = num_samples/30


#20 sec into the session

baseline_idx = 23000

# event before baseline

baseline = 



probe_map_flatten = ephys.probe_map.flatten()
new_probe_flatten_test =[103,7,21,90,75,30]


#remove the first early trials
downsampled_event_idx = downsampled_end[1:]

f0 =plt.figure(figsize=(20,20))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
 

for ch, channel in enumerate(probe_map_flatten):
    try:
                
        #data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        ch_downsampled=ch_raw_uV
                
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        


        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        baseline_chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)


   
        baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
            
            
            
        ch_downsampled = None
        
        chunk_lenght = offset*2
            
        p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

        p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)


        p_ch_avg = np.mean(p_ch, axis =0)
        p_ch_sem = stats.sem(p_ch, axis = 0)




        ax = f0.add_subplot(11, 11, 1+ch, frameon=False)
        
        #plt.figure()
        plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
        plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)

        
        plt.plot(f_base, p_base, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
        #plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                         #alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')
       
        plt.ylim(0,300000)
        plt.xticks(fontsize=4, rotation=0)
        plt.yticks(fontsize=4, rotation=0)
        #plt.title('ch_'+ str(channel))
        #plt.legend(loc='best') 
         
            
    except Exception:
        continue 
       

 
f0.subplots_adjust(wspace=.02, hspace=.02)


###############################################################################################

downsampled_ball= downsampled_ball[:-1]

#previous event as baseline 
freq = 30000
offset = 3000
num_raw_channels = 128


data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
down_sample_lenght = num_samples/30


baseline_event_idx = downsampled_end[1:]

#remove the first early trials
downsampled_event_idx = downsampled_ball[1:]

f0 =plt.figure(figsize=(20,20))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
 
probe_map_flatten = ephys.probe_map.flatten()
for ch, channel in enumerate(probe_map_flatten):
    try:
                
        #data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        ch_downsampled=ch_raw_uV
                
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        


        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        baseline_chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)


   
        baseline_chunk_around_event = np.zeros((len(baseline_event_idx),offset*2))


        for b, base in enumerate(baseline_event_idx):
   
            baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            print(b)
            

            
            
        ch_downsampled = None
        
        chunk_lenght = offset*2
            
        p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 60, bandwidth = 2.5, n_jobs = 8)

        p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 60, bandwidth = 2.5, n_jobs = 8)


        p_ch_avg = np.mean(p_ch, axis =0)
        p_ch_sem = stats.sem(p_ch, axis = 0)

 
        p_base_avg = np.mean(p_base, axis =0)
        p_base_sem = stats.sem(p_base, axis=0)



        ax = f0.add_subplot(11, 11, 1+ch, frameon=False)
        
        #plt.figure()
        plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'event', linewidth= 1)
        plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)

        'plt.figure()
        plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)   
        plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                         alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')
       
        plt.ylim(0,300000)
        plt.xticks(fontsize=4, rotation=0)
        plt.yticks(fontsize=4, rotation=0)
        #plt.title('ch_'+ str(channel))
   
            
    except Exception:
        continue 



#f0.legend(loc='best')  
f0.subplots_adjust(wspace=.02, hspace=.02)

bad_trials = []

for p, power in enumerate(p_base):
    max_value = max(power)
    bad_trials.append(max_value)
        

bad_t = []       
for b,bad in enumerate(bad_trials):
    if bad >=10000000.0:
        bad_t.append(b)
        



for t,trial in enumerate(p_base_final):
    plt.figure()
    plt.plot(f_base,trial)
    #plt.plot(f_ch, p_ch[t])
    #plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         #alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)
    

test = np.delete(test,15,0)
test = p_base_final




        p_base_avg_t = np.mean(test, axis =0)
        p_base_sem_t = stats.sem(test, axis=0)
        plt.figure()
        plt.plot(f_base, p_base_avg_t, color = '#1E90FF',alpha=0.3, label = 'event', linewidth= 1)
        plt.fill_between(f_base, p_base_avg_t-p_base_sem_t, p_base_avg_t+p_base_sem_t,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')








min_freq = 10
max_freq = 30

frequency_finder = [index for index,value in enumerate(f_ch) if value >= min_freq and value <= max_freq]

min_range = min(frequency_finder)
max_range = max(frequency_finder)

power_finder_in_range = p_ch[min_range:max_range]

avg_power = np.mean(power_finder_in_range)






























####test####################################################################################














p, f = time_frequency.psd_array_multitaper(ch_lowpass[15000:30000], sfreq= 30000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#
#plt.figure()
#plt.plot(f,p)
#
#
#
#pd, fd = time_frequency.psd_array_multitaper(data_downsampled[500:1000], sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#plt.figure()
#plt.plot(fd,pd)











#create a list if tuples (end, lenght) for each trials excluding the first one

#count = 0
#trial_end_and_lenght = []
#
#for idx in np.arange(len(end_samples)):
#    
#    if idx == 0:
#        count =+ 1
#    else:          
#        start_sample = end_samples[idx-1]
#        num_samples = samples_lenght_end_to_end[idx]
#        trial_end_and_lenght.append((start_sample, num_samples))
#
#



















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
test =  stats.sem(p_ch, axis = 0)
test_b = stats.sem(p_base)

std = p_ch_std
std_base = p_base_std

plt.plot(x, y, 'k-')
plt.fill_between(x, y-error, y+error)
plt.show()

plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)    
plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
    alpha=0.5, edgecolor='#1E90FF', facecolor='#00BFFF')

plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
    alpha=0.5, edgecolor='#228B22', facecolor='#32CD32')
    
plt.legend()   
 




   
plt.figure()
plt.plot(f_ch,p_ch_avg[:])
plt.title('ch_'+str(channel))


sns.set()



p_avg_base = np.mean(p_base, axis =0)
f_avg_base  = np.mean(f_base)


sns.set()

#plt.figure()
plt.plot(f_base,p_base_avg[:],'r')
plt.title('baseline_ch_'+str(channel))


           
            
            
            
            
    except Exception:
        continue 


sns.set()


# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')

    
    
    

#num_channels = 128
#data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
#num_samples = int(int(len(data))/num_channels)
#freq = 30000
#recording_time_sec = num_samples/freq
#recording_time_min = recording_time_sec/60
#reshaped_data = np.reshape(data,(num_samples,128))
##to have 128 rows
#reshaped_data_T= reshaped_data.T
#data = None
#
#
##signal_reshaped = ephys.apply_probe_map_to_amplifier(reshaped_data_T)
##
## Extract data chunk for single channel
#channel = 4
#
#channel_data = reshaped_data_T[channel,:]
#reshaped_data_T = None


#ch_mean = np.mean(channel_data, axis=0)

#ch_std = np.std(channel_data, axis=0)

#channel_data_Z = channel_data - ch_mean



# Z-score each channel



#raw_Z = np.zeros(raw_uV.shape)
#for ch in range(128):
#    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch]) / raw_std[ch]

# Specify channels to exclude

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
channel_data = None





data_lowpass = butter_filter_lowpass(channel_data_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
#channel_data_highpass = butter_filter(channel_data_uV, 500, 5000, fs=30000, order=3, btype='bandpass')
plt.figure()
plt.plot(data_lowpass[30000:45000])



data_downsampled = data_lowpass[::30]

plt.figure()
plt.plot(data_downsampled[1000:1500])


#
##working test mne fx for multitaper 
#        
#   
#
p, f = time_frequency.psd_array_multitaper(ch_lowpass[15000:30000], sfreq= 30000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#
#plt.figure()
#plt.plot(f,p)
#
#
#
#pd, fd = time_frequency.psd_array_multitaper(data_downsampled[500:1000], sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#plt.figure()
#plt.plot(fd,pd)


offset = 3000

downsampled_touch = np.uint32(np.array(touching_light)/30)

chunk_around_event = np.zeros((len(downsampled_touch),offset*2))

for e, event in enumerate(downsampled_touch):
    try:  
        chunk_around_event[e,:] = data_downsampled[event-offset : event+offset]
        print(e)
    except Exception:
        continue



baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))


for b, base in enumerate(baseline_idx):
    try:  
        baseline_chunk_around_event[b,:] = data_downsampled[base-offset : base+offset]
        print(b)
    except Exception:
        continue 










chunk_lenght = offset*2

p_test, f_test = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)




plt.figure()
plt.plot(f_ch,p_ch[40,:],label='1')
plt.label()

#plt.figure()
plt.plot(f_base,p_base[40,:],'r')





p_avg = np.mean(p_test, axis =0)
f_avg  = np.mean(f_test)



plt.figure()
plt.plot(f_test,p_avg[:])
plt.title('ch_'+str(channel))




p_avg_base = np.mean(p_base, axis =0)
f_avg_base  = np.mean(f_base)



#plt.figure()
plt.plot(f_base,p_avg_base[:],'r')
plt.title('baseline_ch_'+str(channel))
























































#samples_fft = np.fft.rfft(chunk_around_event)
#frequencies = np.abs(samples_fft)
freq_mean = np.mean(frequencies, axis=0)
plt.plot(freq_mean[:100])




#f, t, Sxx = signal.spectrogram(chunk_around_event, 1000, nperseg=1000, nfft=1000, noverlap=500)

### lowpass filter LFP

lowcut = 250

lowpass_data = np.zeros((len(probe_Z),num_samples))
lowpass_downsampled = [[] for _ in range(len(probe_Z))]  

for channel in np.arange(len(probe_Z)):
    try:  
        channel_data = probe_Z[channel,:]
        lowpass_cleaned = ephys.butter_filter_lowpass(channel_data,lowcut, fs=30000, order=3, btype='lowpass')
        downsampled_ch = lowpass_cleaned[::30]
        lowpass_data[channel,:] = lowpass_cleaned
        lowpass_downsampled[channel] = downsampled_ch
        print(channel)
        
    except Exception:




        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 f, t, Zxx = signal.stft(lowpass_cleaned, fs, nperseg=20000)

plt.pcolormesh(t, f, np.abs(Zxx))


# Downsample each channel
num_ds_samples = np.int(np.floor(num_samples / 30))
downsampled = np.zeros((128, num_ds_samples))
for ch in range(128):
    raw_ch = raw[ch,:]
    lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
    downsampled_ch = lowpass_ch[::30]
    downsampled[ch, :] = downsampled_ch[:num_ds_samples]




lowpass_data[22,:]

# Store downsampled data in a binary file







# Report
ch = 21
raw_ch = raw[ch,:]
lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
downsampled_ch = downsampled[ch, :]
plt.figure()
plt.plot(raw_ch, 'r')
plt.plot(lowpass_ch, 'g')
plt.plot(np.arange(num_ds_samples) * 30, downsampled_ch, 'b')
plt.show()

# LORY (spectral analysis, LFP, etc.)

#FIN    
        
#2 and 65 opposite phase        
        
        
plt.plot(lowpass_data[100,:150000],alpha = 0.4)




##### downsampling from 30kHz to 1kHz


# Spectrogram test
plt.figure()
shank = 4
for depth in range(11):
    plt.subplot(11,2,depth*2 + 1)
    probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
    fs = 30000
    ch = (depth * 11) + shank
    f, t, Sxx = signal.spectrogram(probe_Z[ch,:], fs, nperseg=30000, nfft=30000, noverlap=27000)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim([0, 30])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(11,2,depth*2 + 2)
    plt.plot(probe_Z[ch,:])
plt.show()

#FIN