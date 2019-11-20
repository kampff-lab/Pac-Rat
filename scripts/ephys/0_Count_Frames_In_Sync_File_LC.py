# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import os



hardrive_path = r'F:/' 

rat_ID = r'/AK_41.1/'

rat_folder = hardrive_path + rat_ID
sessions = os.listdir(rat_folder)


for session in sessions[:]:
    try:
        # Load Video File Details
        session_path = rat_folder + session + r'/'
        video_csv_file_path =session_path + 'video.csv'    
        video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)
        video_num_frames = video_counter[-1] - video_counter[0] + 1
        
        # Load Sync File Details                       
        sync_file_path = session_path + 'sync.bin'
        sync_dtype = np.uint8
        fs = 30000
        frame_rate = 120.0
        sync_data = np.fromfile(sync_file_path,sync_dtype)
        frame_sync_data = np.int8(sync_data & 1)
        frame_transitions = np.diff(frame_sync_data)
        frame_starts = np.where(frame_transitions == 1)[0]
        sync_num_frames = len(frame_starts)
    except Exception:
        continue
    if video_num_frames != sync_num_frames:
        print(video_num_frames)
        print(sync_num_frames)
        print('DISASTER')
        print (session)
    else:
        print('OK')
        print (session)



trial_end='E:/AK_33.2_test/2018_04_29-15_43/events/TrialEnd.csv'        
        
def idx_Event(trial_end):
    RewardOutcome_file=np.genfromtxt(trial_end,usecols=[1], dtype= str)
    RewardOutcome_idx=[]
    count=0
    for i in RewardOutcome_file:
        count += 1
        if i =='Missed':
            RewardOutcome_idx.append(count-1)
    reward=np.array(RewardOutcome_idx)
    return RewardOutcome_idx




sync_file= 'E:/AK_33.2_test/2018_04_29-15_43/Sync.bin'     
sync_dtype = np.uint8
fs = 30000
sync_data = np.fromfile(sync_file,sync_dtype)
tone_sync_data = np.int8(sync_data & 8)
tone_transitions = np.diff(tone_sync_data)
tone_starts = np.where(tone_transitions == 8)[0]
tone_starts=tone_starts.tolist()
del tone_starts[2]
list_availability_tone = list(filter(lambda varX: varX % 2 == 0,tone_starts))
list_reward_tone = list(filter(lambda varX: varX % 2 == 1,tone_starts))



video_avi_file_path='E:/AK_33.2_test/2018_04_29-15_43/Video.avi'
target_dir= r'E:\AK_33.2_test\2018_04_29-15_43\tone_frames'

def tone_frame(target_dir,video_avi_file_path,nearest):
    video=cv2.VideoCapture(video_avi_file_path)
    success, image=video.read()
    success=True
    count = 0
    for i in nearest:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image



nearest=closest_value_in_array(sample_for_each_video_frame,list_availability_tone)


outpath = 'E:/AK_33.2_test/2018_04_29-15_43/test_folder/'
window_size = 90000
offset_list=list(range(0,30000 * 60 * 30, 30000))


#random_samples = random.sample(range(0,ten_min_samples), num_corr_samples)
#corr_matrix2 = np.zeros((121,121),dtype=float)
#norm_corr_matrix2 = np.zeros((121,121),dtype = float)
for i in range(len(offset_list)):
    corr_matrix2 = np.zeros((121,121),dtype=float)
    data_zero_mean_remapped = GET_data_zero_mean_remapped_window(filename, offset_list[i], window_size)
    
    for e in range(0, window_size, 30):
        outer_product = np.outer(data_zero_mean_remapped[:, e], data_zero_mean_remapped[:, e])
        corr_matrix2 = corr_matrix2 + outer_product
    corr_matrix2 = corr_matrix2 / window_size / 30   
    
    norm_corr_matrix2 = np.zeros((121,121),dtype = float)
    for r in range(121):
        for c in range(121):
            normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
            norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
    plt.figure()
    ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right"))
    plt.savefig(outpath +"correlation{filecount}.png".format(filecount=i))
    plt.close('all')
    print("Current offset: " + str(i))








num_samples=90000

filename = 'E:/AK_33.2_test/2018_04_29-15_43/Amplifier.bin'

def GET_data_zero_mean_remapped_window(filename, offset, num_samples):
    
    num_channels = 128
    bytes_per_sample = 2
    offset_position = offset * num_channels * bytes_per_sample
    
    # Open file and jump to offset position
    f = open(filename, "rb")
    f.seek(offset_position, os.SEEK_SET)

    # Load data from this file position
    data = np.fromfile(f, dtype=np.uint16, count=(num_channels * num_samples))
    f.close()
    
    # Reshape data
    reshaped_data = np.reshape(data,(num_samples,128)).T
    #to have 128 rows
    
    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    data_uV = (reshaped_data.astype(np.float32) - 32768) * 0.195
    
    # Subtract channel mean from each channel
    mean_per_channel_data_uV = np.mean(data_uV,axis=1,keepdims=True)
    data_zero_mean = data_uV - mean_per_channel_data_uV
    
    # Extract (remapped) 121 probe channels
    probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))
    data_zero_mean_remapped = data_zero_mean[probe_map_as_vector,:]
    
    return data_zero_mean_remapped



















