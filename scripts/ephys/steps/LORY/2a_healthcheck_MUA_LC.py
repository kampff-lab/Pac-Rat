# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 2a: health check analysis and plots for MUA signal (high-frequency)

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)

# Ephys Constants
num_raw_channels = 128
bytes_per_sample = 2
raw_sample_rate = 30000

# Specify session folder
session_path =  'F:/Videogame_Assay/AK_33.2/2018_04_29-15_43'
#session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Specify cleaned data path
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')




# Measure file size (and number of samples)
bytes_per_sample = 16
statinfo = os.stat(data_path)
num_samples = np.int(statinfo.st_size / bytes_per_sample)
num_samples_per_channel = np.int(num_samples / num_raw_channels)

# Memory map amplifier data
tmp = np.memmap(data_path, dtype=np.uint16, mode = 'r')
data = np.reshape(tmp,(num_samples_per_channel,128)).T
tmp = None


ch= 100
# Measure stats for each channel
for ch in range(128):
    
    # Report
    print("Starting channel {0}".format(ch))

    # Extract channel data and convert to uV (float32)
    data_ch = reshaped_data_T[ch,:]
    reshaped_data_T = None
    data_ch_uV = (data_ch.astype(np.float32) - 32768) * 0.195
    print("- converted to uV")
    data_ch = None
    
    
    # High-pass filter at 500 Hz
    highpass_ch_uV = ephys.highpass(data_ch_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    print("- highpassed")
    highpass_ch_uV = None
    data_ch_uV = None

    # Determine RMS noise after filtering
    abs_highpass_ch_uV = np.abs(highpass_ch_uV)
    sigma_n = np.median(abs_highpass_ch_uV) / 0.6745
    print("- Noise sigma measured")

# Measure RMS of each channel (in uV) and save
# Make example plots...of unfiltered and high-pass filtered data
# Should compare this with both "cleaned" and raw data

#FIN



ch = 21

start_sample = 5800000
end_sample = 5890000
samples_diff = end_sample-start_sample

data_raw = os.path.join(session_path +'/Amplifier.bin')
raw=  np.memmap(data_raw, dtype = np.uint16, mode = 'r')

num_samples = int(int(len(raw))/num_raw_channels)
reshaped_raw =  np.reshape(raw,(num_samples,128))
raw = None

raw_T = reshaped_raw.T
reshaped_raw=None

raw_ch = raw_T[ch,:]
raw_T=None

raw_ch_uV = (raw_ch.astype(np.float32) - 32768) * 0.195
raw_ch = None

plt.figure()
plt.plot(raw_ch_uV[start_sample:end_sample],alpha=.5, color='red')
raw_ch_uV = None



data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')
data = np.memmap(data_path, dtype = np.uint16, mode = 'r')

num_samples = int(int(len(data))/num_raw_channels)

#recording_time_sec = num_samples/raw_sample_rate
#recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows

data = None
reshaped_data_T= reshaped_data.T

reshaped_data = None

data_ch = reshaped_data_T[ch,:]
reshaped_data_T = None

data_ch_uV = (data_ch.astype(np.float32) - 32768) * 0.195
plt.plot(data_ch_uV[start_sample:end_sample],alpha=.4, color='b')

highpass_ch_uV = ephys.highpass(data_ch_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)

plt.plot(highpass_ch_uV[start_sample:end_sample],alpha = .5, color='green')
data_ch=None 
data_ch_uV=None
highpass_ch_uV=None


data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(down))/num_raw_channels)
reshaped_down=  np.reshape(down,(num_samples,128))
down=None 
down_T = reshaped_down.T
down_ch = down_T[ch,:] 
down_T=None
down_ch_uV = (down_ch.astype(np.float32) - 32768) * 0.195
down_ch=None 
#plt.figure()
plt.plot(np.arange(0,samples_diff,30),down_ch_uV[int(start_sample/30):int(end_sample/30)],'k',alpha=.5)
down_ch_uV=None 

plt.title('RAW = red, CLEANED = blue, HIGH = green, DOWN = black')

