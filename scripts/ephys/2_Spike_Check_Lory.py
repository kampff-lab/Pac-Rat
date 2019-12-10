# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import highpass

# Load and pre-process Data
# -----------------------------------------------------------------------------

# Probe from superficial to deep electrode, left side is shank 11 (far back)
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


# Load Data as uint16 from binary file, use memory mapping (i.e. do not load into RAM)
#   - use read-only mode "r+" to prevent overwriting the original file
filename ='F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/Amplifier.bin'
num_channels = 128
data = np.memmap(filename, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data = None
reshaped_data = None


# Extract data chunk
minutes = 30
seconds = minutes*60
ten_min_samples = seconds*freq
centre = int(num_samples/2)
interval = int(ten_min_samples/2)
data_thirty_min_chunk = reshaped_data_T[:,centre-interval:centre+interval]
reshaped_data_T = None

#?????????????????

#data_thirty_min_chunk=reshaped_data_T



# Select one channel
channel = 28
channel_data = data_thirty_min_chunk[channel, :]
channel_data_float = channel_data.astype(np.float32)

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data_float - 32768) * 0.195

# FILTERS (one ch at the time)
channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
#data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')


#lowcut = 500
#highvut = 2000
#channel_data_bandpass =  butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order=3, btype='bandstop')

#plt.plot(channel_data_bandpass[2000000:3100000])
plt.plot(channel_data_highpass[2000000:3100000])
#plt.plot(data_zero_mean[55][100000:105000])

# Find spikes
spike_times = []
spike_threshold_high = -90
spike_threshold_low = -75
spiking = False

if(spike_threshold_high > 0):
    for i, v in enumerate(channel_data_highpass):
        if(not spiking):
            if(v > spike_threshold_high):
                spiking = True
                spike_times.append(i)
        else:
            if(v < spike_threshold_low):
                spiking = False
else:
   for i, v in enumerate(channel_data_highpass):
        if(not spiking):
            if(v < spike_threshold_high):
                spiking = True
                spike_times.append(i)
        else:
            if(v > spike_threshold_low):
                spiking = False       

# Remove too early and too late spikes
spike_times = np.array(spike_times)
spike_times = spike_times[(spike_times > 100) * (spike_times < (len(channel_data_highpass)-200))]

# Plot all spikes
spikes = np.zeros((len(spike_times), 300))
for i, s in enumerate(spike_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.figure()
plt.plot(spikes[range(0,len(spike_times), 2),:].T, '-', Color=[0,0,0,.002])
plt.ylim(-300, +200)
#sns.plt.xlim(0, None)
avg_spike = np.mean(spikes, axis=0)
plt.plot(avg_spike, '-', Color=[1,1,1,.5])




# FIN                                
data = channel_data_highpass

def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):
    
    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) *tf

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0] 
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window*2])
    for i in pos:
        if i < data.shape[0] - (spike_window+1):
            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i:i+spike_window*2]
            
            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) +i
                
                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

                # Append data
                spike_samp = np.append(spike_samp, tmp_samp)
                wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window*2), axis=0)
    
    # Remove duplicates
    ind = np.where(np.diff(spike_samp) > 1)[0]
    spike_samp = spike_samp[ind]
    wave_form = wave_form[ind]
    
    return spike_samp, wave_form    



spike_samp, wave_form = get_spikes(data, spike_window=50, tf=8, offset=20)

np.random.seed(10)
fig, ax = plt.subplots(figsize=(15, 5))

for i in range(100):
    spike = np.random.randint(0, wave_form.shape[0])
    ax.plot(wave_form[spike, :])

ax.set_xlim([0, 90])
ax.set_xlabel('# sample', fontsize=20)
ax.set_ylabel('amplitude [uV]', fontsize=20)
ax.set_title('spike waveforms', fontsize=23)
plt.show()