# -*- coding: utf-8 -*-
"""
Pac-Rat Electrophysiology Library

@author: You
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import math
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Ephys Constants
num_raw_channels = 128
bytes_per_sample = 2

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

flatten_probe = probe_map.flatten()

# Get raw ephys clip from amplifier.bin
def get_channel_raw_clip_from_amplifier(filename, depth, shank, start_sample, num_samples):

    # Compute offset in binary file
    offset = start_sample * num_raw_channels * bytes_per_sample
    
    # Open file and jump to offset
    f = open(filename, "rb")
    f.seek(offset, os.SEEK_SET)

    # Load data from this file position
    data = np.fromfile(f, dtype=np.uint16, count=(num_raw_channels * num_samples))
    f.close()
    
    # Reshape data to have 128 rows
    reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T

    # Extract selected channel (using probe map)
    amp_channel = probe_map[depth, shank]
    raw = reshaped_data[amp_channel, :]
    
    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    raw_uV = (raw.astype(np.float32) - 32768) * 0.195
    #mean_raw_ch = np.mean(raw_uV)
    #median_raw_ch = np.median(raw_uV)
    
    return raw_uV #, mean_raw_ch, median_raw_ch

# Low pass single channel raw ephys (in uV)
def butter_filter_lowpass(data,lowcut, fs=30000, order=3, btype='lowpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype=btype)
    y = filtfilt(b, a, data)
    return y
    

# High pass single channel raw ephys (in uV)
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


# Add filters...
def butter_bandstop(data,lowcut, highcut, fs=30000, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, data)    

    freq, h = signal.freqz(b, a, fs=fs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([-25, 10])
    ax[0].grid()
    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()
    plt.show()

    return y


def butter_bandpass(data,lowcut, highcut, fs=30000, order=3, btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    y = filtfilt(b, a, data)
    return y

def iirnotch_50(data, fs=30000, quality=30):
    fs = 2000
    quality = 30

    f0 = 60.0  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2 )  # Normalized Frequency
    b, a = signal.iirnotch(w0, quality)
    y = filtfilt(b, a, data)


    freq, h = signal.freqz(b, a, fs=fs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([-25, 10])
    ax[0].grid()
    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()
    plt.show()

    return y









#FIN