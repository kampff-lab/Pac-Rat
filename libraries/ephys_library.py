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
raw_sample_rate = 30000

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

# Clean raw binary data (Amplifier.bin) and save cleaned file (Amplifier_cleaned.bin)
def measure_raw_amplifier_stats(filename):

    # Specifiy paths
    in_path = filename
    out_path = in_path[:-4] + '_stats.csv'

    # Measure file size (and number of samples)
    statinfo = os.stat(in_path)
    num_samples = np.int(statinfo.st_size / bytes_per_sample)

    # Determine number of full chunks (1 minute)
    chunk_size = (num_raw_channels * raw_sample_rate * 60)
    num_chunks = np.int(np.floor(num_samples / chunk_size))

    # Create an array of chunk sizes (edit final chunk size)
    chunk_sizes = np.ones(num_chunks, dtype=np.int) * chunk_size

    # Open input raw amplifier file (in binary read mode)
    in_file = open(in_path, 'rb')

    # Allocate space for channel stats
    chunk_channel_means = np.zeros((num_raw_channels, num_chunks))
    chunk_channel_stds = np.zeros((num_raw_channels, num_chunks))

    # Clean all raw data and store
    for i, chunk_size in enumerate(chunk_sizes):
        
        # Load next chunk (1 second)
        data = np.fromfile(in_file, count=chunk_size, dtype=np.uint16)

        # Reshape
        data = np.reshape(data, (-1, num_raw_channels)).T

        # Measure channel stats
        chunk_channel_means[:, i] = np.mean(data, axis=1)
        chunk_channel_stds[:, i] = np.std(data, axis=1)

        # Report progress
        print("{0}: Chunk {1} of {2}".format(in_path, i, num_chunks))

    # Close input file
    in_file.close()

    # Measure channel stats
    channel_means = np.mean(chunk_channel_means, axis=1)
    channel_stds = np.mean(chunk_channel_stds, axis=1)

    # Save channel stats to CSV file
    channel_stats = np.float32(np.vstack((channel_means, channel_stds)).T)
    np.savetxt(out_path, channel_stats, delimiter=',', fmt='%.3f')

    return

# Clean raw data (Amplifier.bin) and save cleaned binary file (Amplifier_cleaned.bin)
def clean_raw_amplifier(filename, exclude_channels):

    # Specifiy paths
    in_path = filename
    out_path = in_path[:-4] + '_cleaned.bin'

    # Measure file size (and number of samples)
    statinfo = os.stat(in_path)
    num_samples = np.int(statinfo.st_size / bytes_per_sample)

    # Determine number of full chunks (1 minute) and final chunk size
    chunk_size = (num_raw_channels * raw_sample_rate * 60)
    num_full_chunks = np.int(np.floor(num_samples / chunk_size))
    final_chunk_size = num_samples % chunk_size
    num_chunks = num_full_chunks + 1

    # Create an array of chunk sizes (edit final chunk size)
    chunk_sizes = np.ones(num_chunks, dtype=np.int) * chunk_size
    chunk_sizes[-1] = final_chunk_size

    # Determine channels to exclude on each headstage
    A_exclude_channels = exclude_channels[exclude_channels < 64]
    B_exclude_channels = exclude_channels[exclude_channels >= 64]

    # Determine headstage channels
    A_channels = np.arange(64)
    B_channels = np.arange(64, 128)

    # Remove excluded channels
    A_channels = np.delete(A_channels, A_exclude_channels)
    B_channels = np.delete(B_channels, B_exclude_channels)

    # Open input raw amplifier file (in binary read mode)
    in_file = open(in_path, 'rb')

    # Open output cleaned amplifier file (in binary write mode)
    out_file = open(out_path, 'wb')

    # Clean all raw data and store
    for i, chunk_size in enumerate(chunk_sizes):
        
        # Load next chunk (1 second)
        data = np.fromfile(in_file, count=chunk_size, dtype=np.uint16)

        # Reshape
        data = np.reshape(data, (-1, num_raw_channels)).T

        # Compute mean values for each headstage relative to 0 (32767)
        A_mean = np.int16(np.mean(data[A_channels,:], axis=0) - 32767) 
        B_mean = np.int16(np.mean(data[B_channels,:], axis=0) - 32767)

        # Rereference each channel
        clean = np.zeros(data.shape, dtype=np.uint16)
        for ch in A_channels:
            raw_ch = data[ch, :]
            clean_ch = raw_ch - A_mean
            clean[ch,:] = clean_ch
        for ch in B_channels:
            raw_ch = data[ch, :]
            clean_ch = raw_ch - B_mean
            clean[ch,:] = clean_ch

        # Store cleaned binary data
        clean = clean.T
        clean.tofile(out_file)

        # Report progress
        print("{0}: Chunk {1} of {2}".format(in_path, i, num_chunks))

    # Close files
    in_file.close()
    out_file.close()

    return

# Get raw ephys clip from amplifier.bin (all channels)
def get_raw_clip_from_amplifier(filename, start_sample, num_samples):

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
    
    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    raw_uV = (reshaped_data.astype(np.float32) - 32768) * 0.195
    
    return raw_uV

# Get raw ephys clip from amplifier.bin (single channel)
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
    
    return raw_uV

# Get probe map
def get_probe_map():
    return probe_map

# Apply probe map
def apply_probe_map_to_amplifier(amp_data):
    num_samples = np.size(amp_data, 1)
    probe_data = np.zeros((121, num_samples))
    for i, ch in enumerate(flatten_probe):
        probe_data[i,:] = amp_data[ch,:]
    return probe_data







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

# Are spikes downward or upward?
def threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft):
    
    spike_start_times = []
    spike_stop_times = []
    spiking = False
    
    for i, voltage in enumerate(channel_data_highpass):
        # Look for a new spike
        if(not spiking):
            if(voltage < spike_threshold_hard):
                spiking = True
                spike_start_times.append(i)
        # Track ongoing spike            
        else:
            # Keep track of max (negative) voltage until npo longer spiking
            if(voltage > spike_threshold_soft):
                spiking = False       
                spike_stop_times.append(i)
                  
    return spike_start_times, spike_stop_times








#FIN