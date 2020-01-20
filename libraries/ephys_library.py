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
    
    return raw_uV

# Low pass single channel raw ephys (in uV)

# High pass single channel raw ephys (in uV)

# Add filters...


#FIN