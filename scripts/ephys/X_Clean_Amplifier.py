# -*- coding: utf-8 -*-
"""
Clean amplifier raw binary data with commom-mean-(re)referencing for each headtsage
and re-save as Amplifier_cleaned.bin

@author: Adam Kampff
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')

# For Lory
#rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
#hardrive_path = r'F:/' 
#Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
#sessions_subset = Level_2_post
sessions_subset = [0]

# Ephys parameters
num_channels = 128
freq = 30000

for session in sessions_subset:
    try:

        # For testing...
        #session_path = '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'
        session_path = '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
        
        #session_path = os.path.join(hardrive_path, session )
        recording_path = os.path.join( session_path + '/Amplifier.bin')
        save_path = os.path.join(session_path + '/Amplifier_cleaned.bin')

        # Open amplifier file (in binary read mode)
        in_file = open(recording_path, 'rb')

        # Open output amplifier file (in binary write mode)
        out_file = open(save_path, 'wb')

        # Allocate space for cleaned sample
        cleaned = np.zeros(128, dtype=np.uint16)

        # Loop through all samples in binary input file
        count = 0
        while(count < 35600000):

            # Load next sample
            sample = np.fromfile(in_file, count=num_channels, dtype=np.uint16)

            # Check for EOF (end of file)
            if(len(sample) != 128):
                break
            else:
                count = count+1

            # Compute sample mean for each headstage
            mean_Front = np.mean(sample[:64])
            mean_Back = np.mean(sample[64:])

            # Compute offset from u16 zero (32768)
            offset_Front = np.int(mean_Front - 32768)
            offset_Back = np.int(mean_Back - 32768)

            # Subtract offset from relevant channels
            cleaned[:64] = sample[:64] - offset_Front 
            cleaned[64:] = sample[64:] - offset_Back 

            # Write to output file
            bytes_written = out_file.write(cleaned)

        # Close files
        in_file.close()
        out_file.close()

    except:
        print('Session fail')

# Test cleaned file

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

# Load cleaned binary data
data = np.memmap(save_path, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
reshaped_data_T = reshaped_data.T
data = None
reshaped_data = None

chunk_data = reshaped_data_T[:, 35500000:35600000]
#chunk_data = reshaped_data_T[:, 500000:530000]
reshaped_data_T = None

# Plot
for ch, channel in enumerate(flatten_probe):
    plt.plot((ch*1000) + np.float32(chunk_data[channel, :]))
plt.show()

# FIN