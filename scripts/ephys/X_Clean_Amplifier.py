# -*- coding: utf-8 -*-
"""
Clean amplifier raw binary data with commom-mean-(re)referencing for each headtsage
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

# Ephys parameters
num_channels = 128
freq = 30000

for session in sessions_subset:
    try:

        # For testing...
        session_path = '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'
        
        #session_path = os.path.join(hardrive_path, session )
        recording_path = os.path.join( session_path + '/Amplifier.bin')
        save_path = os.path.join(session_path + '/Amplifier_cleaned.bin')

        # Open amplifier file (in binary read mode)
        in_file = open(recording_path, 'rb')

        # Open output amplifier file (in binary write mode)
        out_file = open(save_path, 'wb')

        count = 0
        while(True):

            # Load next sample
            sample = np.fromfile(in_file, count=num_channels, dtype=np.uint16)

            # Check for EOF
            if(len(sample) != 128):
                break
            else:
                count = count+1

            # Compute sample mean for each headstage

            # Subtract from relevant channels

            # Write to output file

        # Close files
        in_file.close()
        out_file.close()

# FIN