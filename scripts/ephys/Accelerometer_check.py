# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs

## Load accelerometer data

# Specify path
accel_path = '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43/Accelerometer.bin'

# Load accelerometer data (7500 Hz)
accel_flat_u16 = np.fromfile(accel_path, dtype=np.uint16)
accel_channels = np.reshape(accel_flat_u16, (-1,3)).T

# Plot
plt.plot(accel_channels[:,100000:120000].T)
plt.show()

# Upsample to 30,000 Hz (4x)