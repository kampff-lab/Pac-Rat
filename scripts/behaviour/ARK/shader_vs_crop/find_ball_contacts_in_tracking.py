import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse


# Load timestamp strings
filename = '/home/kampff/Dropbox/LC_THESIS/ARK/shader_vs_crop/BallTracking.csv'
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamps = split_data[:,0]

# Convert to elapsed seconds
elapsed = []
start_time = parse(timestamps[0])
for ts in timestamps:
    current_time = parse(ts)
    delta_time = current_time - start_time
    elapsed.append(delta_time.total_seconds())
elapsed = np.array(elapsed)

# Find timestamp jumps
deltas = np.diff(elapsed, prepend=[0])
#plt.plot(deltas)
#plt.show()
contact_indices = np.where(deltas > 1)
contact_timestamp_strings = timestamps[contact_indices]
