import numpy as np
import datetime as dt

filename = '/home/kampff/Dropbox/LC_THESIS/ARK/shader_vs_crop/BallPosition.csv'
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamps = split_data[:,0]
positions_strings = split_data[:,1]
for index, s in enumerate(positions_strings):
    tmp = s.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('\n', '')
    tmp = tmp.replace(' ', '')
    positions_strings[index] = tmp
positions = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
