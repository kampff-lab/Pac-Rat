# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:46 2015

@author: lorenza
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance

def furthest_estremes(extremes_list_A,extremes_list_B):
    count=0
    shape=len(extremes_list_A)
    nose=np.zeros((shape,2),dtype=float)
    for i,e in zip(extremes_list_A,extremes_list_B):
        if any(np.isnan(i)) or any(np.isnan(e)):
            nose[count,:]=np.nan
        else:
            dist1 = distance.euclidean(i[:2],e[:2]) + distance.euclidean(i[:2],e[2:])
            dist2 = distance.euclidean(i[2:],e[:2]) + distance.euclidean(i[2:],e[2:])
            if dist1> dist2:
                furthest_extreme = i[:2]
            else:
                furthest_extreme = i[2:]
            nose[count,:]=furthest_extreme
        count += 1
    return nose
    

def find_trajectories_around_light(nose,closest_start,closest_end):
    count=0
    shape=len(closest_start)
    trajectory=np.zeros((shape,330,2),dtype=float)
    for i,j in zip(closest_start,closest_end):
        trajectory[count,:,:]=nose[i-200:j]
        count += 1
    return trajectory
    
def find_trajectories_around_light(nose,closest_start,closest_end):
    count=0
    trajectory=[]
    for i,j in zip(closest_start,closest_end):
        trajectory.append(nose[i-200:j])
        count += 1
    return trajectory    
#with list works    
# plt.plot(trajectories[79][:,0],trajectories[79][:,1])   


def find_trajectories_around_light_array(nose,closest_start):
    count=0
    shape=len(closest_start)
    trajectory=np.zeros((shape,400,2),dtype=float)
    for i in closest_start:
        trajectory[count,:,:]=nose[i-200:i+200]
        count += 1
    diff_x=np.diff(trajectory[:,:,0])
    diff_y=np.diff(trajectory[:,:,1])    
    diff_x_square=diff_x**2
    diff_y_square=diff_y**2
    speed=np.sqrt(diff_x_square+diff_y_square)
    mean_speed=np.average(speed,0)
    mean_speed_no_nan=np.nanmean(speed,axis=0)
    plt.figure(1)
    plt.plot(diff_x.T)
    plt.figure(2)
    plt.plot(diff_y.T)
    plt.figure(3)
    plt.plot(trajectory[:,:,0].T,trajectory[:,:,1].T)
    plt.figure(4)
    plt.plot(speed)
    plt.figure(5)
    plt.plot(mean_speed) 
    plt.figure(6)
    plt.plot(mean_speed_no_nan)
    return diff_x,diff_y,trajectory,speed, mean_speed, mean_speed_no_nan

plt.xlim(0,400)
