# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:52:11 2017

@author: KAMPFF-LAB-BEHAVIOR
"""

import numpy as np
import pandas as pd

def read_sync(syncpath,counterpath):
    sync = np.fromfile(syncpath,dtype=np.uint8)
    syncidx = np.flatnonzero(np.diff(np.int8(sync > 0)) < 0)
    counter = pd.read_csv(counterpath,names=['counter'],
                          usecols=[1],delimiter=' ',header=None)
    drops = counter.diff() - 1
    pulsecount = len(syncidx)
    framecount = len(counter)
    dropframes = drops.sum()[0]
    if pulsecount != (framecount + dropframes):
        print("WARNING: Number of frames does not match number of sync pulses!")
    status = "Sync pulses: {0} Counted frames: {1} Drop frames: {2}"+\
             " Total frames: {3}"
    print(status.format(pulsecount,
                        framecount,
                        dropframes,
                        framecount + dropframes))
    matchedpulses = drops.counter.fillna(0).cumsum() + np.arange(len(drops))
    syncidx = syncidx[matchedpulses.astype(np.int32).values]
    syncidx = pd.DataFrame(syncidx,columns=['syncidx'])
    
    frameidx = np.zeros(sync.shape,dtype=np.int)
    frameidx[0] = -1
    frameidx[syncidx.values] = 1
    frameidx = pd.DataFrame(frameidx,columns=['frame']).cumsum()
    return syncidx,frameidx

# Set the following path to be the 'video.csv' and 'sync.bin' file of the folder to test
syncpath = 'C:/Projects/Videogame/Data/AK_14.2/2017_05_27-12_04/sync.bin'
counterpath = 'C:/Projects/Videogame/Data/AK_14.2/2017_05_27-12_04/video.csv'
syncidx, frameidx = read_sync(syncpath, counterpath)


#def read_sync(syncpath,counterpath):
#    sync = np.fromfile(syncpath,dtype=np.uint8)
#    syncidx = np.flatnonzero(np.diff(np.int8(sync > 0)) < 0)
#    counter = pd.read_csv(counterpath,names=['counter'],
#                          usecols=[1],delimiter=' ',header=None)
#    drops = counter.diff() - 1
#    pulsecount = len(syncidx)
#    framecount = len(counter)
#    dropframes = drops.sum()[0]
#    if pulsecount != (framecount + dropframes):
#        print("WARNING: Number of frames does not match number of sync pulses!")
#    status = "Sync pulses: {0} Counted frames: {1} Drop frames: {2}"+\
#             " Total frames: {3}"
#    print(status.format(pulsecount,
#                        framecount,
#                        dropframes,
#                        framecount + dropframes))
#    matchedpulses = drops.counter.fillna(0).cumsum() + np.arange(len(drops))
#    syncidx = syncidx[matchedpulses.astype(np.int32).values]
#    syncidx = pd.DataFrame(syncidx,columns=['syncidx'])
#    
#    frameidx = np.zeros(sync.shape,dtype=np.int)
#    frameidx[0] = -1
#    frameidx[syncidx.values] = 1
#    frameidx = pd.DataFrame(frameidx,columns=['frame']).cumsum()
#    return syncidx,frameidx
#
## Set the following path to be the 'video.csv' and 'sync.bin' file of the folder to test
#syncpath = 'C:/Projects/Videogame/Data/AK_14.2/2017_05_27-12_04/sync.bin'
#counterpath = 'C:/Projects/Videogame/Data/AK_14.2/2017_05_27-12_04/video.csv'
#syncidx, frameidx = read_sync(syncpath, counterpath)
#
