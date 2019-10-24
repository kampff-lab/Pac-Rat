# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:34:09 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

from scipy import interpolate
from scipy.interpolate import interp1d

#DLC tracking file

#after removing first row
#row 1,0 = bodyparts 
#'Nose' = 0,1 = x
# 0,2  = y
# 2,3 likelihood nose

#
#
#dlc_vieo_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/10-22T15_37_43_CROPPED.avi'
#dlc_tracking_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/BEST_croppedDeepCut_resnet50_Cuttle-ShuttleOct14shuffle1_250000.csv'
#
#dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)
#
#
#x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
#y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])
#
#
#
#cleaned_x = [x for x in x_nan if str(x) != 'nan']
#cleaned_y = [x for x in y_nan if str(x) != 'nan']
#
##cuttlefish
#hist2d(cleaned_x, cleaned_y, bins=250, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)



#rat


#
#dlc_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_07-15_42/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'
#
#dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)
#
#
#x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
#y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])
#
#
#
#cleaned_x = [x for x in x_nan if str(x) != 'nan']
#cleaned_y = [x for x in y_nan if str(x) != 'nan']

hardrive_path = r'F:/'

crop_size = 640

centroid_tracking_path = hardrive_path + 'Videogame_Assay/AK_33.2/2018_03_26-10_54/crop.csv'
dlc_tracking_path = hardrive_path +'Videogame_Assay/AK_33.2/2018_03_26-10_54/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'


# Load Centroid tracking
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)

# Load DLC tracking
dlc_tracking = np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)

#select x and y from centroid file 
centroid_x = centroid_tracking[:, 0] 
centroid_y = centroid_tracking[:, 1] 


#fill with Nan x and yf if the likehood of dlc is less than 0.99
x_nan_nose = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan_nose =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])

x_nan_tail_base = np.where(dlc_tracking[:,12]<= 0.99, np.NaN, dlc_tracking[:,10])
y_nan_tail_base =  np.where(dlc_tracking[:,12]<= 0.99, np.NaN, dlc_tracking[:,11])




dlc_centered_x_nose = x_nan_nose - (crop_size / 2)
dlc_centered_y_nose = y_nan_nose - (crop_size / 2) 

dlc_centered_x_tail_base = x_nan_tail_base - (crop_size / 2)
dlc_centered_y_tail_base = y_nan_tail_base - (crop_size / 2) 



x_correct_nose = centroid_x + dlc_centered_x_nose
y_correct_nose = centroid_y + dlc_centered_y_nose

x_correct_tail_base = centroid_x + dlc_centered_x_tail_base
y_correct_tail_base = centroid_y + dlc_centered_y_tail_base





#remove nans
#cleaned_x = [x for x in x_nose if str(x) != 'nan']
#cleaned_y = [x for x in y_nose if str(x) != 'nan']





# interpolate 1



x_nan_int_2 = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan_int_2=  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])





ok = ~np.isnan(x_correct)
xp = ok.ravel().nonzero()[0]
fp = x_correct[~np.isnan(x_correct)]
x  = np.isnan(x_correct).ravel().nonzero()[0]
x_correct[np.isnan(x_correct)] = np.interp(x, xp, fp)

ok2 = ~np.isnan(y_correct)
xp2 = ok2.ravel().nonzero()[0]
fp2 = y_correct[~np.isnan(y_correct)]
x2  = np.isnan(y_correct).ravel().nonzero()[0] 
y_correct[np.isnan(y_correct)] = np.interp(x2, xp2, fp2)










def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data



#import numpy as np
#nan = np.nan
#
#A = np.array([1, nan, nan, 2, 2, nan, 0])
#
#ok = ~np.isnan(A)
#xp = ok.ravel().nonzero()[0]
#fp = A[~np.isnan(A)]
#x  = np.isnan(A).ravel().nonzero()[0]
#
#A[np.isnan(A)] = np.interp(x, xp, fp)
#
#print (A)
#[1.         1.33333333 1.66666667 2.         2.         1.
# 0.        ]


centroid_tracking_path = r'F:/Videogame_Assay/AK_33.2/2018_03_27-14_34/crop.csv'
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)


centroid_tracking_wo_nan= centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
x_centroid = centroid_tracking_wo_nan[:,0]
y_centroid = centroid_tracking_wo_nan[:,1]


from matplotlib.colors import PowerNorm
        
from matplotlib.colors import LogNorm        
        
plt.hist2d(x_centroid, y_centroid, bins=150, norm=LogNorm()) #norm=PowerNorm(0.3)






norm=PowerNorm(0.3)

x_centroid = centroid_tracking[:,0]
y_centroid = centroid_tracking[:,1]
plt.plot(x_centroid,y_centroid, marker ='o',color= [0,0,0,0.002])


nan_finder = np.argwhere(np.isnan(centroid_tracking))

nan_finder_x= np.argwhere(np.isnan(x_centroid))
nan_finder_y = np.argwhere(np.isnan(y_centroid))


tracking_diff=np.diff(centroid_tracking)


plt.hexbin(x_centroid,y_centroid)


plt.hist2d(x_centroid,y_centroid)

arr = np.array(data)

df = pd.DataFrame(centroid_tracking)

sns.heatmap(df, cmap='coolwarm')


