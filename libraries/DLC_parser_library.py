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


#DLC tracking file

#after removing first row
#row 1,0 = bodyparts 
#'Nose' = 0,1 = x
# 0,2  = y
# 2,3 likelihood nose



dlc_vieo_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/10-22T15_37_43_CROPPED.avi'
dlc_tracking_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/BEST_croppedDeepCut_resnet50_Cuttle-ShuttleOct14shuffle1_250000.csv'

dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)


x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])



cleaned_x = [x for x in x_nan if str(x) != 'nan']
cleaned_y = [x for x in y_nan if str(x) != 'nan']

#cuttlefish
hist2d(cleaned_x, cleaned_y, bins=250, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)



#rat



dlc_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_07-15_42/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'

dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)


x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])



cleaned_x = [x for x in x_nan if str(x) != 'nan']
cleaned_y = [x for x in y_nan if str(x) != 'nan']



centroid_tracking_path = hardrive_path + 'Videogame_Assay/AK_33.2/2018_04_22-14_53/crop.csv'
dlc_tracking_path = hardrive_path +'Videogame_Assay/AK_33.2/2018_04_22-14_53/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'


# Load Centroid tracking
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)

# Load DLC tracking
dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)



# Compensate for crop window shift in DLC coordinates
crop_size = 640

centroid_x = centroid_tracking[:, 0] 
centroid_y = centroid_tracking[:, 1] 



dlc_x = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
dlc_y =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])




dlc_centered_x = dlc_x - (crop_size / 2)
dlc_centered_y = dlc_y - (crop_size / 2) 
x_nose = centroid_x + dlc_centered_x
y_nose = centroid_y + dlc_centered_y




cleaned_x = [x for x in x_nose if str(x) != 'nan']
cleaned_y = [x for x in y_nose if str(x) != 'nan']

f0=plt.figure()
hist2d(cleaned_x, cleaned_y, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)


norm = matplotlib.colors.Normalize(vmin=0, vmax=len(cleaned_x), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
plt.plot(cleaned_x,cleaned_y,color=mapper,alpha=0.4)


centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]

centroid_x = centroid_tracking_wo_nan[:, 0] 
centroid_y = centroid_tracking_wo_nan[:, 1] 
f1=plt.figure()
hist2d(centroid_x, centroid_y, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)

plt.plot()

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(cleaned_x))


# interpolation methods


def linearly_interpolate_nans(y):
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]

    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y




#x_zeros = np.where(dlc_tracking[:,3]<= 0.99, 0, dlc_tracking[:,1])
#y_zeros =  np.where(dlc_tracking[:,3]<= 0.99, 0, dlc_tracking[:,2])



x_nan_int_1 = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan_int_1=  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])


x_nan_fx_inter = linearly_interpolate_nans(x_nan_int_1)
y_nan_fx_inter = linearly_interpolate_nans(y_nan_int_1)




#2nd way to interpolate



x_nan_int_2 = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan_int_2=  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])




ok = ~np.isnan(x_nan_int_2)
xp = ok.ravel().nonzero()[0]
fp = x_nan[~np.isnan(x_nan_int_2)]
x  = np.isnan(x_nan_int_2).ravel().nonzero()[0]
x_nan_int_2[np.isnan(x_nan_int_2)] = np.interp(x, xp, fp)

ok2 = ~np.isnan(y_nan_int_2)
xp2 = ok2.ravel().nonzero()[0]
fp2 = y_nan[~np.isnan(y_nan_int_2)]
x2  = np.isnan(y_nan_int_2).ravel().nonzero()[0] 
y_nan_int_2[np.isnan(y_nan_int_2)] = np.interp(x2, xp2, fp2)








#
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


