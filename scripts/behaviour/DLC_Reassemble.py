# -*- coding: utf-8 -*-
"""
Reassemble tracking 

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
from scipy import interpolate


# Specify hardrive path
#hardrive_path = r'F:/'
hardrive_path = r'/home/kampff/'
crop_size = 640

# DLC tracking file

#after removing first row
#row 1,0 = bodyparts 
#'Nose' = 0,1 = x
# 0,2  = y
# 2,3 likelihood nose

# Specify track data paths
centroid_tracking_path = hardrive_path + r'LC/tracking/centroid_tracking.csv'
dlc_tracking_path = hardrive_path +r'LC/tracking/dlc_tracking.csv'
video_path = hardrive_path +r'LC/tracking/video.avi'

# Load Centroid tracking
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)

# Load DLC tracking
dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)

subset = np.arange(10000, dtype=np.int) + 10000

# Compensate for crop window shift in DLC coordinates
centroid_x = centroid_tracking[:, 0] 
centroid_y = centroid_tracking[:, 1] 
dlc_x = dlc_tracking[:, 1] 
dlc_y = dlc_tracking[:, 2] 
dlc_centered_x = dlc_x - (crop_size / 2)
dlc_centered_y = dlc_y - (crop_size / 2) 
x = centroid_x + dlc_centered_x
y = centroid_y + dlc_centered_y

# Quick plot
plt.figure()
plt.plot(x[subset], y[subset], 'b.')
plt.show()

# Open named window
cv2.namedWindow("Video")

# Overlay tracked points on video frame
video = cv2.VideoCapture(video_path)
for f in subset:
    video.set(cv2.CAP_PROP_POS_FRAMES, f)
    success, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    x_int = np.int(x[f])
    y_int = np.int(y[f])
    cv2.circle(rgb, (x_int, y_int), 5, (0, 255, 0))
    cv2.imshow("Video", rgb)
    cv2.waitKey(1)

# Close named window
cv2.destroyAllWindows()


x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])


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


