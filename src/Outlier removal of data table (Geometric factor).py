#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:23:21 2024

@author: hector.ortiz
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate

#%%

#fname_csv = 'r_gf_8000-CircularV1.3-1340um-10720px.png.csv'
#fname_csv = 'r_gf_4000-CircularV1.3-1340um-5360px.png.csv'
#fname_csv = 'r_gf_4000-CircularV1.6-1340um-5360px.png.csv'
#fname_csv = 'r_gf_8000-CircularV1.6-1340um-10720px.png.csv'
fname_csv = 'r_gf_3124-Annular-W10-G02-2000um.png.csv'
data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
r_d = data_table[:,0]
gf_d = data_table[:,1]
print(data_table.shape)

#%% Outlier removal (I will check the gradient after appllied a filter)
#b, a = scipy.signal.iirfilter(6, Wn=0.5, fs=15, btype="low", ftype="butter") # Cutt off 2.5 Hz
N  = 2    # Filter order
Wn = 0.03 # Cutoff frequency
b, a = signal.butter(N, Wn, output='ba')
filt = scipy.signal.filtfilt(b, a, gf_d)

y = signal.detrend(np.gradient(filt,r_d))

plt.plot(filt,'.')
plt.plot(gf_d, '-')
plt.xlim(0,300)
plt.ylim(649000,750000)
#%%
plt.plot(filt-gf_d)
#%%
plt.plot(gf_d,'.')
plt.show()
plt.plot(y)
plt.plot(signal.detrend(np.gradient(gf_d,r_d)))
#plt.plot(np.gradient(gf_d,r_d))
plt.show()
z = np.abs(stats.zscore(y))
idx = np.where(y > 4)
# New array

#%%
max_grad = 200
for i in range(1):
    r_d2 = r_d[np.where(np.abs(y) < max_grad)]
    gf_d2 = filt[np.where(np.abs(y) < max_grad)]
    print(np.where(np.abs(y) > max_grad ))
    y = signal.detrend(np.gradient(gf_d2,r_d2))
    z = np.abs(stats.zscore(y))

#%%

#f = interp1d(r_d,gf_d, bounds_error = False)
plt.plot(y,'-')
plt.show()
#plt.plot(np.gradient(gf_d,r_d),'.', label='Orig')
#plt.plot(np.gradient(gf_d2, r_d2),'.')
plt.plot(r_d2,gf_d2,'.')
plt.plot(r_d,gf_d,'-')
#plt.plot(np.gradient(gf_d2, r_d2),'.')
#plt.plot(np.gradient(filt, r_d2),'.')
plt.legend()
#plt.plot(np.gradient(gf_d2))
#%%
data = np.column_stack((r_d2, gf_d2))
#data = np.unique(data,axis=0)

#%%
#fname_csv = 'r_gf_' +fname+'.csv'
print(fname_csv)
with open(fname_csv,'wb') as f:
    np.savetxt(f,data,delimiter =",")
    
#%%
fname_csv = 'r_gf_3125-CircularV1.4-3200.png.csv'
data_table = np.genfromtxt(fname_csv, delimiter=',', dtype=None)
r_d = data_table[:, 0]
gf_d = data_table[:, 1]

# Number of additional points you want between each pair of original points
n_interpolation = 10

# Create the interpolation function
interp_func = interpolate.interp1d(r_d, gf_d, kind='linear')

# Generate new r_d values between the original ones
r_d_interpolated = np.linspace(r_d.min(), r_d.max(), len(r_d) * n_interpolation)

# Interpolate gf_d values
gf_d_interpolated = interp_func(r_d_interpolated)

# Combine the new data
data_table_interpolated = np.column_stack((r_d_interpolated, gf_d_interpolated))

print(data_table_interpolated.shape)    

#%%
data = np.column_stack((r_d_interpolated, gf_d_interpolated))

print(fname_csv)
with open(fname_csv,'wb') as f:
    np.savetxt(f,data,delimiter =",")
    
    
    
