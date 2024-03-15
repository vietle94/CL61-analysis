import skimage as ski
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import requests
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import netCDF4
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, median_filter
import pywt
from sklearn.mixture import GaussianMixture
myFmt = mdates.DateFormatter('%H:%M')
import glob

# %%
file_path = 'G:\CloudnetData\Kenttarova\CL61/Raw/'
files = glob.glob(file_path + '*.nc')

# %%
df = xr.open_dataset([x for x in files if '20230101_12' in x][0])
df = df.swap_dims({'profile': 'time'})
co = df['p_pol']/(df['range']**2)
cross = df['x_pol']/(df['range']**2)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True,
                       figsize=(16, 7))
ax[0].plot(co.isel(time=0), co.range, '.')
ax[0].set_xlim([-1e-13, 1e-13])
ax[1].plot(cross.isel(time=0), cross.range, '.')

for ax_ in ax.flatten():
    ax_.grid()