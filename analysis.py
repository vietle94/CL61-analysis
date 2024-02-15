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

# %%
df = xr.open_dataset('20230417_kenttarova_cl61d_clu-generated-daily.nc')
df = df.swap_dims({'profile': 'time'})

# %%
df = df.isel(range=slice(1, None))
x = df['p_pol']/(df['range']**2)
xx = df['x_pol']/(df['range']**2)

# %%
x_profile = x.isel(time=slice(0, 12))

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(df['time'].values, df['range'],
                  x.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax)

# %%

fig, ax = plt.subplots()
ax.hist(x_profile.values.flatten(), bins=np.linspace(-1e-12, 2e-12, 100))
ax.grid()

# %%
temp = x_profile.values.flatten()
temp = temp[temp < 2e-13]
np.std(temp)

# %%
x_profile.values.reshape(-1, -1, 20)

