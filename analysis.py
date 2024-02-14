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
myFmt = mdates.DateFormatter('%H:%M')

# %%
df = xr.open_dataset('20230101_kenttarova_cl61d_clu-generated-daily.nc')
df = df.swap_dims({'profile': 'time'})

# %%
df = df.isel(range=slice(1, None))
x = df['p_pol']/(df['range']**2)
xx = df['x_pol']/(df['range']**2)

# %% Remove inf values
# x = x.isel(range=slice(1, None))
# xx = xx.isel(range=slice(1, None))

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(df['time'].values, df['range'],
                  x.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax)


# %%
f = np.fft.fft2(x)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# %%
fig, ax = plt.subplots()
ax.imshow(magnitude_spectrum.T)

# %%


def estimate_noise(I):

    H, W = I.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma


# %%
estimate_noise(xx)
estimate_noise(x)

# %%
fig, ax = plt.subplots()
ax.hist(x.values.flatten(), bins=np.linspace(-1e-13, 2e-13, 100))
ax.grid()

# %%
temp = x.values.flatten()
temp = temp[temp < 2e-13]

# %%
x
np.std(temp)
np.std(x.values)

np.std(x.values - gaussian_filter(x.values, 0.2))

# %%
temp = x.copy()
temp = temp.where(temp > (1e-13))

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
p = ax[0].pcolormesh(df['time'].values, df['range'],
                     x.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax[0])

p = ax[1].pcolormesh(df['time'].values, df['range'],
                     gaussian_filter(x.values, 2.693192770818292e-12).T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
# median_filter(x.values, 10).T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax[1])

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
p = ax[0].pcolormesh(df['time'].values, df['range'],
                     x.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax[0])

p = ax[1].pcolormesh(df['time'].values, df['range'],
                     # gaussian_filter(x.values, 50).T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
                     temp.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
fig.colorbar(p, ax=ax[1])

# %%
a = np.arange(50, step=2).reshape((5, 5))

gaussian_filter(a, 0.5)
x
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x.isel(time=4), x.range, '.')
ax.set_xlim([-0.25e-12, 0.25e-12])

ski.restoration.estimate_sigma(x)
dir(ski)
