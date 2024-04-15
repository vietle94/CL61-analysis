import skimage as ski
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import requests
from matplotlib.colors import LogNorm, SymLogNorm
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
df = xr.open_dataset([x for x in files if '20240303_093' in x][0])
df = df.isel(range=slice(1, None))
co = df['p_pol']/(df['range']**2)
cross = df['x_pol']/(df['range']**2)

# %%
fig, ax = plt.subplots(1, 3, sharey=True,
                       figsize=(16, 7))
ax[0].plot(co.isel(time=0), co.range, '.')
# ax[0].set_xlim([0, 1e-11])
ax[0].set_xlim([-2e-13, 1e-12])

ax[1].plot(cross.isel(time=0), cross.range, '.')
ax[1].set_xlim([-2e-13, 1e-12])

ax[2].plot(df['linear_depol_ratio'].isel(time=0), df.range, '.')
ax[2].set_xlim([-0.05, 0.05])
ax[0].set_ylim([0, 12000])
for ax_ in ax.flatten():
    ax_.grid()
    
# %%
fig, ax = plt.subplots()
ax.plot(df['overlap_function'], df['range'], '.')
ax.set_ylim([0, 500])
ax.grid()

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
p = ax[0].pcolormesh(df['time'], df['range'], df['p_pol'].T, shading='nearest',
                     norm=SymLogNorm(linthresh=1e-14, vmin=-1e-12, vmax=1e-12),
                     cmap='RdGy_r')
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(df['time'], df['range'],
              df['linear_depol_ratio'].T, shading='nearest',
              vmin=-0.02, vmax=0.02)
fig.colorbar(p, ax=ax[1])
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
# ax.set_ylim([0, 1000])

# %%
fig, ax = plt.subplots(1, 3, sharey=True,
                       figsize=(16, 7))
ax[0].plot(np.convolve(co.mean(dim='time'),
                       [1/5, 1/5, 1/5, 1/5, 1/5],
                       'same'), co.range, '.')
# ax[0].set_xlim([-2e-13, 1e-12])
ax[0].set_xlim([-3e-14, 1e-13])

ax[1].plot(cross.mean(dim='time'), cross.range, '.')
ax[1].set_xlim([-2e-13, 1e-12])

ax[2].plot(df['linear_depol_ratio'].mean(dim='time'), df.range, '.')
ax[2].set_xlim([-0.05, 0.05])
ax[0].set_ylim([0, 12000])
for ax_ in ax.flatten():
    ax_.grid()

# %%
co_mean = co.mean(dim='time')

# %%
co_mean.size
plt.hist(co_mean, bins=np.arange(-1e-14, 1e-12, 1e-14))

# %%
# for wavelet in ['bior3.1', 'coif17', 'db28']: # level 7
# for wavelet in ['sym6', 'bior3.9']: # level 9
# for wavelet in ['bior6.8']: # level 9
# for wavelet in ['bior3.1', 'bior3.9', 'bior6.8', 'coif3', 'db6',
#                 'db7', 'rbio5.5', 'rbio6.8', 'sym5', 'sym6', 'sym7', 'sym8']: # level 9
for wavelet in pywt.wavelist():
    fig, ax = plt.subplots(3, 2)
    for profile, ax_ in zip(
            [df.sel(time=pd.to_datetime('20240303 093500'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240303 091000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 001000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 040000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 200000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 080000'), method='nearest')['p_pol']/(df['range']**2)], 
            ax.flatten()):
        # profile = df.sel(time=pd.to_datetime('20240303 091000'), method='nearest')['p_pol']/(df['range']**2)
        try:
            n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
            coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
                             wavelet, level=7)
        except ValueError:
            continue
        # coeff = pywt.swt(np.pad(profile, (0, (len(profile) // 2**5 + 3) * 2**5 - len(profile)), 'constant', constant_values=(0, 0)),
        #                  wavelet, level=5)
        uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        filtered = pywt.iswt(coeff, wavelet)
        filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]
        
        ax_.plot(profile, df['range'], '.')
        ax_.plot(filtered, df['range'], '.')
        ax_.set_xlim([-1.5e-13, 1.5e-12])
    fig.suptitle(wavelet)

# %%
for wavelet in pywt.wavelist():
    
    profile = co.mean(dim='time')
    try:
        n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
        coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
                             wavelet, level=7)
        # coeff = pywt.swt(np.pad(profile, (0, (len(profile) // 2**5 + 3) * 2**5 - len(profile)), 'constant', constant_values=(0, 0)),
        #                  wavelet, level=5)
    except ValueError:
        continue
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]
    
    fig, ax_ = plt.subplots()
    ax_.plot(profile, df['range'], '.')
    ax_.plot(filtered, df['range'], '.')
    ax_.set_xlim([-0.5e-13, 2e-13])
    ax_.axvline(x=uthresh)
    fig.suptitle(wavelet)
    break

# %%
np.std(profile.sel(range=slice(10000, None))) * 3



