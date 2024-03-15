import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import pywt
from matplotlib.colors import LogNorm, SymLogNorm
import xarray as xr
myFmt = mdates.DateFormatter('%H:%M')

# %%
file_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_2024030[5-6]*.nc")
file_save = "G:\CloudnetData\Kenttarova\CL61\Diag/"

# %%
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))

# %%
df_ = df.sel(time=slice(pd.to_datetime('20240305 131100'),
                       pd.to_datetime('20240306 104200')))

ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
df_ = df.sel(time=slice(pd.to_datetime('20240305 000000'),
                       pd.to_datetime('20240307 000000')))

ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(df.time, df.range,
              ppol.T, shading='nearest', cmap='RdBu',
              norm=SymLogNorm(linthresh=1e-17, vmin=-2e-12, vmax=2e-12))
ax.xaxis.set_major_formatter(myFmt)
fig.colorbar(p, ax=ax)

# %%
h24 = ppol.sel(range=slice(2000, 4000))
snr24 = h24.mean(dim='range')
snr24_std = h24.std(dim='range')

# %%
fig, ax = plt.subplots(2, 1)
ax[0].plot(snr24.time, snr24.values, '.')

ax[1].plot(snr24_std.time, snr24_std.values, '.')
ax[1].set_ylim([2.5e-14, 4.5e-14])
for ax_ in ax.flatten():
    ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
                    alpha=0.3)
    ax_.xaxis.set_major_formatter(myFmt)
