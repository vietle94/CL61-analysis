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
# file_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_2024030[5-6]*.nc")
# file_save = "G:\CloudnetData\Kenttarova\CL61\Calibration/"

file_path = glob.glob("G:/rovaniemi/20240311/*.nc")
file_save = "G:/rovaniemi\Calibration/"

# %%
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))

# %%
# df_ = df.sel(time=slice(pd.to_datetime('20240305 000000'),
#                        pd.to_datetime('20240307 000000')))

# ppol = df_.p_pol/(df_.range**2)
# xpol = df_.x_pol/(df_.range**2)

# %%
ppol = df.p_pol/(df_.range**2)
xpol = df.x_pol/(df_.range**2)

# %%
# fig, ax = plt.subplots()
# p = ax.pcolormesh(df.time, df.range,
#               ppol.T, shading='nearest', cmap='RdBu',
#               norm=SymLogNorm(linthresh=1e-17, vmin=-2e-12, vmax=2e-12))
# ax.xaxis.set_major_formatter(myFmt)
# fig.colorbar(p, ax=ax)

# %%
for hmin, hmax in ((0, 1000), (1000, 3000), (3000, 5000), (5000, 7000),
                   (7000, 9000), (9000, 11000), (11000, 13000), (13000, 15000),
                   (15000, 16000)):
    ppol24 = ppol.sel(range=slice(hmin, hmax))
    ppol_24 = ppol24.mean(dim='range')
    ppol_24_std = ppol24.std(dim='range')
    
    xpol24 = xpol.sel(range=slice(hmin, hmax))
    xpol_24 = xpol24.mean(dim='range')
    xpol_24_std = xpol24.std(dim='range')
    
    fig, ax = plt.subplots(2, 2, sharex=True, constrained_layout=True,
                           figsize=(12, 6))
    ax[0, 0].plot(ppol_24.time, ppol_24.values, '.')
    # ax[0, 0].set_ylim([-1e-13, 1e-13])
    ax[0, 0].set_ylabel(r"$\mu_{ppol}$")
    
    ax[1, 0].plot(ppol_24_std.time, ppol_24_std.values, '.')
    # ax[1, 0].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 0].set_ylabel(r"$\sigma_{ppol}$")
    
    
    ax[0, 1].plot(xpol_24.time, xpol_24.values, '.')
    # ax[0, 1].set_ylim([-1e-13, 1e-13])
    ax[0, 1].set_ylabel(r"$\mu_{xpol}$")
    
    ax[1, 1].plot(xpol_24_std.time, xpol_24_std.values, '.')
    # ax[1, 1].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 1].set_ylabel(r"$\sigma_{xpol}$")
    
    for ax_ in ax.flatten():
        # ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
        #                 alpha=0.3)
        ax_.axvspan(pd.to_datetime('20240311 095500'), pd.to_datetime('20240311 145000'), facecolor='gray',
                        alpha=0.3)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.grid()
    break
    fig.suptitle(f"From {hmin} m to {hmax} m")
    fig.savefig(file_save + f'ppol_xpol_{hmin}_{hmax}.png', dpi=600)

# %%
ppol_mean_time = ppol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).mean(dim='time')
ppol_std_time = ppol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).std(dim='time')
xpol_mean_time = xpol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).mean(dim='time')
xpol_std_time = xpol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).std(dim='time')

# %%
fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 6))
ax[0, 0].plot(ppol_mean_time, ppol_mean_time.range, '.')
ax[0, 0].set_xlim([-1e-13, 1e-13])
ax[0, 0].set_xlabel(r"$\mu_{ppol}$")

ax[1, 0].plot(ppol_std_time, ppol_std_time.range, '.')
ax[1, 0].set_xlim([2.5e-14, 4.5e-14])
ax[1, 0].set_xlabel(r"$\sigma_{ppol}$")


ax[0, 1].plot(xpol_mean_time, xpol_mean_time.range, '.')
ax[0, 1].set_xlim([-1e-13, 1e-13])
ax[0, 1].set_xlabel(r"$\mu_{xpol}$")

ax[1, 1].plot(xpol_std_time, xpol_std_time.range, '.')
ax[1, 1].set_xlim([2.5e-14, 4.5e-14])
ax[1, 1].set_xlabel(r"$\sigma_{xpol}$")

for ax_ in ax.flatten():
    ax_.grid()
fig.savefig(file_save + 'ppol_xpol_profile.png', dpi=600)

ax[0, 0].set_ylim([0, 3000])
fig.savefig(file_save + 'ppol_xpol_profile_near.png', dpi=600)

ax[0, 0].set_xlim([-1e-14, 1e-14])
ax[0, 1].set_xlim([-1e-14, 1e-14])
ax[1, 0].set_xlim([3.25e-14, 3.75e-14])
ax[1, 1].set_xlim([3.25e-14, 3.75e-14])
fig.savefig(file_save + 'ppol_xpol_profile_nearnear.png', dpi=600)

