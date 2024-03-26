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
file_path = glob.glob("G:/rovaniemi/20240311/*.nc") #'sw_version': '1.2.7',
file_save = "G:/rovaniemi\Calibration/"

# %%
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))

# %%
df_ = df.sel(time=slice(pd.to_datetime('20240311 090000'),
                        pd.to_datetime('20240311 170000')))
# df_ = df
ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
for hmin, hmax in ((1000, 3000), (3000, 5000), (5000, 7000),
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
    ax[0, 0].set_ylim([-5e-15, 5e-15])
    ax[0, 0].set_ylabel(r"$\mu_{ppol}$")
    
    ax[1, 0].plot(ppol_24_std.time, ppol_24_std.values, '.')
    # ax[1, 0].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 0].set_ylim([4.5e-15, 7.5e-15])
    ax[1, 0].set_ylabel(r"$\sigma_{ppol}$")
    
    ax[0, 1].plot(xpol_24.time, xpol_24.values, '.')
    # ax[0, 1].set_ylim([-1e-13, 1e-13])
    ax[0, 1].set_ylim([-5e-15, 5e-15])
    ax[0, 1].set_ylabel(r"$\mu_{xpol}$")
    
    ax[1, 1].plot(xpol_24_std.time, xpol_24_std.values, '.')
    # ax[1, 1].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 1].set_ylim([4.5e-15, 7.5e-15])
    ax[1, 1].set_ylabel(r"$\sigma_{xpol}$")
    for ax_ in ax.flatten():
        ax_.axvspan(pd.to_datetime('20240311 095500'), pd.to_datetime('20240311 141000'), facecolor='gray',
                        alpha=0.3)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.grid()
    fig.suptitle(f"From {hmin} m to {hmax} m")
    fig.savefig(file_save + f'ppol_xpol_{hmin}_{hmax}.png', dpi=600)

# %% rovaniemi
time0, time1 = (pd.to_datetime('20240311 115500'), pd.to_datetime('20240311 141000'))
ppol_mean_time = ppol.sel(time=slice(time0, time1)).mean(dim='time')
ppol_std_time = ppol.sel(time=slice(time0, time1)).std(dim='time')
xpol_mean_time = xpol.sel(time=slice(time0, time1)).mean(dim='time')
xpol_std_time = xpol.sel(time=slice(time0, time1)).std(dim='time')

fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 6))
ax[0, 0].plot(ppol_mean_time, ppol_mean_time.range, '.')
ax[0, 0].set_xlim([-1e-13, 1e-13])
ax[0, 0].set_xlabel(r"$\mu_{ppol}$")

ax[1, 0].plot(ppol_std_time, ppol_std_time.range, '.')
ax[1, 0].set_xlim([4e-15, 1e-14])
ax[1, 0].set_xlabel(r"$\sigma_{ppol}$")


ax[0, 1].plot(xpol_mean_time, xpol_mean_time.range, '.')
ax[0, 1].set_xlim([-1e-13, 1e-13])
ax[0, 1].set_xlabel(r"$\mu_{xpol}$")

ax[1, 1].plot(xpol_std_time, xpol_std_time.range, '.')
ax[1, 1].set_xlim([4e-15, 1e-14])
ax[1, 1].set_xlabel(r"$\sigma_{xpol}$")

for ax_ in ax.flatten():
    ax_.grid()
fig.savefig(file_save + 'ppol_xpol_profile.png', dpi=600)
ax[0, 0].set_ylim([0, 3000])
ax[0, 0].set_xlim([-0.25e-13, 0.75e-13])
ax[0, 1].set_xlim([-0.25e-13, 0.75e-13])
ax[1, 0].set_xlim([4e-15, 8e-15])
ax[1, 1].set_xlim([4e-15, 8e-15])
fig.savefig(file_save + 'ppol_xpol_profile_near.png', dpi=600)

ax[0, 0].set_xlim([-2e-15, 6e-15])
ax[0, 1].set_xlim([-2e-15, 6e-15])
fig.savefig(file_save + 'ppol_xpol_profile_nearnear.png', dpi=600)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True,
                       constrained_layout=True)
p = ax[1].pcolormesh(df['time'], df['range'], df['overlap_function'].T)
fig.colorbar(p, ax=ax[1])
ax[1].set_ylim([0, 500])
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax[1].xaxis.set_major_formatter(myFmt)

ax[0].plot(df['overlap_function'].isel(time=0).values, df.range, '.')
ax[0].set_ylim([0, 500])
ax[0].grid()
fig.savefig(file_save + 'overlap_function.png', dpi=600)

# %%
df_monitoring = xr.open_mfdataset(file_path, group='monitoring')

# %%
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 9),
                       constrained_layout=True)
for val, ax_ in zip(['background_radiance', 'internal_temperature',
            'laser_temperature', 'transmitter_enclosure_temperature'],
                    ax.flatten()):
    ax_.plot(df_monitoring['time'], df_monitoring[val])
    ax_.set_ylabel(val)
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.grid()
    ax_.axvspan(pd.to_datetime('20240311 095500'), pd.to_datetime('20240311 141000'), facecolor='gray',
                    alpha=0.3)
fig.savefig(file_save + 'monitoring.png', dpi=600)

# %% Temperature dependence
df_ = df.sel(time=slice(pd.to_datetime('20240311 095500'),
                        pd.to_datetime('20240311 141000')))
# df_ = df
ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
df_monitoring_ = df_monitoring.sel(time=slice(pd.to_datetime('20240311 095500'),
                        pd.to_datetime('20240311 141000')))

# %%
df_total = xr.concat([df_.resample(time='1min').mean(),
                      df_monitoring_.resample(time='1min').mean()], dim='time')

ppol = df_total.p_pol/(df_total.range**2)
xpol = df_total.x_pol/(df_total.range**2)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 4))
for lab, grp in df_total.groupby_bins("internal_temperature", [19, 21, 23]):
    print(lab)
    ppol = grp.p_pol/(grp.range**2)
    xpol = grp.x_pol/(grp.range**2)
    
    ax[0].plot(ppol.mean(dim='time'), ppol.range, '.')
    ax[0].set_xlim([-1e-13, 1e-13])
    ax[0].set_xlabel(r"$\mu_{ppol}$")
  
    ax[1].plot(xpol.mean(dim='time'), xpol.range, '.')
    ax[1].set_xlim([-1e-13, 1e-13])
    ax[1].set_xlabel(r"$\mu_{xpol}$")
    
for ax_ in ax.flatten():
    ax_.grid()
    
# %%
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 4))

ppol = grp.p_pol/(grp.range**2)
xpol = grp.x_pol/(grp.range**2)

ax[0].plot(ppol.mean(dim='time'), ppol.range, '.')
ax[0].set_xlim([-1e-9, 1e-9])
ax[0].set_xlabel(r"$\mu_{ppol}$")
  
ax[1].plot(xpol.mean(dim='time'), xpol.range, '.')
ax[1].set_xlim([-1e-9, 1e-9])
ax[1].set_xlabel(r"$\mu_{xpol}$")
    
for ax_ in ax.flatten():
    ax_.grid()
    
# %%
grp['time'].values

