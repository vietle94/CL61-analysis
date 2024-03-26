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
file_save = "G:\CloudnetData\Kenttarova\CL61\Calibration/"

# %%
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))

# %%
df_ = df.sel(time=slice(pd.to_datetime('20240305 000000'),
                        pd.to_datetime('20240307 000000')))

ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
# fig, ax = plt.subplots()
# p = ax.pcolormesh(df.time, df.range,
#               ppol.T, shading='nearest', cmap='RdBu',
#               norm=SymLogNorm(linthresh=1e-17, vmin=-2e-12, vmax=2e-12))
# ax.xaxis.set_major_formatter(myFmt)
# fig.colorbar(p, ax=ax)

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
        ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
                        alpha=0.3)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.grid()
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
# fig.savefig(file_save + 'ppol_xpol_profile.png', dpi=600)

# ax[0, 0].set_ylim([0, 3000])
ax[0, 0].set_ylim([0, 500])
# fig.savefig(file_save + 'ppol_xpol_profile_near.png', dpi=600)

# ax[0, 0].set_xlim([-1e-14, 1e-14])
# ax[0, 1].set_xlim([-1e-14, 1e-14])
# ax[1, 0].set_xlim([3.25e-14, 3.75e-14])
# ax[1, 1].set_xlim([3.25e-14, 3.75e-14])
# fig.savefig(file_save + 'ppol_xpol_profile_nearnear.png', dpi=600)

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
    ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
                    alpha=0.3)
fig.savefig(file_save + 'monitoring.png', dpi=600)

# %%
file_path_case = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_20230920*.nc")
df_case = xr.open_mfdataset(file_path_case)
df_case = df_case.isel(range=slice(1, None))

# %%
df_case_ = df_case.sel(time=slice(pd.to_datetime('20230920 220000'),
                        pd.to_datetime('20230920 235900')))

ppol_case = df_case_.p_pol/(df_case_.range**2)
xpol_case = df_case_.x_pol/(df_case_.range**2)

ppol_case = ppol_case.mean(dim='time')
xpol_case = xpol_case.mean(dim='time')

# %%
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(19, 6), constrained_layout=True)

p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlabel('Depo study case')

ax[1].plot(xpol_case/ppol_case, xpol_case.range, '.')
ax[1].set_xlim([-0.004, 0.01])
ax[1].set_ylim([0, 400])
# ax[1].set_ylim([0, 2000])
ax[1].set_xlabel('Averaged depo study case')

ax[2].plot(xpol_mean_time/ppol_mean_time, xpol_mean_time.range, '.')
ax[2].set_xlim([-1, 2])
ax[2].set_xlabel('Averaged depo calibration')

ax[3].plot(ppol_mean_time, ppol_mean_time.range, '.')
ax[3].set_xlim([-1e-13, 1e-13])
ax[3].set_xlabel(r"$\mu_{ppol}$")

ax[4].plot(xpol_mean_time, xpol_mean_time.range, '.')
ax[4].set_xlim([-1e-13, 1e-13])
ax[4].set_xlabel(r"$\mu_{xpol}$")

for ax_ in ax.flatten():
    ax_.grid()
fig.savefig(file_save + 'study_case.png', dpi=600)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True,
                       constrained_layout=True)
p = ax[1].pcolormesh(df_case_['time'], df_case_['range'], df_case_['overlap_function'].T)
fig.colorbar(p, ax=ax[1])
ax[1].set_ylim([0, 500])
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax[1].xaxis.set_major_formatter(myFmt)

ax[0].plot(df_case_['overlap_function'].isel(time=0).values, df_case_.range, '.')
ax[0].set_ylim([0, 500])
ax[0].grid()
fig.savefig(file_save + 'overlap_function.png', dpi=600)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6), constrained_layout=True)

p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlabel('Depo study case')
ax[0].set_ylim([0, 400])

p = ax[1].pcolormesh(df_case_.time, df_case_.range,
                 (((df_case_.x_pol/(df_case_.range**2))/xpol_mean_time)/((df_case_.p_pol/(df_case_.range**2))/ppol_mean_time)).T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[1])
ax[1].xaxis.set_major_formatter(myFmt)
ax[1].set_xlabel('Depo study case corrected')

for ax_ in ax.flatten():
    ax_.grid()
    
fig.savefig(file_save + 'study_case_corrected.png', dpi=600)