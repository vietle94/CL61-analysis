import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
from scipy import signal
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
site = "Vehmasmaki"
plot_lim = {'Vehmasmaki': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                           LogNorm(vmin=1e-15, vmax=1e-12), 0],
            'Kenttarova': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                         LogNorm(vmin=1e-15, vmax=1e-12), 0],
            'Hyytiala': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                         LogNorm(vmin=1e-15, vmax=1e-12), 1]}

raw_dir = r"G:\CloudnetData\/" + site + r"\CL61\Raw/"
merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"
save_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary/"

# %% 
df_signal = xr.open_mfdataset(glob.glob(merged_dir + '*_signal.nc'))
df_diag = xr.open_mfdataset(glob.glob(merged_dir + '*_diag.nc'))

df_diag = df_diag.reindex(time=df_signal.time.values, method='nearest', tolerance='8s')
df_diag = df_diag.dropna(dim='time')

# %%
df = df_diag.merge(df_signal, join='inner')
df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

# %%
temp_range = np.arange(np.floor(df.internal_temperature.min(skipna=True).values),
          np.ceil(df.internal_temperature.max(skipna=True)).values+1)
df_gr = df.groupby_bins("internal_temperature", temp_range,
                        labels=temp_range[:-1])
df_mean = df_gr.mean(dim='time', skipna=True)
df_std = df_gr.std(dim='time', skipna=True)

# %%
sample_path = glob.glob(raw_dir + "live_20241029*.nc")

df_sample_signal = xr.open_mfdataset(sample_path)
df_sample_diag = xr.open_mfdataset(sample_path, group='monitoring')

df_sample_diag = df_sample_diag.reindex(time=df_sample_signal.time.values,
                                        method='nearest', tolerance='8s')
df_sample_diag = df_sample_diag.dropna(dim='time')

# %%
df_sample = df_sample_diag.merge(df_sample_signal, join='inner')
save_name = df_sample.time.isel(time=0).values.astype(str)[:10]
df_sample = df_sample.isel(range=slice(1, None))
df_sample['ppol_r'] = df_sample.p_pol/(df_sample.range**2)
df_sample['xpol_r'] = df_sample.x_pol/(df_sample.range**2)

# %%
df_sample = df_sample.sel(time=slice(pd.to_datetime("20241029 120000"), None))
df_sample['internal_temperature_bins'] = np.floor(df_sample.internal_temperature)
df_sample_mean = df_sample.mean(dim='time')

# %%
df_sample_corrected = df_mean.sel(internal_temperature_bins=df_sample.internal_temperature_bins)
df_sample_corrected['ppol_r_c'] = df_sample['ppol_r'] - df_sample_corrected['ppol_r']
df_sample_corrected['xpol_r_c'] = df_sample['xpol_r'] - df_sample_corrected['xpol_r']

df_sample_corrected_mean = df_sample_corrected.mean(dim='time')

# %%
cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
cmap.set_under(color='w')
fig, ax = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True,
                       sharex=True, sharey=True)

p = ax[0, 0].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.ppol_r.T, norm=LogNorm(vmin=1e-13, vmax=1e-8))
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.set_label('ppol_r')

p = ax[0, 1].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.xpol_r.T, norm=LogNorm(vmin=1e-13, vmax=1e-8))
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.set_label('xpol_r')

p = ax[0, 2].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.xpol_r.T/df_sample.ppol_r.T, vmin=0, vmax=0.3,
                    cmap=cmap)
cbar = fig.colorbar(p, ax=ax[0, 2], extend='min')
cbar.set_label('Depo')

p = ax[1, 0].pcolormesh(df_sample_corrected.time, df_sample_corrected.range,
                    df_sample_corrected.ppol_r_c.T, norm=LogNorm(vmin=1e-13, vmax=1e-8))
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.set_label('ppol_r_c')

p = ax[1, 1].pcolormesh(df_sample_corrected.time, df_sample_corrected.range,
                    df_sample_corrected.xpol_r_c.T, norm=LogNorm(vmin=1e-13, vmax=1e-8))
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.set_label('xpol_r_c')

p = ax[1, 2].pcolormesh(df_sample_corrected.time, df_sample_corrected.range,
                    df_sample_corrected.xpol_r_c.T/df_sample_corrected.ppol_r_c.T, vmin=0, vmax=0.3,
                    cmap=cmap)
cbar = fig.colorbar(p, ax=ax[1, 2], extend='min')
cbar.set_label('Depo_c')

for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax_.set_ylim([0, 500])
fig.savefig(save_dir + save_name + "_corrected_overview.png",
            bbox_inches='tight', dpi=600)
    
# %%
t_frame = slice(pd.to_datetime("20241029 120000"), pd.to_datetime("20241029 130000"))
profile = df_sample.sel(time=t_frame).mean(dim='time')
profile_c = df_sample_corrected.sel(time=t_frame).mean(dim='time')

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
ax[0].plot(profile.xpol_r, profile.range, label='original xpol')
ax[0].plot(profile_c.xpol_r_c, profile_c.range, label='original xpol')

ax[1].plot(profile.ppol_r, profile.range, label='original ppol')
ax[1].plot(profile_c.ppol_r_c, profile_c.range, label='original ppol')

ax[2].plot(profile.xpol_r/profile.ppol_r, profile.range, label='original')
ax[2].plot(profile_c['xpol_r_c']/profile_c['ppol_r_c'], profile_c.range,
        label='corrected')
for ax_ in ax.flatten():
    ax_.legend()
    ax_.grid()
ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[2].set_xlim([0, 0.1])
ax[0].set_ylim([0, 500])
# ax[0].set_xlim([1e-14, 1e-13])
# ax[0].set_xscale('log')
# %%
fig, ax = plt.subplots()
ax.plot(df_sample.isel(time=400).xpol_r, df_sample.range, '.')
ax.plot(df_mean.sel(internal_temperature_bins=df_sample.isel(time=400).internal_temperature_bins).xpol_r, df_mean.range, '.')
ax.set_xlim([-0.5e-13, 1e-13])
ax.set_ylim([0, 500])
ax.grid()

# %%
cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
cmap.set_under(color='w')
fig, ax = plt.subplots()
p = ax.pcolormesh(df_mean.internal_temperature_bins,
              df_mean.range,
              df_mean.xpol_r.T, shading='nearest',
              vmin=0, vmax=0.5e-13, cmap=cmap)
fig.colorbar(p, ax=ax)
ax.set_ylim([0, 500])

# %%
fig, ax = plt.subplots()
ax.plot(df_mean.isel(internal_temperature_bins=2).xpol_r, df_mean.range, '.')
ax.set_xlim([-0.5e-13, 1e-13])
ax.set_ylim([0, 500])
ax.grid()

