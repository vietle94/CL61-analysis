import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
import matplotlib.dates as mdates
from scipy import signal
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
site = "Kenttarova"
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
sample_path = glob.glob(raw_dir + "live_20231002*.nc")

df_sample_signal = xr.open_mfdataset(sample_path)
df_sample_diag = xr.open_mfdataset(sample_path, group='monitoring')

df_sample_diag = df_sample_diag.reindex(time=df_sample_signal.time.values,
                                        method='nearest', tolerance='8s')
df_sample_diag = df_sample_diag.dropna(dim='time')

# %%
df_sample = df_sample_diag.merge(df_sample_signal, join='inner')
save_name = df_sample.time.isel(time=10).values.astype(str)[:10]
df_sample = df_sample.isel(range=slice(1, None))
df_sample['ppol_r'] = df_sample.p_pol/(df_sample.range**2)
df_sample['xpol_r'] = df_sample.x_pol/(df_sample.range**2)

t1 = pd.to_datetime("20231002 120000")
t2 = pd.to_datetime("20231002 130000")

# %%
cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
cmap.set_under(color='w')
fig, ax = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True,
                       sharex=True, sharey=True)

p = ax[0].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.ppol_r.T, norm=LogNorm(vmin=1e-13, vmax=1e-10))
cbar = fig.colorbar(p, ax=ax[0])
cbar.set_label('ppol_r')

p = ax[1].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.xpol_r.T, norm=LogNorm(vmin=1e-13, vmax=1e-11))
cbar = fig.colorbar(p, ax=ax[1])
cbar.set_label('xpol_r')

p = ax[2].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.xpol_r.T/df_sample.ppol_r.T, vmin=0, vmax=0.3,
                    cmap=cmap)
cbar = fig.colorbar(p, ax=ax[2], extend='min')
cbar.set_label('Depo')

ax[0].set_ylim([0, 8000])
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=6))

# for ax_ in ax.flatten():
#     ax_.axvspan(t1, t2, facecolor='gray',
#                 alpha=0.5)
fig.savefig(save_dir + save_name + "_profiles_sample.png",
                bbox_inches='tight', dpi=600)

# %%
df_sample = df_sample.sel(time=slice(t1, t2))
df_sample['internal_temperature_bins'] = np.floor(df_sample.internal_temperature)
df_sample_mean = df_sample.mean(dim='time')

# %%
df_mean_ref_sample = df_mean.sel(internal_temperature_bins=df_sample.internal_temperature_bins).drop_vars('internal_temperature_bins')
df_std_ref_sample = df_std.sel(internal_temperature_bins=df_sample.internal_temperature_bins).drop_vars('internal_temperature_bins')

df_sample['ppol_r_c'] = df_sample['ppol_r'] - df_mean_ref_sample['ppol_r']
df_sample['xpol_r_c'] = df_sample['xpol_r'] - df_mean_ref_sample['xpol_r']

df_sample['ppol_r_c_std'] = df_std_ref_sample.ppol_r
df_sample['xpol_r_c_std'] = df_std_ref_sample.xpol_r

df_sample['depo_c'] = df_sample['xpol_r_c']/df_sample['ppol_r_c']
df_sample['depo_c_std'] = np.abs(df_sample['depo_c'])*\
    np.sqrt((df_sample['xpol_r_c_std']/df_sample['xpol_r_c'])**2 + (df_sample['ppol_r_c_std']/df_sample['ppol_r_c'])**2)

df_sample_corrected_mean = df_sample.mean(dim='time')
df_sample_corrected_mean['depo_c'] = df_sample_corrected_mean['xpol_r_c']/df_sample_corrected_mean['ppol_r_c']
df_sample_corrected_mean['depo_c_std'] = np.sqrt((df_sample['depo_c_std']**2).sum(dim='time'))/df_sample.time.size

# %% Test depo profile
temp = df_sample.sel(time=pd.to_datetime("20231002 180000"), method='nearest')
fig, ax = plt.subplots()
ax.plot(temp.xpol_r/temp.ppol_r, temp.range)
ax.errorbar(x=temp.depo_c, y=temp.range,
            xerr=temp.depo_c_std, fmt='-.')
ax.set_ylim([0, 1000])
ax.set_xlim([-0.01, 0.03])
ax.grid()

# %% Corrected mean profile
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
ax[0].plot(df_sample_mean.xpol_r, df_sample_mean.range, label='original')
ax[0].plot(df_sample_corrected_mean.xpol_r_c, df_sample_corrected_mean.range, label='corrected')
ax[0].set_xlabel('xpol_r')

ax[1].plot(df_sample_mean.ppol_r, df_sample_mean.range, label='original ppol')
ax[1].plot(df_sample_corrected_mean.ppol_r_c, df_sample_corrected_mean.range, label='corrected')
ax[1].set_xlabel('ppol_r')

ax[2].plot(df_sample_mean.xpol_r/df_sample_mean.ppol_r, df_sample.range, label='original')
ax[2].errorbar(x = df_sample_corrected_mean['depo_c'], y = df_sample.range,
               xerr = df_sample_corrected_mean['depo_c_std'],
               label='corrected')
ax[2].set_xlabel('depo')
for ax_ in ax.flatten():
    ax_.legend()
    ax_.grid()

ax[0].set_xlim([-1e-13, 5e-13])
ax[1].set_xlim([0, 1e-10])
# ax[0].set_xscale('log')
# ax[1].set_xscale('log')

ax[2].set_xlim([0, 0.01])
ax[0].set_ylim([0, 500])

fig.savefig(save_dir + save_name + "_corrected_profiles.png",
            bbox_inches='tight', dpi=600)

# %%
cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
cmap.set_under(color='w')
fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True,
                        sharex=True, sharey=True)

# p = ax[0].pcolormesh(df_sample.time, df_sample.range,
#                     df_sample.xpol_r.T/df_sample.ppol_r.T, vmin=0, vmax=0.3)
# fig.colorbar(p, ax=ax[0])

p = ax[0].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.depo_c.T, vmin=0, vmax=0.3, cmap=cmap)
cbar = fig.colorbar(p, ax=ax[0])
cbar.set_label('Depo')

p = ax[1].pcolormesh(df_sample.time, df_sample.range,
                    df_sample.depo_c_std.T, vmin=0, vmax=0.1, cmap=cmap)
cbar = fig.colorbar(p, ax=ax[1])
cbar.set_label('Depo std')

ax[1].set_ylim([0, 8000])
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
    ax_.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
fig.savefig(save_dir + save_name + "_corrected_depo.png",
            bbox_inches='tight', dpi=600)

# %%
df_plot = df_sample.sel(time=slice(pd.to_datetime("20231002 170000"),
                                   pd.to_datetime("20231002 230000")))
df_plot = df_plot.sel(range=slice(None, 1000))

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True, sharex=True,
                       constrained_layout=True)
p = ax[0].pcolormesh(df_plot.time, df_plot.range,
              df_plot.xpol_r.T/df_plot.ppol_r.T, 
              vmin=-0.01, vmax=0.01,
              cmap='RdBu')
cbar = fig.colorbar(p, ax=ax[0])
cbar.set_label('Depo')

p = ax[1].pcolormesh(df_plot.time, df_plot.range,
                    df_plot.depo_c_std.T, vmin=0, vmax=0.01)
cbar = fig.colorbar(p, ax=ax[1])
cbar.set_label('Depo std')

for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=1))
fig.savefig(save_dir + save_name + "_negative_depo.png",
            bbox_inches='tight', dpi=600)

# %%
df_plot = df_sample.sel(time=pd.to_datetime("20231002 183000"),
                        method='nearest')
df_plot = df_plot.sel(range=slice(None, 1000))

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 4), sharey=True,
                       constrained_layout=True)
ax[0].plot(df_plot['ppol_r'], df_plot['range'], '.')
ax[1].plot(df_plot['xpol_r'], df_plot['range'], '.')
ax[2].plot(df_plot['xpol_r']/df_plot['ppol_r'], df_plot['range'], '.')

ax[0].set_ylim([0, 1000])
ax[0].set_xlabel('ppol')

ax[0].set_xscale('log')
ax[0].set_xlim([1e-14, 1e-7])

ax[1].set_xlim([-1e-11, 1e-11])
ax[1].set_xlabel('xpol')

ax[2].set_xlim([-0.01, 0.01])
ax[2].set_xlabel('depol')

for ax_ in ax.flatten():
    ax_.grid()
     
name = np.datetime_as_string(df_plot.time.values, unit='h')
fig.suptitle(name)
fig.savefig(save_dir + save_name + "_negative_depo_profile.png",
            bbox_inches='tight', dpi=600)

# %%
sample_path
temp = xr.open_dataset(sample_path[0])
xx = glob.glob(raw_dir + "live_20230102*.nc")
temp= xr.open_mfdataset(xx, group='diagnostics')
Dataset(temp)
# %%
temp.plot()
list(temp.keys())
# %%
# temp.Receiver_voltage.plot()
for val in list(temp.keys()):
    fig, ax = plt.subplots()
    ax.plot(temp.time, temp[val], '.')
    ax.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel(val)
    
# %%
diag = pd.concat([pd.read_csv(x) for x in glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Diag/live_20230225*_diag.csv")], ignore_index=True)
diag['datetime'] = pd.to_datetime(diag['datetime'], format='mixed')
diag = diag.set_index('datetime')

# %%
for x in diag:
    fig, ax = plt.subplots()
    ax.plot(diag.index, diag[x], '.')
    ax.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel(x)

# %%
fig, ax = plt.subplots()
ax.
    