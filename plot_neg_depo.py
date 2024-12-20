import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import glob
import os
import matplotlib.pyplot as plt
import func
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from scipy.ndimage import maximum_filter, median_filter
myFmt = mdates.DateFormatter('%H:%M')

# %%
files = glob.glob(r'G:\CloudnetData\Kenttarova\CL61\Files_analysis\Negative_depo/*.csv')
# selected_files = files[[i for i, x in enumerate(files) if '20230621' in x][0]:]
selected_files = [x for x in files if '20230920' in x]
df = pd.concat([pd.read_csv(x) for x in selected_files], ignore_index=True)
df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')

# %%
fig, ax = plt.subplots()
p = ax.hist2d(df['xpol'], df['ppol'], bins=1000, norm=LogNorm())
fig.colorbar(p[3], ax=ax)

# %%
fig, ax = plt.subplots()
ax.hist(np.log10(np.abs(df['ppol'])), bins=100)


# %%


# %%
df_ = df[df['datetime'].between(pd.to_datetime('20230920'),
                                pd.to_datetime('20230921'))]
df['datetime'].loc[pd.to_datetime('20230920'):pd.to_datetime('20230921')]
# %%
fig, ax = plt.subplots()
p = ax.hist2d(df_['xpol'], df_['ppol'], bins=1000)
fig.colorbar(p[3], ax=ax)


# %%
df_ = df.loc[(df['datetime'] > pd.to_datetime('20230920')) & \
   (df['datetime'] < pd.to_datetime('20230921'))]
    
# %%
df['ppol'].min()
df['xpol'].min()

df['ppol'].max()
df['xpol'].max()

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['datetime'], df['xpol'], '.')
ax.set_ylim([-1e-6, 0])
# ax.set_ylim([-1e-6, 4e-6])
ax.xaxis.set_major_formatter(myFmt)
ax.grid()

# %%
df_ = df.loc[(df['datetime'].dt.hour == 20) & (df['range'] < 400)]
plt.hist(df_['ppol'])

# %%
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw/' + '*.nc')

# %%
date = '20230920'
df = xr.open_mfdataset([x for x in files if date in x])
df = df.isel(range=slice(1, None))

# %%
# df = xr.open_dataset([x for x in files if '20230920_2100' in x][0])
df = xr.open_dataset([x for x in files if '20230920_17' in x][0])

# %%
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
p = ax[0].pcolormesh(df['time'], df['range'], df['p_pol'].T,
                 norm=LogNorm(vmin=1e-7, vmax=1e-4))
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(df['time'], df['range'], cumsum.T)
fig.colorbar(p, ax=ax[1])
for ax_ in ax.flatten():
    ax_.set_ylim([0, 1000])
    
# %%
log_beta = np.log10(df['p_pol'])
mask = log_beta > -4
mask = maximum_filter(mask, size=5)
mask = median_filter(mask, size=13)
mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
mask_col = np.nanargmax(mask[mask_row, :], axis=1)
for row, col in zip(mask_row, mask_col):
    mask[row, col:] = True

# %%
fig, ax = plt.subplots()
ax.pcolormesh(df.time, df.range,
        mask.T)

# %%
df = xr.open_mfdataset([x for x in files if '20230920' in x])
# %%
mask = df['p_pol'] > 1e-4
mask = maximum_filter(mask, size=5)
mask = median_filter(mask, size=13)
mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
mask_col = np.nanargmax(mask[mask_row, :], axis=1)
for row, col in zip(mask_row, mask_col):
    mask[row, col:] = True
    
# %%
df_range = np.tile(df['range'], df['time'].size).reshape(df['time'].size, -1)
df['range_full'] = (['time', 'range'], df_range)
df_range = df['range_full'].where(~df['noise']).where(df['x_pol'] < 0).values
xpol = df['x_pol'].where(~(df['noise'])).where(~(mask)).where(df['x_pol'] < 0)
ppol = df['p_pol'].where(~df['noise']).where(~(mask)).where(df['x_pol'] < 0)

# %%
mask_ = ~(df['p_pol_smooth']<10*df.attrs['minimax_thresh']) * ~mask
mask_ = median_filter(mask_, size=100)
# neg = (df['p_pol'].where(~(df['p_pol_smooth']<10*df.attrs['minimax_thresh'])).where(~(mask)).where(df['x_pol'] < 0)).T
neg = (df['p_pol'].where(mask_).where(df['x_pol'] < 0)).T
fig, ax = plt.subplots(figsize=(9, 4))
ax.pcolormesh(df.time, df.range,
             neg,
              norm=LogNorm(vmin=1e-7, vmax=1e-4))
ax.xaxis.set_major_formatter(myFmt)

# %%
for d in ['20230920_16', '20230920_17', '20230920_18', '20230920_19', 
          '20230920_20', '20230920_21', '20230920_22', '20230920_23', 
          '20230921_01', '20230921_02', '20230921_03', '20230921_04',
          '20230921_05', '20230921_06', '20230921_07', '20230921_08',
          '20230921_09', '20230921_10', '20230921_11', '20230921_12',
          '20230921_13', '20230921_14', '20230921_15', '20230921_16',
          '20230921_17', '20230927_16', '20231001_07', '20231003_06',
          '20231003_12', '20231012_17', '20231012_21', '20231012_22',
          '20231013_04', '20231017_11', '20231017_14', '20231022_23']:
    df = xr.open_dataset([x for x in files if d in x][0])
    df = df.sel(range=slice(0, 2000))
    fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True,
                           constrained_layout=True)
    ax[1].plot(df['p_pol'].mean(dim='time')/(df['range']**2), df['range'], '.')
    ax[2].plot(df['x_pol'].mean(dim='time')/(df['range']**2), df['range'], '.')
    ax[3].plot(df['linear_depol_ratio'].mean(dim='time'), df['range'], '.')
    
    ax[1].set_ylim([0, 2000])
    ax[1].set_xlabel('ppol')
    
    ax[1].set_xscale('log')
    ax[1].set_xlim([1e-14, 1e-7])
    
    ax[2].set_xlim([-1e-11, 1e-11])
    ax[2].set_xlabel('xpol')
    
    ax[3].set_xlim([-0.01, 0.01])
    ax[3].set_xlabel('depol')
    
    p = ax[0].pcolormesh(df.time, df.range, df.p_pol.T, 
                     norm=LogNorm(vmin=1e-7, vmax=1e-4))
    fig.colorbar(p, ax=ax[0])
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].xaxis.set_major_locator(mdates.HourLocator())
    ax[0].set_xlabel('ppol')
    for ax_ in ax.flatten():
        ax_.grid()
        
    name = np.datetime_as_string(df.time[0].values, unit='h')
    fig.suptitle(name)
    fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Img\Neg_dep/" + name + '.png',
                bbox_inches='tight')
    plt.close(fig)
# %%
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw/' + '*.nc')
df = xr.open_dataset([x for x in files if '20230920_22' in x][0])
df = df.isel(range=slice(1, None))
fig, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True,
                       constrained_layout=True)
ax[1].plot((df['p_pol'].mean(dim='time')/(df['range']**2)+1)/ppol_correction - 1, df['range'], '.')
ax[2].plot((df['x_pol'].mean(dim='time')/(df['range']**2)+1)/xpol_correction - 1, df['range'], '.')
ax[3].plot(df['linear_depol_ratio'].mean(dim='time'), df['range'], '.')

ax[1].set_ylim([0, 2000])
ax[1].set_xlabel('ppol')

ax[1].set_xscale('log')
ax[1].set_xlim([1e-14, 1e-7])

ax[2].set_xlim([-1e-11, 1e-11])
ax[2].set_xlabel('xpol')

ax[3].set_xlim([-0.01, 0.01])
ax[3].set_xlabel('depol')

p = ax[0].pcolormesh(df.time, df.range, df.p_pol.T, 
                 norm=LogNorm(vmin=1e-7, vmax=1e-4))
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].xaxis.set_major_locator(mdates.HourLocator())
ax[0].set_xlabel('ppol')
for ax_ in ax.flatten():
    ax_.grid()
    
name = np.datetime_as_string(df.time[0].values, unit='h')
fig.suptitle(name)