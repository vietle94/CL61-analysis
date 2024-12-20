import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
import func
from netCDF4 import Dataset, MFDataset
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
raw_dir = r"G:\CloudnetData\Vehmasmaki\CL61\Raw/"
file_dirs = glob.glob(raw_dir + "live_20210524*.nc")
# df_raw = xr.open_mfdataset(file_dirs[0])

# %%
# denoise = func.noise_detection(df)
# %%
# df_noise = denoise.where(denoise['noise'])
df_raw['p_pol_r'] = df_raw['p_pol']/(df_raw['range']**2)
df_raw['x_pol_r'] = df_raw['x_pol']/(df_raw['range']**2)
grp_range = df_raw[
    ['p_pol_r', 'x_pol_r']].groupby_bins(
        "range", [4000, 6000, 8000, 10000, 12000, 14000])
        
grp_mean = grp_range.mean(dim=['range', 'time'])
grp_std = grp_range.std(dim=['range', 'time'])

# %%
fig, ax = plt.subplots()
for label, grp in grp_range:
    ax.plot(grp.time, grp.p_pol_r.std(dim='range'), label=label)
ax.legend()
ax.xaxis.set_major_formatter(myFmt)
ax.set_ylim([1.2e-14, 2.2e-14])

# %%
fig, ax = plt.subplots()
for label, grp in denoise:
    
# %%
df_raw = xr.open_mfdataset(file_dirs[:10], group='diagnostics')
df_raw.plot()

# %%
df_raw = xr.open_dataset(file_dirs[0], group='diagnostics')
df_raw = xr.open_dataset(file_dirs[0])
df_raw

# %%
for val in list(df_raw.keys()):
    fig, ax = plt.subplots()
    ax.plot(df_raw.time, df_raw[val], '.')
    ax.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel(val)
    
# %%
df_raw = MFDataset(file_dirs[:10])
