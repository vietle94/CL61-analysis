import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import glob
import os
import func
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')

# %%
fig_save = "G:\CloudnetData\Kenttarova\CL61\Summary/"
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw/' + '*.nc')

# %%
date = '20231229'
df = xr.open_mfdataset([x for x in files if date in x])

# %%
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
p = ax.pcolormesh(df.time, df.range,
               np.isnan(df.p_pol).T)
              # np.isinf(df.p_pol).T)
cbar = fig.colorbar(p, ax=ax)
cbar.ax.set_ylabel("is ppol nan")
# cbar.ax.set_ylabel("is ppol inf")
ax.set_ylim([0, 200])
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.set_ylabel('Height [m]')
fig.savefig(fig_save + f"studycase_{date}.png", dpi=600)

# %%
df_ = df.sel(time=slice(date + " 132000",
                         date + " 133000"))
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
p = ax.pcolormesh(df_.time, df_.range,
              # np.isnan(df_.p_pol).T)
              np.isinf(df_.p_pol).T)
cbar = fig.colorbar(p, ax=ax)
# cbar.ax.set_ylabel("is ppol nan")
cbar.ax.set_ylabel("is ppol inf")
ax.set_ylim([0, 200])
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.set_ylabel('Height [m]')

# %%
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
p = ax.pcolormesh(df_.time, df_.range,
              df_.p_pol.T, )
cbar = fig.colorbar(p, ax=ax)
ax.set_ylim([0, 200])