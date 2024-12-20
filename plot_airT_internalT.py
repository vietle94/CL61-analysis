import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import pywt
from matplotlib.colors import LogNorm, SymLogNorm
import xarray as xr
import func
import matplotlib as mpl
myFmt = mdates.DateFormatter('%H:%M\n%Y-%m-%d')

# %%
file_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_*.nc")
file_path = file_path[[i for i, x in enumerate(file_path) if '20230621' in x][0]:]
monitoring = xr.open_mfdataset(file_path, group='monitoring')

# %%
monitoring.to_netcdf(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Monitoring/monitoring.nc")

# %%
monitoring = xr.open_dataset(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Monitoring/monitoring.nc")

# %%
# monitoring = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/monitoring.csv")
# monitoring['datetime'] = pd.to_datetime(monitoring['datetime'], format='mixed')
# monitoring_10min = monitoring.set_index('datetime').resample('10min').mean().reset_index()

# %%
weather = pd.read_csv(glob.glob(r"G:\CloudnetData\Calibration\Kenttarova/*.csv")[0])
weather['Time [UTC]'] = weather['Time [UTC]'] + ':00'
weather['datetime'] = pd.to_datetime(weather[['Year', 'Month', 'Day']])
weather['datetime'] = pd.to_datetime(weather['datetime'].astype(str) + ' ' + weather['Time [UTC]'])
weather = weather[['datetime', 'Air temperature [°C]']]
weather = weather.rename({'Air temperature [°C]': 'AirT', 'datetime': 'time'}, axis=1)

# %%
monitoring_10min = monitoring.resample(time='10min').mean()


# %%
monitoring_10min = monitoring_10min.dropna(dim='time')
itemp = monitoring_10min.internal_temperature.to_series()
itemp = itemp.to_frame().reset_index()

# %%
iweather = itemp.merge(weather)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(iweather['internal_temperature'], iweather['AirT'], '.', alpha=0.1)
ax.set_box_aspect(1)
ax.grid()
ax.plot(np.arange(-5, 30),np.arange(-5, 30))
ax.set_xlabel('Internal T')
ax.set_ylabel('Air T')

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df['datetime'], df['internal_temperature'], '.', label='Internal T')
ax.plot(df['datetime'], df['AirT'], '.', label='Air T')
ax.legend()
1
