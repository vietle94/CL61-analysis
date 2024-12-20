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
from functools import partial
myFmt = mdates.DateFormatter('%H:%M\n%Y-%m-%d')

# %%
calibration = pd.read_csv(r"G:\CloudnetData\Calibration\Hyytiala\calibration.txt")
calibration['t1'] = pd.to_datetime(calibration['t1'])
calibration['t2'] = pd.to_datetime(calibration['t2'])

# %%
for i, row in calibration.iterrows():
    t1 = row['t1']
    t2 = row['t2']
    file_path = glob.glob(r"G:\CloudnetData\Hyytiala\CL61\Raw/" + row['file'])
    
    save_name = t1.strftime("%Y%m%d")
    print(save_name)
    df = xr.open_mfdataset(file_path)
    df = df.isel(range=slice(1, None))
    
    df_ = df.sel(time=slice(t1, t2))
    
    if i == 0:
        df_diag = xr.open_mfdataset(file_path, group='monitoring', preprocess=_preprocess)
    else:
        df_diag = xr.open_mfdataset(file_path, group='monitoring')

    df_diag_ = df_diag.sel(time=slice(t1, t2))
    break
    
    
    # Merge with diag
    df_diag_ = df_diag_.reindex(time=df_.time.values, method='nearest', tolerance='8s')
    df_merge = df_diag_.merge(df_)
    
    if i == 0:
        df_merge = df_merge[['laser_temperature', 'p_pol', 'x_pol', 'linear_depol_ratio',
                       'internal_temperature']]
    else:
        df_merge = df_merge[['laser_temperature', 'p_pol', 'x_pol', 'linear_depol_ratio',
                       'internal_temperature', 'transmitter_enclosure_temperature']]
    df_merge['internal_temperature'] = df_merge['internal_temperature'].ffill(dim='time')
    df_merge.to_netcdf(r"G:\CloudnetData\Calibration\Hyytiala\Summary\Data_merged/" + save_name + ".nc")
    
    break

# %%
df_merge['internal_temperature'] = df_merge['internal_temperature'].ffill(dim='time')

# %%
def _preprocess(x):
    x['internal_temperature'] = (['time'], [x.attrs['internal_temperature']])
    x['laser_temperature'] = (['time'], [x.attrs['laser_temperature']])
    return x.assign_coords(time=[pd.to_datetime(float(x.attrs['Timestamp']), unit='s')])
    

# %%
xr.open_dataset(file_path[-1], group='monitoring')