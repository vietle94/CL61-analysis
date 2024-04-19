import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.dates as mdates
import netCDF4
import pywt
import glob
import os
import func
myFmt = mdates.DateFormatter('%H:%M')

# %%
file_save = 'G:\CloudnetData\Kenttarova\CL61/Raw_processed/'
diag_save = 'G:\CloudnetData\Kenttarova\CL61/Diag_new/'
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw/' + '*.nc')

# %%
for file in files[[i for i, x in enumerate(files) if '20230621' in x][0]:]:
    df = xr.open_dataset(file)
    df = func.noise_detection(df)
    df.to_netcdf(file_save + os.path.basename(file))
    df_noise = df.where(df['noise'])
    grp_range = df_noise[
        ['p_pol', 'x_pol']].groupby_bins(
            "range", [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
            
    grp_mean = grp_range.mean(dim=['range', 'time'])
    grp_std = grp_range.std(dim=['range', 'time'])
    
    result = pd.DataFrame({
        'datetime': df.time[0].values,
        'co_mean': grp_mean['p_pol'],
        'co_std': grp_std['p_pol'],
        'cross_mean': grp_mean['x_pol'],
        'cross_std': grp_std['x_pol'],
        'range': grp_mean.range_bins.values.astype(str)
    })
    file_name = file.split('.')[0].split('\\')[-1]
    result.to_csv(diag_save + file_name + '_noise.csv', index=False)