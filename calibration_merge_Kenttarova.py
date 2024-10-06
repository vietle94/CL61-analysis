import numpy as np
import pandas as pd
import glob
import xarray as xr

# %%
site = "Kenttarova"

# %% Set calibration files
calibration = pd.read_csv(r"G:\CloudnetData\Calibration\/" + site + r"\calibration.txt")
calibration['t1'] = pd.to_datetime(calibration['t1'])
calibration['t2'] = pd.to_datetime(calibration['t2'])

# %% Set dir
raw_dir = r"G:\CloudnetData\/" + site + r"\CL61\Raw/"
merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"

# %%
for i, row in calibration.iterrows():
    t1 = row['t1']
    t2 = row['t2']
    save_name = t1.strftime("%Y%m%d")
    print(save_name)
    file_path = glob.glob(raw_dir + row['file']) # Select data in a whole day
    
    df_path = pd.DataFrame({'path': file_path})
    df_path['date'] = df_path['path'].str.split("_").str[1]
    df_path['time'] = df_path['path'].str.split("_").str[2].str.split(".").str[0]
    df_path['datetime'] = pd.to_datetime(df_path['date'] + ' ' + df_path['time'])

    path_cal = df_path.loc[(df_path['datetime'] > t1) & (df_path['datetime'] < t2), 'path'].values
    
    print("Done indexing")
    
    
    
    df = xr.open_mfdataset(df_path)
    df = df.isel(range=slice(1, None))
    df_cal = df.sel(time=slice(t1, t2))
    df_cal = df_cal[['p_pol', 'x_pol']]
    df_cal.to_netcdf(merged_dir + save_name + "_signal.nc")
    
    df_diag = xr.open_mfdataset(df_path, group='monitoring')
    df_diag_cal = df_diag.sel(time=slice(t1, t2))
    df_diag_cal = df_diag_cal[['laser_temperature',
                   'internal_temperature', 'transmitter_enclosure_temperature']]
    df_diag_cal.to_netcdf(merged_dir + save_name + "_diag.nc")
