import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import glob
import os
import func
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# %%
depo_save = r'G:\CloudnetData\Kenttarova\CL61\Files_analysis\Negative_depo/'
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw_processed/' + '*.nc')

# %%
# for file in files[:[i for i, x in enumerate(files) if '20230621' in x][0]]:
#     print(file)
#     file_name = file.split('.')[0].split('\\')[-1]
#     check_file = depo_save + file_name + '_neg_depo.csv'
#     if os.path.isfile(check_file):
#         print('yes')
#         continue
#     try:
#         df = xr.open_dataset(file)
#     except OSError:
#         print('Bad file')
#         continue

#     df_range = np.tile(df['range'], df['time'].size).reshape(df['time'].size, -1)
#     df['range_full'] = (['time', 'range'], df_range)
#     df_range = df['range_full'].where(~df['noise']).where(df['x_pol'] < 0).values
#     xpol = df['x_pol'].where(~df['noise']).where(df['x_pol'] < 0)
#     ppol = df['p_pol'].where(~df['noise']).where(df['x_pol'] < 0)
    
#     mask = ~np.isnan(xpol)
#     xpol = xpol.values[mask]
#     ppol = ppol.values[mask]
#     df_range = df_range[mask]
#     if xpol.size < 1:
#         continue
    
#     result = pd.DataFrame({
#         'datetime': df.time[0].values,
#         'xpol': xpol,
#         'ppol': ppol,
#         'range': df_range
#     })
#     file_name = file.split('.')[0].split('\\')[-1]
#     result.to_csv(check_file, index=False)
    
# %% new device software
for file in files[[i for i, x in enumerate(files) if '20230621' in x][0]:]:
    print(file)
    file_name = file.split('.')[0].split('\\')[-1]
    check_file = depo_save + file_name + '_neg_depo.csv'
    df = xr.open_dataset(file)
    df_range = np.tile(df['range'], df['time'].size).reshape(df['time'].size, -1)
    df['range_full'] = (['time', 'range'], df_range)
    df_range = df['range_full'].where(~df['noise']).where(df['x_pol'] < 0).values
    xpol = df['x_pol'].where(~(df['noise'])).where(df['x_pol'] < 0)
    ppol = df['p_pol'].where(~df['noise']).where(df['x_pol'] < 0)
    
    mask = ~np.isnan(xpol)
    xpol = xpol.values[mask]
    ppol = ppol.values[mask]
    df_range = df_range[mask]
    if xpol.size < 1:
        continue
    
    result = pd.DataFrame({
        'datetime': df.time[0].values,
        'xpol': xpol,
        'ppol': ppol,
        'range': df_range
    })
    file_name = file.split('.')[0].split('\\')[-1]
    result.to_csv(check_file, index=False)