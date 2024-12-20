import glob
import numpy as np
import pandas as pd
import xarray as xr
import os

file_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_*.nc")
file_save = "G:\CloudnetData\Kenttarova\CL61\Diag/"

for file in file_path[[i for i, x in enumerate(file_path) if '20230621' in x][0]:]:
    print(file)
    file_name = file.split('.')[0].split('\\')[-1]
    check_file = file_save + file_name + '_monitoring.csv'
    if os.path.isfile(check_file):
        print('yes')
        continue
    try:
        df_status = xr.open_dataset(file, group='status')
        df_monitoring = xr.open_dataset(file, group='monitoring')
        df = xr.open_dataset(file)
    except OSError:
        print('Bad file')
        continue
    
    df_status = df_status.to_dataframe().reset_index(drop=True).mean()
    df_status['datetime'] = df.time[0].values
    status = pd.DataFrame([df_status])
    status.to_csv(file_save + file_name + '_status.csv', index=False)

    df_monitoring = df_monitoring.to_dataframe().reset_index(drop=True).mean()
    df_monitoring['datetime'] = df.time[0].values
    monitoring = pd.DataFrame([df_monitoring])
    monitoring.to_csv(file_save + file_name + '_monitoring.csv', index=False)

# %%
for file in file_path[[i for i, x in enumerate(file_path) if '20230621' in x][0]:]:
    print(file)
    file_name = file.split('.')[0].split('\\')[-1]
    check_file = file_save + file_name + '_monitoring.csv'
    break
    if os.path.isfile(check_file):
        print('yes')
        continue
    df_status = xr.open_dataset(file, group='status')
    df_monitoring = xr.open_dataset(file, group='monitoring')
    df = xr.open_dataset(file)
    break