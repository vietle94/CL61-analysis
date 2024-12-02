import requests
import os
import xarray as xr
import io
import pandas as pd

url = 'https://cloudnet.fmi.fi/api/raw-files'
pr = pd.period_range(start='2023-06-21',end='2024-12-01', freq='D') 

save_path = r'G:\CloudnetData\Kenttarova\CL61\Diag/'
for i in pr:
    idate = i.strftime("%Y-%m-%d")
    print(idate)
    df_save_monitoring = pd.DataFrame({})
    df_save_status = pd.DataFrame({})
    params = {
        'dateFrom': idate,
        'dateTo': idate,
        'site': 'kenttarova',
        'instrument': 'cl61d'
    }
    metadata = requests.get(url, params).json()
    if not metadata:
        continue
    
    for row in metadata:
        if 'live' in row['filename']:
            print(row['filename'])
            if int(row['size']) < 100000:
                continue
            res = requests.get(row['downloadUrl'])
            break

            with io.BytesIO(res.content) as file:
                try:
                    df_monitoring = xr.open_dataset(file, group='monitoring')
                    df_status = xr.open_dataset(file, group='status')
                except OSError:
                    print('Bad file')
                    continue
                monitoring = df_monitoring.to_dataframe().reset_index()
                monitoring = monitoring.rename({'time': 'datetime'}, axis=1)
                
                status = df_status.to_dataframe().reset_index()
                status = status.rename({'time': 'datetime'}, axis=1)
                
                df_save_monitoring = pd.concat([df_save_monitoring, monitoring])
                df_save_status = pd.concat([df_save_status, status])
                     
    print('saving')
    df_save_monitoring.to_csv(save_path + i.strftime("%Y%m%d") + '_monitoring.csv', index=False)
    df_save_status.to_csv(save_path + i.strftime("%Y%m%d") + '_status.csv', index=False)
