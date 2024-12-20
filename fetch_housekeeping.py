import requests
import xarray as xr
import io
import pandas as pd
import netCDF4
url = 'https://cloudnet.fmi.fi/api/raw-files'
pr = pd.period_range(start='2023-01-01',end='2023-12-01', freq='D') 

save_path = r'G:\CloudnetData\Vehmasmaki\CL61\Diag/'
for i in pr:
    idate = i.strftime("%Y-%m-%d")
    print(idate)
    df_save = pd.DataFrame({})
    params = {
        'dateFrom': idate,
        'dateTo': idate,
        'site': 'vehmasmaki',
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
            with io.BytesIO(res.content) as file:
                try:
                    df = xr.open_dataset(file)
                    df_diag = xr.open_dataset(file, group='diagnostics')
                except OSError:
                    print('Bad file')
                    continue
                diag = pd.DataFrame([df_diag.attrs])
                diag['datetime'] = df.time[0].values
                df_save = pd.concat([df_save, diag])
     
    print('saving')
    df_save.to_csv(save_path + i.strftime("%Y%m%d") + '_diag.csv', index=False)
