import requests
import xarray as xr
import io
import pandas as pd

def fetch_housekeeping(site, start_date, end_date, save_path):
    url = 'https://cloudnet.fmi.fi/api/raw-files'
    pr = pd.period_range(start=start_date,end=end_date, freq='D') 

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        df_save = pd.DataFrame({})
        params = {
            'dateFrom': idate,
            'dateTo': idate,
            'site': site,
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


def fetch_housekeeping_v2(site, start_date, end_date, save_path):
    url = 'https://cloudnet.fmi.fi/api/raw-files'
    pr = pd.period_range(start=start_date,end=end_date, freq='D')  

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        df_save_monitoring = pd.DataFrame({})
        df_save_status = pd.DataFrame({})
        params = {
            'dateFrom': idate,
            'dateTo': idate,
            'site': site,
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
                while True:
                    bad_file=False
                    with io.BytesIO(res.content) as file:
                        try:
                            df_monitoring = xr.open_dataset(file, group='monitoring')
                            df_status = xr.open_dataset(file, group='status')
                            monitoring = df_monitoring.to_dataframe().reset_index()
                            monitoring = monitoring.rename({'time': 'datetime'}, axis=1)
                            
                            status = df_status.to_dataframe().reset_index()
                            status = status.rename({'time': 'datetime'}, axis=1)
                            
                            df_save_monitoring = pd.concat([df_save_monitoring, monitoring])
                            df_save_status = pd.concat([df_save_status, status])
                        except ValueError:
                            continue
                        except OSError:
                            bad_file = True
                            print('Bad file')
                            break
                    break
                if bad_file:
                    continue
                
                        
        print('saving')
        df_save_monitoring.to_csv(save_path + i.strftime("%Y%m%d") + '_monitoring.csv', index=False)
        df_save_status.to_csv(save_path + i.strftime("%Y%m%d") + '_status.csv', index=False)


def fetch_housekeeping_v3(site, start_date, end_date, save_path):
    # For Vehmasmaki 2023
    url = 'https://cloudnet.fmi.fi/api/raw-files'
    pr = pd.period_range(start=start_date,end=end_date, freq='D') 

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        df_save_monitoring = pd.DataFrame({})
        df_save_status = pd.DataFrame({})
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
                        df_monitoring = xr.open_dataset(file, group='monitoring')
                        df_status = xr.open_dataset(file, group='status')
                    except OSError:
                        print('Bad file')
                        continue
                    monitoring = pd.DataFrame([df_monitoring.attrs])
                    monitoring = monitoring.rename({'Timestamp': 'datetime'}, axis=1)
                    monitoring.datetime = monitoring.datetime.astype(float)
                    monitoring['datetime'] = pd.to_datetime(monitoring['datetime'], unit='s')
                    
                    status = pd.DataFrame([df_status.attrs])
                    status = status.rename({'Timestamp': 'datetime'}, axis=1)
                    status.datetime = status.datetime.astype(float)
                    status['datetime'] = pd.to_datetime(status['datetime'], unit='s')
                    
                    df_save_monitoring = pd.concat([df_save_monitoring, monitoring])
                    df_save_status = pd.concat([df_save_status, status])
                        
        print('saving')
        df_save_monitoring.to_csv(save_path + i.strftime("%Y%m%d") + '_monitoring.csv', index=False)
        df_save_status.to_csv(save_path + i.strftime("%Y%m%d") + '_status.csv', index=False)
