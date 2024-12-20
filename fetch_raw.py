import requests
import os
import xarray as xr
import io

url = 'https://cloudnet.fmi.fi/api/raw-files'
params = {
    'dateFrom': '2024-11-10',
    # 'dateTo': '2024-05-29',
    'dateTo': '2024-11-10',
    'site': 'vehmasmaki',
    # 'site': 'hyytiala',
    'instrument': 'cl61d'
}
metadata = requests.get(url, params).json()

for row in metadata:
    # file_path = r'G:\CloudnetData\Hyytiala\CL61\Raw/'+ row['filename']
    file_path = r'G:\CloudnetData\Vehmasmaki\CL61\Raw/'+ row['filename']
    # if os.path.isfile(file_path):
    #     print('Done')
    #     continue
    if 'live' in row['s3key']:
        print(row['filename'])
        res = requests.get(row['downloadUrl'])
        # with open(file_path, 'wb') as f:
            # f.write(res.content)
        df = xr.open_dataset(io.BytesIO(res.content))
        print(df)
