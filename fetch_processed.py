import requests
import os

url = 'https://cloudnet.fmi.fi/api/raw-files'
params = {
    'dateFrom': '2023-08-01',
    'dateTo': '2024-02-01',
    'site': 'kenttarova',
    'instrument': 'cl61d'
}
metadata = requests.get(url, params).json()

for row in metadata:
    file_path = 'D:/CloudnetData/Kenttarova/CL61/Processed/' + row['filename']
    if os.path.isfile(file_path):
        continue
    if 'live' in row['s3key']:
        continue
    else:
        print(row['filename'])
        res = requests.get(row['downloadUrl'])
        with open(file_path, 'wb') as f:
            f.write(res.content)
