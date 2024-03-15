import requests
import os

url = 'https://cloudnet.fmi.fi/api/raw-files'
params = {
    'dateFrom': '2023-09-01',
    'dateTo': '2024-03-06',
    'site': 'kenttarova',
    'instrument': 'cl61d'
}
metadata = requests.get(url, params).json()

for row in metadata:
    file_path = 'G:/CloudnetData/Kenttarova/CL61/Raw/' + row['filename']
    if os.path.isfile(file_path):
        continue
    if 'live' in row['s3key']:
        print(row['filename'])
        res = requests.get(row['downloadUrl'])
        with open(file_path, 'wb') as f:
            f.write(res.content)
