import requests

url = 'https://cloudnet.fmi.fi/api/raw-files'
params = {
    'dateFrom': '2022-01-01',
    'dateTo': '2024-02-06',
    'site': 'kenttarova',
    'instrument': 'cl61d'
}
metadata = requests.get(url, params).json()

for row in metadata:
    if 'live' in row['s3key']:
        res = requests.get(row['downloadUrl'])
        with open('D:/CloudnetData/Raw/' + row['filename'], 'wb') as f:
            f.write(res.content)
