import glob
import numpy as np
import pandas as pd
import xarray as xr
import os

file_path = glob.glob("G:\CloudnetData\Hyytiala\CL61\Raw/*.nc")
file_save = "G:\CloudnetData\Hyytiala\CL61\Diag/"

for file in file_path:
    print(file)
    file_name = file.split('.')[0].split('\\')[-1]
    check_file = file_save + file_name + '_noise.csv'
    if os.path.isfile(check_file):
        print('yes')
        continue
    try:
        df_monitoring = xr.open_dataset(file, group='monitoring')
        df = xr.open_dataset(file)
    except OSError:
        print('Bad file')
        continue
    # df = df.swap_dims({'profile': 'time'})
    df = df.isel(range=slice(25, None))
    co = df['p_pol']/(df['range']**2)
    cross = df['x_pol']/(df['range']**2)

    co_chunks = np.array_split(co.values, 50, axis=1)
    cross_chunks = np.array_split(cross.values, 50, axis=1)

    data = pd.DataFrame({
        'co_mean': [np.mean(chunk) for chunk in co_chunks],
        'co_std': [np.std(chunk) for chunk in co_chunks],
        'co_chunk': [chunk.flatten() for chunk in co_chunks],
        'cross_mean': [np.mean(chunk) for chunk in cross_chunks],
        'cross_std': [np.std(chunk) for chunk in cross_chunks],
        'cross_chunk': [chunk.flatten() for chunk in cross_chunks]})
    data_sort = data.sort_values(by=['co_mean', 'co_std']).reset_index()
    noise = data_sort.loc[:10]

    result = pd.DataFrame({
        'datetime': df.time[0].values,
        'co_mean': [noise['co_chunk'].explode().mean()],
        'co_std': [noise['co_chunk'].explode().std()],
        'cross_mean': [noise['cross_chunk'].explode().mean()],
        'cross_std': [noise['cross_chunk'].explode().std()]
    })
    result.to_csv(file_save + file_name + '_noise.csv', index=False)

    df_monitoring = pd.DataFrame([df_monitoring.attrs])
    df_monitoring['datetime'] = df.time[0].values
    monitoring = df_monitoring
    monitoring.to_csv(file_save + file_name + '_monitoring.csv', index=False)
