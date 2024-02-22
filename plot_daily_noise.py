import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
myFmt = mdates.DateFormatter('%H:%M')

# %%
noise = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise.csv")
noise['datetime'] = pd.to_datetime(noise['datetime'])

# %%
file_path = glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Processed/*.nc")
file_path = [x for x in file_path if '2023' in x]

for file in file_path:
    print(file)
    df = xr.open_dataset(file)
    df = df.swap_dims({'profile': 'time'})

    noise_ = noise[(noise['datetime'] > df.time.min().values) &
                   (noise['datetime'] < df.time.max().values)]

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(12, 9),
                           constrained_layout=True)
    p = ax[0, 0].pcolormesh(df['time'].values, df['range'], df['p_pol'].T,
                            norm=LogNorm(vmin=1e-7, vmax=1e-4))
    fig.colorbar(p, ax=ax[0, 0])
    p = ax[0, 1].pcolormesh(df['time'].values, df['range'], df['x_pol'].T,
                            norm=LogNorm(vmin=1e-7, vmax=1e-4))
    fig.colorbar(p, ax=ax[0, 1])

    ax[1, 0].plot(noise_['datetime'], noise_['co_mean'], '.')
    ax[1, 0].set_ylabel('co_mean')
    ax[1, 1].plot(noise_['datetime'], noise_['cross_mean'], '.')
    ax[1, 1].set_ylabel('cross_mean')

    ax[2, 0].plot(noise_['datetime'], noise_['co_std'], '.')
    ax[2, 0].set_ylabel('co_std')

    ax[2, 1].plot(noise_['datetime'], noise_['cross_std'], '.')
    ax[2, 1].set_ylabel('cross_std')

    for ax_ in ax.flatten():
        ax_.xaxis.set_major_formatter(myFmt)

    for ax_ in ax.flatten()[2:]:
        ax_.grid()

    date_save = np.datetime_as_string(df.time.min().values, 'D')
    fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Img/" + date_save + '_noise.png',
                dpi=600)
    plt.close('all')
