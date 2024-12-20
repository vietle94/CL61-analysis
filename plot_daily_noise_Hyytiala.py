import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import xarray as xr
from matplotlib.colors import LogNorm
myFmt = mdates.DateFormatter('%H:%M')

# %% Plot daily noise
noise = pd.read_csv(r"G:\CloudnetData\Hyytiala\CL61\Summary/noise.csv")
noise['datetime'] = pd.to_datetime(noise['datetime'], format='mixed')
noise = noise[noise['datetime'] > '20230101'] # why does it take too much time to plot?
# noise = noise[noise['datetime'] > '20230604']
raw_path = glob.glob(r"G:\CloudnetData\Hyytiala\CL61\Raw_processed/*.nc")
print('finnished loading initial data')
for date, grp_date in noise.groupby(noise['datetime'].dt.date):
    print(date.strftime("%Y%m%d"))
    file_name_save = r"G:\CloudnetData\Hyytiala\CL61\Img\Background/" +\
                date.strftime("%Y%m%d") + '_noise.png'
    if os.path.isfile(file_name_save):
        print('yes')
        continue
    print('opening data')
    df = xr.open_mfdataset(
        [x for x in raw_path if date.strftime("%Y%m%d") in x])
    print('plotting')
    fig, ax = plt.subplots(
        4, 2, constrained_layout=True, figsize=(12, 8),
        sharex=True, sharey='row')
    
    p = ax[0, 0].pcolormesh(df['time'], df['range'],
                        df['p_pol'].T,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel('p_pol')
    
    p = ax[0, 1].pcolormesh(df['time'], df['range'],
                        df['x_pol'].T,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel('x_pol')
    
    p = ax[1, 0].pcolormesh(df['time'], df['range'],
                        df['p_pol'].where(df['noise']).T,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[1, 0])
    cbar.ax.set_ylabel('p_pol')
    
    p = ax[1, 1].pcolormesh(df['time'], df['range'],
                        df['x_pol'].where(df['noise']).T,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[1, 1])
    cbar.ax.set_ylabel('x_pol')
    
    for height, grp_height in grp_date.groupby(grp_date['range']):

        ax[2, 0].plot(grp_height['datetime'], grp_height['co_mean'], '.',
                      label=height)
        ax[2, 0].set_ylabel('ppol_mean')
        
        ax[2, 1].plot(grp_height['datetime'], grp_height['cross_mean'], '.',
                      label=height)
        ax[2, 1].set_ylabel('xpol_mean')
        
        ax[3, 0].plot(grp_height['datetime'], grp_height['co_std'], '.',
                      label=height)
        ax[3, 0].set_ylabel('ppol_std')
        
        ax[3, 1].plot(grp_height['datetime'], grp_height['cross_std'], '.',
                      label=height)
        ax[3, 1].set_ylabel('xpol_std')
        
    handles, labels = ax[3, 1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles),
                                  key=lambda t: int(t[0][-6:-1])))
    fig.legend(handles, labels, ncol=7, loc = "outside lower center")
    for ax_ in ax.flatten()[4:]:
        ax_.grid()
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
    fig.savefig(file_name_save, dpi=600)
    plt.close(fig)
