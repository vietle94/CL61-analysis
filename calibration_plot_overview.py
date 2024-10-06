import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
site = "Hyytiala"
merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"

# %%
for dir_signal, dir_diag in zip(glob.glob(merged_dir + '*_signal.nc'),
                                glob.glob(merged_dir + '*_diag.nc')):
    df = xr.open_dataset(dir_signal)
    df_diag = xr.open_dataset(dir_diag)
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True,
                           constrained_layout=True)

    p = ax[0, 0].pcolormesh(df.time, df.range, (df.p_pol/(df.range**2)).T,
                  norm=LogNorm(vmin=1e-14, vmax=1e-11), shading='nearest')
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel(r"ppol_r")

    p = ax[0, 1].pcolormesh(df.time, df.range, (df.x_pol/(df.range**2)).T,
                  norm=LogNorm(vmin=1e-14, vmax=1e-11), shading='nearest')
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel(r"xpol_r")

    ax[1, 0].plot(df_diag.time, df_diag.laser_temperature)
    ax[1, 0].set_ylabel('Laser Temperature')

    ax[1, 1].plot(df_diag.time, df_diag.internal_temperature, '.')
    ax[1, 1].set_ylabel('Internal Temperature')

    for ax_ in ax.flatten():
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        ax_.grid()
        
    fig.savefig(r"G:\CloudnetData\Calibration/" + site + "/Summary/" + \
                np.datetime_as_string(df.time.values[0], unit='D') + \
                    "_overview.png")
    plt.close('all')
