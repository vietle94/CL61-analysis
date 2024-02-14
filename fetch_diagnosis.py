import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import requests
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import netCDF4
myFmt = mdates.DateFormatter('%H:%M')

# %%
file_list = glob.glob(r'D:\CloudnetData\Kenttarova\CL61\Raw/*.nc')
file_list

# %%
df = xr.open_mfdataset(file_list, group='monitoring')

# %%
for var in list(df.data_vars.keys()):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(df['time'], df[var], '.')
    fig.savefig(r'D:\Paper2\Kenttarova\Diag/' + var + '.png',
                bbox_inches='tight', dpi=600)
    plt.close('all')
