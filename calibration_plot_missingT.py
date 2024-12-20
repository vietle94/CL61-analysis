import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
myFmt = mdates.DateFormatter('%Y-%m-%d')

# %%
fig, ax = plt.subplots(figsize=(9, 3))
for site in ["Kenttarova", "Hyytiala", "Vehmasmaki"]:
    merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"
    diag = xr.open_mfdataset(glob.glob(merged_dir + '*_diag.nc'))
    ax.plot(diag.time, diag.internal_temperature, '.', label=site)
ax.set_ylabel("T")
ax.legend()
ax.grid()
ax.xaxis.set_major_formatter(myFmt)
fig.savefig(r"G:\CloudnetData\Calibration/missing_T.png", bbox_inches='tight')
