import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import pywt
from matplotlib.colors import LogNorm, SymLogNorm
import xarray as xr
import func
import matplotlib as mpl
myFmt = mdates.DateFormatter('%H:%M\n%Y-%m-%d')

# %%
file_path = glob.glob(r"G:\CloudnetData\Calibration\Hyytiala\Summary\Data_merged/*.nc")
df = xr.open_mfdataset(file_path)
df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

# %%
df['internal_temperature'].bfill(dim='time')

# %%
temp_range = np.arange(np.floor(df.internal_temperature.min(skipna=True).values),
          np.ceil(df.internal_temperature.max(skipna=True)).values+1)
df_gr = df.groupby_bins("internal_temperature", temp_range)
df_mean = df_gr.mean(dim='time', skipna=True)
df_std = df_gr.std(dim='time', skipna=True)
df_count = df_gr.count(dim='time')

df_mean = df_mean.dropna(dim='internal_temperature_bins', how='all')
df_std = df_std.dropna(dim='internal_temperature_bins', how='all')
df_count = df_count.dropna(dim='internal_temperature_bins', how='all')

n1 = df_mean.internal_temperature_bins.size // 2
n2 = df_mean.internal_temperature_bins.size - n1
my_c = mpl.colormaps['RdBu_r'](np.append(np.linspace(0, 0.4, n1),
                                         np.linspace(0.6, 1, n2)))
    
# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 9),
                       sharex='row', sharey=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t)
    t_std = df_std.sel(internal_temperature_bins=t)
   
    ax[0, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 0].plot(t_std['ppol_r'], t_std.range, color=myc, label=t.values)
        
    ax[0, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 1].plot(t_std['xpol_r'], t_std.range, color=myc, label=t.values)
    
for ax_ in ax.flatten():
    # ax_.legend()
    ax_.grid()
    
ax[0, 0].set_xlabel(r"$\mu_{ppol/range^2}$")
ax[1, 0].set_xlabel(r"$\sigma_{ppol/range^2}$")
ax[0, 1].set_xlabel(r"$\mu_{xpol/range^2}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/range^2}$")
ax[0, 0].set_ylabel('Height a.g.l [m]')
ax[1, 0].set_ylabel('Height a.g.l [m]')

ax[0, 0].set_xlim(-2e-13, 1.5e-13)
ax[1, 0].set_xscale('log')
ax[1, 1].set_xscale('log')

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)
fig.subplots_adjust(bottom=0.12)

fig.savefig(r"G:\CloudnetData\Calibration\Hyytiala\Summary/" + \
            "internalT.png", bbox_inches='tight', dpi=600)

ax[0, 0].set_ylim([0, 500])
fig.savefig(r"G:\CloudnetData\Calibration\Hyytiala\Summary/" + \
            "internalT500.png", bbox_inches='tight', dpi=600)

# ax[0, 0].autoscale()
ax[0, 0].set_ylim([0, 30])
ax[0, 0].set_xlim(1e-12, 1e-7)
ax[0, 0].set_xscale('log')
ax[0, 1].set_xscale('log')

fig.savefig(r"G:\CloudnetData\Calibration\Hyytiala\Summary/" + \
            "internalT30.png", bbox_inches='tight', dpi=600)
    