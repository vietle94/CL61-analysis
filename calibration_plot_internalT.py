import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
site = "Hyytiala"
plot_lim = {'Vehmasmaki': [[-5e-15, 5e-15], [5e-15, 1e-14], 0.12,
                            [-0.5e-13, 2e-13]],
            'Kenttarova': [[-5e-15, 5e-15], [1e-14, 1e-13], 0.12,
                                        [-2e-13, 2e-13]],
            'Hyytiala': [[-5e-15, 5e-15], [9e-15, 3e-14], 0.12,
                                        [-4e-13, 2e-13]]}

merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"
save_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary/"

# %%
df_signal = xr.open_mfdataset(glob.glob(merged_dir + '*_signal.nc'))
df_diag = xr.open_mfdataset(glob.glob(merged_dir + '*_diag.nc'))

df_diag = df_diag.reindex(time=df_signal.time.values, method='nearest', tolerance='8s')
df_diag = df_diag.dropna(dim='time')

df = df_diag.merge(df_signal, join='inner')

df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

# %%
temp_range = np.arange(np.floor(df.internal_temperature.min(skipna=True).values),
          np.ceil(df.internal_temperature.max(skipna=True)).values+1)
df_gr = df.groupby_bins("internal_temperature", temp_range,
                        labels=temp_range[:-1])
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
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(df_count.internal_temperature_bins,
       df_count.internal_temperature)
ax.set_xlabel("Temperature")
ax.set_ylabel("N")
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid()
fig.savefig(save_dir + "internalT_count.png", dpi=600,
            bbox_inches='tight')

# %% Full range, but smooth 100 gates
fig, ax = plt.subplots(2, 2, figsize=(16, 9),
                       sharex='row', sharey=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t).rolling(range=100, center=True).mean()
    t_std = df_std.sel(internal_temperature_bins=t).rolling(range=100, center=True).mean()
   
    ax[0, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 0].plot(t_std['ppol_r'], t_std.range, color=myc, label=t.values)
        
    ax[0, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 1].plot(t_std['xpol_r'], t_std.range, color=myc, label=t.values)
    
for ax_ in ax.flatten():
    ax_.grid()
    
ax[0, 0].set_xlabel(r"$\mu_{ppol/range^2}$")
ax[1, 0].set_xlabel(r"$\sigma_{ppol/range^2}$")
ax[0, 1].set_xlabel(r"$\mu_{xpol/range^2}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/range^2}$")
ax[0, 0].set_ylabel('Height a.g.l [m]')
ax[1, 0].set_ylabel('Height a.g.l [m]')

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10)
fig.subplots_adjust(bottom=plot_lim[site][2])

ax[0, 0].set_xlim(plot_lim[site][0])
ax[1, 0].set_xlim(plot_lim[site][1])

fig.savefig(save_dir + "internalT_full.png", dpi=600,
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 9),
                       sharex='row', sharey=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t, range=slice(None, 1000))
    t_std = df_std.sel(internal_temperature_bins=t, range=slice(None, 1000))
   
    ax[0, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 0].plot(t_std['ppol_r'], t_std.range, color=myc, label=t.values)
        
    ax[0, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 1].plot(t_std['xpol_r'], t_std.range, color=myc, label=t.values)
    
for ax_ in ax.flatten():
    ax_.grid()
    
ax[0, 0].set_xlabel(r"$\mu_{ppol/range^2}$")
ax[1, 0].set_xlabel(r"$\sigma_{ppol/range^2}$")
ax[0, 1].set_xlabel(r"$\mu_{xpol/range^2}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/range^2}$")
ax[0, 0].set_ylabel('Height a.g.l [m]')
ax[1, 0].set_ylabel('Height a.g.l [m]')

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10)
fig.subplots_adjust(bottom=plot_lim[site][2])

ax[0, 0].set_xlim(plot_lim[site][3])
ax[1, 0].set_xscale('log')

fig.savefig(save_dir + "internalT_1000.png", dpi=600,
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 9),
                       sharex='row', sharey=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t, range=slice(None, 30))
    t_std = df_std.sel(internal_temperature_bins=t, range=slice(None, 30))
   
    ax[0, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 0].plot(t_std['ppol_r'], t_std.range, color=myc, label=t.values)
        
    ax[0, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 1].plot(t_std['xpol_r'], t_std.range, color=myc, label=t.values)
    
for ax_ in ax.flatten():
    ax_.grid()
    
ax[0, 0].set_xlabel(r"$\mu_{ppol/range^2}$")
ax[1, 0].set_xlabel(r"$\sigma_{ppol/range^2}$")
ax[0, 1].set_xlabel(r"$\mu_{xpol/range^2}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/range^2}$")
ax[0, 0].set_ylabel('Height a.g.l [m]')
ax[1, 0].set_ylabel('Height a.g.l [m]')

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10)
fig.subplots_adjust(bottom=plot_lim[site][2])

ax[1, 0].set_xscale('log')
fig.savefig(save_dir + "internalT_30.png", dpi=600,
            bbox_inches='tight')
