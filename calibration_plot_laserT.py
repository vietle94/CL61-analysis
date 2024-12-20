import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
from scipy import signal
myFmt = mdates.DateFormatter('%Y\n%m-%d\n%H:%M')

# %%
site = "Vehmasmaki"
plot_lim = {'Vehmasmaki': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                           LogNorm(vmin=1e-15, vmax=1e-12), 0],
            'Kenttarova': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                         LogNorm(vmin=1e-15, vmax=1e-12), 0],
            'Hyytiala': [SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
                         LogNorm(vmin=1e-15, vmax=1e-12), 1]}

merged_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary\Data_merged/"
save_dir = r"G:\CloudnetData\Calibration\/" + site + r"\Summary/"

# %% 
# Old software
#######################################################
df_signal = xr.open_dataset(glob.glob(merged_dir + '*_signal.nc')[0])
df_diag = xr.open_dataset(glob.glob(merged_dir + '*_diag.nc')[0])

df_diag = df_diag.reindex(time=df_signal.time.values, method='nearest', tolerance='8s')
df_diag = df_diag.dropna(dim='time')

df = df_diag.merge(df_signal, join='inner')

df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4),
                       constrained_layout=True)
ax[0].plot(df.time, df.laser_temperature)
ax[0].set_ylabel('Laser T')
ax[1].plot(df.time, df.ppol_r.isel(range=5))
ax[1].set_ylabel('ppol_r')
ax[0].xaxis.set_major_formatter(myFmt)
for ax_ in ax.flatten():
    ax_.grid()
# fig.savefig(save_dir + 'old_software_laser_T.png')

# %%
# New software
#################################################
for signal_path, diag_path in zip(glob.glob(merged_dir + '*_signal.nc')[plot_lim[site][2]:], 
                                  glob.glob(merged_dir + '*_diag.nc')[plot_lim[site][2]:]):
    
    df_signal = xr.open_dataset(signal_path)
    df_diag = xr.open_dataset(diag_path)

    df_diag = df_diag.reindex(time=df_signal.time.values, method='nearest', tolerance='8s')
    df_diag = df_diag.dropna(dim='time')

    df = df_diag.merge(df_signal, join='inner')

    df['ppol_r'] = df.p_pol/(df.range**2)
    df['xpol_r'] = df.x_pol/(df.range**2)
    
    timestep = np.median(df.time.values[1:] - df.time.values[:-1]).astype('timedelta64[s]').astype(np.int32) # for fft
    
    save_name = df.time.isel(time=0).values.astype(str)[:10]
    df_ = df.sel(range=slice(0, 300))
    
    fft_ppol = df_.ppol_r.T
    fft_ppol = np.fft.fft(fft_ppol.values)/fft_ppol.size
    fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)
    
    fft_xpol = df_.xpol_r.T
    fft_xpol = np.fft.fft(fft_xpol.values)/fft_xpol.size
    fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)
    
    freqx = np.fft.fftfreq(fft_ppol.shape[1], d=timestep)
    freqx = np.fft.fftshift(freqx)

    fig, ax = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True,
                            sharex='col')
    
    p = ax[0, 0].pcolormesh(df_.time, df_.range,
                      df_.ppol_r.T, shading='nearest',
                      norm=plot_lim[site][0],
                      cmap='RdBu')
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel('ppol')
    
    
    p = ax[1, 0].pcolormesh(df_.time, df_.range,
                      df_.xpol_r.T, shading='nearest',
                      norm=plot_lim[site][0],
                      cmap='RdBu')
    cbar = fig.colorbar(p, ax=ax[1, 0])
    cbar.ax.set_ylabel('xpol')
    
    ax[2, 0].plot(df_.time, df_.laser_temperature)
    ax[2, 0].set_ylabel('Laser T')
    ax[2, 0].grid()

    p = ax[0, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_ppol), shading='nearest',
                      norm=plot_lim[site][1])
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    p = ax[1, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_xpol), shading='nearest',
                      norm=plot_lim[site][1])
    cbar = fig.colorbar(p, ax=ax[1, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    fft_laser = np.fft.fft(df_.laser_temperature.values)/df_.laser_temperature.size
    fft_laser = np.fft.fftshift(fft_laser)
    freqx = np.fft.fftfreq(fft_laser.size, d=timestep)
    freqx = np.fft.fftshift(freqx)
    
    max_freq = freqx[np.max(np.argsort(np.abs(fft_laser), axis=0)[-3:-1])] # extract the second highest frequency
    ax[2, 1].plot(freqx, np.abs(fft_laser))
    ax[2, 1].grid()
    ax[2, 1].set_ylabel('FFT Amplitude/2')
    ax[2, 1].axvline(x=max_freq,
               color='r', linestyle='--')
    ax[2, 1].annotate(f'Frequency: {max_freq:.2g}Hz\nTime: {1/max_freq:.2f}s',
                      (0.05, 0.8), xycoords='axes fraction',
                color='r')
    
    for ax_ in ax[:, 0]:
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        
    
    ax[-1, -1].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_xlim(df_.time.values.min(),
                df_.time.values.min() + pd.Timedelta('1h'))

    fig.savefig(save_dir + save_name + "_fft_laserT_overview.png",
                bbox_inches='tight', dpi=600)

# %%
# 30m
#################################################
fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey='row', sharex='col')
for signal_path, diag_path, ax in zip(glob.glob(merged_dir + '*_signal.nc'), 
                                  glob.glob(merged_dir + '*_diag.nc'),
# for signal_path, diag_path, ax in zip(glob.glob(merged_dir + '*_signal.nc')[plot_lim[site][2]:], 
#                                   glob.glob(merged_dir + '*_diag.nc')[plot_lim[site][2]:],
                                  axes.T):
    df_signal = xr.open_dataset(signal_path)
    df_diag = xr.open_dataset(diag_path)

    df_diag = df_diag.reindex(time=df_signal.time.values, method='nearest', tolerance='8s')
    df_diag = df_diag.dropna(dim='time')

    df = df_diag.merge(df_signal, join='inner')

    df['ppol_r'] = df.p_pol/(df.range**2)
    df['xpol_r'] = df.x_pol/(df.range**2)
    
    ax[0].plot(df.time, df['ppol_r'].isel(range=10), '.')
    ax[1].plot(df.time, df['internal_temperature'])
    ax[2].plot(df.time, df['laser_temperature'])
    # ax[0].set_yscale('log')
    ax[0].xaxis.set_major_formatter(myFmt)
    for ax_ in ax.flatten():
        ax_.grid()

# %%
fig, ax = plt.subplots()
# ax.plot(df.time, df.internal_temperature)
ax.plot(df.time, df.laser_temperature)

# %%
    
    timestep = np.median(df.time.values[1:] - df.time.values[:-1]).astype('timedelta64[s]').astype(np.int32) # for fft
    
    save_name = df.time.isel(time=0).values.astype(str)[:10]
    df_ = df.sel(range=slice(0, 30))
    
    fft_ppol = df_.ppol_r.T
    fft_ppol = np.fft.fft(fft_ppol.values)/fft_ppol.size
    fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)
    
    fft_xpol = df_.xpol_r.T
    fft_xpol = np.fft.fft(fft_xpol.values)/fft_xpol.size
    fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)
    
    freqx = np.fft.fftfreq(fft_ppol.shape[1], d=timestep)
    freqx = np.fft.fftshift(freqx)

    fig, ax = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True,
                            sharex='col')
    
    p = ax[0, 0].pcolormesh(df_.time, df_.range,
                      df_.ppol_r.T, shading='nearest',
                      norm=plot_lim[site][0],
                      cmap='RdBu')
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel('ppol')
    
    
    p = ax[1, 0].pcolormesh(df_.time, df_.range,
                      df_.xpol_r.T, shading='nearest',
                      norm=plot_lim[site][0],
                      cmap='RdBu')
    cbar = fig.colorbar(p, ax=ax[1, 0])
    cbar.ax.set_ylabel('xpol')
    
    ax[2, 0].plot(df_.time, df_.laser_temperature)
    ax[2, 0].set_ylabel('Laser T')
    ax[2, 0].grid()

    p = ax[0, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_ppol), shading='nearest',
                      norm=plot_lim[site][1])
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    p = ax[1, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_xpol), shading='nearest',
                      norm=plot_lim[site][1])
    cbar = fig.colorbar(p, ax=ax[1, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    fft_laser = np.fft.fft(df_.laser_temperature.values)/df_.laser_temperature.size
    fft_laser = np.fft.fftshift(fft_laser)
    freqx = np.fft.fftfreq(fft_laser.size, d=timestep)
    freqx = np.fft.fftshift(freqx)
    
    max_freq = freqx[np.max(np.argsort(np.abs(fft_laser), axis=0)[-3:-1])] # extract the second highest frequency
    ax[2, 1].plot(freqx, np.abs(fft_laser))
    ax[2, 1].grid()
    ax[2, 1].set_ylabel('FFT Amplitude/2')
    ax[2, 1].axvline(x=max_freq,
               color='r', linestyle='--')
    ax[2, 1].annotate(f'Frequency: {max_freq:.2g}Hz\nTime: {1/max_freq:.2f}s',
                      (0.05, 0.8), xycoords='axes fraction',
                color='r')
    
    for ax_ in ax[:, 0]:
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        
    
    ax[-1, -1].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_xlim(df_.time.values.min(),
                df_.time.values.min() + pd.Timedelta('1h'))
    break
    # fig.savefig(save_dir + save_name + "_fft_laserT_30m.png",
    #             bbox_inches='tight', dpi=600)
    
# %%
fig, ax = plt.subplots()
ax.plot(df.time, df['ppol_r'].isel(range=3))
ax.xaxis.set_major_formatter(myFmt)