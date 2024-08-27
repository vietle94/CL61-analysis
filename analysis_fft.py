import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import pywt
from matplotlib.colors import LogNorm, SymLogNorm
import xarray as xr
myFmt = mdates.DateFormatter('%H:%M')

# %% ############################################################################
file_path_case = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_20230920*.nc")
df_case = xr.open_mfdataset(file_path_case)
df_case = df_case.isel(range=slice(1, None))

# %%
df_case_ = df_case.sel(time=slice(pd.to_datetime('20230920 220000'),
                        pd.to_datetime('20230920 235900')))

ppol_case_ = df_case_.p_pol/(df_case_.range**2)
ppol_case = ppol_case_.mean(dim='time')
ppol_case_std = ppol_case_.std(dim='time')

xpol_case_ = df_case_.x_pol/(df_case_.range**2)
xpol_case = xpol_case_.mean(dim='time')
xpol_case_std = xpol_case_.std(dim='time')

depo_case = xpol_case/ppol_case
depo_case_std = np.abs(depo_case) * np.sqrt((xpol_case_std/xpol_case)**2 + (ppol_case_std/ppol_case)**2)

# %%
df_case_ = df_case_.sel(range=slice(0, 1000))
fft_depo = df_case_.linear_depol_ratio.T
fft_depo = np.fft.fft(fft_depo.values)/fft_depo.size
fft_depo = np.fft.fftshift(fft_depo, axes=-1)

fft_ppol = df_case_.p_pol.T
fft_ppol = np.fft.fft(fft_ppol.values)/fft_ppol.size
fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)

fft_xpol = df_case_.x_pol.T
fft_xpol = np.fft.fft(fft_xpol.values)/fft_xpol.size
fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)

freqx = np.fft.fftfreq(fft_depo.shape[1], d=10)
freqx = np.fft.fftshift(freqx)

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True,
                       sharex='row', sharey=True)
p = ax[0, 0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.p_pol.T, shading='nearest',
                 norm=LogNorm(vmin=1e-7, vmax=1e-5))
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel('ppol')

p = ax[0, 1].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.x_pol.T, shading='nearest',
                 norm=SymLogNorm(linthresh=1e-10, vmin=-1e-8, vmax=1e-8),
                 cmap='RdBu')
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel('xpol')

p = ax[0, 2].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, shading='nearest', 
                 vmin=-0.004, vmax=0.004)
cbar = fig.colorbar(p, ax=ax[0, 2])
cbar.ax.set_ylabel('depo')

p = ax[1, 0].pcolormesh(freqx, df_case_.range,
                 np.abs(fft_ppol), shading='nearest',
                 norm=LogNorm(vmin=1e-13, vmax=1e-8))
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel('FFT_Amplitude/2')

p = ax[1, 1].pcolormesh(freqx, df_case_.range,
                 np.abs(fft_xpol), shading='nearest',
                 norm=LogNorm(vmin=1e-13, vmax=1e-10))
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel('FFT_Amplitude/2')

p = ax[1, 2].pcolormesh(freqx, df_case_.range,
                 np.abs(fft_depo), shading='nearest',
                  norm=LogNorm(vmin=1e-7, vmax=1e-4))
cbar = fig.colorbar(p, ax=ax[1, 2])
cbar.ax.set_ylabel('FFT_Amplitude/2')

for ax_ in ax[0, :]:
    ax_.xaxis.set_major_formatter(myFmt)
    
for ax_ in ax[1, :]:
    ax_.set_xlabel('Frequency (Hz)')
fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Calibration/" + "fft.png",
            dpi=600, bbox_inches='tight')


# %%
df_diag = xr.open_mfdataset(file_path_case, group='monitoring')

# %%
df_ = df_diag.sel(time=slice(pd.to_datetime('20230920 220000'),
                        pd.to_datetime('20230920 235900')))
fft_laser = np.fft.fft(df_.laser_temperature.values)/df_.laser_temperature.size
fft_laser = np.fft.fftshift(fft_laser)
freqx = np.fft.fftfreq(fft_laser.size, d=10)
freqx = np.fft.fftshift(freqx)

# %%
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(freqx, np.abs(fft_laser))
ax[0].grid()
ax[0].set_ylabel('FFT Amplitude/2')

ax[1].plot(df_.time, df_.laser_temperature)
ax[1].grid()
ax[1].xaxis.set_major_formatter(myFmt)
ax[1].set_ylabel('Laser temperature')
fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Calibration/" + "fft_laser_temperature.png",
            dpi=600, bbox_inches='tight')