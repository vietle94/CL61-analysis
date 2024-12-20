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
from scipy import signal
myFmt = mdates.DateFormatter('%H:%M\n%Y-%m-%d')

# %%
calibration = pd.read_csv(r"G:\CloudnetData\Calibration\Kenttarova\calibration.txt")
calibration['t1'] = pd.to_datetime(calibration['t1'])
calibration['t2'] = pd.to_datetime(calibration['t2'])

# %%
# pm_all = pd.DataFrame()
# pms_all = pd.DataFrame()
# xm_all = pd.DataFrame()
# xms_all = pd.DataFrame()

for i, row in calibration.iterrows():
    t1 = row['t1']
    t2 = row['t2']
    file_path = glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Raw/" + row['file'])
    
    save_name = t1.strftime("%Y%m%d")
    print(save_name)
    df = xr.open_mfdataset(file_path)
    df = df.isel(range=slice(1, None))
    break
# %%  
    df_ = df.sel(time=slice(t1 - pd.Timedelta('2h'),
                            t2 + pd.Timedelta('2h')))
    
    fig, ax = plt.subplots(4, 2, figsize=(16, 9), sharex='col', sharey='row',
                           constrained_layout=True)
    
    p = ax[0, 0].pcolormesh(df_.time, df_.range, df_.beta_att.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel(r"$\beta'$")
    
    p = ax[1, 0].pcolormesh(df_.time, df_.range, df_.p_pol.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-5))
    cbar = fig.colorbar(p, ax=ax[1, 0])
    cbar.ax.set_ylabel(r"ppol")
    
    p = ax[2, 0].pcolormesh(df_.time, df_.range, df_.x_pol.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-5))
    cbar = fig.colorbar(p, ax=ax[2, 0])
    cbar.ax.set_ylabel(r"xpol")
    
    p = ax[3, 0].pcolormesh(df_.time, df_.range, df_.linear_depol_ratio.T,
                  vmin=0, vmax=0.5)
    cbar = fig.colorbar(p, ax=ax[3, 0])
    cbar.ax.set_ylabel(r"$\delta'$")
    
    df_ = df.sel(time=slice(t1, t2))
    
    p = ax[0, 1].pcolormesh(df_.time, df_.range, df_.beta_att.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-4))
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel(r"$\beta'$")
    
    p = ax[1, 1].pcolormesh(df_.time, df_.range, df_.p_pol.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-5))
    cbar = fig.colorbar(p, ax=ax[1, 1])
    cbar.ax.set_ylabel(r"ppol")
    
    p = ax[2, 1].pcolormesh(df_.time, df_.range, df_.x_pol.T,
                  norm=LogNorm(vmin=1e-7, vmax=1e-5))
    cbar = fig.colorbar(p, ax=ax[2, 1])
    cbar.ax.set_ylabel(r"xpol")
    
    p = ax[3, 1].pcolormesh(df_.time, df_.range, df_.linear_depol_ratio.T,
                  vmin=0, vmax=0.5)
    cbar = fig.colorbar(p, ax=ax[3, 1])
    cbar.ax.set_ylabel(r"$\delta'$")
    
    for ax_ in ax.flatten():
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        
    for ax_ in ax[:, 0]:
        ax_.set_ylabel('Height a.g.l [m]')
    fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
                save_name + "overview.png", bbox_inches='tight', dpi=600)
    
# %%   

    # ppol = df_.p_pol/(df_.range**2)
    # xpol = df_.x_pol/(df_.range**2)
    
    # pm = ppol.mean(dim='time')
    # pms = ppol.std(dim='time')
    # fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    # ax[0, 0].plot(pm, ppol.range)
    # ax[0, 0].fill_betweenx(ppol.range, pm - pms, pm + pms, alpha=0.3)
    # ax[0, 0].set_xlim(-1e-13, 1e-13)
    # ax[0, 0].grid()  
    # ax[0, 0].set_xlabel(r"$ppol/range^2$")
    
    # ax[1, 0].plot(pm, ppol.range)
    # ax[1, 0].fill_betweenx(ppol.range, pm - pms, pm + pms, alpha=0.3)
    # ax[1, 0].set_xlim(-1e-13, 2e-13)
    # ax[1, 0].set_ylim([0, 500])
    # ax[1, 0].grid()
    # ax[1, 0].set_xlabel(r"$ppol/range^2$")
    
    # xm = xpol.mean(dim='time')
    # xms = xpol.std(dim='time')
    # ax[0, 1].plot(xm, xpol.range)
    # ax[0, 1].fill_betweenx(xpol.range, xm - xms, xm + xms, alpha=0.3)
    # ax[0, 1].set_xlim(-1e-13, 1e-13)
    # ax[0, 1].grid()  
    # ax[0, 1].set_xlabel(r"$xpol/range^2$")
    
    # ax[1, 1].plot(xm, ppol.range)
    # ax[1, 1].fill_betweenx(xpol.range, xm - xms, xm + xms, alpha=0.3)
    # ax[1, 1].set_xlim(-1e-13, 2e-13)
    # ax[1, 1].set_ylim([0, 500])
    # ax[1, 1].grid()  
    # ax[1, 1].set_xlabel(r"$xpol/range^2$")
    # for ax_ in ax.flatten():
    #     ax_.set_ylabel('Height a.g.l [m]')
    # fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
    #             save_name + "cal_mean.png", bbox_inches='tight', dpi=600)
    plt.close('all')
    df_diag = xr.open_mfdataset(file_path, group='monitoring')

    df_diag_ = df_diag.sel(time=slice(t1, t2))
    
    # Merge with diag
    df_diag_ = df_diag_.reindex(time=df_.time.values, method='nearest', tolerance='8s')
    df_merge = df_diag_.merge(df_)

    df_merge = df_merge[['laser_temperature', 'p_pol', 'x_pol', 'linear_depol_ratio',
                   'internal_temperature', 'transmitter_enclosure_temperature']]
    df_merge.to_netcdf(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/" + save_name + ".nc")
        

# %%
fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
for col in pm_all:
    print(col)
    ax[0, 0].plot(pm_all[col], ppol.range)
    ax[0, 0].fill_betweenx(ppol.range, pm_all[col] - pms_all[col],
                           pm_all[col] + pms_all[col], alpha=0.3, label=col)
    ax[0, 0].set_xlim(-1e-13, 1e-13)
    ax[0, 0].grid()  
    ax[0, 0].set_xlabel(r"$ppol/range^2$")
    
    ax[1, 0].plot(pm_all[col], ppol.range)
    ax[1, 0].fill_betweenx(ppol.range, pm_all[col] - pms_all[col],
                           pm_all[col] + pms_all[col], alpha=0.3, label=col)
    ax[1, 0].set_xlim(-1e-13, 2e-13)
    ax[1, 0].set_ylim([0, 500])
    ax[1, 0].grid()
    ax[1, 0].set_xlabel(r"$ppol/range^2$")

    ax[0, 1].plot(xm_all[col], xpol.range)
    ax[0, 1].fill_betweenx(xpol.range, xm_all[col] - xms_all[col],
                           xm_all[col] + xms_all[col], alpha=0.3, label=col)
    ax[0, 1].set_xlim(-1e-13, 1e-13)
    ax[0, 1].grid()  
    ax[0, 1].set_xlabel(r"$xpol/range^2$")
    
    ax[1, 1].plot(xm_all[col], ppol.range)
    ax[1, 1].fill_betweenx(xpol.range, xm_all[col] - xms_all[col],
                           xm_all[col] + xms_all[col], alpha=0.3, label=col)
    ax[1, 1].set_xlim(-1e-13, 2e-13)
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].grid()  
    ax[1, 1].set_xlabel(r"$xpol/range^2$")
for ax_ in ax.flatten():
    ax_.set_ylabel('Height a.g.l [m]')
    ax_.legend()
fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
            "all_cal_mean.png", bbox_inches='tight', dpi=600)

# %%######################################3
# Laser temperature
file_path = glob.glob(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/*.nc")
for file in file_path:

    df = xr.open_dataset(file)
    df['ppol_r'] = df.p_pol/(df.range**2)
    df['xpol_r'] = df.x_pol/(df.range**2)
    
    df_gr = df.groupby_bins("laser_temperature", np.arange(18, 24))
    df_mean = df_gr.mean(dim='time')
    df_std = df_gr.std(dim='time')
    

    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for t in df_mean.laser_temperature_bins:
        t_mean = df_mean.sel(laser_temperature_bins=t)
        t_std = df_std.sel(laser_temperature_bins=t)
        ax[0, 0].plot(t_mean['ppol_r'], t_mean.range)
        ax[0, 0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
                               t_mean['ppol_r'] + t_std['ppol_r'],
                               alpha=0.3, label=t.values)
        ax[0, 0].set_xlim(-2e-13, 1.5e-13)
        ax[0, 0].grid()  
        ax[0, 0].set_xlabel(r"$ppol/range^2$")
        
        ax[1, 0].plot(t_mean['ppol_r'], t_mean.range)
        ax[1, 0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
                               t_mean['ppol_r'] + t_std['ppol_r'],
                               alpha=0.3, label=t.values)
        ax[1, 0].set_xlim(-2e-13, 1.5e-13)
        ax[1, 0].grid()  
        ax[1, 0].set_xlabel(r"$ppol/range^2$")
        ax[1, 0].set_ylim([0, 500])
        
        ax[0, 1].plot(t_mean['xpol_r'], t_mean.range)
        ax[0, 1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
                               t_mean['xpol_r'] + t_std['xpol_r'],
                               alpha=0.3, label=t.values)
        ax[0, 1].set_xlim(-2e-13, 1.5e-13)
        ax[0, 1].grid()  
        ax[0, 1].set_xlabel(r"$xpol/range^2$")
        
        ax[1, 1].plot(t_mean['xpol_r'], t_mean.range)
        ax[1, 1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
                               t_mean['xpol_r'] + t_std['xpol_r'],
                               alpha=0.3, label=t.values)
        ax[1, 1].set_xlim(-2e-13, 1.5e-13)
        ax[1, 1].grid()  
        ax[1, 1].set_xlabel(r"$xpol/range^2$")
        ax[1, 1].set_ylim([0, 500])
    
    for ax_ in ax.flatten():
        ax_.legend()
    break
    fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
                file[-11:-3] + "laserT.png", bbox_inches='tight', dpi=600)
        
# %%######################################3
# Laser temperature near range
file_path = glob.glob(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/*.nc")
for file in file_path:

    df = xr.open_dataset(file)
    df['ppol_r'] = df.p_pol/(df.range**2)
    df['xpol_r'] = df.x_pol/(df.range**2)
    
    df_gr = df.groupby_bins("laser_temperature", np.arange(18, 24))
    df_mean = df_gr.mean(dim='time')
    df_std = df_gr.std(dim='time')
    
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True,
                           sharex=True, sharey=True)
    for t, myc in zip(df_mean.laser_temperature_bins):
        t_mean = df_mean.sel(laser_temperature_bins=t)
        t_std = df_std.sel(laser_temperature_bins=t)
        ax[0].plot(np.abs(t_mean['ppol_r']), t_mean.range, label=t.values)
        # ax[0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
        #                        t_mean['ppol_r'] + t_std['ppol_r'],
        #                        alpha=0.3)
        # ax[0].set_xlim(-2e-13, 1.5e-13)
        ax[0].grid()  
        ax[0].set_xlabel(r"$ppol/range^2$")
        ax[0].set_ylim([0, 100])
        ax[0].set_ylabel('Height a.g.l [m]')
                
        ax[1].plot(np.abs(t_mean['xpol_r']), t_mean.range, label=t.values)
        # ax[1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
        #                        t_mean['xpol_r'] + t_std['xpol_r'],
        #                        alpha=0.3)
        ax[1].set_xlim(1e-15, 1e-6)
        ax[1].grid()  
        ax[1].set_xlabel(r"$xpol/range^2$")
        ax[1].set_ylim([0, 100])
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        
    for ax_ in ax.flatten():
        ax_.legend()
    fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
                file[-11:-3] + "laserT100shifted.png", bbox_inches='tight', dpi=600)

# %% ########################################################
# Internal Temperature
file_path = glob.glob(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/*.nc")
df = xr.open_mfdataset(file_path)
df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

temp_range = np.arange(np.floor(df.internal_temperature.values.min()),
          np.ceil(df.internal_temperature.values.max())+1)
my_c = mpl.colormaps['RdBu_r'](np.linspace(0, 1, temp_range.size))
df_gr = df.groupby_bins("internal_temperature", temp_range)
df_mean = df_gr.mean(dim='time', skipna=True)
df_std = df_gr.std(dim='time', skipna=True)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t)
    t_std = df_std.sel(internal_temperature_bins=t)
    ax[0, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[0, 0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
                           t_mean['ppol_r'] + t_std['ppol_r'],
                           alpha=0.3, color=myc)
    ax[0, 0].set_xlim(-2e-13, 1.5e-13)
    ax[0, 0].grid()  
    ax[0, 0].set_xlabel(r"$ppol/range^2$")
    
    ax[1, 0].plot(t_mean['ppol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
                           t_mean['ppol_r'] + t_std['ppol_r'],
                           alpha=0.3, color=myc)
    ax[1, 0].set_xlim(-2e-13, 1.5e-13)
    ax[1, 0].grid()  
    ax[1, 0].set_xlabel(r"$ppol/range^2$")
    ax[1, 0].set_ylim([0, 500])
    
    ax[0, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[0, 1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
                           t_mean['xpol_r'] + t_std['xpol_r'],
                           alpha=0.3, color=myc)
    ax[0, 1].set_xlim(-2e-13, 1.5e-13)
    ax[0, 1].grid()  
    ax[0, 1].set_xlabel(r"$xpol/range^2$")
    
    ax[1, 1].plot(t_mean['xpol_r'], t_mean.range, color=myc, label=t.values)
    ax[1, 1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
                           t_mean['xpol_r'] + t_std['xpol_r'],
                           alpha=0.3, color=myc)
    ax[1, 1].set_xlim(-2e-13, 1.5e-13)
    ax[1, 1].grid()  
    ax[1, 1].set_xlabel(r"$xpol/range^2$")
    ax[1, 1].set_ylim([0, 500])

for ax_ in ax.flatten():
    ax_.legend()
    ax_.grid()
fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
            "internalT.png", bbox_inches='tight', dpi=600)
    
# %% ########################################################
# Internal Temperature near range
file_path = glob.glob(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/*.nc")
for file in file_path:
    df = xr.open_mfdataset(file)
    df['ppol_r'] = df.p_pol/(df.range**2)
    df['xpol_r'] = df.x_pol/(df.range**2)
    
    temp_range = np.arange(np.floor(df.internal_temperature.values.min()),
              np.ceil(df.internal_temperature.values.max())+1)
    my_c = mpl.colormaps['RdBu_r'](np.linspace(0, 1, temp_range.size))
    df_gr = df.groupby_bins("internal_temperature", temp_range)
    df_mean = df_gr.mean(dim='time', skipna=True)
    df_std = df_gr.std(dim='time', skipna=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True,
                           sharex=True, sharey=True)
    for t, myc in zip(df_mean.internal_temperature_bins, my_c):
        t_mean = df_mean.sel(internal_temperature_bins=t)
        t_std = df_std.sel(internal_temperature_bins=t)
        ax[0].plot(np.abs(t_mean['ppol_r']), t_mean.range, color=myc, label=t.values)
        # ax[0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
        #                        t_mean['ppol_r'] + t_std['ppol_r'],
        #                        alpha=0.3, color=myc)
        # ax[0].set_xlim(-2e-13, 1.5e-13)
        ax[0].set_xlabel(r"$ppol/range^2$")
        
        ax[1].plot(np.abs(t_mean['xpol_r']), t_mean.range, color=myc, label=t.values)
        # ax[1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
        #                        t_mean['xpol_r'] + t_std['xpol_r'],
        #                        alpha=0.3, color=myc)
        ax[1].set_xlabel(r"$xpol/range^2$")
        
        # ax[0].set_xlim(-2e-13, 1.5e-13)
        ax[0].set_xscale('log')
        ax[0].set_ylim([0, 100])
      
    for ax_ in ax.flatten():
        ax_.legend()
        ax_.grid()
    
    
    fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
                file[-11:-3] + "internalT100.png", bbox_inches='tight', dpi=600)

# %% ###############################
# Laser temperature vs xpol
fig, ax = plt.subplots(5, 2, figsize=(16, 9), constrained_layout=True)
df_ = df.sel(time=slice(pd.to_datetime('2023-09-26T12:15:00'),
            pd.to_datetime('2023-09-26T13:15:00')))
for i, ax_ in enumerate(ax.flatten()):
    ax2 = ax_.twinx()
    ax2.plot(df_['time'], df_['laser_temperature'], label='Laser T', c='tab:orange')
    ax_.plot(df_['time'], df_['xpol_r'].isel(range=i))
    ax_.grid()
    ax_.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_.set_title(f"range #{i+1}", c='tab:blue')
fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
            save_name + "laser_temp_xpol.png", bbox_inches='tight', dpi=600)

# %%
fig, ax = plt.subplots()
ax.plot(df.time, df.internal_temperature, '.')
ax.plot(weather.datetime, weather['Air temperature [°C]'], '.')
ax.grid()
ax.set_xlim(pd.to_datetime('2024-03-04T00:00:00'),
            pd.to_datetime('2024-03-07T00:00:00'))
ax.set_ylim([-10, 20])

# %%
fig, ax = plt.subplots()
ax.hist(df.internal_temperature)

# %%
weather = pd.read_csv(glob.glob(r"G:\CloudnetData\Calibration\Kenttarova/*.csv")[0])
weather['Time [UTC]'] = weather['Time [UTC]'] + ':00'
weather['datetime'] = pd.to_datetime(weather[['Year', 'Month', 'Day']])
weather['datetime'] = pd.to_datetime(weather['datetime'].astype(str) + ' ' + weather['Time [UTC]'])

# %%
df_10min = df.resample(time='10min').mean()
df_10min = df_10min.dropna(dim='time')
itemp = df_10min.internal_temperature.to_series()
itemp = itemp.to_frame().reset_index()

# %%
weather['time'] = weather['datetime']
weather = weather[['time', 'Air temperature [°C]']]
weather = weather.rename({'Air temperature [°C]': 'AirT'}, axis=1)
iweather = itemp.merge(weather)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(iweather['internal_temperature'], iweather['AirT'], '.')
ax.set_box_aspect(1)
ax.grid()
ax.plot(np.arange(-5, 30),np.arange(-5, 30))
ax.set_xlabel('Internal T')
ax.set_ylabel('Air T')

# %%
file_path = glob.glob(r"G:\CloudnetData\Calibration\Kenttarova\Summary\Data_merged/*.nc")
for file in file_path:
    df = xr.open_dataset(file)
    save_name = df.time.isel(time=0).values.astype(str)[:10]
    df_ = df.sel(range=slice(0, 500))
    
    fft_ppol = df_.p_pol.T
    fft_ppol = np.fft.fft(fft_ppol.values)/fft_ppol.size
    fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)
    
    fft_xpol = df_.x_pol.T
    fft_xpol = np.fft.fft(fft_xpol.values)/fft_xpol.size
    fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)
    
    freqx = np.fft.fftfreq(fft_ppol.shape[1], d=10)
    freqx = np.fft.fftshift(freqx)
    
    fig, ax = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True,
                            sharex='col')
    p = ax[0, 0].pcolormesh(df_.time, df_.range,
                      df_.p_pol.T, shading='nearest',
                      norm=LogNorm(vmin=1e-7, vmax=1e-5))
    cbar = fig.colorbar(p, ax=ax[0, 0])
    cbar.ax.set_ylabel('ppol')
    
    
    p = ax[1, 0].pcolormesh(df_.time, df_.range,
                      df_.x_pol.T, shading='nearest',
                      norm=SymLogNorm(linthresh=1e-10, vmin=-1e-8, vmax=1e-8),
                      cmap='RdBu')
    cbar = fig.colorbar(p, ax=ax[1, 0])
    cbar.ax.set_ylabel('xpol')
    
    ax[2, 0].plot(df_.time, df_.laser_temperature)
    ax[2, 0].set_ylabel('Laser T')
    ax[2, 0].grid()
    
    p = ax[0, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_ppol), shading='nearest',
                      norm=LogNorm(vmin=1e-13, vmax=1e-8))
    cbar = fig.colorbar(p, ax=ax[0, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    p = ax[1, 1].pcolormesh(freqx, df_.range,
                      np.abs(fft_xpol), shading='nearest',
                      norm=LogNorm(vmin=1e-13, vmax=1e-10))
    cbar = fig.colorbar(p, ax=ax[1, 1])
    cbar.ax.set_ylabel('FFT_Amplitude/2')
    
    fft_laser = np.fft.fft(df_.laser_temperature.values)/df_.laser_temperature.size
    fft_laser = np.fft.fftshift(fft_laser)
    freqx = np.fft.fftfreq(fft_laser.size, d=10)
    freqx = np.fft.fftshift(freqx)
    
    ax[2, 1].plot(freqx, np.abs(fft_laser))
    ax[2, 1].grid()
    ax[2, 1].set_ylabel('FFT Amplitude/2')
    
    for ax_ in ax[:, 0]:
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        
    
    ax[-1, -1].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_xlim(df_.time.values.min(),
                df_.time.values.min() + pd.Timedelta('1h'))
        
    fig.savefig(r"G:\CloudnetData\Calibration\Kenttarova\Summary/" + \
                save_name + "fft_laserT.png", bbox_inches='tight', dpi=600)

# %%
for t, xpol_mean, xpol_std in zip(df_mean['internal_temperature_bins'].values,
                 df_mean.sel(range=9000, method='nearest')['xpol_r'].values,
                 df_std.sel(range=9000, method='nearest')['xpol_r'].values):
    print(t, xpol_mean, xpol_std)
    
# %%
summary_9000 = pd.DataFrame({})
summary_9000['internal_temp_range'] = df_mean['internal_temperature_bins'].values
summary_9000['xpol_mean'] = df_mean.sel(range=9000, method='nearest')['xpol_r'].values
summary_9000['xpol_std'] = df_std.sel(range=9000, method='nearest')['xpol_r'].values
summary_9000['ppol_mean'] = df_mean.sel(range=9000, method='nearest')['ppol_r'].values
summary_9000['ppol_std'] = df_std.sel(range=9000, method='nearest')['ppol_r'].values

# %%
df = xr.open_dataset(file_path[0])
df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)
# df = df.sel(time=slice(df.time.values.min(),
#             df.time.values.min() + pd.Timedelta('1h')))

# %%
fig, ax = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
ax[0].plot(df.time[:-5], df.isel(range=1).ppol_r[:-5])
ax[1].plot(df.time[:-5], df.laser_temperature[:-5])
# ax[0].plot(df.time[:-5], df.isel(range=1).ppol_r[:-5])
# ax[1].plot(df.time[:-5], df.laser_temperature[5:])
# ax[0].plot(df.time[:-8], df.isel(range=1).ppol_r[8:])
# ax[1].plot(df.time[:-8], df.laser_temperature[:-8])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlim([df.time.values.min(),
            df.time.values.min() + pd.Timedelta('30min')])
for ax_ in ax.flatten():
    ax_.grid()
    
# %%
x = df.isel(range=2).xpol_r.values
y = df.laser_temperature.values
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
lag = lags[np.argmax(abs(correlation))]
lag

# %%
fig, ax = plt.subplots()
ax.plot(lags, correlation)
ax.grid()

# %%
df = xr.open_dataset(file_path[0])
df['laser_temperature'] = df['laser_temperature'].shift(time=8)


df['ppol_r'] = df.p_pol/(df.range**2)
df['xpol_r'] = df.x_pol/(df.range**2)

df_gr = df.groupby_bins("laser_temperature", np.arange(18, 24))
df_mean = df_gr.mean(dim='time')
df_std = df_gr.std(dim='time')


fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True,
                       sharex=True, sharey=True)
for t in df_mean.laser_temperature_bins:
    t_mean = df_mean.sel(laser_temperature_bins=t)
    t_std = df_std.sel(laser_temperature_bins=t)
    ax[0].plot(np.abs(t_mean['ppol_r']), t_mean.range, label=t.values)
    # ax[0].fill_betweenx(t_mean.range, t_mean['ppol_r'] - t_std['ppol_r'],
    #                        t_mean['ppol_r'] + t_std['ppol_r'],
    #                        alpha=0.3)
    # ax[0].set_xlim(-2e-13, 1.5e-13)
    ax[0].grid()  
    ax[0].set_xlabel(r"$ppol/range^2$")
    ax[0].set_ylim([0, 100])
    ax[0].set_ylabel('Height a.g.l [m]')
            
    ax[1].plot(np.abs(t_mean['xpol_r']), t_mean.range, label=t.values)
    # ax[1].fill_betweenx(t_mean.range, t_mean['xpol_r'] - t_std['xpol_r'],
    #                        t_mean['xpol_r'] + t_std['xpol_r'],
    #                        alpha=0.3)
    ax[1].set_xlim(1e-15, 1e-6)
    ax[1].grid()  
    ax[1].set_xlabel(r"$xpol/range^2$")
    ax[1].set_ylim([0, 100])
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    
for ax_ in ax.flatten():
    ax_.legend()

