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
myFmt = mdates.DateFormatter('%H:%M')

# %% Kenttarova ######################################
file_path = glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Raw/live_20240304*.nc")
save_path = r"G:\CloudnetData\Calibration\Kenttarova\20240304/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240304 125700')
t2 = pd.to_datetime('20240304 150900')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-3e-14, 3e-14])

# %%
df_monitoring = xr.open_mfdataset(file_path, group='monitoring')
df_monitoring_ = df_monitoring.sel(time=slice(t1, t2))

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(9, 6),
                       constrained_layout=True)
for val, ax_ in zip(['background_radiance', 'internal_temperature',
            'laser_temperature', 'transmitter_enclosure_temperature'],
                    ax.flatten()):
    ax_.plot(df_monitoring_['time'], df_monitoring_[val], '.')
    ax_.set_ylabel(val)
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.grid()
    fig.savefig(save_path + "monitoring.png", dpi=600)
    
# %% Kenttarova ######################################
file_path = glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Raw/live_2024030[5-6]*.nc")
save_path = r"G:\CloudnetData\Kenttarova\CL61\Calibration/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240305 131500')
t2 = pd.to_datetime('20240306 103800')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-3e-14, 3e-14])

# %%
df_monitoring = xr.open_mfdataset(file_path, group='monitoring')
df_monitoring_ = df_monitoring.sel(time=slice(t1, t2))

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(9, 6),
                       constrained_layout=True)
for val, ax_ in zip(['background_radiance', 'internal_temperature',
            'laser_temperature', 'transmitter_enclosure_temperature'],
                    ax.flatten()):
    ax_.plot(df_monitoring_['time'], df_monitoring_[val], '.')
    ax_.set_ylabel(val)
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.grid()
    
# %% Kenttarova #########################################
file_path = glob.glob(r"G:\CloudnetData\Kenttarova\CL61\Raw/live_20230926*.nc")
save_path = r"G:\CloudnetData\Calibration\Kenttarova\20230926/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20230926 111500')
t2 = pd.to_datetime('20230926 161000')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-3e-14, 3e-14])

# %%
df_monitoring = xr.open_mfdataset(file_path, group='monitoring')
df_monitoring_ = df_monitoring.sel(time=slice(t1, t2))

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(9, 6),
                       constrained_layout=True)
for val, ax_ in zip(['background_radiance', 'internal_temperature',
            'laser_temperature', 'transmitter_enclosure_temperature'],
                    ax.flatten()):
    ax_.plot(df_monitoring_['time'], df_monitoring_[val], '.')
    ax_.set_ylabel(val)
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.grid()
    fig.savefig(save_path + "monitoring.png", dpi=600)


# %% Rovaniemi ###################3
file_path = glob.glob(r"G:\CloudnetData\Rovaniemi\20240311/*.nc")
save_path = r"G:\CloudnetData\Calibration\Rovaniemi/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240311 095700')
t2 = pd.to_datetime('20240311 140500')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-5e-15, 5e-15],
                      time_std_lim=[4.5e-15, 7.5e-15],
                      profile_std_lim=[5e-15, 10e-15])

# %% Vehmsmaki ########################3
file_path = glob.glob(r"G:\CloudnetData\Vehmasmaki\20240411/live_20240411*.nc")
save_path = r"G:\CloudnetData\Vehmasmaki/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240411 090000')
t2 = pd.to_datetime('20240411 110000')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-5e-15, 5e-15],
                      time_std_lim=[6.5e-15, 9.5e-15],
                      profile_std_lim=[6e-15, 11e-15])

# %% Vehmsmaki ########################
file_path = glob.glob(r"G:\CloudnetData\Vehmasmaki\20240528/live_2024052*.nc")
save_path = r"G:\CloudnetData\Calibration\Vehmasmaki\20240528/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240528 092000')
t2 = pd.to_datetime('20240529 053000')
df_ = df.sel(time=slice(None, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-5e-15, 5e-15],
                      time_std_lim=[6.5e-15, 9.5e-15],
                      profile_std_lim=[6e-15, 11e-15])

# %% Hyytiala ########################
file_path = glob.glob(r"G:\CloudnetData\Hyytiala\CL61\Raw\*20231103*.nc")
save_path = r"G:\CloudnetData\Calibration\Hyytiala\20231103/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20231103 105000')
t2 = pd.to_datetime('20231103 123000')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-5e-15, 5e-15],
                      time_std_lim=[8e-15, 11.5e-15],
                      profile_std_lim=[8e-15, 11.5e-15])

# %% Hyytiala ########################
file_path = glob.glob(r"G:\CloudnetData\Hyytiala\CL61\Raw\*20240202*.nc")
save_path = r"G:\CloudnetData\Calibration\Hyytiala\20240202/"
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))
t1 = pd.to_datetime('20240202 105000')
t2 = pd.to_datetime('20240202 125000')
df_ = df.sel(time=slice(t1, t2))

func.plot_calibration(df_,
                      save_path=save_path,
                      time_mean_lim=[-15e-15, 15e-15],
                      time_std_lim=[1.4e-14, 2.4e-14],
                      profile_std_lim=[1.4e-14, 2.4e-14])

# %% ##############################
fig, ax = plt.subplots(figsize=(9, 4))
p = ax.pcolormesh(df_.time, df_.range,
              df_.p_pol.T, shading='nearest', cmap='jet',
              norm=LogNorm(vmin=1e-7, vmax=1e-4))
ax.xaxis.set_major_formatter(myFmt)
fig.colorbar(p, ax=ax)

# %%
temp = (df_.p_pol/df_.range**2).std(dim='time').values

# %%
fig, ax = plt.subplots(figsize=(9, 4))
p = ax.pcolormesh(df_.time, df_.range,
              ppol.T, shading='nearest', cmap='RdBu',
              norm=SymLogNorm(linthresh=1e-17, vmin=-2e-14, vmax=2e-14))
ax.xaxis.set_major_formatter(myFmt)
fig.colorbar(p, ax=ax)

# %%
ppol_mean_time = ppol.mean(dim='time')
xpol_mean_time = xpol.mean(dim='time')

# %%
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, constrained_layout=True)
ax[0].plot(ppol_mean_time, ppol_mean_time.range)
ax[1].plot(xpol_mean_time, xpol_mean_time.range)
ax[0].set_xlim([-1e-13, 1e-13])
ax[0].set_ylim([0, 1000])
for ax_ in ax.flatten():
    ax_.grid()

# %%
fig, ax = plt.subplots()
ax.hist(ppol_mean_time.sel(range=slice(3000, 7000)))

# %%
ppol_ref_base = ppol_mean_time.sel(range=slice(3000, 15000))
df_['ppol_ref'] = ppol_mean_time.copy()
df_['ppol_ref'].loc[dict(range=slice(None, 3000))] = np.random.normal(
    ppol_ref_base.mean(),
    ppol_ref_base.std(),
    ppol_mean_time.sel(range=slice(None, 3000)).size)

xpol_ref_base = xpol_mean_time.sel(range=slice(3000, 15000))
df_['xpol_ref'] = xpol_mean_time.copy()
df_['xpol_ref'].loc[dict(range=slice(None, 3000))] = np.random.normal(
    xpol_ref_base.mean(),
    xpol_ref_base.std(),
    xpol_mean_time.sel(range=slice(None, 3000)).size)

# %%
ppol_correction = (ppol_mean_time)/(df_['ppol_ref'])
xpol_correction = (xpol_mean_time)/(df_['xpol_ref'])

# %%
fig, ax = plt.subplots(1, 2)
ax[0].plot(df_['ppol_ref'], df_.range, label='ref')
ax[0].plot(ppol_mean_time, df_.range, label='mean')
ax[1].plot(df_['xpol_ref'], df_.range, label='ref')
ax[1].plot(xpol_mean_time, df_.range, label='mean')

for ax_ in ax.flatten():
    ax_.set_xlim([-1e-13, 1e-13])
    ax_.legend()
    ax_.set_ylim([0, 2000])
    ax_.grid()

# %%
fig, ax = plt.subplots(1, 2)
ax[0].plot(ppol_correction, df.range)
ax[1].plot(xpol_correction, df.range)

# %%
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
ax[0].plot(df['x_pol'].mean(dim='time')/(df['range']**2),
           df['range'])
ax[1].plot(df['x_pol'].mean(dim='time')/(df['range']**2)/xpol_correction,
           df['range'])
ax[0].set_xlim([-0.5e-11, 1e-11])
ax[0].set_ylim([0, 2000])

# %%
df_ = df.sel(time=slice(pd.to_datetime('20240305 000000'),
                        pd.to_datetime('20240307 000000')))

ppol = df_.p_pol/(df_.range**2)
xpol = df_.x_pol/(df_.range**2)

# %%
# fig, ax = plt.subplots()
# p = ax.pcolormesh(df.time, df.range,
#               ppol.T, shading='nearest', cmap='RdBu',
#               norm=SymLogNorm(linthresh=1e-17, vmin=-2e-12, vmax=2e-12))
# ax.xaxis.set_major_formatter(myFmt)
# fig.colorbar(p, ax=ax)

# %%
for hmin, hmax in ((1000, 3000), (3000, 5000), (5000, 7000),
                   (7000, 9000), (9000, 11000), (11000, 13000), (13000, 15000),
                   (15000, 16000)):
    ppol24 = ppol.sel(range=slice(hmin, hmax))
    ppol_24 = ppol24.mean(dim='range')
    ppol_24_std = ppol24.std(dim='range')
    
    xpol24 = xpol.sel(range=slice(hmin, hmax))
    xpol_24 = xpol24.mean(dim='range')
    xpol_24_std = xpol24.std(dim='range')
    
    fig, ax = plt.subplots(2, 2, sharex=True, constrained_layout=True,
                           figsize=(12, 6))
    ax[0, 0].plot(ppol_24.time, ppol_24.values, '.')
    # ax[0, 0].set_ylim([-1e-13, 1e-13])
    ax[0, 0].set_ylim([-5e-15, 5e-15])
    ax[0, 0].set_ylabel(r"$\mu_{ppol}$")
    
    ax[1, 0].plot(ppol_24_std.time, ppol_24_std.values, '.')
    # ax[1, 0].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 0].set_ylim([4.5e-15, 7.5e-15])
    ax[1, 0].set_ylabel(r"$\sigma_{ppol}$")
    
    ax[0, 1].plot(xpol_24.time, xpol_24.values, '.')
    # ax[0, 1].set_ylim([-1e-13, 1e-13])
    ax[0, 1].set_ylim([-5e-15, 5e-15])
    ax[0, 1].set_ylabel(r"$\mu_{xpol}$")
    
    ax[1, 1].plot(xpol_24_std.time, xpol_24_std.values, '.')
    # ax[1, 1].set_ylim([2.5e-14, 4.5e-14])
    ax[1, 1].set_ylim([4.5e-15, 7.5e-15])
    ax[1, 1].set_ylabel(r"$\sigma_{xpol}$")
    for ax_ in ax.flatten():
        ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
                        alpha=0.3)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.grid()
    fig.suptitle(f"From {hmin} m to {hmax} m")
    fig.savefig(file_save + f'ppol_xpol_{hmin}_{hmax}.png', dpi=600)

# %%
ppol_mean_time = ppol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).mean(dim='time')
ppol_std_time = ppol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).std(dim='time')
xpol_mean_time = xpol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).mean(dim='time')
xpol_std_time = xpol.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800'))).std(dim='time')

# %%
fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 6))
ax[0, 0].plot(ppol_mean_time + 1, ppol_mean_time.range, '.')
ax[0, 0].set_xlim([-1e-13, 1e-13])
ax[0, 0].set_xlabel(r"$\mu_{ppol}$")

ax[1, 0].plot(ppol_std_time, ppol_std_time.range, '.')
ax[1, 0].set_xlim([3e-14, 2e-13])
ax[1, 0].set_xlabel(r"$\sigma_{ppol}$")


ax[0, 1].plot(xpol_mean_time, xpol_mean_time.range, '.')
ax[0, 1].set_xlim([-1e-13, 1e-13])
ax[0, 1].set_xlabel(r"$\mu_{xpol}$")

ax[1, 1].plot(xpol_std_time, xpol_std_time.range, '.')
ax[1, 1].set_xlim([3e-14, 2e-13])
ax[1, 1].set_xlabel(r"$\sigma_{xpol}$")

for ax_ in ax.flatten():
    ax_.grid()
# fig.savefig(file_save + 'ppol_xpol_profile.png', dpi=600)

# ax[0, 0].set_ylim([0, 3000])
ax[0, 0].set_ylim([0, 500])
# fig.savefig(file_save + 'ppol_xpol_profile_near.png', dpi=600)

# ax[0, 0].set_xlim([-1e-14, 1e-14])
# ax[0, 1].set_xlim([-1e-14, 1e-14])
# ax[1, 0].set_xlim([3.25e-14, 3.75e-14])
# ax[1, 1].set_xlim([3.25e-14, 3.75e-14])
# fig.savefig(file_save + 'ppol_xpol_profile_nearnear.png', dpi=600)

# %%
fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True,
                       figsize=(9, 6))
ax[0, 0].plot(ppol_mean_time, ppol_mean_time.range, '.')
# ax[0, 0].set_xlim([1-1e-13, 1+1e-13])
ax[0, 0].set_xlabel(r"$\mu_{ppol}$")
ax[0, 0].set_xscale('log')

# %%
df_monitoring = xr.open_mfdataset(file_path, group='monitoring')
df_monitoring_ = df_monitoring.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800')))
# %%
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 9),
                       constrained_layout=True)
for val, ax_ in zip(['background_radiance', 'internal_temperature',
            'laser_temperature', 'transmitter_enclosure_temperature'],
                    ax.flatten()):
    ax_.plot(df_monitoring_['time'], df_monitoring_[val])
    ax_.set_ylabel(val)
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.grid()
    ax_.axvspan(pd.to_datetime('20240305 131100'), pd.to_datetime('20240306 104200'), facecolor='gray',
                    alpha=0.3)
# fig.savefig(file_save + 'monitoring.png', dpi=600)

# %%
df_monitoring_ = df_monitoring.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800')))
df_ = df.sel(time=slice(pd.to_datetime('20240305 131500'), pd.to_datetime('20240306 103800')))
df_calibration = xr.merge([df_.resample(time='1min').mean(),
                      df_monitoring_.resample(time='1min').mean()])

# %%
ppol_cal = df_calibration.p_pol/(df_calibration.range**2)
ppol_cal_mean = ppol_cal.mean(dim='time')
ppol_cal_std = ppol_cal.std(dim='time')

xpol_cal = df_calibration.x_pol/(df_calibration.range**2)
xpol_cal_mean = xpol_cal.mean(dim='time')
xpol_cal_std = xpol_cal.std(dim='time')

depo_cal = xpol_cal_mean/ppol_cal_mean
depo_cal_std = np.abs(depo_cal) * np.sqrt((xpol_cal_std/xpol_cal_mean)**2 + (ppol_cal_std/ppol_cal_mean)**2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True,
                       constrained_layout=True)
p = ax[0].pcolormesh(df_calibration['time'], df_calibration['range'], df_calibration['p_pol'].T,
                     vmin=-1e-13, vmax=1e-13)
fig.colorbar(p, ax=ax[0])
ax[0].set_ylim([0, 500])
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax[0].xaxis.set_major_formatter(myFmt)

ax[1].plot(df_calibration['time'], df_calibration['internal_temperature'])

# %%
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True,
                       sharex=True,
                       figsize=(9, 4))
for lab, grp in df_calibration.groupby_bins("internal_temperature",
                                            [15, 17, 19, 21]):
    print(lab)
    ppol = grp.p_pol/(grp.range**2)
    xpol = grp.x_pol/(grp.range**2)
    
    ax[0].scatter(ppol.mean(dim='time'), ppol.range, s=3, label= "T=" + str(lab))
    ax[1].scatter(xpol.mean(dim='time'), xpol.range, s=3, label= "T=" + str(lab))
    
ax[0].set_xlim([-1e-13, 1e-13])
ax[1].set_xlabel(r"$\mu_{xpol}$")
ax[0].set_xlabel(r"$\mu_{ppol}$")   
for ax_ in ax.flatten():
    ax_.grid()
    ax_.legend()
fig.savefig(file_save + 'ppol_xpol_profile_t.png', dpi=600)

ax[0].set_ylim([0, 500])
ax[0].set_xlim([-1.75e-13, 1e-13])
fig.savefig(file_save + 'ppol_xpol_profile_t_near.png', dpi=600)

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
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(19, 6), constrained_layout=True)

p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlabel('Depo study case')

ax[1].plot(xpol_case/ppol_case, xpol_case.range, '.')
ax[1].set_xlim([-0.004, 0.01])
ax[1].set_ylim([0, 400])
# ax[1].set_ylim([0, 2000])
ax[1].set_xlabel('Averaged depo study case')

ax[2].plot(xpol_mean_time/ppol_mean_time, xpol_mean_time.range, '.')
ax[2].set_xlim([-1, 2])
ax[2].set_xlabel('Averaged depo calibration')

ax[3].plot(ppol_mean_time, ppol_mean_time.range, '.')
ax[3].set_xlim([-1e-13, 1e-13])
ax[3].set_xlabel(r"$\mu_{ppol}$")

ax[4].plot(xpol_mean_time, xpol_mean_time.range, '.')
ax[4].set_xlim([-1e-13, 1e-13])
ax[4].set_xlabel(r"$\mu_{xpol}$")

for ax_ in ax.flatten():
    ax_.grid()
fig.savefig(file_save + 'study_case.png', dpi=600)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True,
                       constrained_layout=True)
p = ax[1].pcolormesh(df_case_['time'], df_case_['range'], df_case_['overlap_function'].T)
fig.colorbar(p, ax=ax[1])
ax[1].set_ylim([0, 500])
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax[1].xaxis.set_major_formatter(myFmt)

ax[0].plot(df_case_['overlap_function'].isel(time=0).values, df_case_.range, '.')
ax[0].set_ylim([0, 500])
ax[0].grid()
fig.savefig(file_save + 'overlap_function.png', dpi=600)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6), constrained_layout=True)

p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlabel('Depo study case')
ax[0].set_ylim([0, 400])

p = ax[1].pcolormesh(df_case_.time, df_case_.range,
                 (((df_case_.x_pol/(df_case_.range**2))/xpol_mean_time)/((df_case_.p_pol/(df_case_.range**2))/ppol_mean_time)).T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[1])
ax[1].xaxis.set_major_formatter(myFmt)
ax[1].set_xlabel('Depo study case corrected')

for ax_ in ax.flatten():
    ax_.grid()
    
fig.savefig(file_save + 'study_case_corrected.png', dpi=600)

# %%
fig, ax = plt.subplots(3, 3, sharey=True, figsize=(16, 9), constrained_layout=True)

p = ax[0, 0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0, 0])
ax[0, 0].xaxis.set_major_formatter(myFmt)
ax[0, 0].set_xlabel(r"$\delta}$")
ax[0, 0].set_ylim([0, 400])

ax[0, 1].errorbar(ppol_case, ppol_case.range, xerr=ppol_case_std, fmt='.')
ax[0, 1].set_xlim([0, 1e-9])
ax[0, 1].set_xlabel(r"$\mu_{ppol}$")

ax[0, 2].errorbar(xpol_case, xpol_case.range, xerr=xpol_case_std, fmt='.')
ax[0, 2].set_xlim([-1e-12, 1e-12])
ax[0, 2].set_xlabel(r"$\mu_{xpol}$")

ax[1, 0].errorbar(depo_case, depo_case.range, xerr=depo_case_std, fmt='.')
ax[1, 0].set_xlim([-0.008, 0.004])
ax[1, 0].set_xlabel(r"$\delta}$")

ax[1, 1].errorbar(ppol_cal_mean, ppol_cal_mean.range, xerr=ppol_cal_std, fmt='.')
ax[1, 1].set_xlim([-1.75e-13, 1e-13])
ax[1, 1].set_xlabel(r"$\mu_{ppolCAL}$")

ax[1, 2].errorbar(xpol_cal_mean, xpol_cal_mean.range, xerr=xpol_cal_std, fmt='.')
ax[1, 2].set_xlim([-1.75e-13, 1e-13])
ax[1, 2].set_xlabel(r"$\mu_{xpolCAL}$")

ax[2, 0].plot((xpol_case/xpol_cal_mean)/(ppol_case/ppol_cal_mean),
              depo_case.range, '.')
ax[2, 0].set_xlim([-0.008, 0.004])
ax[2, 0].set_xlabel(r"$\delta_{corrected}}$")

ax[2, 1].errorbar(depo_cal, depo_cal.range, xerr=depo_cal_std, fmt='.')
ax[2, 1].set_xlim([-4, 4])
ax[2, 1].set_xlabel(r"$\delta CAL}$")

for ax_ in ax.flatten()[1:]:
    ax_.grid()
fig.savefig(file_save + 'study_case2.png', dpi=600)

# %%

fig, ax = plt.subplots()
ax.plot(df_case_.time, (df_case_.sel(range=100, method='nearest')['x_pol'])/(df_case_.sel(range=100, method='nearest')['p_pol']))
ax.set_ylim([-0.005, 0])

# %%
from scipy.fft import rfft, rfftfreq
dep = (df_case_.sel(range=100, method='nearest')['x_pol'])/(df_case_.sel(range=100, method='nearest')['p_pol'])
yf = rfft(dep.values)
xf = rfftfreq(dep.values.size, 10)

yf = np.abs(yf)/dep.values.size
yf[1:yf.size] = 2*yf[1:yf.size]
# %%
fig, ax = plt.subplots()
ax.plot(xf, yf, '.')
# ax.set_ylim([0, 0.001])

# %%
temp = df_case_.linear_depol_ratio.sel(range=slice(0, 400)).T
FS = np.fft.fft2(temp)
fshift = np.fft.fftshift(FS)

# %%
from matplotlib.colors import LogNorm
fig, ax = plt.subplots()
p = ax.pcolormesh((np.abs(FS))/temp.size)
fig.colorbar(p, ax=ax)

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(np.abs(fshift), norm=LogNorm(vmin=1e-1, vmax=1e2))
fig.colorbar(p, ax=ax)

# %%
fig, ax = plt.subplots()
ax.plot(np.abs(fshift)[10, :], '.')
ax.set_ylim([0, 20])

# %%
omg = np.abs(fshift)[10, :]
np.argwhere(omg>17)


# %%
fig, ax = plt.subplots()
ax.plot(yf, '.')
ax.set_ylim([0, 0.001])
ax.set_xlim([0, 100])

# %%
fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_ylim([0, 400])

ax[1].plot(dep.time, dep)
ax[1].set_xlim(right=pd.to_datetime('2023-09-20T22:15:00'))

# %%
(54*(1/10/dep.values.size/2))
1/xf[54]
1/xf[55]
xf[54]
