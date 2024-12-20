import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.dates as mdates
import netCDF4
import pywt
import glob
import os
import func
myFmt = mdates.DateFormatter('%H:%M')

# %%
file_save = 'G:\CloudnetData\Kenttarova\CL61/Raw_processed/'
diag_save = 'G:\CloudnetData\Kenttarova\CL61/Diag_new/'
files = glob.glob('G:\CloudnetData\Kenttarova\CL61/Raw/' + '*.nc')

# # %%
# df = xr.open_dataset([x for x in files if '20240303_093' in x][0])
# df = df.isel(range=slice(1, None))
# co = df['p_pol']/(df['range']**2)
# cross = df['x_pol']/(df['range']**2)

# # %%
# fig, ax = plt.subplots(1, 3, sharey=True,
#                        figsize=(16, 7))
# ax[0].plot(co.isel(time=0), co.range, '.')
# ax[0].set_xlim([0, 1e-11])
# # ax[0].set_xlim([-2e-13, 1e-12])

# ax[1].plot(cross.isel(time=0), cross.range, '.')
# ax[1].set_xlim([-2e-13, 1e-12])

# ax[2].plot(df['linear_depol_ratio'].isel(time=0), df.range, '.')
# # ax[2].set_xlim([-0.05, 0.05])
# ax[2].set_xlim([-0.05, 0.02])
# ax[0].set_ylim([0, 5000])
# for ax_ in ax.flatten():
#     ax_.grid()
    
# %%
fig, ax = plt.subplots()
ax.plot(df['overlap_function'], df['range'], '.')
ax.set_ylim([0, 500])
ax.grid()

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True, sharex=True)
p = ax[0].pcolormesh(df['time'], df['range'], df['p_pol'].T, shading='nearest',
                      norm=SymLogNorm(linthresh=1e-14, vmin=-1e-12, vmax=1e-12),
                      cmap='RdGy_r')
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(df['time'], df['range'], df['x_pol'].T, shading='nearest',
                      norm=SymLogNorm(linthresh=1e-14, vmin=-1e-12, vmax=1e-12),
                      cmap='RdGy_r')
fig.colorbar(p, ax=ax[1])
p = ax[2].pcolormesh(df['time'], df['range'],
              df['linear_depol_ratio'].T, shading='nearest',
              vmin=-0.02, vmax=0.02)
fig.colorbar(p, ax=ax[2])
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
ax[0].set_ylim([0, 500])

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
p = ax[0].pcolormesh(df['time'], df['range'], df['p_pol'].T, shading='nearest',
                      # norm=LogNorm(vmin=1e-14, vmax=1e-6),
                      norm=SymLogNorm(linthresh=1e-14, vmin=-1e-12, vmax=1e-12))
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(df['time'], df['range'], df['x_pol'].T, shading='nearest',
                      norm=SymLogNorm(linthresh=1e-14, vmin=-1e-12, vmax=1e-12),
                      cmap='RdGy_r')
fig.colorbar(p, ax=ax[1])
p = ax[2].pcolormesh(df['time'], df['range'],
              df['linear_depol_ratio'].T, shading='nearest',
              vmin=-0.02, vmax=0.02)
fig.colorbar(p, ax=ax[2])
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.set_ylim([0, 1000])

# # %%
# df = xr.open_dataset([x for x in files if '20230920_20' in x][0])
# df = df.isel(range=slice(1, None))
# co = df['p_pol']/(df['range']**2)

# # for wavelet in pywt.wavelist():
# for wavelet in ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
#                 'coif', 'db1', 'db2', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5']:
#     profile = co.mean(dim='time')
#     try:
#         n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
#         coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
#                               wavelet, trim_approx=True, level=7)

#     except ValueError:
#         continue
#     uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1]))) # wrong, need to be calculated for all coeff
#     minimax_thresh = np.median(np.abs(coeff[1]))/0.6745 * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
#     # coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
#     coeff[1:] = (pywt.threshold(i, value=minimax_thresh, mode='hard') for i in coeff[1:])
#     filtered = pywt.iswt(coeff, wavelet)
#     filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]
    
#     fig, ax_ = plt.subplots()
#     ax_.plot(profile, df['range'], '.')
#     ax_.plot(filtered, df['range'], '.')
#     ax_.set_xlim([-0.5e-13, 1e-12])
#     ax_.axvline(x=uthresh, c='red')
#     ax_.axvline(x=minimax_thresh, c='blue')
#     ax_.axvline(x=minimax_thresh * 10, c='grey')
#     fig.suptitle(wavelet + '\n' + np.datetime_as_string(df.time[0].values, unit='D'))
#     print(uthresh, minimax_thresh)
    
# %%
wavelet = 'bior1.1'
co = df['p_pol']/(df['range']**2)
profile = co.mean(dim='time')
n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
                      wavelet, trim_approx=True, level=7)
uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1]))) # wrong, need to be calculated for all coeff
minimax_thresh = np.median(np.abs(coeff[1]))/0.6745 * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
# coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
coeff[1:] = (pywt.threshold(i, value=minimax_thresh, mode='hard') for i in coeff[1:])
filtered = pywt.iswt(coeff, wavelet)
filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]

df_noise = df.sel(range=profile[filtered < 0.5*minimax_thresh].range.values)

fig, ax_ = plt.subplots()
ax_.plot(profile, df['range'], '.')
ax_.plot(filtered, df['range'], '.')
# ax_.set_xlim([-2e-13, 2e-8])
ax_.axvline(x=uthresh, c='red')
ax_.axvline(x=minimax_thresh, c='blue')
ax_.axvline(x=minimax_thresh * 10, c='grey')
ax_.grid()
ax_.plot(df_noise['p_pol'].mean(dim='time')/df_noise.range**2, df_noise['range'], '.', c='green')
fig.suptitle(wavelet + '\n' + np.datetime_as_string(df.time[0].values, unit='D'))
print(uthresh, minimax_thresh)
# ax_.set_ylim([0, 2000])
ax_.set_xscale('log')
   
# %%
df = xr.open_dataset([x for x in files if '20230924_18' in x][0])
df = func.noise_detection(df)

# %%
df_ = df.where(df['noise'])
fig, ax = plt.subplots(2, 2, figsize=(9, 4), constrained_layout=True,
                        sharex=True, sharey=True)
p = ax[0, 0].pcolormesh(df['time'], df['range'], df['beta_att'].T, shading='nearest',
                      norm=LogNorm(vmin=1e-7, vmax=1e-4))
fig.colorbar(p, ax=ax[0, 0])
p = ax[0, 1].pcolormesh(df_['time'], df_['range'], df_['beta_att'].T, shading='nearest',
                      norm=LogNorm(vmin=1e-7, vmax=1e-4))
fig.colorbar(p, ax=ax[0, 1])
p = ax[1, 0].pcolormesh(df['time'], df['range'],
              df['linear_depol_ratio'].T, shading='nearest',
              vmin=-0.0005, vmax=0.0005, cmap='RdGy_r')
fig.colorbar(p, ax=ax[1, 0])
p = ax[1, 1].pcolormesh(df['time'], df['range'],
              # df['linear_depol_ratio'].T,
              (df['x_pol']/df['p_pol']).T,
              shading='nearest',
              vmin=0, vmax=0.5)
fig.colorbar(p, ax=ax[1, 1])

for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.set_ylim([0, 1000])

# %%
# np.median(np.abs(coeff[1]))/0.6745 * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
# np.sum(np.isnan(coeff[1]))
# plt.plot(np.isnan(coeff[1]))
# plt.plot(np.isnan(df.isel(time=0)['p_pol']))

# %%
co = (df['p_pol']/(df['range']**2)).mean(dim='time')
cross = (df['x_pol']/(df['range']**2)).mean(dim='time')
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(co, df['range'], '.')
ax[0].plot(filtered, df['range'], '.')
ax[1].plot(cross, df['range'], '.')
# ax[0].set_ylim([0, 10000])
ax[0].axvline(x=uthresh, c='red')
ax[0].axvline(x=minimax_thresh, c='blue')
ax[0].axvline(x=minimax_thresh * 50, c='grey')
# ax.set_xlim(-1e-6, 1e-6)
for ax_ in ax.flatten():
    ax_.grid()
    ax_.set_xscale('log')

# %%
temp = df['p_pol'].where(~(df['p_pol_smooth']<50*df.attrs['minimax_thresh'])).where(df['x_pol'] < 0)

# %%
fig, ax = plt.subplots()
ax.pcolormesh(temp.time, temp.range, temp.T, norm=LogNorm())
ax.set_ylim([0, 2000])

# 