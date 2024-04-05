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

fig, ax = plt.subplots()
ax.plot(df_case_.time, (df_case_.sel(range=100, method='nearest')['x_pol'])/(df_case_.sel(range=100, method='nearest')['p_pol']))
ax.set_ylim([-0.005, 0])

# %% 1d fft
from scipy.fft import rfft, rfftfreq
dep = (df_case_.sel(range=100, method='nearest')['x_pol'])/(df_case_.sel(range=100, method='nearest')['p_pol'])
yf = rfft(dep.values)
xf = rfftfreq(dep.values.size, 10)

yf = np.abs(yf)/dep.values.size
yf[1:yf.size] = 2*yf[1:yf.size]

fig, ax = plt.subplots()
ax.plot(xf, yf, '.')
# ax.set_ylim([0, 0.001])

# %%
temp = df_case_.linear_depol_ratio.sel(range=slice(0, 400)).T
FS = np.fft.fft2(temp)/temp.size
fshift = np.fft.fftshift(FS)

freqx = np.fft.fftfreq(temp.shape[1], d=10)
freqx = np.fft.fftshift(freqx)
# freqx[freqx!=0] = 1/(freqx[freqx!=0])

freqy = np.fft.fftfreq(temp.shape[0], d=5)
freqy = np.fft.fftshift(freqy)
# freqy[freqy!=0] = 1/(freqy[freqy!=0])

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(freqx,
                   freqy,
                  np.abs(fshift), norm=LogNorm(vmin=1e-5, vmax=1e-3),
                  shading='nearest')
fig.colorbar(p, ax=ax)
ax.set_xticklabels([1, 1/-0.04, 1/-0.02, 'inf', 1/0.02, 1/0.04, 9])
# %%
fig, ax = plt.subplots()
ax.plot(np.abs(fshift)[10, :], '.')
ax.set_ylim([0, 20])

# %%
omg = np.abs(fshift)[10, :]
np.argwhere(omg>17)

# %%
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(19, 6), constrained_layout=True)

p = ax[0].pcolormesh(df_case_.time, df_case_.range,
                 df_case_.linear_depol_ratio.T, vmin=-0.004, vmax=0.004, cmap='RdBu')
fig.colorbar(p, ax=ax[0])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_xlabel('Depo study case')

for ax_ in ax.flatten():
    ax_.grid()
# fig.savefig(file_save + 'study_case.png', dpi=600)
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

# %%
hW, hH = 600, 300
hFreq = 10

# Mesh on the square [0,1)x[0,1)
x = np.linspace( 0, 1, 2*hW+1)     # columns (Width)
y = np.linspace( 0, 1, 2*hH+1)     # rows (Height)

[X,Y] = np.meshgrid(x,y)

# %%
A = np.sin(hFreq*2*np.pi*Y)

plt.imshow(A, cmap = 'gray');
H,W = np.shape(A)

# %%

F = np.fft.fft2(A)/(W*H)                          
F = np.fft.fftshift(F)
P = np.abs(F)
x = np.fft.fftfreq(A.shape[1], 1/A.shape[1])
# x = np.fft.fftfreq(A.shape[1])
x = np.fft.fftshift(x)
y = np.fft.fftfreq(A.shape[0], 1/A.shape[0])
# y = np.fft.fftfreq(A.shape[0])
y = np.fft.fftshift(y)
# %%
fig, ax = plt.subplots()                            
ax.pcolormesh(x, y, P)
ax.set_ylim(-100, 100)

# %%
plt.imshow(P)

# %%
np.fft.fftfreq(temp.shape[1])
np.fft.fftfreq(temp.shape[1], d=10)
