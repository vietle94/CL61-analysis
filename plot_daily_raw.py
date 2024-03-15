import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import pywt
from matplotlib.colors import LogNorm
import xarray as xr
myFmt = mdates.DateFormatter('%H:%M')

# %%
file_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/live_202403*.nc")
file_save = "G:\CloudnetData\Kenttarova\CL61\Diag/"

# %%
df = xr.open_mfdataset(file_path)
df = df.isel(range=slice(1, None))

# %%
fig, ax = plt.subplots()
ax.plot(df.sel(time=pd.to_datetime('20240303 235900'), method='nearest')['p_pol']/(df['range']**2), df['range'], '.')
ax.set_xlim([-1.5e-13, 1.5e-13])

# %%
# profile = df.sel(time=pd.to_datetime('20240304 003000'), method='nearest')['p_pol']/(df['range']**2)
profile = df.sel(time=pd.to_datetime('20240303 130000'), method='nearest')['p_pol']/(df['range']**2)

# %%
# for wavelet in ['bior3.1', 'coif17', 'db28']: # level 7
# for wavelet in ['sym6', 'bior3.9']: # level 9
# for wavelet in ['bior6.8']: # level 9
# for wavelet in ['bior3.1', 'bior3.9', 'bior6.8', 'coif3', 'db6',
#                 'db7', 'rbio5.5', 'rbio6.8', 'sym5', 'sym6', 'sym7', 'sym8']: # level 9
for wavelet in pywt.wavelist():
    fig, ax = plt.subplots(3, 2)
    for profile, ax_ in zip(
            [df.sel(time=pd.to_datetime('20240303 093500'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240303 091000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 001000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 040000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 200000'), method='nearest')['p_pol']/(df['range']**2),
             df.sel(time=pd.to_datetime('20240304 080000'), method='nearest')['p_pol']/(df['range']**2)], 
            ax.flatten()):
        # profile = df.sel(time=pd.to_datetime('20240303 091000'), method='nearest')['p_pol']/(df['range']**2)
        try:
            n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
            coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
                             wavelet, level=7)
        except ValueError:
            continue
        # coeff = pywt.swt(np.pad(profile, (0, (len(profile) // 2**5 + 3) * 2**5 - len(profile)), 'constant', constant_values=(0, 0)),
        #                  wavelet, level=5)
        uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        filtered = pywt.iswt(coeff, wavelet)
        filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]
        
        ax_.plot(profile, df['range'], '.')
        ax_.plot(filtered, df['range'], '.')
        ax_.set_xlim([-1.5e-13, 1.5e-12])
    fig.suptitle(wavelet)
    

# %%
temp = df.sel(time=slice(pd.to_datetime('20240303 080000'),
                         pd.to_datetime('20240303 100000')))

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(temp.time, temp.range, temp.p_pol.T,
              norm=LogNorm(vmin=1e-8, vmax=1e-5))
fig.colorbar(p, ax=ax)
ax.xaxis.set_major_formatter(myFmt)

# %%
fig, ax = plt.subplots()
ax.plot(profile, df['range'], '.')
ax.plot(filtered, df['range'], '.')
# ax.set_xlim([-1.5e-13, 1.5e-13])
ax.set_xlim([-1.5e-13, 1.5e-12])
# ax.set_xscale('log')
fig.suptitle(wavelet)