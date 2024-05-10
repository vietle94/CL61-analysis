import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.dates as mdates
import pywt
myFmt = mdates.DateFormatter('%H:%M')

def noise_detection(df, wavelet = 'bior1.1'):
    
    # pre processing
    df = df.isel(range=slice(1, None))
    co = df['p_pol']/(df['range']**2)
    cross = df['x_pol']/(df['range']**2)
    profile = co.mean(dim='time')
    
    # wavelet transform
    n_pad = (len(profile) // 2**7 + 3) * 2**7 - len(profile)
    coeff = pywt.swt(np.pad(profile, (n_pad - n_pad // 2, n_pad // 2), 'constant', constant_values=(0, 0)),
                          wavelet, trim_approx=True, level=7)
    minimax_thresh = np.median(np.abs(coeff[1]))/0.6745 * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=minimax_thresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2):len(profile) + (n_pad - n_pad // 2)]
    
    # half the minimax threshold
    df['noise'] = (['range'], filtered < 0.5*minimax_thresh)
    df['p_pol_smooth'] = (['range'], filtered)
    df.attrs['minimax_thresh'] = minimax_thresh
    return df

def plot_calibration(df_, save_path,
                     time_mean_lim=[-1e-13, 1e-13],
                     time_std_lim=[2.5e-14, 4.5e-14],
                     profile_mean_lim=[-1e-13, 1e-13],
                     profile_std_lim=[3e-14, 2e-13]):
    
    # Calculate pure signals
    ppol = df_.p_pol/(df_.range**2)
    xpol = df_.x_pol/(df_.range**2)
    
    # Time series
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
        ax[0, 0].set_ylim(time_mean_lim)
        ax[0, 0].set_ylabel(r"$\mu_{ppol}/range^2$")
        
        ax[1, 0].plot(ppol_24_std.time, ppol_24_std.values, '.')
        ax[1, 0].set_ylim(time_std_lim)
        ax[1, 0].set_ylabel(r"$\sigma_{ppol}/range^2$")
        
        ax[0, 1].plot(xpol_24.time, xpol_24.values, '.')
        ax[0, 1].set_ylim(time_mean_lim)
        ax[0, 1].set_ylabel(r"$\mu_{xpol}/range^2$")
        
        ax[1, 1].plot(xpol_24_std.time, xpol_24_std.values, '.')
        ax[1, 1].set_ylim(time_std_lim)
        ax[1, 1].set_ylabel(r"$\sigma_{xpol}/range^2$")
        for ax_ in ax.flatten():
            ax_.xaxis.set_major_formatter(myFmt)
            ax_.grid()
        fig.suptitle(f"From {hmin} m to {hmax} m")
        fig.savefig(save_path + f'ppol_xpol_{hmin}_{hmax}.png', dpi=600)
        plt.close(fig)
        
        # Profiles
    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True,
                           figsize=(9, 6))
    ax[0, 0].plot(ppol.mean(dim='time'), ppol.range)
    ax[0, 0].set_xlim(profile_mean_lim)
    ax[0, 0].set_xlabel(r"$\mu_{ppol}/range^2$")
    
    ax[1, 0].plot(ppol.std(dim='time'), ppol.range)
    ax[1, 0].set_xlim(profile_std_lim)
    ax[1, 0].set_xlabel(r"$\sigma_{ppol}/range^2$")
    
    
    ax[0, 1].plot(xpol.mean(dim='time'), xpol.range)
    ax[0, 1].set_xlim(profile_mean_lim)
    ax[0, 1].set_xlabel(r"$\mu_{xpol}/range^2$")
    
    ax[1, 1].plot(xpol.std(dim='time'), xpol.range)
    ax[1, 1].set_xlim(profile_std_lim)
    ax[1, 1].set_xlabel(r"$\sigma_{xpol}/range^2$")
    
    for ax_ in ax.flatten():
        ax_.grid()
    fig.savefig(save_path + 'ppol_xpol_profile.png', dpi=600)
    
    ax[0, 0].set_ylim([0, 1000])
    fig.savefig(save_path + 'ppol_xpol_profile_near.png', dpi=600)
    
    plt.close(fig)