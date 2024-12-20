import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:%M')

# %%
noise = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise.csv")
noise['datetime'] = pd.to_datetime(noise['datetime'], format='mixed')
noise = noise[noise['datetime'] > '20230622']

# %%
fig, ax = plt.subplots(
    4, 2, constrained_layout=True, figsize=(16, 9),
    sharex=True, sharey='row')
grp = noise.groupby(noise['range'])

for height in ['(0, 2000]', '(2000, 4000]', '(4000, 6000]']:
    grp_height = grp.get_group(height)
    ax[0, 0].scatter(grp_height['datetime'], grp_height['co_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[0, 0].set_ylabel('ppol_mean')
    ax[1, 0].scatter(grp_height['datetime'], grp_height['cross_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[1, 0].set_ylabel('xpol_mean')
    
    ax[2, 0].scatter(grp_height['datetime'], grp_height['co_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[2, 0].set_ylabel('ppol_std')
    
    ax[3, 0].scatter(grp_height['datetime'], grp_height['cross_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[3, 0].set_ylabel('xpol_std')
    
for height in ['(6000, 8000]', '(8000, 10000]', '(10000, 12000]', '(12000, 14000]']:
    grp_height = grp.get_group(height)
    ax[0, 1].scatter(grp_height['datetime'], grp_height['co_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[0, 1].set_ylabel('ppol_mean')
    ax[1, 1].scatter(grp_height['datetime'], grp_height['cross_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[1, 1].set_ylabel('xpol_mean')
    
    ax[2, 1].scatter(grp_height['datetime'], grp_height['co_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[2, 1].set_ylabel('ppol_std')
    
    ax[3, 1].scatter(grp_height['datetime'], grp_height['cross_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[3, 1].set_ylabel('xpol_std')

handles, labels = ax[0, 0].get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: int(t[0][-6:-1])))
fig.legend(handles, labels, ncol=7, loc = "outside lower left",
           markerscale=10)

handles, labels = ax[0, 1].get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: int(t[0][-6:-1])))
fig.legend(handles, labels, ncol=7, loc = "outside lower right",
           markerscale=10)

ax[0, 0].set_ylim([-1e-13, 1e-13])
ax[1, 0].set_ylim([-1e-13, 1e-13])
ax[2, 0].set_ylim([0, 1e-12])
ax[3, 0].set_ylim([0, 1e-12])
for ax_ in ax.flatten():
    ax_.grid()
    ax_.yaxis.set_tick_params(which='both', labelleft=True)
fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise_summary.png", dpi=600)
    
# %%
# fig, ax = plt.subplots(
#     4, 2, constrained_layout=True, figsize=(16, 9),
#     sharex=True, sharey='row')
# for height, grp_height in noise.groupby(noise['range']):

#     ax[0].scatter(grp_height['datetime'], grp_height['co_mean'], alpha=0.3,
#                   label=height, linewidths=0.0, edgecolors=None)
#     ax[0].set_ylabel('ppol_mean')
    
#     ax[1].scatter(grp_height['datetime'], grp_height['cross_mean'], alpha=0.3,
#                   label=height, linewidths=0.0, edgecolors=None)
#     ax[1].set_ylabel('xpol_mean')
    
#     ax[2].scatter(grp_height['datetime'], grp_height['co_std'], alpha=0.3,
#                   label=height, linewidths=0.0, edgecolors=None)
#     ax[2].set_ylabel('ppol_std')
    
#     ax[3].scatter(grp_height['datetime'], grp_height['cross_std'], alpha=0.3,
#                   label=height, linewidths=0.0, edgecolors=None)
#     ax[3].set_ylabel('xpol_std')
    
# handles, labels = ax[3].get_legend_handles_labels()
# labels, handles = zip(*sorted(zip(labels, handles),
#                               key=lambda t: int(t[0][-6:-1])))
# fig.legend(handles, labels, ncol=7, loc = "outside lower center")
# ax[0].set_ylim([-5e-6, 5e-6])
# ax[1].set_ylim([-5e-6, 5e-6])
# ax[2].set_ylim([0, 3e-5])
# ax[3].set_ylim([0, 3e-5])
# for ax_ in ax.flatten():
#     ax_.grid()
    # ax_.xaxis.set_major_formatter(myFmt)
    # ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
# fig.savefig(file_name_save, dpi=600)
# plt.close(fig)

