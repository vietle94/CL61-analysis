import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:%M')

# %%
noise = pd.read_csv(r"G:\CloudnetData\Hyytiala\CL61\Summary/noise.csv")
noise['datetime'] = pd.to_datetime(noise['datetime'], format='mixed')

# %%
# temp = noise.loc[noise['datetime'].between('2024-03-20', '2024-03-21')]

# # %%
# fig, ax = plt.subplots(
#     2, 2, constrained_layout=True, figsize=(12, 8),
#     sharex=True, sharey='row')
# for height, grp_height in temp.groupby(temp['range']):

#     ax[0, 0].plot(grp_height['datetime'], grp_height['co_mean'], '.',
#                   label=height)
#     ax[0, 0].set_ylabel('ppol_mean')
    
#     ax[0, 1].plot(grp_height['datetime'], grp_height['cross_mean'], '.',
#                   label=height)
#     ax[0, 1].set_ylabel('xpol_mean')
    
#     ax[1, 0].plot(grp_height['datetime'], grp_height['co_std'], '.',
#                   label=height)
#     ax[1, 0].set_ylabel('ppol_std')
    
#     ax[1, 1].plot(grp_height['datetime'], grp_height['cross_std'], '.',
#                   label=height)
#     ax[1, 1].set_ylabel('xpol_std')
    
# handles, labels = ax[1, 1].get_legend_handles_labels()
# labels, handles = zip(*sorted(zip(labels, handles),
#                               key=lambda t: int(t[0][-6:-1])))
# ax[1, 0].set_ylim([0, 0.5e-12])
# fig.legend(handles, labels, ncol=7, loc = "outside lower center")
# for ax_ in ax.flatten():
#     ax_.grid()
#     ax_.xaxis.set_major_formatter(myFmt)
#     ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))


# %%

grp = noise.groupby(noise['range'])

for height in ['(0, 2000]', '(2000, 4000]', '(4000, 6000]', '(6000, 8000]', '(8000, 10000]', '(10000, 12000]', '(12000, 14000]']:
    fig, ax = plt.subplots(
        4, 1, constrained_layout=True, figsize=(9, 6),
        sharex=True)
    grp_height = grp.get_group(height)
    ax[0].scatter(grp_height['datetime'], grp_height['co_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[0].set_ylabel('ppol_mean')
    ax[1].scatter(grp_height['datetime'], grp_height['cross_mean'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[1].set_ylabel('xpol_mean')
    
    ax[2].scatter(grp_height['datetime'], grp_height['co_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[2].set_ylabel('ppol_std')
    
    ax[3].scatter(grp_height['datetime'], grp_height['cross_std'], alpha=0.3, s=0.5,
                  label=height, linewidths=0.0, edgecolors=None)
    ax[3].set_ylabel('xpol_std')
    
    fig.suptitle(height)

    ax[0].set_ylim([-1e-13, 1e-13])
    ax[1].set_ylim([-0.5e-13, 0.5e-13])
    ax[2].set_ylim([0, 0.2e-12])
    ax[3].set_ylim([0, 0.2e-12])
    for ax_ in ax.flatten():
        ax_.grid()
        ax_.yaxis.set_tick_params(which='both', labelleft=True)
    # fig.savefig(rf"G:\CloudnetData\Hyytiala\CL61\Summary/noise_summary_{height}.png", dpi=600)
    fig.savefig(rf"G:\CloudnetData\Kenttarova\CL61\Summary/noise_summary_{height}.png", dpi=600)
    
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

# %%
grp = noise.groupby(noise['range'])
fig, ax = plt.subplots(
    7, 2, constrained_layout=True, figsize=(9, 9),
    sharex=True)
for i, height in enumerate(['(0, 2000]', '(2000, 4000]', '(4000, 6000]',
                         '(6000, 8000]', '(8000, 10000]', '(10000, 12000]',
                         '(12000, 14000]']):
    
    grp_height = grp.get_group(height)
    ax[i, 0].hist(grp_height['co_std'], bins=np.linspace(0, 0.6e-13, 30))
    ax[i, 0].set_xlim([0, 0.6e-13])
    
    
    ax[i, 1].hist(grp_height['cross_std'], bins=np.linspace(0, 0.6e-13, 30))
    ax[i, 1].set_xlim([0, 0.6e-13])
    
    for ax_ in ax.flatten():
        ax_.grid()
ax[i, 0].set_xlabel('co_std')  
ax[i, 1].set_xlabel('cross_std')    
    # break
    # ax[1].scatter(grp_height['datetime'], grp_height['cross_mean'], alpha=0.3, s=0.5,
    #               label=height, linewidths=0.0, edgecolors=None)
    # ax[1].set_ylabel('xpol_mean')
    
    # ax[2].scatter(grp_height['datetime'], grp_height['co_std'], alpha=0.3, s=0.5,
    #               label=height, linewidths=0.0, edgecolors=None)
    # ax[2].set_ylabel('ppol_std')
    
    # ax[3].scatter(grp_height['datetime'], grp_height['cross_std'], alpha=0.3, s=0.5,
    #               label=height, linewidths=0.0, edgecolors=None)
    # ax[3].set_ylabel('xpol_std')
    
    # fig.suptitle(height)

    # ax[0].set_ylim([-1e-13, 1e-13])
    # ax[1].set_ylim([-0.5e-13, 0.5e-13])
    # ax[2].set_ylim([0, 0.2e-12])
    # ax[3].set_ylim([0, 0.2e-12])
    # for ax_ in ax.flatten():
    #     ax_.grid()
    #     ax_.yaxis.set_tick_params(which='both', labelleft=True)
    # fig.savefig(rf"G:\CloudnetData\Hyytiala\CL61\Summary/noise_summary_{height}.png", dpi=600)
    # fig.savefig(rf"G:\CloudnetData\Kenttarova\CL61\Summary/noise_hist.png", dpi=600)
