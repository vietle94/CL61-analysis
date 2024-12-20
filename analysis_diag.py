import seaborn as sns
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
diag = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/diag.csv")
monitoring = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/monitoring.csv")
status = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/status.csv")

# %%
noise['datetime'] = pd.to_datetime(noise['datetime'], format='mixed')
diag['datetime'] = pd.to_datetime(diag['datetime'], format='mixed')
monitoring['datetime'] = pd.to_datetime(monitoring['datetime'], format='mixed')
status['datetime'] = pd.to_datetime(status['datetime'], format='mixed')

# %%
for date_, grp in monitoring.groupby(monitoring['datetime'].dt.date):
    if pd.to_datetime('20240204') > pd.to_datetime(date_):
        continue
    fig, ax = plt.subplots(3, 4, figsize=(12, 6),
                           sharex=True, constrained_layout=True)
    for ax_, variable in zip(ax.flatten(), grp.columns[:-1]):
        ax_.plot(grp['datetime'], grp[variable], '.')
        ax_.grid()
        ax_.set_ylabel(variable.replace('_', '\n', 1))
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    fig.delaxes(ax.flatten()[-1])
    fig.savefig(r'G:\CloudnetData\Kenttarova\CL61\Img\Diag/' + \
                date_.strftime('%Y-%m-%d') + '_diag.png', dpi=600,
                bbox_inches='tight')
    plt.close(fig)


# %%
pd.to_datetime('20240206') > pd.to_datetime(date_)


# %%
fig, ax = plt.subplots(2, 2, constrained_layout=True,
                       sharex=True, figsize=(12, 6))
for ax_, variable in zip(ax.flatten(), noise.columns[1:].values):
    ax_.plot(noise['datetime'], noise[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()
# fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise.png",
#             dpi=600)

# %%
fig, ax = plt.subplots(7, 5, constrained_layout=True,
                       sharex=True, figsize=(29, 19))

for ax_, variable in zip(ax.flatten()[:4], noise.columns[1:].values):
    ax_.plot(noise['datetime'], noise[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

for ax_, variable in zip(ax.flatten()[4:], diag.columns[:-1].values):
    ax_.plot(diag['datetime'], diag[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

# fig.savefig(r"G:\CloudnetData\Kenttarova\CL61/Summary/noise_diag.png", dpi=600)

# %%
fig, ax = plt.subplots(4, 4, constrained_layout=True,
                       sharex=True, figsize=(19, 9))

for ax_, variable in zip(ax.flatten()[:4], noise.columns[1:].values):
    ax_.plot(noise['datetime'], noise[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

for ax_, variable in zip(ax.flatten()[4:], monitoring.columns[:-1].values):
    ax_.plot(monitoring['datetime'], monitoring[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

# fig.savefig(r"G:\CloudnetData\Kenttarova\CL61/Summary/noise_monitoring.png", dpi=600)

# %%
fig, ax = plt.subplots(4, 4, constrained_layout=True,
                       sharex=True, figsize=(19, 9))

for ax_, variable in zip(ax.flatten()[:4], noise.columns[1:].values):
    ax_.plot(noise['datetime'], noise[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

for ax_, variable in zip(ax.flatten()[4:], status.columns[:-1].values):
    ax_.plot(status['datetime'], status[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()

# fig.savefig(r"G:\CloudnetData\Kenttarova\CL61/Summary/noise_status.png", dpi=600)

# %%
df_merged = noise.merge(diag).iloc[:, 1:]
corr = df_merged.corr()

# %%
fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(corr, ax=ax, cmap='RdBu', center=0, vmin=-0.6, vmax=0.6)
fig.savefig(r"G:\CloudnetData\Kenttarova\CL61/Summary/noise_heatmap.png", dpi=600,
            bbox_inches='tight')
