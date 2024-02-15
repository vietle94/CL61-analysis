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
noise_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/*_noise.csv")
diag_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Raw/*_diag.csv")

noise = pd.concat([pd.read_csv(x) for x in noise_path], ignore_index=True)
diag = pd.concat([pd.read_csv(x) for x in diag_path], ignore_index=True)

# %%
noise['datetime'] = pd.to_datetime(noise['datetime'], format='ISO8601')
diag['datetime'] = pd.to_datetime(diag['datetime'], format='ISO8601')

# %%
fig, ax = plt.subplots(7, 5, constrained_layout=True,
                       sharex=True, figsize=(29, 19))

for ax_, variable in zip(ax.flatten()[:4], noise.columns[1:].values):
    ax_.plot(noise['datetime'], noise[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()
    ax_.xaxis.set_major_formatter(myFmt)

for ax_, variable in zip(ax.flatten()[4:], diag.columns[:-1].values):
    ax_.plot(diag['datetime'], diag[variable], '.')
    ax_.set_ylabel(variable)
    ax_.grid()
    ax_.xaxis.set_major_formatter(myFmt)

# %%
df_merged = noise.merge(diag).iloc[:, 1:]
df_normalized = (df_merged-df_merged.mean())/df_merged.std()
corr = df_normalized.corr()

# %%
fig, ax = plt.subplots(figsize=(29, 19))
sns.heatmap(corr, ax=ax, cmap='RdBu', center=0, vmin=-0.6, vmax=0.6)
