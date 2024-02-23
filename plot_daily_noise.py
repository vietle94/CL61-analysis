import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
myFmt = mdates.DateFormatter('%H:%M')

# %%
noise = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise.csv")
diag = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/diag.csv")
monitoring = pd.read_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/monitoring.csv")
noise['datetime'] = pd.to_datetime(noise['datetime'])
diag['datetime'] = pd.to_datetime(diag['datetime'])
monitoring['datetime'] = pd.to_datetime(monitoring['datetime'])

# %%
df = noise.merge(diag)
df = df[['datetime', 'co_mean', 'co_std', 'cross_mean', 'cross_std',
         'IsolDC_DC(PFB)', 'clomenvironmentalmsensormenvmtemperature',
         'clomenvironmentalmsensormenvmhumidity', 'Window_heater_NTC']]
for id, grp in df.groupby(df['datetime'].dt.date):
    date_save = np.datetime_as_string(grp.datetime.values[0], 'D')
    fig, ax = plt.subplots(2, 4, constrained_layout=True,
                           sharex=True, figsize=(16, 7))
    for ax_, variable in zip(ax.flatten(), grp.columns[1:].values):
        ax_.plot(grp['datetime'], grp[variable], '.')
        ax_.set_ylabel(variable)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
        ax_.grid()

    fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Img/" + date_save + '_noise_diag.png',
                dpi=600)
    print('Done')
    plt.close('all')

# %%
df = noise.merge(monitoring)
df = df.drop(['internal_pressure', 'background_radiance', 'internal_humidity'], axis=1)
for id, grp in df.groupby(df['datetime'].dt.date):
    date_save = np.datetime_as_string(grp.datetime.values[0], 'D')
    fig, ax = plt.subplots(3, 4, constrained_layout=True,
                           sharex=True, figsize=(16, 7))
    for ax_, variable in zip(ax.flatten(), grp.columns[1:].values):
        ax_.plot(grp['datetime'], grp[variable], '.')
        ax_.set_ylabel(variable)
        ax_.xaxis.set_major_formatter(myFmt)
        ax_.xaxis.set_major_locator(mdates.HourLocator([0, 6, 12, 18, 24]))
        ax_.grid()
    fig.savefig(r"G:\CloudnetData\Kenttarova\CL61\Img/" + date_save + '_noise_diag.png',
                dpi=600)
    print('Done')
    plt.close('all')
