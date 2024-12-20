import glob
import glob
import numpy as np
import pandas as pd

# %%
noise_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Diag/*_noise.csv")
noise = pd.concat([pd.read_csv(x) for x in noise_path], ignore_index=True)
noise.to_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/noise.csv", index=False)

# %%
diag_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Diag/*_diag.csv")
diag = pd.concat([pd.read_csv(x) for x in diag_path], ignore_index=True)
diag.to_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/diag.csv", index=False)

# %%
monitoring_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Diag/*_monitoring.csv")
monitoring = pd.concat([pd.read_csv(x) for x in monitoring_path], ignore_index=True)
monitoring.to_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/monitoring.csv", index=False)

# %%
status_path = glob.glob("G:\CloudnetData\Kenttarova\CL61\Diag/*_status.csv")
status = pd.concat([pd.read_csv(x) for x in status_path], ignore_index=True)
status.to_csv(r"G:\CloudnetData\Kenttarova\CL61\Summary/status.csv", index=False)

