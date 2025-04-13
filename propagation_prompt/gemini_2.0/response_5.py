import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Install PyWake if needed
try:
    import py_wake
except ModuleNotFoundError:
    !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils import layouts

# Define site and wind turbines
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Wind farm model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# 1. Propagate realistic measurement uncertainties
# Define uncertainty in inflow (e.g., wind speed, wind direction)
# For simplicity, let's assume wind speed has a normal distribution uncertainty
ws_mean = np.arange(3, 26)  # Default wind speeds [cite: 6]
ws_std = 0.1 * ws_mean  # Example: 10% uncertainty (realistic assumption)
wd_mean = np.arange(0, 360) #default wind directions [cite: 6]
wd_std = np.ones_like(wd_mean) * 2 #assume 2 degree std in wd.

n_samples = 100  # Number of Monte Carlo samples

# Generate random samples of wind speed and direction
ws_samples = np.random.normal(loc=ws_mean, scale=ws_std, size=(n_samples, len(ws_mean)))
wd_samples = np.random.normal(loc=wd_mean, scale=wd_std, size=(n_samples, len(wd_mean)))
ws_samples = np.clip(ws_samples, 0, 50) # Ensure no negative or unrealistic wind speeds
wd_samples = wd_samples % 360 #keep wind directions in 0-360

# 2. Sobol Sensitivity Analysis
# For simplicity, we'll focus on first-order Sobol indices for wind speed
# Sobol requires multiple samples, and we already generated them

WS_eff_samples = np.zeros((n_samples, len(x), len(y), len(ws_mean), len(wd_mean))) #to store all the flow fields

for i in range(n_samples):
    sim_res = wf_model(x, y, ws=ws_samples[i], wd=wd_samples[i])
    for j in range(len(ws_mean)):
      for k in range(len(wd_mean)):
        fm = sim_res.flow_map(ws=ws_samples[i,j], wd=wd_samples[i,k])
        WS_eff_samples[i, :, :, j, k] = fm.WS_eff.isel(h=0)  # Store WS_eff

# Compute Sobol sensitivity indices (first order)
# This is a simplified calculation; a full Sobol analysis is more complex
#Need to compute variance across samples and total variance

WS_eff_mean = np.mean(WS_eff_samples, axis=0)
WS_eff_variance = np.var(WS_eff_samples, axis=0)
WS_eff_total_variance = np.var(WS_eff_mean, axis = (0,1)) #total variance across x and y

# Approximate first-order Sobol index (sensitivity to wind speed)
#This is a simplification and assumes independence, which may not be true
sobol_index_ws = WS_eff_variance / WS_eff_total_variance


# 3. Flow field images
# Generate a series of flow field images showing Sobol sensitivity
n_time_steps = 5  # Number of time steps to visualize
time_steps_idx = np.linspace(0, len(ws_mean) - 1, n_time_steps, dtype=int)

for t_idx in time_steps_idx:
    plt.figure(figsize=(10, 8))
    plt.imshow(sobol_index_ws[:,:,t_idx].T, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], origin='lower', cmap='viridis') # [cite: 52, 53, 54, 55, 56]
    plt.colorbar(label='Sobol Index (WS_eff)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sobol Sensitivity of WS_eff at Wind Speed {ws_mean[t_idx]} m/s')
    plt.savefig(f'sobol_sensitivity_t{t_idx}.png')
    plt.close()
