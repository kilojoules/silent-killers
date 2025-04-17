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
from py_wake.utils.plotting import setup_plot

# Load site and wind turbine data
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define wind farm model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# --- Uncertainty Propagation and Sensitivity Analysis ---

# 1. Define Uncertainty in Inflow (example: wind speed)
#    - Assume a normal distribution for wind speed with a standard deviation
#    - This is a simplified example; in reality, you'd get this from measurements
ws_mean = np.linspace(5, 15, 5)  # Example mean wind speeds
ws_std = 1.0  # Example standard deviation for wind speed

# 2. Generate Wind Speed Samples
n_samples = 100
ws_samples = np.random.normal(loc=ws_mean, scale=ws_std, size=(len(ws_mean), n_samples))

# 3.  Run PyWake for each wind speed sample
all_sim_res = []
for i, ws_set in enumerate(ws_samples):
    sim_res = wf_model(x, y, ws=ws_set)  # Run simulation for each wind speed set
    all_sim_res.append(sim_res)

# 4. Calculate Sobol Sensitivity Indices
#    -  This requires a more complex sensitivity analysis library (not in the basic PyWake examples)
#    -  Here's a placeholder for the concept:
#    -  Sobol indices quantify how much of the output variance (WS_eff) is due to the input variance (wind speed)

sobol_sensitivity = []

for t_index in range(len(ws_mean)):

  ws_eff_at_t = []
  for sim_res in all_sim_res:
    ws_eff_at_t.append(sim_res.WS_eff.isel(ws=t_index).values) #extracting the effective wind speed for a given wind speed

  ws_eff_at_t = np.array(ws_eff_at_t)

  first_order = np.var(np.mean(ws_eff_at_t, axis=1)) / np.var(ws_eff_at_t)
  sobol_sensitivity.append(first_order)


# 5. Generate Flow Field Images

for t_index in range(len(ws_mean)):

    #create a horizontal grid
    grid = py_wake.HorizontalGrid(resolution=100, extend=0.2)

    #calculate the flow map
    flow_map = all_sim_res[0].flow_map(grid=grid,ws=ws_mean[t_index]) #using the first sim_res just for geometry

    plt.figure(figsize=(10, 8))
    flow_map.plot_wake_map()

    plt.title(f'Sobol Sensitivity at Time {t_index+1} : {sobol_sensitivity[t_index]:.2f}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(f'flow_map_t_{t_index}.png')
    plt.close()
