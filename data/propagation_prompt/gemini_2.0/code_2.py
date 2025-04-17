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

# Define site and wind turbine
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Wind farm model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# --- Uncertainty Propagation ---
# This is a placeholder for the uncertainty propagation.
# In a real scenario, you would replace this with your actual
# uncertainty analysis method (e.g., Monte Carlo, Polynomial Chaos).
# For simplicity, we'll assume a normal distribution for wind speed
# and direction and propagate it through the wake model.

n_simulations = 10  # Number of Monte Carlo simulations
ws_mean = site.default_ws  # Default wind speeds
wd_mean = site.default_wd  # Default wind directions
ws_std = 1  # Example standard deviation for wind speed
wd_std = 5  # Example standard deviation for wind direction

# Generate random samples
ws_samples = np.random.normal(loc=ws_mean, scale=ws_std, size=(n_simulations, len(ws_mean)))
wd_samples = np.random.normal(loc=wd_mean, scale=wd_std, size=(n_simulations, len(wd_mean)))

# Store results
all_sim_res = []

for i in range(n_simulations):
    sim_res = wf_model(x, y, ws=ws_samples[i], wd=wd_samples[i])
    all_sim_res.append(sim_res)

# --- Sobol Sensitivity Analysis ---
# This is a placeholder. You'll need a library like 'SALib'
# to perform a proper Sobol sensitivity analysis.
# Here's a simplified version to illustrate the concept:

# 1.  Calculate the mean and variance of WS_eff
ws_eff_values = np.array([res.WS_eff.values for res in all_sim_res])
ws_eff_mean = np.mean(ws_eff_values, axis=0)
ws_eff_variance = np.var(ws_eff_values, axis=0)

# 2.  Approximate first-order Sobol indices (simplified)
#     In reality, SALib or a similar library would do this rigorously.
sobol_first_order = np.var(ws_eff_values, axis=0) / ws_eff_variance

# --- Plotting ---

# Create flow maps for a few timesteps to visualize sensitivity
num_time_steps_to_plot = 5
time_step_indices = np.linspace(0, len(time_stamp) - 1, num_time_steps_to_plot, dtype=int)  # Select some time steps

for t_idx in time_step_indices:
    plt.figure(figsize=(12, 10))
    
    # Get the wind speed and direction for this time step
    ws_t = ws[t_idx]
    wd_t = wd[t_idx]
    
    # Calculate the flow map
    flow_map = all_sim_res[0].flow_map(wd=wd_t, ws=ws_t)  # Use the first simulation for the base flow
    flow_map.plot_wake_map()
    
    # Overlay Sobol sensitivity (this is a simplification)
    #  -  The 'sobol_first_order' needs to be reshaped or handled
    #     appropriately to match the flow_map grid.
    #  -  This is just a conceptual visualization; adapt as needed.
    
    plt.contourf(flow_map.X, flow_map.Y, sobol_first_order[0, ..., t_idx], alpha=0.2, cmap='viridis')  # Example
    
    plt.title(f'Flow Map and Sobol Sensitivity at Time Step {time_stamp[t_idx]:.2f}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label='Sobol First-Order Index (Approx.)')
    plt.savefig(f'flow_map_sobol_t{t_idx}.png')  # Save each plot
    plt.close()

print("Flow maps with Sobol sensitivity (approximation) saved as png files.")
