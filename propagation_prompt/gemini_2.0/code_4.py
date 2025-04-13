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

# Define a function to run the wind farm simulation with uncertain wind speed
def run_simulation_with_ws_uncertainty(ws_mean, ws_std, wd):
    """
    Runs the wind farm simulation with a wind speed that has added uncertainty.

    Args:
        ws_mean (float): Mean wind speed.
        ws_std (float): Standard deviation of wind speed (uncertainty).
        wd (array-like): Wind directions to simulate.

    Returns:
        xarray.SimulationResult: Simulation results.
    """

    # Generate wind speeds with uncertainty (normal distribution)
    ws = np.random.normal(loc=ws_mean, scale=ws_std, size=len(wd))
    ws = np.clip(ws, 3, 25)  # Ensure wind speeds are within realistic bounds [cite: 6]

    sim_res = wf_model(x, y, wd=wd, ws=ws)  # Run simulation with uncertain wind speed [cite: 7, 8]
    return sim_res

# Define wind conditions
wd = np.arange(0, 360, 10)  # Wind directions [cite: 6]
ws_mean = 10  # Mean wind speed
ws_std = 2    # Standard deviation of wind speed (example uncertainty)

# Run simulation with wind speed uncertainty
sim_res_uncertain = run_simulation_with_ws_uncertainty(ws_mean, ws_std, wd)

# Calculate Sobol sensitivity (simplified - assumes independent samples)
# For a proper Sobol analysis, you'd need a specific sampling strategy
# This example approximates sensitivity by looking at the variance of WS_eff

def calculate_sobol_sensitivity(sim_res, x, y):
    """
    Calculates a simplified Sobol-like sensitivity of WS_eff to wind speed uncertainty.

    Args:
        sim_res (xarray.SimulationResult): Simulation results.
        x (array-like): x coordinates of turbines.
        y (array-like): y coordinates of turbines.

    Returns:
        numpy.ndarray: Sobol sensitivity approximation for each point in the flow field.
    """

    ws_eff = sim_res.WS_eff.values  # Effective wind speed [cite: 9]
    sensitivity = np.var(ws_eff, axis=0) / np.mean(np.var(ws_eff, axis=0))  # Normalize by mean variance

    # Map sensitivity to x, y grid (simplified - assumes sensitivity is the same for all turbines)
    grid_x, grid_y = np.meshgrid(np.unique(x), np.unique(y))
    sensitivity_grid = np.zeros_like(grid_x, dtype=float)
    for i, turbine_x in enumerate(np.unique(x)):
        for j, turbine_y in enumerate(np.unique(y)):
            #Find a corresponding index -- assumes x and y are ordered.
            turbine_index = np.where((x == turbine_x) & (y == turbine_y))[0][0]
            sensitivity_grid[j, i] = sensitivity[turbine_index]

    return sensitivity_grid

sensitivity_map = calculate_sobol_sensitivity(sim_res_uncertain, x, y)

# Plotting the sensitivity map
plt.figure()
plt.imshow(sensitivity_map, origin='lower', extent=(np.min(x), np.max(x), np.min(y), np.max(y)), cmap='viridis')
plt.colorbar(label='Sobol Sensitivity (Approximation)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Sobol Sensitivity of WS_eff to Wind Speed Uncertainty')
plt.savefig('sobol_sensitivity_wseff.png')

print("Sobol sensitivity plot generated: sobol_sensitivity_wseff.png")
