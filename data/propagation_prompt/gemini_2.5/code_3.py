# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
# Assume SALib is installed for Sobol analysis: pip install SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
import os # For saving images

# Import PyWake components (adjust imports based on your specific models)
# Ensure PyWake is installed: pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
try:
    import py_wake
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80 # Example site/turbine [cite: 3]
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014 # Example model [cite: 3]
    from py_wake import HorizontalGrid # For flow maps [cite: 59]
    from py_wake.utils.plotting import setup_plot
except ModuleNotFoundError:
    print("PyWake not found. Please install it.")
    # You might need to exit or handle this error appropriately
    exit()

print(f"Using PyWake version: {py_wake.__version__}")

# --- Setup Simulation ---
# 1. Site and Wind Turbines (Using example from documentation) [cite: 3]
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# 2. Wind Farm Model (Using example from documentation) [cite: 3]
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)
print("Wind Farm Model Setup:", wf_model) # [cite: 3]

# --- Define Uncertain Parameters and Problem for Sensitivity Analysis ---
# This section requires defining which input parameters have uncertainty
# and what their distributions are. This is NOT covered in the provided docs.
# We'll assume uncertainty in mean wind speed (ws), wind direction (wd),
# and turbulence intensity (ti) for a specific time or condition.

# Example: Define the problem for SALib
# We need bounds for each uncertain parameter.
# These bounds should represent the 'realistic measurement uncertainties'.
# *** THESE ARE PLACEHOLDER VALUES - ADJUST BASED ON YOUR DATA ***
problem = {
    'num_vars': 3,
    'names': ['ws', 'wd', 'ti'],
    'bounds': [
        [8.0, 12.0],    # Example bounds for mean wind speed (m/s)
        [265.0, 275.0], # Example bounds for mean wind direction (deg)
        [0.05, 0.15]    # Example bounds for mean turbulence intensity (-)
    ]
}

# --- Generate Parameter Samples using SALib ---
# Generate samples using Saltelli's method for Sobol analysis
# The number of samples N determines the accuracy and computational cost.
# Total runs = N * (2 * num_vars + 2)
N_samples = 64 # Power of 2 often recommended. Adjust as needed.
param_values = saltelli.sample(problem, N_samples, calc_second_order=True)
print(f"Generated {param_values.shape[0]} parameter sets for Sobol analysis.")

# --- Run PyWake Simulations for Each Parameter Sample ---
# We need to run the wind farm model for each row in param_values
# To get WS_eff across the field, we need flow maps for each run.

# Define the grid for the flow map [cite: 59, 60]
# Adjust resolution and extent as needed
flow_map_grid = HorizontalGrid(resolution=100, extend=0.5)

# Store WS_eff results for each point in the grid for each simulation run
# Dimensions: (num_simulations, num_grid_points)
all_ws_eff_results = []

print("Running PyWake simulations for each parameter sample...")
for i, params in enumerate(param_values):
    ws_sample, wd_sample, ti_sample = params
    print(f"  Run {i+1}/{param_values.shape[0]}: ws={ws_sample:.2f}, wd={wd_sample:.1f}, ti={ti_sample:.3f}")

    # Run the simulation for this specific parameter set [cite: 7, 8]
    # Note: We simulate a single flow case (ws, wd, ti) for sensitivity
    # To do this over *time*, you'd need to wrap this in a time loop
    # and likely define time-varying bounds or run SA at representative times.
    sim_res_sample = wf_model(x, y,
                              wd=[wd_sample], # Must be iterable
                              ws=[ws_sample], # Must be iterable
                              ti=ti_sample   # Can be scalar if uniform
                             )

    # Calculate the flow map for this simulation result [cite: 53, 54]
    # We only have one wd/ws, so select index 0
    flow_map_sample = sim_res_sample.flow_map(grid=flow_map_grid, wd=wd_sample, ws=ws_sample)

    # Extract WS_eff from the flow map
    # The flow_map object contains WS_eff on the specified grid.
    # Flatten the WS_eff grid to store results for analysis.
    # The structure might be flow_map_sample.WS_eff.values.flatten()
    # Need to confirm the exact structure from PyWake FlowMap object
    # Let's assume flow_map_sample is an xarray DataArray with dims (x, y)
    ws_eff_flat = flow_map_sample.WS_eff.values.flatten()
    all_ws_eff_results.append(ws_eff_flat)

# Convert results to a NumPy array: rows=simulations, cols=grid_points
results_array = np.array(all_ws_eff_results)
print(f"Shape of results array (simulations, grid_points): {results_array.shape}")

# --- Perform Sobol Sensitivity Analysis ---
# Analyze the results for each output (WS_eff at each grid point)
# This can be computationally intensive if the grid is large.

# Placeholder for Sobol indices (e.g., S1 for first-order)
# Dimensions: (num_grid_points, num_vars)
sobol_indices_s1 = np.zeros((results_array.shape[1], problem['num_vars']))
sobol_indices_st = np.zeros((results_array.shape[1], problem['num_vars']))

print("Performing Sobol analysis for WS_eff at each grid point...")
for i in range(results_array.shape[1]): # Loop through each grid point
    if (i + 1) % 1000 == 0: # Print progress
         print(f"  Analyzing grid point {i+1}/{results_array.shape[1]}")
    Si = sobol.analyze(problem, results_array[:, i], calc_second_order=True, print_to_console=False)
    sobol_indices_s1[i, :] = Si['S1']
    sobol_indices_st[i, :] = Si['ST'] # Total sensitivity index

print("Sobol analysis complete.")

# --- Visualize and Save Sensitivity Maps ---
# Reshape the Sobol indices back into the grid format for plotting

# Get grid coordinates from the last flow map (assuming grid is constant)
x_coords = flow_map_sample.x.values
y_coords = flow_map_sample.y.values
grid_shape = (len(x_coords), len(y_coords)) # Should be (flow_map_sample.WS_eff.shape)

# Create output directory
output_dir = "sobol_sensitivity_maps"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving sensitivity maps to '{output_dir}'...")

for i, param_name in enumerate(problem['names']):
    # Reshape S1 and ST indices for this parameter
    s1_map = sobol_indices_s1[:, i].reshape(grid_shape)
    st_map = sobol_indices_st[:, i].reshape(grid_shape)

    # Plot S1 sensitivity map
    plt.figure(figsize=(10, 8))
    plt.contourf(x_coords, y_coords, s1_map.T, cmap='viridis', levels=50) # Transpose needed if shape is (x,y)
    plt.colorbar(label=f'S1 Index (First Order Sensitivity)')
    windTurbines.plot(x, y, ax=plt.gca()) # Plot turbines on top
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sobol First Order Sensitivity (S1) of WS_eff to {param_name}')
    plt.axis('equal')
    filename_s1 = os.path.join(output_dir, f'ws_eff_s1_sensitivity_{param_name}.png')
    plt.savefig(filename_s1)
    plt.close()
    print(f"  Saved: {filename_s1}")

    # Plot ST sensitivity map
    plt.figure(figsize=(10, 8))
    plt.contourf(x_coords, y_coords, st_map.T, cmap='viridis', levels=50)
    plt.colorbar(label=f'ST Index (Total Sensitivity)')
    windTurbines.plot(x, y, ax=plt.gca()) # Plot turbines on top
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sobol Total Sensitivity (ST) of WS_eff to {param_name}')
    plt.axis('equal')
    filename_st = os.path.join(output_dir, f'ws_eff_st_sensitivity_{param_name}.png')
    plt.savefig(filename_st)
    plt.close()
    print(f"  Saved: {filename_st}")

# --- Note on Time Dependency ---
# The script above performs the sensitivity analysis for *one* set of mean conditions.
# To show sensitivity "at different times", you would need to:
# 1. Define time-varying inputs (like in the time series example [cite: 28, 30, 31])
#    OR define representative time steps/conditions.
# 2. Loop through these time steps.
# 3. Inside the time loop, potentially redefine the 'problem' bounds based on the
#    uncertainty *at that time*.
# 4. Rerun the sampling (or use the same samples if appropriate) and the full set
#    of PyWake simulations for *each* time step.
# 5. Perform Sobol analysis and save the sensitivity maps for *each* time step,
#    likely adding the time step to the filename.
# This significantly increases computational cost.

print("\nScript finished. Remember to adjust placeholder values and logic.")
