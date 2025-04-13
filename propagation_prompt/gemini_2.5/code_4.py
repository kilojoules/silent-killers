import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

# Attempt to import py_wake and SALib, provide install instructions if missing
try:
    import py_wake
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake import HorizontalGrid
except ModuleNotFoundError:
    print("PyWake is not installed.")
    print("Please install it, e.g., using: pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git")
    exit()

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
except ModuleNotFoundError:
    print("SALib is not installed.")
    print("Please install it, e.g., using: pip install salib")
    exit()

# --- 1. Define Uncertain Parameters and Problem for SALib ---

# Define the uncertain inflow parameters and their bounds.
# Example: Uncertainty in mean Wind Speed (WS), Wind Direction (WD), and Turbulence Intensity (TI)
# Adjust bounds and names as needed for realistic measurement uncertainties.
problem = {
    'num_vars': 3,
    'names': ['WS', 'WD', 'TI'],
    'bounds': [
        [8.0, 12.0],  # Range for mean Wind Speed (m/s)
        [265.0, 275.0], # Range for mean Wind Direction (deg)
        [0.05, 0.15]   # Range for mean Turbulence Intensity (-)
    ]
}

# --- 2. Generate Samples using SALib ---

# Generate samples using Saltelli's scheme for Sobol analysis.
# N is the number of samples per parameter. Total runs = N * (2*D + 2) for Sobol
# Reduce N for quicker testing, increase for more robust results.
N = 64 # Base samples (adjust as needed for computational cost vs accuracy)
param_values = saltelli.sample(problem, N, calc_second_order=False) # Use calc_second_order=True for S2 indices
print(f"Generated {param_values.shape[0]} parameter samples using Saltelli sampler.")

# --- 3. PyWake Setup ---

# Standard PyWake setup using example data
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define the wake model
# We will create the wf_model inside the wrapper function to update TI based on samples
# wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Define the grid for the flow map
# Adjust resolution and extent as needed
grid = HorizontalGrid(resolution=50, extend=0.1)

# Create directory for saving images
output_dir = "sobol_sensitivity_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Output images will be saved in: {output_dir}")

# --- 4. Create PyWake Wrapper Function ---

# Define a function to run PyWake for a given parameter set
# and return the WS_eff field on the defined grid.
def evaluate_pywake_flow_map(params, site_obj, wt_obj, x_coords, y_coords, grid_obj):
    """
    Runs a PyWake flow map simulation for a given parameter set.

    Args:
        params (list or np.array): A single sample of parameters [WS, WD, TI].
        site_obj: An initialized PyWake Site object.
        wt_obj: An initialized PyWake WindTurbines object.
        x_coords, y_coords: Wind turbine coordinates.
        grid_obj: The grid definition for the flow map.

    Returns:
        np.array: Flattened array of WS_eff values on the grid, or None if simulation fails.
    """
    ws_val, wd_val, ti_val = params

    try:
        # Create a wind farm model instance for this specific TI value
        # Note: This example assumes TI is constant for the flow case,
        # matching the sampled value. Site definition might need adjustment
        # if the Site object itself depends complexly on these parameters.
        wf_model_instance = Bastankhah_PorteAgel_2014(site_obj, wt_obj, k=0.0324555)

        # Run simulation for the single specified WD and WS
        # Pass the sampled TI value directly to the simulation call
        sim_res = wf_model_instance(x_coords, y_coords, wd=[wd_val], ws=[ws_val], TI=ti_val)

        # Calculate the flow map
        flow_map = sim_res.flow_map(grid=grid_obj, wd=wd_val, ws=ws_val) # Use the same wd, ws

        # Extract WS_eff and flatten
        # Ensure WS_eff exists and handle potential issues
        if hasattr(flow_map, 'WS_eff'):
           ws_eff_field = flow_map.WS_eff.squeeze().values # Squeeze removes single dimensions
           return ws_eff_field.flatten()
        else:
           print(f"Warning: WS_eff not found in flow_map for params {params}")
           return None

    except Exception as e:
        print(f"Error during PyWake simulation for params {params}: {e}")
        return None

# --- 5. Run Model Iteratively ---

print("Running PyWake simulations for all parameter samples...")
# Store results (WS_eff on the grid for each sample)
num_grid_points = grid.n_points # Get number of points from the grid object
model_outputs = np.zeros((param_values.shape[0], num_grid_points)) * np.nan # Initialize with NaN

# Run the simulations (consider parallelization for large N)
for i, params in enumerate(param_values):
    print(f"Running sample {i+1}/{param_values.shape[0]}...")
    ws_eff_flat = evaluate_pywake_flow_map(params, site, windTurbines, x, y, grid)
    if ws_eff_flat is not None and ws_eff_flat.shape[0] == num_grid_points:
        model_outputs[i, :] = ws_eff_flat
    else:
        print(f"  Skipping sample {i+1} due to evaluation error or shape mismatch.")

# Check for simulation failures
if np.isnan(model_outputs).all():
    print("Error: All PyWake simulations failed. Cannot perform sensitivity analysis.")
    exit()
elif np.isnan(model_outputs).any():
    print("Warning: Some PyWake simulations failed. Results may be affected.")
    # Optional: Handle NaN values, e.g., by imputation or removal,
    # but this can bias Sobol results. Here, we proceed but SALib might error.

print("Finished PyWake simulations.")

# --- 6. Perform Sobol Analysis ---

print("Performing Sobol sensitivity analysis...")
# Initialize dictionary to store Sobol indices for each grid point
sobol_indices_st = np.zeros(num_grid_points) * np.nan

# Analyze point-by-point
# This loop can be slow; vectorization might be possible depending on SALib internals
# or consider analyzing only a subset of points for faster results.
analysis_failed = False
for point_idx in range(num_grid_points):
    # Skip points where all simulations failed
    if np.isnan(model_outputs[:, point_idx]).all():
        print(f"  Skipping analysis for grid point {point_idx} (all simulations failed).")
        continue
    # Filter out NaN results for this specific point if some simulations failed
    valid_outputs = model_outputs[~np.isnan(model_outputs[:, point_idx]), point_idx]
    valid_param_values = param_values[~np.isnan(model_outputs[:, point_idx]), :]

    # Ensure sufficient samples remain after filtering NaN
    # SALib's saltelli sampler requires N*(2D+2) or N*(D+2) samples
    required_samples_st = N * (problem['num_vars'] + 2) # For S1/ST only
    if len(valid_outputs) < required_samples_st:
         print(f"  Skipping analysis for grid point {point_idx}: Insufficient valid samples ({len(valid_outputs)} < {required_samples_st}).")
         continue


    try:
       # Ensure problem dict matches the filtered param_values structure if needed
       # (Here assumes filtering doesn't break SALib's expectation of A/B matrix pairs)
       Si = sobol.analyze(problem, valid_outputs, calc_second_order=False, print_to_console=False)
       sobol_indices_st[point_idx] = Si['ST']
    except Exception as e:
       print(f"Error during Sobol analysis for grid point {point_idx}: {e}")
       analysis_failed = True
       # Optionally break or continue based on acceptable failure rate

if analysis_failed:
     print("Warning: Sobol analysis failed for one or more grid points.")

print("Finished Sobol analysis.")


# --- 7. Visualize Sensitivity ---

print("Visualizing Sobol Total Sensitivity (ST) for WS_eff...")

# Reshape the sensitivity indices back to the grid dimensions
st_map = sobol_indices_st.reshape(grid.Ny, grid.Nx)

# Get grid coordinates for plotting
X, Y = grid.XY # Use the meshgrid coordinates from the grid object

plt.figure(figsize=(12, 10))
# Use contourf for filled contours
if not np.isnan(st_map).all():
    levels = plt.MaxNLocator(nbins=20).tick_values(np.nanmin(st_map), np.nanmax(st_map))
    cf = plt.contourf(X, Y, st_map, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(cf, label='Sobol Total Sensitivity Index (ST) for WS_eff')
else:
    print("Cannot plot sensitivity map: All sensitivity values are NaN.")
    plt.text(0.5, 0.5, 'Sensitivity Analysis Failed', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


# Plot wind turbine locations
plt.plot(x, y, 'ko', markersize=5, label='Wind Turbines')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Sobol Total Sensitivity (ST) of WS_eff to Inflow Parameters (WS, WD, TI)')
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
output_filename = os.path.join(output_dir, f"sobol_st_ws_eff_map_N{N}.png")
plt.savefig(output_filename)
print(f"Saved sensitivity map to: {output_filename}")

plt.show()

print("Script finished.")
