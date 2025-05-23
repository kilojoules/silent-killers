# Python Script for Sobol Sensitivity Analysis of WS_eff in PyWake

# --- 1. Import Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

# Import PyWake components (ensure PyWake is installed)
try:
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake import HorizontalGrid
except ModuleNotFoundError:
    print("Error: PyWake not found. Please install it, e.g.,:")
    print("pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git")
    exit()

# Import SALib (ensure SALib is installed: pip install salib)
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.util import read_param_file # Alternative for defining problem
except ModuleNotFoundError:
    print("Error: SALib not found. Please install it: pip install SALib")
    exit()

# --- 2. Setup PyWake Model ---
print("Setting up PyWake model...")
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555) # Example model

# Define the flow map grid
grid_resolution = 50 # Lower resolution for faster SA computation
grid_extend = 0.1 # Smaller extent around the farm
flow_map_grid = HorizontalGrid(resolution=grid_resolution, extend=grid_extend)

# Create output directory for sensitivity maps
output_dir = "sobol_sensitivity_maps"
os.makedirs(output_dir, exist_ok=True)
print(f"Output images will be saved in: {output_dir}")

# --- 3. Define Sensitivity Analysis Problem ---
# Define the uncertain parameters and their bounds/distributions
# Example: Uncertainty in mean wind speed and wind direction
# More realistic uncertainties might involve specific measurement device specs
# and potentially correlated variables (use Chaospy/UQpy for complex dependencies)
problem = {
    'num_vars': 2,
    'names': ['wind_speed', 'wind_direction'],
    # Define realistic bounds based on expected measurement uncertainty
    'bounds': [
        [8.0, 12.0], # Example range for mean wind speed (m/s)
        [260, 280]   # Example range for mean wind direction (degrees)
    ],
    # Assuming uniform distribution for simplicity
    # 'dists': ['unif', 'unif'] # SALib can use distributions if needed
}

# --- 4. Define PyWake Wrapper Function for SALib ---
# Store grid shape for reshaping later
ref_sim_res = wf_model(x, y, ws=10, wd=270) # Run a dummy simulation
ref_flow_map = ref_sim_res.flow_map(grid=flow_map_grid, ws=10, wd=270)
grid_shape = ref_flow_map.WS_eff.shape # (y_points, x_points)
num_grid_points = grid_shape[0] * grid_shape[1]
print(f"Flow map grid shape: {grid_shape}, Total points: {num_grid_points}")

# Cache wf_model results to potentially speed up repeated calls with same base params
_wf_model_cache = {}

def evaluate_pywake_ws_eff(params):
    """
    Wrapper function for SALib.
    Takes a list/array of parameter samples [ws, wd]
    Runs PyWake simulation, calculates WS_eff flow map, returns flattened WS_eff.
    """
    ws_sample, wd_sample = params

    # Simple caching key (consider hashing for more complex scenarios)
    cache_key = (round(ws_sample, 2), round(wd_sample, 2))

    if cache_key in _wf_model_cache:
        sim_res = _wf_model_cache[cache_key]
    else:
        # Run PyWake simulation for the sampled parameters
        # Note: This assumes a single WS/WD defines the 'state' for SA.
        # For time-series sensitivity, the approach would need modification.
        try:
             # Using ws=[ws_sample] forces calculation even if ws_sample matches a default bin
            sim_res = wf_model(x, y, ws=[ws_sample], wd=[wd_sample])
            _wf_model_cache[cache_key] = sim_res # Store result
        except Exception as e:
            print(f"Error during PyWake simulation for ws={ws_sample}, wd={wd_sample}: {e}")
            # Return NaN array of correct size on error to avoid crashing SALib
            return np.full(num_grid_points, np.nan)

    # Calculate the flow map
    try:
        flow_map = sim_res.flow_map(grid=flow_map_grid, ws=[ws_sample], wd=[wd_sample])
        # Extract and flatten the effective wind speed grid
        ws_eff_flat = flow_map.WS_eff.values.flatten()
        return ws_eff_flat
    except Exception as e:
        print(f"Error calculating flow map for ws={ws_sample}, wd={wd_sample}: {e}")
        # Return NaN array of correct size on error
        return np.full(num_grid_points, np.nan)


# --- 5. Generate Samples and Run Model ---
# Choose N (must be a power of 2 for Saltelli)
# Total runs = N * (num_vars + 2)
N = 64 # Power of 2. Increase for more accuracy, but significantly increases runtime.
print(f"Generating {N * (problem['num_vars'] + 2)} parameter samples using Saltelli's method...")
param_values = saltelli.sample(problem, N, calc_second_order=False)

# Evaluate the model for each parameter sample set
# This is the most time-consuming part
print(f"Running PyWake {param_values.shape[0]} times...")
# Initialize results array (important for SALib analysis format)
Y = np.full((param_values.shape[0], num_grid_points), np.nan)

for i, p in enumerate(param_values):
    if (i + 1) % 50 == 0: # Progress indicator
        print(f"  Running simulation {i+1}/{param_values.shape[0]}...")
    Y[i, :] = evaluate_pywake_ws_eff(p)

print("Finished running simulations.")

# Check for NaN values which indicate errors during simulation/flow map calculation
if np.isnan(Y).any():
    nan_runs = np.isnan(Y).any(axis=1).sum()
    print(f"Warning: {nan_runs} simulation runs produced NaN values. Sobol results might be affected.")
    # Option: Fill NaN values, e.g., with the mean of the column, or investigate errors.
    # For simplicity here, we proceed, but SALib might handle NaNs differently or error out.
    # Replacing NaNs with column means (mean WS_eff at that grid point across runs)
    # col_means = np.nanmean(Y, axis=0)
    # nan_indices = np.where(np.isnan(Y))
    # Y[nan_indices] = np.take(col_means, nan_indices[1])


# --- 6. Perform Sobol Analysis ---
print("Performing Sobol analysis (this may take time depending on grid size)...")

# Initialize arrays to store sensitivity indices for each grid point
S1_ws = np.full(num_grid_points, np.nan)
ST_ws = np.full(num_grid_points, np.nan)
S1_wd = np.full(num_grid_points, np.nan)
ST_wd = np.full(num_grid_points, np.nan)

# Analyze sensitivity for each grid point (output variable)
# This loop can be slow for large grids. Vectorization might be possible
# depending on how SALib handles multi-output analysis internally, but
# looping ensures clarity here.
for i in range(num_grid_points):
    if (i + 1) % 100 == 0: # Progress indicator
         print(f"  Analyzing sensitivity for grid point {i+1}/{num_grid_points}...")
    
    # Check if the output for this grid point has variance
    if np.nanvar(Y[:, i]) > 1e-10: # Check variance to avoid SALib errors
       try:
            Si = sobol.analyze(problem, Y[:, i], calc_second_order=False, print_to_console=False)

            # Store the indices
            S1_ws[i] = Si['S1'][0] # First-order index for wind_speed
            ST_ws[i] = Si['ST'][0] # Total-order index for wind_speed
            S1_wd[i] = Si['S1'][1] # First-order index for wind_direction
            ST_wd[i] = Si['ST'][1] # Total-order index for wind_direction
       except Exception as e:
           print(f"  Error analyzing point {i}: {e}. Skipping.")
           # Keep NaNs for points that error out during analysis
    else:
        # If no variance (output is constant), sensitivity is zero or undefined
        S1_ws[i], ST_ws[i], S1_wd[i], ST_wd[i] = 0.0, 0.0, 0.0, 0.0


print("Finished Sobol analysis.")

# --- 7. Visualize and Save Sensitivity Maps ---
print("Generating and saving sensitivity maps...")

# Reshape sensitivity indices back to grid format
S1_ws_map = S1_ws.reshape(grid_shape)
ST_ws_map = ST_ws.reshape(grid_shape)
S1_wd_map = S1_wd.reshape(grid_shape)
ST_wd_map = ST_wd.reshape(grid_shape)

# Get grid coordinates for plotting
X_map, Y_map = ref_flow_map.x.values, ref_flow_map.y.values # Use coords from the reference map

def plot_sensitivity_map(data, title, filename, cmap='viridis'):
    """Helper function to plot sensitivity maps."""
    plt.figure(figsize=(10, 8))
    # Use pcolormesh for compatibility with non-uniform grids if needed
    plt.pcolormesh(X_map, Y_map, data, shading='auto', cmap=cmap, vmin=0) # Sensitivity indices are >= 0
    plt.colorbar(label='Sobol Index')
    # Plot wind turbines
    plt.scatter(x, y, c='red', s=50, marker='x', label='Turbines')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(title)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot and save maps for each index
plot_sensitivity_map(S1_ws_map, 'First-Order Sobol Index (S1) of WS_eff to Wind Speed', 'S1_wind_speed.png')
plot_sensitivity_map(ST_ws_map, 'Total-Order Sobol Index (ST) of WS_eff to Wind Speed', 'ST_wind_speed.png')
plot_sensitivity_map(S1_wd_map, 'First-Order Sobol Index (S1) of WS_eff to Wind Direction', 'S1_wind_direction.png', cmap='plasma')
plot_sensitivity_map(ST_wd_map, 'Total-Order Sobol Index (ST) of WS_eff to Wind Direction', 'ST_wind_direction.png', cmap='plasma')

# Plot an example WS_eff map for context
mean_ws_eff_map = np.nanmean(Y, axis=0).reshape(grid_shape)
plot_sensitivity_map(mean_ws_eff_map, f'Mean WS_eff across {param_values.shape[0]} samples', 'mean_WS_eff.png', cmap='coolwarm')


print("Script finished. Sensitivity maps saved.")
