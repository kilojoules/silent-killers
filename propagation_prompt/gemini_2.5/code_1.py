# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# PyWake imports based on the documentation [cite: 3, 53]
try:
    import py_wake
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80 # Example site and turbine [cite: 3]
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014 # Example wake model [cite: 3]
    from py_wake import HorizontalGrid # For defining flow map grids [cite: 59]
    # Import SALib for sensitivity analysis (needs to be installed separately: pip install SALib)
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ModuleNotFoundError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure PyWake and SALib are installed ('pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git SALib')")
    exit()

# %% Setup PyWake Model [cite: 3]
print("Setting up PyWake model...")
site = Hornsrev1Site()
x, y = site.initial_position.T # Get turbine coordinates
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555) # Example model setup [cite: 3]
print("Model setup complete.")
print(wf_model)

# %% Define Time Series Data (Example based on documentation [cite: 28])
print("Loading example time series data...")
# Using a small sample for demonstration
try:
    # Load example time series data if available, otherwise create dummy data
    from py_wake.examples.data import example_data_path
    d = np.load(example_data_path + "/time_series.npz")
    n_points = 100 # Use fewer points for faster demo
    wd_ts = d['wd'][:n_points]
    ws_ts = d['ws'][:n_points]
    ws_std_ts = d['ws_std'][:n_points]
    ti_ts = np.minimum(ws_std_ts / ws_ts, 0.5)
    time_stamps = np.arange(n_points) # Simple time index
    print(f"Loaded {n_points} time steps.")
except FileNotFoundError:
    print("Example time series file not found. Creating dummy data.")
    n_points = 10
    wd_ts = np.linspace(260, 280, n_points)
    ws_ts = np.linspace(8, 12, n_points)
    ti_ts = np.linspace(0.05, 0.15, n_points)
    time_stamps = np.arange(n_points)

# %% Define Uncertain Parameters and Sensitivity Analysis Problem
print("Defining sensitivity analysis problem...")
# Define the parameters for sensitivity analysis and their bounds/distributions
# This requires knowledge of "realistic measurement uncertainties"
# Example: Uncertainty in Wind Speed (WS), Wind Direction (WD), Turbulence Intensity (TI)
# Let's assume Normal distributions and sample within +/- 3 standard deviations
ws_mean, ws_std_unc = ws_ts, 0.5 # Mean from time series, assumed std dev for uncertainty
wd_mean, wd_std_unc = wd_ts, 2.0 # Mean from time series, assumed std dev for uncertainty (degrees)
ti_mean, ti_std_unc = ti_ts, 0.02 # Mean from time series, assumed std dev for uncertainty

# SALib problem definition
# We analyze sensitivity for *one time step* at a time
problem = {
    'num_vars': 3,
    'names': ['WS', 'WD', 'TI'],
    # Define bounds roughly based on +/- 3 sigma for Sobol sampling
    # Note: SALib's Sobol sampler uses uniform distributions within bounds.
    # For non-uniform distributions (like Normal), transformation or different sampling methods might be needed.
    # This is a simplified approach for demonstration.
    'bounds': [[ws_mean.min() - 3*ws_std_unc, ws_mean.max() + 3*ws_std_unc],
               [wd_mean.min() - 3*wd_std_unc, wd_mean.max() + 3*wd_std_unc],
               [ti_mean.min() - 3*ti_std_unc, ti_mean.max() + 3*ti_std_unc]]
}
print("Problem defined:")
print(problem)

# Generate Sobol samples
# N determines the number of samples: Total samples = N * (2D + 2), where D is num_vars
N = 64 # Power of 2 often recommended. Adjust based on computational resources.
param_values = sobol_sample.sample(problem, N, calc_second_order=False)
print(f"Generated {param_values.shape[0]} Sobol samples.")

# Define the grid for flow map calculation [cite: 59, 60]
grid = HorizontalGrid(resolution=50, extend=0.1) # Lower resolution for faster demo

# %% Function to run PyWake for a given parameter set and time step
def evaluate_pywake_flowmap(params, time_index):
    """Runs PyWake simulation for one parameter sample and returns WS_eff flow map."""
    ws_val, wd_val, ti_val = params
    try:
        # Run simulation for a single flow case corresponding to the sample
        # Note: We run the wf_model for the specific sample, not the full time series simulation object.
        sim_res_sample = wf_model(x, y, wd=[wd_val], ws=[ws_val], ti=ti_val)

        # Calculate the flow map for this specific case [cite: 51, 53]
        flow_map_sample = sim_res_sample.flow_map(grid=grid, wd=wd_val, ws=ws_val)

        # Return the effective wind speed (WS_eff) field flattened
        # WS_eff is typically accessed from the flow_map object like this:
        ws_eff_map = flow_map_sample.WS_eff_xylk # Shape (x, y, wd_index, ws_index)
        # Since we run for single wd/ws, indices are 0. Flatten for SALib.
        return ws_eff_map[:, :, 0, 0].flatten()

    except Exception as e:
        print(f"Error during PyWake evaluation for params {params}: {e}")
        # Return an array of NaNs matching the expected output size if an error occurs
        # Get grid dimensions
        X, Y, x_flat, y_flat, h_flat = grid._get_grid_point_xyh(x_0=x, y_0=y, h_0=windTurbines.hub_height())
        num_grid_points = x_flat.shape[0]
        return np.full(num_grid_points, np.nan)


# %% Run Simulations for Samples and Perform Sensitivity Analysis for Selected Time Steps
# Select a few time steps to analyze sensitivity
# time_indices_to_analyze = [0, len(time_stamps) // 2, len(time_stamps) - 1]
time_indices_to_analyze = [0, n_points // 2, n_points -1] # Analyze start, middle, end


for t_idx in time_indices_to_analyze:
    print(f"\n--- Analyzing Time Step Index: {t_idx} ---")
    print(f"  WS={ws_ts[t_idx]:.2f}, WD={wd_ts[t_idx]:.2f}, TI={ti_ts[t_idx]:.3f}")

    # Update problem bounds for the current time step's mean values (optional refinement)
    problem['bounds'] = [
        [ws_ts[t_idx] - 3*ws_std_unc, ws_ts[t_idx] + 3*ws_std_unc],
        [wd_ts[t_idx] - 3*wd_std_unc, wd_ts[t_idx] + 3*wd_std_unc],
        [ti_ts[t_idx] - 3*ti_std_unc, ti_ts[t_idx] + 3*ti_std_unc]
    ]
    # Regenerate samples for the specific time step bounds (more accurate)
    param_values_t = sobol_sample.sample(problem, N, calc_second_order=False)
    print(f"  Generated {param_values_t.shape[0]} Sobol samples for time step {t_idx}.")

    # Evaluate the model for each sample point at this time step
    print(f"  Running {param_values_t.shape[0]} simulations for time step {t_idx}...")
    # The output Y will have shape (num_samples, num_grid_points)
    Y = np.array([evaluate_pywake_flowmap(p, t_idx) for p in param_values_t])
    print("  Simulations complete.")

    # Check if all outputs were NaN (due to errors)
    if np.isnan(Y).all():
        print(f"  Skipping sensitivity analysis for time step {t_idx} due to evaluation errors.")
        continue

    # Perform Sobol analysis on the results for each grid point
    print(f"  Performing Sobol analysis for {Y.shape[1]} grid points...")
    num_grid_points = Y.shape[1]
    Si_ST = np.full((num_grid_points, 1), np.nan) # Store Total Sobol Index (ST)

    # Analyze point-by-point (can be slow for high-res grids)
    # Potential optimization: Vectorize if SALib allows or use parallel processing
    analysis_errors = 0
    for i in range(num_grid_points):
        # Filter out NaN results for this grid point before analysis
        y_point = Y[:, i]
        valid_indices = ~np.isnan(y_point)
        if np.sum(valid_indices) < 2: # Need at least 2 valid points for variance analysis
             # print(f"    Skipping grid point {i}: Not enough valid data.")
             Si_ST[i] = np.nan
             analysis_errors += 1
             continue

        y_point_valid = y_point[valid_indices]
        param_values_valid = param_values_t[valid_indices]

        # Check variance - SALib requires non-zero variance
        if np.var(y_point_valid) < 1e-10:
             # print(f"    Skipping grid point {i}: Zero variance in output.")
             Si_ST[i] = 0.0 # Or NaN, depending on desired handling
             analysis_errors += 1
             continue

        try:
             # Perform analysis only with valid data
             Si = sobol_analyze.analyze(problem, y_point_valid, calc_second_order=False, print_to_console=False)
             Si_ST[i] = Si['ST'] # Total order index
        except Exception as e:
             # print(f"    Error during Sobol analysis for grid point {i}: {e}")
             Si_ST[i] = np.nan # Mark as NaN if analysis fails
             analysis_errors += 1

    if analysis_errors > 0:
        print(f"  Sobol analysis completed with {analysis_errors} errors/skips out of {num_grid_points} grid points.")
    else:
        print("  Sobol analysis complete.")


    # %% Visualize Sensitivity Map
    print("  Visualizing sensitivity map...")
    # Get grid coordinates for plotting
    X_plot, Y_plot, _, _, _ = grid._get_grid_point_xyh(x_0=x, y_0=y, h_0=windTurbines.hub_height())

    # Reshape the Sobol index (ST) back into the grid shape
    # Need to know the original shape from the grid object
    x_unique = np.sort(np.unique(X_plot))
    y_unique = np.sort(np.unique(Y_plot))
    nx, ny = len(x_unique), len(y_unique)

    if nx * ny == num_grid_points:
        try:
            # Ensure Si_ST is reshaped correctly according to X_plot, Y_plot correspondence
            # Typically, meshgrid output needs careful handling for reshaping flattened arrays
            # Create a mapping from flattened index back to 2D grid location
            sensitivity_map = np.full((ny, nx), np.nan) # Initialize with NaNs

            # Find the mapping (assuming regular grid)
            x_map = {val: i for i, val in enumerate(x_unique)}
            y_map = {val: i for i, val in enumerate(y_unique)}

            if Si_ST.shape[0] == X_plot.flatten().shape[0]: # Check if sizes match
                 for idx_flat, (x_coord, y_coord) in enumerate(zip(X_plot.flatten(), Y_plot.flatten())):
                     if x_coord in x_map and y_coord in y_map:
                         row_idx, col_idx = y_map[y_coord], x_map[x_coord]
                         sensitivity_map[row_idx, col_idx] = Si_ST[idx_flat, 0]
                     else:
                         # This case might happen with irregular grids or if grid points outside bounds were generated
                         print(f"Warning: Coordinate ({x_coord}, {y_coord}) not found in unique grid coordinates.")

                 plt.figure(figsize=(10, 8))
                 cf = plt.contourf(x_unique, y_unique, sensitivity_map, levels=50, cmap='viridis')
                 plt.colorbar(cf, label='Total Sobol Index (ST) for WS_eff')
                 # Plot turbines for context
                 plt.scatter(x, y, c='red', marker='x', label='Turbines')
                 plt.xlabel('x [m]')
                 plt.ylabel('y [m]')
                 plt.title(f'WS_eff Sobol Sensitivity (ST) - Time Step {t_idx} (WD={wd_ts[t_idx]:.1f}°, WS={ws_ts[t_idx]:.1f} m/s)')
                 plt.axis('equal')
                 plt.legend()
                 # Save the figure
                 filename = f'sobol_sensitivity_ws_eff_time_{t_idx}.png'
                 plt.savefig(filename)
                 print(f"  Saved sensitivity map: {filename}")
                 # plt.show() # Optionally display plot interactively
                 plt.close() # Close plot to free memory

            else:
                 print(f"  Error: Mismatch between Sobol results ({Si_ST.shape[0]}) and grid points ({X_plot.flatten().shape[0]})")

        except ValueError as e:
            print(f"  Error reshaping sensitivity results: {e}. Check grid dimensions ({ny}x{nx}={nx*ny}) vs results ({num_grid_points}).")
        except Exception as e:
            print(f"  An unexpected error occurred during plotting: {e}")
    else:
        print(f"  Error: Grid dimensions ({ny}x{nx}={nx*ny}) do not match the number of analyzed points ({num_grid_points}). Cannot reshape.")


print("\nScript finished.")
