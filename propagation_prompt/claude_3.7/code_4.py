import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os
from datetime import datetime
from SALib.sample import saltelli
from SALib.analyze import sobol
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid
from py_wake.utils.profiling import timeit

# Create output directory for images
output_dir = "sobol_sensitivity_images"
os.makedirs(output_dir, exist_ok=True)

# Define the site and wind turbines
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define the wake model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Load time series data (if not available, we'll create synthetic data)
try:
    data = np.load("time_series.npz")
    time_series_wd = data['wd']
    time_series_ws = data['ws']
    time_series_ti = data['ws_std'] / data['ws']
    time_series_ti = np.minimum(time_series_ti, 0.5)
    time_stamps = np.arange(len(time_series_wd)) / 6 / 24  # Assuming 10-minute data
except:
    # Create synthetic time series data if file not available
    print("Creating synthetic time series data...")
    n_times = 48  # 8 hours of 10-minute data
    time_stamps = np.arange(n_times) / 6 / 24
    
    # Base values
    time_series_wd = np.ones(n_times) * 270  # westerly wind
    time_series_ws = np.ones(n_times) * 8    # 8 m/s
    time_series_ti = np.ones(n_times) * 0.1  # 10% TI
    
    # Add some variability
    time_series_wd += np.random.normal(0, 15, n_times)  # 15 deg std
    time_series_ws += np.random.normal(0, 1, n_times)   # 1 m/s std
    time_series_ti += np.random.normal(0, 0.02, n_times)  # 2% std
    
    # Ensure realistic bounds
    time_series_wd = np.mod(time_series_wd, 360)
    time_series_ws = np.maximum(time_series_ws, 3)
    time_series_ti = np.clip(time_series_ti, 0.05, 0.3)

print(f"Time series length: {len(time_stamps)} points")

# Define the problem for Sobol sensitivity analysis
problem = {
    'num_vars': 3,
    'names': ['wind_direction', 'wind_speed', 'turbulence_intensity'],
    'bounds': [
        [0.9, 1.1],  # Multiplier for wind direction (±10%)
        [0.9, 1.1],  # Multiplier for wind speed (±10%)
        [0.7, 1.3],  # Multiplier for turbulence intensity (±30%)
    ]
}

# Number of Sobol samples
n_samples = 256  # This controls the number of model evaluations: N*(2D+2) where D=3 variables

# Define a fixed grid for flow map evaluation
resolution = 100
grid = HorizontalGrid(x=np.linspace(min(x)-500, max(x)+500, resolution),
                     y=np.linspace(min(y)-500, max(y)+500, resolution),
                     h=windTurbines.hub_height())

# Function to evaluate the model with given parameters
def evaluate_model(wd, ws, ti, grid):
    # Run the wind farm simulation
    sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
    
    # Get the flow map
    flow_map = sim_res.flow_map(grid=grid, wd=wd, ws=ws)
    
    # Return the effective wind speed field
    return flow_map.WS_eff_xylk.values

# Number of timesteps to analyze (limit to save computation time)
n_timesteps = min(10, len(time_stamps))

# Loop through selected timesteps
for t_idx in range(n_timesteps):
    print(f"Processing timestep {t_idx+1}/{n_timesteps} - Time: {time_stamps[t_idx]*24:.1f} hours")
    
    # Get base values for this timestep
    base_wd = time_series_wd[t_idx]
    base_ws = time_series_ws[t_idx]
    base_ti = time_series_ti[t_idx]
    
    print(f"  Base values: WD={base_wd:.1f}°, WS={base_ws:.1f} m/s, TI={base_ti:.3f}")
    
    # Generate samples for this timestep using Saltelli's method
    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
    
    # Apply the multipliers to the base values
    wd_samples = base_wd * param_values[:, 0]
    ws_samples = base_ws * param_values[:, 1]
    ti_samples = base_ti * param_values[:, 2]
    
    # Evaluate the model for all samples
    print(f"  Running {len(param_values)} model evaluations...")
    Y = np.zeros((len(param_values), resolution, resolution))
    
    for i, (wd, ws, ti) in enumerate(tqdm(zip(wd_samples, ws_samples, ti_samples), total=len(param_values))):
        Y[i] = evaluate_model(wd, ws, ti, grid)
    
    # Reshape Y for SALib
    Y_reshaped = Y.reshape(len(param_values), -1)
    
    # Calculate Sobol indices for each grid point
    print("  Calculating Sobol indices...")
    total_Si = np.zeros((3, resolution, resolution))
    
    # Analyze each grid point
    for i in tqdm(range(Y_reshaped.shape[1])):
        Yi = Y_reshaped[:, i]
        # Skip points with no variation
        if np.std(Yi) < 1e-10:
            continue
        
        # Calculate first-order Sobol indices
        Si = sobol.analyze(problem, Yi, calc_second_order=False)
        total_Si[0, i // resolution, i % resolution] = Si['ST'][0]  # Total effect index for wind direction
        total_Si[1, i // resolution, i % resolution] = Si['ST'][1]  # Total effect index for wind speed
        total_Si[2, i // resolution, i % resolution] = Si['ST'][2]  # Total effect index for turbulence intensity
    
    # Create plots for each parameter's sensitivity
    param_names = ['Wind Direction', 'Wind Speed', 'Turbulence Intensity']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Generate x, y grid for plotting
    X, Y = np.meshgrid(grid.x, grid.y)
    
    for p in range(3):
        im = axes[p].contourf(X, Y, total_Si[p].T, levels=np.linspace(0, 1, 11), cmap='viridis')
        axes[p].set_title(f'Sensitivity to {param_names[p]}')
        axes[p].set_xlabel('x [m]')
        axes[p].set_ylabel('y [m]')
        axes[p].scatter(x, y, color='red', s=10, marker='x')  # Mark turbine positions
        fig.colorbar(im, ax=axes[p])
    
    fig.suptitle(f'Sobol Sensitivity Analysis - Time: {time_stamps[t_idx]*24:.1f} hours\n'
                f'WD={base_wd:.1f}°, WS={base_ws:.1f} m/s, TI={base_ti:.3f}')
    
    # Save the figure
    timestamp_str = f"{time_stamps[t_idx]*24:.1f}".replace('.', '_')
    fig_path = os.path.join(output_dir, f"sobol_sensitivity_t{timestamp_str}h.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print(f"  Saved figure to {fig_path}")

print("Completed Sobol sensitivity analysis and saved all figures.")

# Also create a base flow map without uncertainty to show reference conditions
plt.figure(figsize=(10, 6))
for t_idx in range(min(3, n_timesteps)):
    plt.figure(figsize=(10, 6))
    base_wd = time_series_wd[t_idx]
    base_ws = time_series_ws[t_idx]
    
    # Run the wind farm simulation with base values
    sim_res = wf_model(x, y, wd=base_wd, ws=base_ws)
    
    # Get and plot the flow map
    flow_map = sim_res.flow_map(grid=grid, wd=base_wd, ws=base_ws)
    flow_map.plot_wake_map()
    plt.title(f'Reference Wake Map - Time: {time_stamps[t_idx]*24:.1f} hours\n'
             f'WD={base_wd:.1f}°, WS={base_ws:.1f} m/s, TI={time_series_ti[t_idx]:.3f}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    
    # Save the figure
    timestamp_str = f"{time_stamps[t_idx]*24:.1f}".replace('.', '_')
    ref_path = os.path.join(output_dir, f"reference_flow_t{timestamp_str}h.png")
    plt.tight_layout()
    plt.savefig(ref_path, dpi=200)
    plt.close()

print("Script execution completed successfully!")
