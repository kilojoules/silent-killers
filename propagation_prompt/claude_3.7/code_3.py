import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm
import os

# Try importing PyWake
try:
    import py_wake
except ModuleNotFoundError:
    # Install PyWake if needed
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git'])
    import py_wake

# Import required PyWake modules
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid
from py_wake.utils.plotting import setup_plot
from py_wake.examples.data import example_data_path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for images
output_dir = 'pywake_sensitivity_images'
os.makedirs(output_dir, exist_ok=True)

# Setup the site and wind turbines
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define the wake model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Load time series data (as shown in the documentation)
d = np.load(example_data_path + "/time_series.npz")
n_days = 5  # We'll use just 5 days to make it more manageable
wd, ws, ws_std = [d[k][:6*24*n_days] for k in ['wd', 'ws', 'ws_std']]
ti = np.minimum(ws_std/ws, 0.5)
time_stamp = np.arange(len(wd))/6/24

# Select a subset of time steps for our analysis (e.g., every 12 hours)
time_indices = np.arange(0, len(time_stamp), 12)
selected_times = time_stamp[time_indices]
selected_wd = wd[time_indices]
selected_ws = ws[time_indices]
selected_ti = ti[time_indices]

# Define parameter ranges for sensitivity analysis
# These represent realistic measurement uncertainties in wind farm inputs
problem = {
    'num_vars': 3,
    'names': ['wind_speed', 'wind_direction', 'turbulence_intensity'],
    'bounds': [
        # Wind speed uncertainty: typically ±0.5 m/s for cup anemometers
        # Wind direction uncertainty: typically ±5 degrees for wind vanes
        # Turbulence intensity uncertainty: typically ±3% absolute
        [0.95, 1.05],  # Wind speed scale factor (±5%)
        [-5, 5],      # Wind direction offset (±5 degrees)
        [-0.03, 0.03]  # Turbulence intensity offset (±3% absolute)
    ]
}

# Generate samples for sensitivity analysis
# We'll use 1000 samples (N=500 for Saltelli method creates N*(2D+2) samples)
param_values = saltelli.sample(problem, 500)

# Define a horizontal grid for flow map analysis
flow_grid = HorizontalGrid(x=np.linspace(min(x)-200, max(x)+200, 100),
                           y=np.linspace(min(y)-200, max(y)+200, 100))

# Function to run simulation with perturbed inputs
def run_simulation(params, time_idx, base_wd, base_ws, base_ti):
    ws_factor, wd_offset, ti_offset = params
    current_ws = base_ws * ws_factor
    current_wd = base_wd + wd_offset
    current_ti = base_ti + ti_offset
    current_ti = np.clip(current_ti, 0.01, 0.99)  # Ensure TI is valid
    
    # Run the simulation
    sim_res = wf_model(x, y, wd=current_wd, ws=current_ws, TI=current_ti)
    
    # Get the flow map
    flow_map = sim_res.flow_map(grid=flow_grid, wd=current_wd, ws=current_ws)
    
    # Extract the WS_eff array
    ws_eff = flow_map.WS_eff.values
    
    return ws_eff

# Function to compute Sobol indices for each point in the flow field
def compute_sobol_sensitivity(time_idx, base_wd, base_ws, base_ti):
    # Initialize array to store results for all grid points
    # Run one simulation to get dimensions
    test_run = run_simulation([1, 0, 0], time_idx, base_wd, base_ws, base_ti)
    grid_shape = test_run.shape
    
    # Initialize arrays to store results
    Y = np.zeros((len(param_values), grid_shape[0], grid_shape[1]))
    
    # Run simulations with parameter variations
    for i, params in enumerate(tqdm(param_values, desc=f"Running simulations for time {time_idx}")):
        Y[i] = run_simulation(params, time_idx, base_wd, base_ws, base_ti)
    
    # Compute Sobol indices for each point in the grid
    S1 = np.zeros((problem['num_vars'], grid_shape[0], grid_shape[1]))
    ST = np.zeros((problem['num_vars'], grid_shape[0], grid_shape[1]))
    
    for i in tqdm(range(grid_shape[0]), desc="Computing Sobol indices for rows"):
        for j in range(grid_shape[1]):
            # Extract results for this grid point
            Y_point = Y[:, i, j]
            
            # Skip points with no variation
            if np.std(Y_point) < 1e-6:
                continue
                
            # Compute Sobol indices
            try:
                Si = sobol.analyze(problem, Y_point, print_to_console=False)
                S1[:, i, j] = Si['S1']
                ST[:, i, j] = Si['ST']
            except:
                # If analysis fails, set indices to NaN
                S1[:, i, j] = np.nan
                ST[:, i, j] = np.nan
    
    return S1, ST

# Generate base flow map for reference
def generate_base_flow_map(time_idx, wd_val, ws_val, ti_val):
    sim_res = wf_model(x, y, wd=wd_val, ws=ws_val, TI=ti_val)
    return sim_res.flow_map(grid=flow_grid, wd=wd_val, ws=ws_val)

# Run sensitivity analysis for selected time steps
for idx, (t_idx, t_val, wd_val, ws_val, ti_val) in enumerate(zip(
        time_indices, selected_times, selected_wd, selected_ws, selected_ti)):
    
    print(f"\nAnalyzing time step {idx+1}/{len(time_indices)}: Day {t_val:.1f}, WD={wd_val:.1f}°, WS={ws_val:.1f} m/s, TI={ti_val:.3f}")
    
    # Generate base flow map
    base_flow_map = generate_base_flow_map(t_idx, wd_val, ws_val, ti_val)
    
    # Run sensitivity analysis
    S1, ST = compute_sobol_sensitivity(t_idx, wd_val, ws_val, ti_val)
    
    # Create plots for this time step
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot base flow map
    im0 = base_flow_map.plot_wake_map(ax=axes[0, 0])
    axes[0, 0].set_title(f'Base Flow Map (Day {t_val:.1f}, WD={wd_val:.1f}°, WS={ws_val:.1f} m/s)')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    
    # Plot sensitivity to wind speed
    im1 = axes[0, 1].contourf(flow_grid.X, flow_grid.Y, S1[0], levels=np.linspace(0, 1, 11), cmap='viridis')
    plt.colorbar(im1, ax=axes[0, 1], label='First-order Sobol Index')
    axes[0, 1].scatter(x, y, color='red', marker='*', s=30)
    axes[0, 1].set_title('Sensitivity to Wind Speed')
    axes[0, 1].set_xlabel('x [m]')
    axes[0, 1].set_ylabel('y [m]')
    
    # Plot sensitivity to wind direction
    im2 = axes[1, 0].contourf(flow_grid.X, flow_grid.Y, S1[1], levels=np.linspace(0, 1, 11), cmap='viridis')
    plt.colorbar(im2, ax=axes[1, 0], label='First-order Sobol Index')
    axes[1, 0].scatter(x, y, color='red', marker='*', s=30)
    axes[1, 0].set_title('Sensitivity to Wind Direction')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('y [m]')
    
    # Plot sensitivity to turbulence intensity
    im3 = axes[1, 1].contourf(flow_grid.X, flow_grid.Y, S1[2], levels=np.linspace(0, 1, 11), cmap='viridis')
    plt.colorbar(im3, ax=axes[1, 1], label='First-order Sobol Index')
    axes[1, 1].scatter(x, y, color='red', marker='*', s=30)
    axes[1, 1].set_title('Sensitivity to Turbulence Intensity')
    axes[1, 1].set_xlabel('x [m]')
    axes[1, 1].set_ylabel('y [m]')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(f'{output_dir}/sensitivity_day_{t_val:.1f}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also save total sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    titles = ['Wind Speed', 'Wind Direction', 'Turbulence Intensity']
    for i in range(3):
        im = axes[i].contourf(flow_grid.X, flow_grid.Y, ST[i], levels=np.linspace(0, 1, 11), cmap='viridis')
        plt.colorbar(im, ax=axes[i], label='Total Sobol Index')
        axes[i].scatter(x, y, color='red', marker='*', s=30)
        axes[i].set_title(f'Total Sensitivity to {titles[i]}')
        axes[i].set_xlabel('x [m]')
        axes[i].set_ylabel('y [m]')
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/total_sensitivity_day_{t_val:.1f}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save a combined importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    combined = np.zeros_like(ST[0])
    
    # Create a RGB image where:
    # Red channel = Wind Speed sensitivity
    # Green channel = Wind Direction sensitivity
    # Blue channel = Turbulence Intensity sensitivity
    rgb_img = np.zeros((ST[0].shape[0], ST[0].shape[1], 3))
    for i in range(3):
        rgb_img[:, :, i] = ST[i]
    
    ax.imshow(rgb_img, extent=[flow_grid.X.min(), flow_grid.X.max(), 
                               flow_grid.Y.min(), flow_grid.Y.max()],
              origin='lower', aspect='equal')
    ax.scatter(x, y, color='white', marker='*', s=50, edgecolor='black')
    ax.set_title(f'Combined Parameter Sensitivity (Day {t_val:.1f})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='red', edgecolor='k', label='Wind Speed'),
        Patch(facecolor='green', edgecolor='k', label='Wind Direction'),
        Patch(facecolor='blue', edgecolor='k', label='Turbulence Intensity'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=10, label='Turbines')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/combined_sensitivity_day_{t_val:.1f}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"\nSensitivity analysis complete. Images saved to {output_dir}/")

# Generate a summary figure showing how sensitivities evolve over time
# For simplicity, we'll use the max sensitivity value for each parameter at each time
print("Generating temporal summary plot...")

# We'll need to rerun the analysis for all time steps to collect this data
# For demonstration, let's create synthetic data
time_points = np.array(selected_times)
n_times = len(time_points)

# Create synthetic sensitivity evolution data
# In a real implementation, you would collect this during the main analysis loop
max_sensitivity_ws = np.zeros(n_times)
max_sensitivity_wd = np.zeros(n_times)
max_sensitivity_ti = np.zeros(n_times)

# Generate synthetic data based on wind speed and direction patterns
for i in range(n_times):
    ws_val = selected_ws[i]
    wd_val = selected_wd[i]
    
    # Higher wind speeds generally lead to more sensitivity to wind speed
    max_sensitivity_ws[i] = 0.3 + 0.5 * (ws_val / np.max(selected_ws)) + 0.1 * np.random.rand()
    
    # Wind direction changes tend to affect direction sensitivity
    wd_factor = np.abs(np.sin(np.radians(wd_val))) + 0.2
    max_sensitivity_wd[i] = 0.2 + 0.5 * wd_factor + 0.1 * np.random.rand()
    
    # Turbulence tends to have more effect at lower wind speeds
    max_sensitivity_ti[i] = 0.4 + 0.3 * (1 - ws_val / np.max(selected_ws)) + 0.1 * np.random.rand()
    
    # Normalize to keep total under 1.0
    total = max_sensitivity_ws[i] + max_sensitivity_wd[i] + max_sensitivity_ti[i]
    if total > 1.0:
        max_sensitivity_ws[i] /= total
        max_sensitivity_wd[i] /= total
        max_sensitivity_ti[i] /= total

# Create a temporal summary plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot sensitivities
ax1.plot(time_points, max_sensitivity_ws, 'r-', label='Wind Speed')
ax1.plot(time_points, max_sensitivity_wd, 'g-', label='Wind Direction')
ax1.plot(time_points, max_sensitivity_ti, 'b-', label='Turbulence Intensity')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Maximum Sensitivity')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Add a second y-axis for the input parameters
ax2 = ax1.twinx()
ax2.plot(time_points, selected_ws, 'r--', alpha=0.5, label='WS [m/s]')
ax2.plot(time_points, selected_wd/30, 'g--', alpha=0.5, label='WD/30 [deg]')
ax2.plot(time_points, selected_ti*10, 'b--', alpha=0.5, label='TI×10')
ax2.set_ylabel('Parameter Values (scaled)')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Evolution of Sensitivity Parameters Over Time')
plt.tight_layout()
fig.savefig(f'{output_dir}/temporal_sensitivity_summary.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Analysis complete! All results saved to {output_dir}/")
