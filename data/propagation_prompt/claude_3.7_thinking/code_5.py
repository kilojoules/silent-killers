# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os
from datetime import datetime

# Import PyWake components
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid
from py_wake.utils.plotting import setup_plot

# Import SALib for Sobol sensitivity analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

# Create output directory for saving images
output_dir = "flow_sensitivity_images"
os.makedirs(output_dir, exist_ok=True)

# Set up the wind farm using the Hornsrev1 example data
print("Setting up wind farm model...")
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Define a flow map grid for visualization
# Using moderate resolution for balance between detail and computation time
horizontal_grid = HorizontalGrid(resolution=40, extend=0.5)

# Create time series data for analysis
# We'll simulate a 24-hour period with varying wind conditions
n_hours = 24
time_stamps = np.linspace(0, n_hours, n_hours+1)  # hourly data
# Wind speed varies between 6-10 m/s through the day
base_wind_speed = 8 + 2 * np.sin(2 * np.pi * time_stamps / 24)
# Wind direction varies between 135-225 degrees through the day  
base_wind_direction = 180 + 45 * np.sin(2 * np.pi * time_stamps / 24)

# Define realistic measurement uncertainties
# These are typical values for meteorological measurements
ws_uncertainty = 0.5  # m/s (typical for cup anemometers)
wd_uncertainty = 5.0  # degrees (typical for wind vanes)
ti_uncertainty = 0.02  # absolute uncertainty in turbulence intensity

# Define the parameter space for Sobol sensitivity analysis
problem = {
    'num_vars': 3,
    'names': ['wind_speed', 'wind_direction', 'turbulence_intensity'],
    'bounds': [[-1, 1],  # Normalized bounds for wind speed uncertainty
               [-1, 1],  # Normalized bounds for wind direction uncertainty
               [-1, 1]]  # Normalized bounds for TI uncertainty
}

# Generate samples for Sobol analysis
# N determines the number of model evaluations: N*(2D+2) where D is the number of parameters
# Smaller N is faster but less accurate
N = 24  # This gives (2*N+2)*3 = 150 model runs
print(f"Generating {N*(2*len(problem['names'])+2)} Sobol samples...")
param_values = saltelli.sample(problem, N, calc_second_order=False)

# Function to calculate Sobol sensitivity indices for a specific time point
def calculate_sobol_indices(time_idx):
    ref_ws = base_wind_speed[time_idx]
    ref_wd = base_wind_direction[time_idx]
    ref_ti = 0.1  # Reference turbulence intensity
    
    print(f"Calculating Sobol indices at time {time_stamps[time_idx]:.1f}h")
    print(f"Reference conditions: WS={ref_ws:.1f}m/s, WD={ref_wd:.1f}deg, TI={ref_ti:.2f}")
    
    # Generate perturbed samples around the reference point
    ws_samples = ref_ws + ws_uncertainty * param_values[:, 0]
    wd_samples = ref_wd + wd_uncertainty * param_values[:, 1]
    ti_samples = ref_ti + ti_uncertainty * param_values[:, 2]
    
    # Run model for all samples and collect WS_eff fields
    ws_eff_fields = []
    for i, (ws, wd, ti) in enumerate(zip(ws_samples, wd_samples, ti_samples)):
        print(f"  Running sample {i+1}/{len(ws_samples)}")
        # Run the wind farm model with the perturbed parameters
        sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
        flow_map = sim_res.flow_map(grid=horizontal_grid, wd=wd, ws=ws)
        ws_eff_fields.append(flow_map.WS_eff.values)
    
    # Convert to numpy array
    ws_eff_fields = np.array(ws_eff_fields)
    
    # For efficiency, subsample the grid points for Sobol analysis
    # This reduces computation while still providing a good representation
    stride = 2  # Analyze every 2nd point in each direction
    grid_shape = ws_eff_fields[0].shape
    subsampled_shape = (grid_shape[0]//stride, grid_shape[1]//stride)
    
    S1_ws = np.zeros(subsampled_shape)
    S1_wd = np.zeros(subsampled_shape)
    S1_ti = np.zeros(subsampled_shape)
    
    # Analyze subsampled grid points
    for i in tqdm(range(0, grid_shape[0], stride), desc="Calculating Sobol indices"):
        for j in range(0, grid_shape[1], stride):
            # Extract model outputs for this grid point
            Y = ws_eff_fields[:, i, j]
            
            # Calculate Sobol indices (first order only)
            Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            
            # Store first-order indices
            S1_ws[i//stride, j//stride] = Si['S1'][0]  # Wind speed sensitivity
            S1_wd[i//stride, j//stride] = Si['S1'][1]  # Wind direction sensitivity
            S1_ti[i//stride, j//stride] = Si['S1'][2]  # Turbulence intensity sensitivity
    
    # Upscale the results to match the original grid
    S1_ws_full = np.repeat(np.repeat(S1_ws, stride, axis=0), stride, axis=1)
    S1_wd_full = np.repeat(np.repeat(S1_wd, stride, axis=0), stride, axis=1)
    S1_ti_full = np.repeat(np.repeat(S1_ti, stride, axis=0), stride, axis=1)
    
    # Trim to match original grid dimensions exactly
    S1_ws_full = S1_ws_full[:grid_shape[0], :grid_shape[1]]
    S1_wd_full = S1_wd_full[:grid_shape[0], :grid_shape[1]]
    S1_ti_full = S1_ti_full[:grid_shape[0], :grid_shape[1]]
    
    # Get reference flow field
    sim_res_ref = wf_model(x, y, wd=ref_wd, ws=ref_ws, TI=ref_ti)
    flow_map_ref = sim_res_ref.flow_map(grid=horizontal_grid, wd=ref_wd, ws=ref_ws)
    ws_eff_ref = flow_map_ref.WS_eff.values
    
    return S1_ws_full, S1_wd_full, S1_ti_full, ref_ws, ref_wd, ref_ti, ws_eff_ref

# Select time points to analyze
# We'll use fewer points to reduce computation time
selected_times = [0, 6, 12, 18, 23]  # Indices for specific hours

for time_idx in selected_times:
    # Calculate Sobol indices
    S1_ws, S1_wd, S1_ti, ref_ws, ref_wd, ref_ti, ws_eff_ref = calculate_sobol_indices(time_idx)
    
    # Create figure for visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    axs = axs.flatten()
    
    # Plot reference flow field
    im0 = axs[0].imshow(ws_eff_ref, origin='lower', cmap='Blues',
                      extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                              horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[0].set_title(f'Reference WS_eff (WS={ref_ws:.1f}m/s, WD={ref_wd:.1f}deg, TI={ref_ti:.2f})')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
    plt.colorbar(im0, ax=axs[0], label='Effective Wind Speed [m/s]')
    
    # Plot Sobol sensitivity to wind speed
    im1 = axs[1].imshow(S1_ws, origin='lower', cmap='viridis', vmin=0, vmax=1,
                      extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                              horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[1].set_title(f'Sobol Sensitivity to Wind Speed (±{ws_uncertainty}m/s)')
    axs[1].set_xlabel('x [m]')
    axs[1].set_ylabel('y [m]')
    plt.colorbar(im1, ax=axs[1], label='Sobol First-Order Index')
    
    # Plot Sobol sensitivity to wind direction
    im2 = axs[2].imshow(S1_wd, origin='lower', cmap='viridis', vmin=0, vmax=1,
                      extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                              horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[2].set_title(f'Sobol Sensitivity to Wind Direction (±{wd_uncertainty}deg)')
    axs[2].set_xlabel('x [m]')
    axs[2].set_ylabel('y [m]')
    plt.colorbar(im2, ax=axs[2], label='Sobol First-Order Index')
    
    # Plot Sobol sensitivity to turbulence intensity
    im3 = axs[3].imshow(S1_ti, origin='lower', cmap='viridis', vmin=0, vmax=1,
                      extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                              horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[3].set_title(f'Sobol Sensitivity to Turbulence Intensity (±{ti_uncertainty})')
    axs[3].set_xlabel('x [m]')
    axs[3].set_ylabel('y [m]')
    plt.colorbar(im3, ax=axs[3], label='Sobol First-Order Index')
    
    # Add turbine positions to all plots
    for ax in axs:
        ax.scatter(x, y, color='red', s=10, label='Wind Turbines')
        ax.legend()
    
    # Add an overall title
    fig.suptitle(f'Sobol Sensitivity Analysis of WS_eff - Time: {time_stamps[time_idx]:.1f} hours', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/sobol_sensitivity_time_{time_stamps[time_idx]:.1f}h.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    print(f"Saved Sobol sensitivity analysis for time {time_stamps[time_idx]:.1f}h to {filename}")

    # Also save the raw sensitivity data for later analysis
    np.savez(f"{output_dir}/sobol_data_time_{time_stamps[time_idx]:.1f}h.npz",
             S1_ws=S1_ws, S1_wd=S1_wd, S1_ti=S1_ti, 
             ws_eff_ref=ws_eff_ref,
             ref_ws=ref_ws, ref_wd=ref_wd, ref_ti=ref_ti,
             x_grid=horizontal_grid.x, y_grid=horizontal_grid.y)

print(f"\nAnalysis complete! {len(selected_times)} sensitivity images saved to {output_dir}/")

# Create a summary visualization comparing all time points
plt.figure(figsize=(15, 10))

# Create subplots grid with rows for times and columns for parameter sensitivities
fig, axs = plt.subplots(len(selected_times), 3, figsize=(18, 4*len(selected_times)))

# Load saved data for each time point
for i, time_idx in enumerate(selected_times):
    data = np.load(f"{output_dir}/sobol_data_time_{time_stamps[time_idx]:.1f}h.npz")
    
    # Extract data
    S1_ws = data['S1_ws']
    S1_wd = data['S1_wd']
    S1_ti = data['S1_ti']
    ref_ws = data['ref_ws']
    ref_wd = data['ref_wd']
    
    # Plot wind speed sensitivity
    im1 = axs[i, 0].imshow(S1_ws, origin='lower', cmap='viridis', vmin=0, vmax=1,
                          extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                                  horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[i, 0].set_title(f'WS Sensitivity (t={time_stamps[time_idx]:.1f}h, WS={ref_ws:.1f}, WD={ref_wd:.1f})')
    axs[i, 0].set_xlabel('x [m]')
    axs[i, 0].set_ylabel('y [m]')
    
    # Plot wind direction sensitivity
    im2 = axs[i, 1].imshow(S1_wd, origin='lower', cmap='viridis', vmin=0, vmax=1,
                          extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                                  horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[i, 1].set_title(f'WD Sensitivity (t={time_stamps[time_idx]:.1f}h, WS={ref_ws:.1f}, WD={ref_wd:.1f})')
    axs[i, 1].set_xlabel('x [m]')
    
    # Plot turbulence intensity sensitivity
    im3 = axs[i, 2].imshow(S1_ti, origin='lower', cmap='viridis', vmin=0, vmax=1,
                          extent=[horizontal_grid.x.min(), horizontal_grid.x.max(), 
                                  horizontal_grid.y.min(), horizontal_grid.y.max()])
    axs[i, 2].set_title(f'TI Sensitivity (t={time_stamps[time_idx]:.1f}h, WS={ref_ws:.1f}, WD={ref_wd:.1f})')
    axs[i, 2].set_xlabel('x [m]')
    
    # Add turbine positions to all plots in this row
    for j in range(3):
        axs[i, j].scatter(x, y, color='red', s=10)

# Add colorbars
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Sobol Sensitivity Index')

# Add an overall title
fig.suptitle('Comparison of Sobol Sensitivity Indices Across Different Times', fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

# Save the comparison figure
plt.savefig(f"{output_dir}/sobol_sensitivity_comparison.png", dpi=300)
plt.close(fig)

print(f"Saved comparison visualization to {output_dir}/sobol_sensitivity_comparison.png")
