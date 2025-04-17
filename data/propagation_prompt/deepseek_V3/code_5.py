import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.plotting import setup_plot
from SALib.analyze import sobol
from SALib.sample import saltelli
import time
import os

# Set up output directory for saving images
output_dir = "sensitivity_flow_fields"
os.makedirs(output_dir, exist_ok=True)

# Initialize PyWake model
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Define the problem for sensitivity analysis
problem = {
    'num_vars': 3,
    'names': ['wind_speed', 'wind_direction', 'turbulence_intensity'],
    'bounds': [[8.0, 12.0],  # Wind speed range [m/s]
               [260.0, 280.0],  # Wind direction range [deg]
               [0.05, 0.15]]   # TI range
}

# Generate samples using Saltelli's method
n_samples = 100  # Reduced for demonstration - increase for production runs
param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

# Time points to analyze (days)
time_points = np.linspace(0, 29, 6)  # 6 time points over 30 days

# Load time series data for realistic conditions
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz")
full_wd, full_ws, full_ws_std = [d[k][:6*24*30] for k in ['wd', 'ws', 'ws_std']]
full_ti = np.minimum(full_ws_std/full_ws, 0.5)
full_time_stamp = np.arange(len(full_wd))/6/24

# Function to evaluate the model with given parameters
def evaluate_model(params, time_idx):
    """Run the PyWake model with given parameters at a specific time point."""
    ws, wd, ti = params
    
    # Get the time point and surrounding data for realistic conditions
    time_point = time_points[time_idx]
    idx = np.argmin(np.abs(full_time_stamp - time_point))
    
    # Create a window around the time point with modified parameters
    window_size = 6  # 1 hour window (6 * 10-min intervals)
    start_idx = max(0, idx - window_size//2)
    end_idx = min(len(full_wd), idx + window_size//2)
    
    # Modify the parameters in the window
    modified_wd = full_wd.copy()
    modified_ws = full_ws.copy()
    modified_ti = full_ti.copy()
    
    modified_wd[start_idx:end_idx] = wd
    modified_ws[start_idx:end_idx] = ws
    modified_ti[start_idx:end_idx] = ti
    
    # Run the simulation
    sim_res = wf_model(x, y,
                       wd=modified_wd[start_idx:end_idx],
                       ws=modified_ws[start_idx:end_idx],
                       time=full_time_stamp[start_idx:end_idx],
                       TI=modified_ti[start_idx:end_idx])
    
    # Get the flow map for the center time point
    center_idx = min(window_size//2, len(sim_res.WS_eff.time)-1)
    flow_map = sim_res.flow_map(grid=None, 
                               time=sim_res.WS_eff.time[center_idx].values)
    
    return flow_map.WS_eff.values

# Perform sensitivity analysis at each time point
for i, time_point in enumerate(time_points):
    print(f"Processing time point {i+1}/{len(time_points)} (Day {time_point:.1f})")
    
    # Run the sensitivity analysis for this time point
    Y = np.zeros((param_values.shape[0], 500, 500))  # Grid size from flow_map
    
    for j, params in enumerate(param_values):
        if j % 10 == 0:
            print(f"  Sample {j}/{param_values.shape[0]}")
        try:
            Y[j] = evaluate_model(params, i)
        except Exception as e:
            print(f"Error in sample {j}: {e}")
            Y[j] = np.nan
    
    # Calculate Sobol indices for each grid point
    print("Calculating Sobol indices...")
    S1 = np.zeros((problem['num_vars'], *Y.shape[1:]))
    ST = np.zeros_like(S1)
    
    # Calculate sensitivity indices for each grid point
    for x_idx in range(Y.shape[1]):
        for y_idx in range(Y.shape[2]):
            if np.all(np.isnan(Y[:, x_idx, y_idx])):
                continue
                
            try:
                Si = sobol.analyze(problem, Y[:, x_idx, y_idx], 
                                  calc_second_order=False)
                S1[:, x_idx, y_idx] = Si['S1']
                ST[:, x_idx, y_idx] = Si['ST']
            except:
                pass
    
    # Plot and save results
    fig, axes = plt.subplots(2, problem['num_vars'], figsize=(15, 10))
    fig.suptitle(f"Sensitivity Analysis at Day {time_point:.1f}")
    
    for k, var in enumerate(problem['names']):
        # First order indices
        im = axes[0, k].imshow(S1[k], vmin=0, vmax=1, cmap='viridis')
        axes[0, k].set_title(f"First-order ({var})")
        plt.colorbar(im, ax=axes[0, k])
        
        # Total indices
        im = axes[1, k].imshow(ST[k], vmin=0, vmax=1, cmap='viridis')
        axes[1, k].set_title(f"Total-order ({var})")
        plt.colorbar(im, ax=axes[1, k])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_day_{time_point:.1f}.png")
    plt.close()
    
    # Also save the flow field with nominal parameters
    nominal_params = [np.mean(bounds) for bounds in problem['bounds']]
    flow_map = evaluate_model(nominal_params, i)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(flow_map, cmap='coolwarm')
    plt.colorbar(label='WS_eff [m/s]')
    plt.title(f"Flow Field at Day {time_point:.1f} (Nominal Parameters)")
    plt.savefig(f"{output_dir}/flow_field_day_{time_point:.1f}.png")
    plt.close()

print("Analysis complete. Results saved to", output_dir)
