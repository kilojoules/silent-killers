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
    'names': ['wd_uncertainty', 'ws_uncertainty', 'ti_uncertainty'],
    'bounds': [[-5, 5],        # Wind direction uncertainty (±5 degrees)
               [-0.5, 0.5],    # Wind speed uncertainty (±0.5 m/s)
               [-0.02, 0.02]]  # TI uncertainty (±2%)
}

# Generate samples using Saltelli's method
n_samples = 100  # Reduced for demonstration - increase for production runs
param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

# Time series parameters
n_days = 3  # Analyze 3 days of data
time_stamp = np.arange(6*24*n_days)/6/24  # 10-minute intervals for n_days

# Load example time series data
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz")
wd_base, ws_base, ws_std = [d[k][:6*24*n_days] for k in ['wd', 'ws', 'ws_std']]
ti_base = np.minimum(ws_std/ws_base, 0.5)

# Function to run simulation with perturbed inputs
def run_simulation_with_uncertainty(perturbation):
    wd_pert = wd_base + perturbation[0]
    ws_pert = ws_base + perturbation[1]
    ti_pert = ti_base + perturbation[2]
    
    # Ensure TI stays within reasonable bounds
    ti_pert = np.clip(ti_pert, 0.01, 0.5)
    
    sim_res = wf_model(x, y, 
                       wd=wd_pert,
                       ws=ws_pert,
                       time=time_stamp,
                       TI=ti_pert)
    return sim_res

# Function to analyze sensitivity at a specific time point
def analyze_sensitivity_at_time(t_idx, grid_resolution=100):
    print(f"Analyzing time point {t_idx} ({time_stamp[t_idx]:.2f} days)")
    
    # Create grid for flow analysis
    from py_wake import HorizontalGrid
    grid = HorizontalGrid(resolution=grid_resolution, extend=0.5)
    
    # Run all samples for this time point
    ws_eff_samples = []
    for params in param_values:
        # Create perturbation that's only active at our time point
        wd_pert = np.zeros_like(wd_base)
        ws_pert = np.zeros_like(ws_base)
        ti_pert = np.zeros_like(ti_base)
        
        wd_pert[t_idx] = params[0]
        ws_pert[t_idx] = params[1]
        ti_pert[t_idx] = params[2]
        
        sim_res = run_simulation_with_uncertainty([wd_pert, ws_pert, ti_pert])
        flow_map = sim_res.flow_map(grid=grid, wd=wd_base[t_idx], ws=ws_base[t_idx])
        ws_eff_samples.append(flow_map.WS_eff.values.flatten())
    
    # Convert to numpy array
    ws_eff_samples = np.array(ws_eff_samples)
    
    # Perform Sobol analysis for each grid point
    n_grid_points = ws_eff_samples.shape[1]
    S1 = np.zeros((n_grid_points, problem['num_vars']))
    ST = np.zeros((n_grid_points, problem['num_vars']))
    
    for i in range(n_grid_points):
        Y = ws_eff_samples[:, i]
        try:
            Si = sobol.analyze(problem, Y, calc_second_order=False)
            S1[i,:] = Si['S1']
            ST[i,:] = Si['ST']
        except:
            S1[i,:] = np.nan
            ST[i,:] = np.nan
    
    # Reshape results back to grid
    grid_shape = (grid_resolution, grid_resolution)
    S1_grid = {name: S1[:,i].reshape(grid_shape) for i, name in enumerate(problem['names'])}
    ST_grid = {name: ST[:,i].reshape(grid_shape) for i, name in enumerate(problem['names'])}
    
    return S1_grid, ST_grid, grid

# Analyze sensitivity at several time points
time_indices = [0, 100, 200, 300]  # Example time points to analyze

for t_idx in time_indices:
    # Perform sensitivity analysis
    S1_grid, ST_grid, grid = analyze_sensitivity_at_time(t_idx)
    
    # Get base flow map for reference
    base_sim = run_simulation_with_uncertainty([np.zeros_like(wd_base), 
                                              np.zeros_like(ws_base), 
                                              np.zeros_like(ti_base)])
    base_flow = base_sim.flow_map(grid=grid, wd=wd_base[t_idx], ws=ws_base[t_idx])
    
    # Plot results
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Time: {time_stamp[t_idx]:.2f} days - WD: {wd_base[t_idx]:.1f}°, WS: {ws_base[t_idx]:.1f} m/s")
    
    # Plot base flow
    base_flow.plot_wake_map(ax=axes[0,0])
    axes[0,0].set_title("Base Flow (WS_eff)")
    
    # Plot first order indices
    for i, name in enumerate(problem['names']):
        im = axes[1,i].imshow(S1_grid[name], origin='lower', 
                             extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
                             vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(im, ax=axes[1,i])
        axes[1,i].set_title(f"S1 - {name}")
        axes[1,i].set_xlabel("x [m]")
        axes[1,i].set_ylabel("y [m]")
    
    # Plot total order indices
    for i, name in enumerate(problem['names']):
        im = axes[2,i].imshow(ST_grid[name], origin='lower', 
                             extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
                             vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(im, ax=axes[2,i])
        axes[2,i].set_title(f"ST - {name}")
        axes[2,i].set_xlabel("x [m]")
        axes[2,i].set_ylabel("y [m]")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_t{time_stamp[t_idx]:.2f}_wd{wd_base[t_idx]:.0f}_ws{ws_base[t_idx]:.0f}.png")
    plt.close()
    print(f"Saved results for time point {t_idx}")

print("Sensitivity analysis complete. Results saved to:", output_dir)
