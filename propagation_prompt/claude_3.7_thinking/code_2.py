# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os
from SALib.sample import saltelli
from SALib.analyze import sobol

# Import PyWake packages
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid
from py_wake.utils.plotting import setup_plot

# Set up the wind farm and site
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define the wake model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Define grid for flow map calculation - use a coarser grid for computational efficiency
grid = HorizontalGrid(x=np.linspace(-1000, 5000, 50), 
                      y=np.linspace(-2000, 2000, 50))

# Define realistic measurement uncertainties and their distributions
# These are the input parameters for our Sobol analysis
problem = {
    'num_vars': 3,
    'names': ['wind_speed', 'wind_direction', 'turbulence_intensity'],
    'bounds': [[0.9, 1.1],   # Relative wind speed range (multiplier)
               [-5, 5],      # Wind direction offset range [degrees]
               [0.8, 1.2]]   # Relative turbulence intensity range (multiplier)
}

# Generate Sobol samples
n_samples = 128  # Number of base samples
param_values = saltelli.sample(problem, n_samples)  # This creates n_samples*(2*D+2) samples

# Create synthetic time series data for demonstration
n_timesteps = 6  # Smaller number for computational feasibility
time_stamps = np.linspace(0, 24, n_timesteps)  # 6 time steps over 24 hours
base_ws = 8 + 4 * np.sin(np.pi * time_stamps / 12)  # Wind speed varying through the day
base_wd = 270 + 45 * np.sin(np.pi * time_stamps / 6)  # Wind direction varying through the day
base_ti = 0.1 + 0.05 * np.cos(np.pi * time_stamps / 12)  # TI varying through the day

# Create output directory for results
os.makedirs('sobol_sensitivity_results', exist_ok=True)

# For each time step, perform Sobol sensitivity analysis
for t_idx, t in enumerate(time_stamps):
    print(f"Processing time step {t_idx+1}/{len(time_stamps)}: t={t:.1f}h")
    
    # Get base values for this time step
    ws_base = base_ws[t_idx]
    wd_base = base_wd[t_idx]
    ti_base = base_ti[t_idx]
    
    # Print info
    print(f"Base conditions: WS={ws_base:.1f}m/s, WD={wd_base:.1f}°, TI={ti_base:.3f}")
    
    # Initialize arrays to store WS_eff for each sample at each grid point
    ws_eff_samples = np.zeros((len(param_values), len(grid.x), len(grid.y)))
    
    # Run simulations for all parameter combinations
    for i, params in enumerate(tqdm(param_values)):
        # Apply uncertainties to base values
        ws_rel, wd_offset, ti_rel = params
        
        ws = ws_base * ws_rel
        wd = (wd_base + wd_offset) % 360
        ti = ti_base * ti_rel
        
        # Run the simulation
        sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
        
        # Generate flow map
        flow_map = sim_res.flow_map(grid=grid, wd=wd, ws=ws)
        
        # Store WS_eff
        ws_eff_samples[i] = flow_map.WS_eff.values
    
    # Initialize arrays to store Sobol indices at each grid point
    S1 = np.zeros((problem['num_vars'], len(grid.x), len(grid.y)))
    ST = np.zeros_like(S1)
    
    # Calculate Sobol indices for each grid point
    for i in tqdm(range(len(grid.x))):
        for j in range(len(grid.y)):
            # Extract WS_eff at this grid point for all samples
            Y = ws_eff_samples[:, i, j]
            
            # Calculate Sobol indices
            Si = sobol.analyze(problem, Y, print_to_console=False)
            
            # Store the results
            S1[:, i, j] = Si['S1']
            ST[:, i, j] = Si['ST']
    
    # Create and save the visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Run baseline simulation for reference
    sim_res_base = wf_model(x, y, wd=wd_base, ws=ws_base, TI=ti_base)
    flow_map_base = sim_res_base.flow_map(grid=grid, wd=wd_base, ws=ws_base)
    
    # Plot baseline flow
    im = axes[0, 0].contourf(grid.x, grid.y, flow_map_base.WS_eff.values.T, cmap='viridis', levels=20)
    axes[0, 0].scatter(x, y, c='red', marker='x', s=30)
    cbar = plt.colorbar(im, ax=axes[0, 0])
    cbar.set_label('WS_eff [m/s]')
    axes[0, 0].set_title(f'Baseline Flow at t={t:.1f}h\nWS={ws_base:.1f}m/s, WD={wd_base:.1f}°, TI={ti_base:.3f}')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    
    # Plot first-order Sobol indices for each parameter
    for p_idx, param_name in enumerate(problem['names']):
        row, col = divmod(p_idx + 1, 3)
        
        im = axes[row, col].contourf(grid.x, grid.y, S1[p_idx].T, cmap='Reds', levels=np.linspace(0, 1, 21))
        axes[row, col].scatter(x, y, c='black', marker='x', s=30)
        cbar = plt.colorbar(im, ax=axes[row, col])
        cbar.set_label('First-order Sobol index')
        axes[row, col].set_title(f'Sensitivity to {param_name}\n(First-order)')
        axes[row, col].set_xlabel('x [m]')
        axes[row, col].set_ylabel('y [m]')
    
    # Fill the remaining subplot with total-order indices sum
    # This shows regions where interaction effects are important
    total_interactions = np.sum(ST, axis=0) - np.sum(S1, axis=0)
    im = axes[1, 2].contourf(grid.x, grid.y, total_interactions.T, cmap='Blues', levels=np.linspace(0, 2, 21))
    axes[1, 2].scatter(x, y, c='black', marker='x', s=30)
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label('Interaction Effects')
    axes[1, 2].set_title('Parameter Interactions\n(Sum of total-order - Sum of first-order)')
    axes[1, 2].set_xlabel('x [m]')
    axes[1, 2].set_ylabel('y [m]')
    
    plt.tight_layout()
    plt.savefig(f'sobol_sensitivity_results/sobol_sensitivity_t{t_idx:02d}.png', dpi=300)
    plt.close()
    
    # Also save total-order sensitivity plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for p_idx, param_name in enumerate(problem['names']):
        im = axes[p_idx].contourf(grid.x, grid.y, ST[p_idx].T, cmap='plasma', levels=np.linspace(0, 1, 21))
        axes[p_idx].scatter(x, y, c='black', marker='x', s=30)
        cbar = plt.colorbar(im, ax=axes[p_idx])
        cbar.set_label('Total-order Sobol index')
        axes[p_idx].set_title(f'Total Sensitivity to {param_name}')
        axes[p_idx].set_xlabel('x [m]')
        axes[p_idx].set_ylabel('y [m]')
    
    plt.tight_layout()
    plt.savefig(f'sobol_sensitivity_results/sobol_total_sensitivity_t{t_idx:02d}.png', dpi=300)
    plt.close()
    
    # Save Sobol indices as numpy arrays for later use
    np.savez(f'sobol_sensitivity_results/sobol_indices_t{t_idx:02d}.npz',
             S1=S1, ST=ST, 
             grid_x=grid.x, grid_y=grid.y,
             ws_base=ws_base, wd_base=wd_base, ti_base=ti_base,
             time=t)

print("Sobol sensitivity analysis complete. Results saved in 'sobol_sensitivity_results' directory.")
