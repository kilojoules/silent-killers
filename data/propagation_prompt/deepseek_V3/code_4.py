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

# 1. Setup PyWake model
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# 2. Define parameter space for uncertainty analysis
problem = {
    'num_vars': 3,
    'names': ['ws', 'wd', 'ti'],
    'bounds': [[8.0, 12.0],  # Wind speed range [m/s]
               [260.0, 280.0],  # Wind direction range [deg]
               [0.05, 0.15]]   # Turbulence intensity range
}

# 3. Generate samples using Saltelli's method
n_samples = 100  # Reduced for demonstration - increase for production
param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

# 4. Time series setup
n_timesteps = 5  # Number of time steps to analyze
time_stamps = np.linspace(0, 24, n_timesteps)  # Over a 24-hour period

# 5. Main analysis loop
for i, t in enumerate(time_stamps):
    print(f"Processing timestep {i+1}/{n_timesteps} (t={t:.1f} hours)")
    
    # Create perturbed inflow conditions for this timestep
    ws_perturbed = param_values[:, 0]
    wd_perturbed = param_values[:, 1]
    ti_perturbed = param_values[:, 2]
    
    # Initialize array to store WS_eff results
    n_points = 100  # Resolution of flow field grid
    ws_eff_results = np.zeros((len(param_values), n_points, n_points))
    
    # Run simulations for all parameter combinations
    for j, (ws, wd, ti) in enumerate(zip(ws_perturbed, wd_perturbed, ti_perturbed)):
        if j % 10 == 0:
            print(f"  Running simulation {j+1}/{len(param_values)}")
        
        # Run simulation with current parameters
        sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
        
        # Get flow map
        flow_map = sim_res.flow_map(
            grid=HorizontalGrid(resolution=n_points, extend=1.5),
            wd=wd,
            ws=ws
        )
        
        # Store WS_eff values
        ws_eff_results[j] = flow_map.WS_eff.values
    
    # Reshape results for Sobol analysis
    ws_eff_flat = ws_eff_results.reshape(len(param_values), -1)
    
    # Calculate Sobol indices at each grid point
    Si = np.zeros((ws_eff_flat.shape[1], problem['num_vars']))
    for k in range(ws_eff_flat.shape[1]):
        if k % 1000 == 0:
            print(f"  Analyzing point {k}/{ws_eff_flat.shape[1]}")
        Si[k] = sobol.analyze(problem, ws_eff_flat[:, k], calc_second_order=False)['S1']
    
    # Reshape sensitivity indices back to grid
    Si_grid = Si.reshape(n_points, n_points, problem['num_vars'])
    
    # Create plots for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Wind Speed Sensitivity', 'Wind Direction Sensitivity', 'Turbulence Intensity Sensitivity']
    
    for ax, param_idx, title in zip(axes, range(3), titles):
        im = ax.imshow(Si_grid[:, :, param_idx], 
                      extent=[flow_map.x.min(), flow_map.x.max(), 
                              flow_map.y.min(), flow_map.y.max()],
                      origin='lower', 
                      cmap='viridis',
                      vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(im, ax=ax, label='First-order Sobol index')
    
    # Plot wind turbines
    for ax in axes:
        windTurbines.plot(x, y, ax=ax)
    
    plt.suptitle(f'Sensitivity Analysis at t={t:.1f} hours', y=1.05)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'sensitivity_t{i:02d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

print("Analysis complete. Results saved to:", output_dir)
