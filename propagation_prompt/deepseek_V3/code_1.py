import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils import layouts
from py_wake.utils.profiling import timeit
from SALib.analyze import sobol
from SALib.sample import saltelli
import xarray as xr
import os
from tqdm import tqdm

# Create output directory for images
os.makedirs('sensitivity_flow_maps', exist_ok=True)

# 1. Setup PyWake model
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# 2. Define the problem with uncertainty parameters
problem = {
    'num_vars': 3,
    'names': ['wd_uncertainty', 'ws_uncertainty', 'ti_uncertainty'],
    'bounds': [[-5, 5],        # Wind direction uncertainty (±5 degrees)
               [-0.5, 0.5],    # Wind speed uncertainty (±0.5 m/s)
               [-0.02, 0.02]]  # TI uncertainty (±2%)
}

# 3. Generate samples using Saltelli's method
n_samples = 100  # Reduced for demonstration; use more for production
param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

# 4. Time series setup
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz")
n_days = 30
wd, ws, ws_std = [d[k][:6*24*n_days] for k in ['wd', 'ws', 'ws_std']]
ti = np.minimum(ws_std/ws, 0.5)
time_stamp = np.arange(len(wd))/6/24

# Select specific time points to analyze
time_points = [0, len(time_stamp)//4, len(time_stamp)//2, 3*len(time_stamp)//4, len(time_stamp)-1]

# 5. Sensitivity analysis at each selected time point
for i, time_idx in enumerate(tqdm(time_points, desc="Processing time points")):
    # Get the nominal conditions at this time point
    nominal_wd = wd[time_idx]
    nominal_ws = ws[time_idx]
    nominal_ti = ti[time_idx]
    
    # Prepare model evaluation function
    def evaluate_ws_eff(params):
        # Apply uncertainties to nominal values
        perturbed_wd = nominal_wd + params[0]
        perturbed_ws = nominal_ws + params[1]
        perturbed_ti = nominal_ti + params[2]
        
        # Run simulation with perturbed inputs
        sim_res = wf_model(x, y, 
                          wd=perturbed_wd,
                          ws=perturbed_ws,
                          TI=perturbed_ti)
        
        # Get flow map and WS_eff values
        flow_map = sim_res.flow_map()
        ws_eff = flow_map.WS_eff.values
        
        # Return flattened array for sensitivity analysis
        return ws_eff.flatten()
    
    # Run model for all parameter samples
    Y = np.array([evaluate_ws_eff(v) for v in tqdm(param_values, desc=f"Time point {i+1}/{len(time_points)}", leave=False)])
    
    # Reshape Y for analysis (samples × grid_points)
    Y = Y.reshape(len(param_values), -1)
    
    # Perform Sobol analysis for each grid point
    Si_list = []
    for j in tqdm(range(Y.shape[1]), desc="Analyzing grid points", leave=False):
        Si = sobol.analyze(problem, Y[:, j], calc_second_order=False)
        Si_list.append(Si['ST'])
    
    # Reshape sensitivity indices back to flow map grid
    flow_map = wf_model(x, y, wd=nominal_wd, ws=nominal_ws, TI=nominal_ti).flow_map()
    sensitivity_map = np.array(Si_list).reshape(flow_map.WS_eff.shape)
    
    # Create xarray for the sensitivity map
    sensitivity_da = xr.DataArray(
        sensitivity_map,
        dims=flow_map.WS_eff.dims,
        coords=flow_map.WS_eff.coords,
        name='WS_eff_Sobol_Total'
    )
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot wake map
    plt.subplot(2, 1, 1)
    flow_map.plot_wake_map()
    plt.title(f'Wake Map - Time: {time_stamp[time_idx]:.2f} days\nWD: {nominal_wd:.1f}°, WS: {nominal_ws:.1f} m/s, TI: {nominal_ti:.3f}')
    
    # Plot sensitivity map
    plt.subplot(2, 1, 2)
    im = plt.contourf(flow_map.x, flow_map.y, sensitivity_map, levels=20, cmap='viridis')
    plt.colorbar(im, label='Total Sobol Sensitivity Index')
    plt.scatter(x, y, c='r', s=20, label='Turbines')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Total Sobol Sensitivity of WS_eff to Inflow Uncertainties')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'sensitivity_flow_maps/sensitivity_time_{time_stamp[time_idx]:.2f}_days.png')
    plt.close()

print("Sensitivity analysis complete. Results saved in 'sensitivity_flow_maps' directory.")
