#!/usr/bin/env python
"""
Script to propagate realistic measurement uncertainties through PyWake and perform a Sobol sensitivity analysis
of the effective wind speed (WS_eff) field with respect to inflow uncertainties. For each chosen time in a 
time‐series of inflow conditions, the script:
  1. Loads the nominal wind speed (ws) and wind direction (wd) along with wind speed uncertainty (ws_std).
  2. Defines a two‐parameter uncertainty problem (ws and wd) with bounds based on the measured values.
  3. Uses SALib’s Saltelli scheme to generate samples of (ws, wd).
  4. For each sample, computes the flow field (WS_eff) via PyWake’s flow map.
  5. At every grid point, computes the first‐order Sobol sensitivity indices for each input.
  6. Plots the sensitivity maps (for ws and wd separately) on the (x,y) grid and saves the figures.
  
This example is based on the PyWake documentation (see :contentReference[oaicite:1]{index=1}).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Import PyWake components
from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid

# Import SALib for Sobol sensitivity analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

# --- Function definitions ---

def evaluate_flow_field(ws, wd, sim_res, grid):
    """
    Given a wind speed (ws) and wind direction (wd), compute the flow field.
    This function calls the flow_map method on the PyWake SimulationResult object.
    
    Parameters:
      ws      : Wind speed [m/s] (float)
      wd      : Wind direction [deg] (float)
      sim_res : The PyWake SimulationResult object from wf_model(x, y)
      grid    : A HorizontalGrid object for the flow field calculation
      
    Returns:
      WS_field: 2D numpy array of effective wind speed (WS_eff) on the grid
    """
    flow_map = sim_res.flow_map(grid=grid, ws=ws, wd=wd)
    # Assumes that the FlowMap object contains an attribute WS_eff holding the 2D field.
    # (Consult the PyWake documentation for your version if different.)
    WS_field = flow_map.WS_eff  
    return WS_field

def main():
    # ------------------------------
    # 1. Set up the PyWake simulation
    # ------------------------------
    # Use the IEA37 site and turbine layout as in the example docs
    site = IEA37Site(16)
    x_pos, y_pos = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)
    
    # Pre-compute the SimulationResult object (allows for flow_map calls with varying inflow parameters)
    sim_res = wf_model(x_pos, y_pos)
    
    # ------------------------------
    # 2. Set up the flow grid for the wake map
    # ------------------------------
    # Here we use a HorizontalGrid with a moderate resolution (adjust resolution as needed)
    grid = HorizontalGrid(resolution=50, extend=0.2)
    
    # ------------------------------
    # 3. Load time series inflow data (if available)
    # ------------------------------
    # The example data includes time series of wind direction, wind speed, and the wind speed standard deviation (ws_std)
    try:
        from py_wake.examples.data import example_data_path
        d = np.load(example_data_path + "/time_series.npz")
        n_days = 30
        wd_series, ws_series, ws_std_series = [d[k][:6*24*n_days] for k in ['wd', 'ws', 'ws_std']]
        time_stamp = np.arange(len(wd_series)) / 6 / 24  # time in days
        # Choose a few indices (times) for analysis (adjust these indices as desired)
        time_indices = [100, 1000, 3000]
    except Exception as e:
        print("Time series data not found, using default nominal conditions.")
        wd_series = None
        ws_series = None
        ws_std_series = None
        time_indices = [0]
        time_stamp = np.array([0])
    
    # ------------------------------
    # 4. Prepare output directory for sensitivity plots
    # ------------------------------
    output_dir = "sensitivity_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ------------------------------
    # 5. Loop over selected time instants and perform Sobol sensitivity analysis
    # ------------------------------
    # For each time, we define the uncertain input parameters:
    # - Wind speed (ws): nominal value ws_nom ± its standard deviation (ws_std)
    # - Wind direction (wd): nominal value wd_nom ± a fixed uncertainty (here, ±3°)
    for idx in time_indices:
        # Select nominal inflow conditions (from time series if available)
        if ws_series is not None:
            ws_nom = ws_series[idx]
            wd_nom = wd_series[idx]
            ws_std_val = ws_std_series[idx]
        else:
            ws_nom = 9.8   # default nominal wind speed [m/s]
            wd_nom = 270   # default nominal wind direction [deg]
            ws_std_val = 0.5  # default wind speed uncertainty
        
        # Define bounds for the uncertain parameters
        ws_low = max(ws_nom - ws_std_val, 0.1)  # ensure positive wind speed
        ws_high = ws_nom + ws_std_val
        wd_unc = 3.0  # assumed ±3 deg uncertainty in wind direction
        wd_low = wd_nom - wd_unc
        wd_high = wd_nom + wd_unc

        # Define the uncertainty problem for SALib
        problem = {
            'num_vars': 2,
            'names': ['ws', 'wd'],
            'bounds': [[ws_low, ws_high],
                       [wd_low, wd_high]]
        }
        
        # ------------------------------
        # 6. Generate samples using the Saltelli scheme
        # ------------------------------
        # Choose a modest base sample size (N). For 2 parameters the total sample count will be N*(D+2).
        N = 50
        param_values = saltelli.sample(problem, N)
        n_samples = param_values.shape[0]
        
        # ------------------------------
        # 7. Evaluate the flow field for each uncertain sample
        # ------------------------------
        # Run one evaluation to determine the grid dimensions
        test_field = evaluate_flow_field(param_values[0, 0], param_values[0, 1], sim_res, grid)
        nx, ny = test_field.shape
        all_fields = np.zeros((n_samples, nx, ny))
        
        print(f"Time index {idx}: Nominal ws = {ws_nom:.2f} m/s, wd = {wd_nom:.2f}°")
        print(f"Evaluating {n_samples} samples on a grid of size {nx}x{ny}...")
        for i in range(n_samples):
            ws_sample, wd_sample = param_values[i]
            field = evaluate_flow_field(ws_sample, wd_sample, sim_res, grid)
            all_fields[i] = field
        
        # ------------------------------
        # 8. Compute Sobol first-order indices point-by-point
        # ------------------------------
        # Initialize arrays to hold the first-order sensitivity for each input parameter
        S1_ws = np.zeros((nx, ny))
        S1_wd = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                Y_ij = all_fields[:, i, j]
                try:
                    Si = sobol.analyze(problem, Y_ij, print_to_console=False)
                    S1_ws[i, j] = Si['S1'][0]  # Sensitivity index for wind speed
                    S1_wd[i, j] = Si['S1'][1]  # Sensitivity index for wind direction
                except Exception as e:
                    S1_ws[i, j] = np.nan
                    S1_wd[i, j] = np.nan
        
        # ------------------------------
        # 9. Plot and save the sensitivity maps for the current time instant
        # ------------------------------
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Plot sensitivity with respect to wind speed uncertainty
        im0 = axs[0].contourf(grid.x, grid.y, S1_ws.T, levels=20, cmap='viridis')
        title0 = f"Sobol S1 (ws) at time index {idx}"
        if ws_series is not None:
            title0 += f" (t={time_stamp[idx]:.2f} days)"
        axs[0].set_title(title0)
        axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        fig.colorbar(im0, ax=axs[0])
        
        # Plot sensitivity with respect to wind direction uncertainty
        im1 = axs[1].contourf(grid.x, grid.y, S1_wd.T, levels=20, cmap='viridis')
        title1 = f"Sobol S1 (wd) at time index {idx}"
        if wd_series is not None:
            title1 += f" (t={time_stamp[idx]:.2f} days)"
        axs[1].set_title(title1)
        axs[1].set_xlabel("x [m]")
        axs[1].set_ylabel("y [m]")
        fig.colorbar(im1, ax=axs[1])
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"sensitivity_time_{idx}.png")
        plt.savefig(output_file)
        plt.close(fig)
        print(f"Saved sensitivity plot to {output_file}")

if __name__ == "__main__":
    main()

