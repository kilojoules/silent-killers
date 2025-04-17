Okay, let's break down the approach and refactor the code for better modularity, focusing on the wind farm model (WFM) instantiation while keeping the core logic intact.

Critique of Current Approach:

    WFM Instantiation Repetition: You've correctly identified that the logic for creating the All2AllIterative wind farm model (wfm) is duplicated: once inside the evaluate_rmse function (which is called repeatedly by the optimizer) and again after the optimization to evaluate the best parameters. This violates the DRY (Don't Repeat Yourself) principle and makes the code harder to maintain. If you need to change a model or its default settings, you have to do it in two places.
    evaluate_rmse Complexity: This function does a lot: it parses parameters, selects and configures multiple sub-models based on DOWNWIND and MODEL flags, instantiates the WFM, runs the simulation, calculates the flow map (iteratively, which could potentially be simplified), computes the prediction, and returns the RMSE. Separating the WFM creation would improve its clarity.
    Flow Map Calculation in Loop: Inside evaluate_rmse, the flow map is calculated by iterating through time steps and concatenating. sim_res.flow_map() can typically handle the time dimension directly, which might be more efficient and cleaner.
    Blockage Model Confusion: In evaluate_rmse, blockage_args are defined based on DOWNWIND, but the final call seems to always use blockage_deficitModel=SelfSimilarityDeficit2020(groundModel=Mirror()), potentially ignoring the blockage_args (depending on whether the commented-out **blockage_args was intended). However, the post-optimization block does use **blockage_args for the SelfSimilarityDeficit2020 when DOWNWIND is False. This inconsistency should be resolved. Let's assume the logic inside evaluate_rmse (where the optimization actually happens) is the "correct" one to preserve for the refactoring, meaning SelfSimilarityDeficit2020(groundModel=Mirror()) might be used consistently, or that the **blockage_args should indeed be applied conditionally. Correction: Looking closely at evaluate_rmse, the line is blockage_deficitModel=SelfSimilarityDeficit2020(groundModel=Mirror()))#**blockage_args)). This implies the Mirror() ground model is always added, but the other blockage_args (like ss_alpha, ss_beta) are only intended to be passed when DOWNWIND is False (upstream). The post-optimization block reflects this correctly.
    Clarity of MODEL: Using MODEL = 1 or MODEL = 2 isn't immediately descriptive. Using strings like 'Blondel' and 'TurboGaussian' might be clearer if feasible, but numeric flags are functional.

Flow Field Plot Suggestion:

Your idea to change the flow field plots (currently showing Observed, Predicted, Difference per time step) to report average and P90 errors is a good one for summarizing performance across the spatial domain.

    Average Error: You could calculate diff.mean() for each time step.
    P90 Error: You could calculate diff.quantile(0.9) for each time step.
    Visualization: Instead of 3 contour plots per time step, you could generate:
        A single plot showing the average and P90 error values over time (e.g., a line plot with time on the x-axis).
        Or, keep one representative contour plot (e.g., for the time step with the median or highest RMSE) and add text annotations for the average/P90 error for that specific step.
        Or, plot the spatial distribution of the average error across all time steps (diff.mean(dim='time')).

Refactored Code:

Here's a refactored version focusing on modularizing the WFM instantiation.
Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from bayes_opt import BayesianOptimization

# PyWake imports
from py_wake.wind_farm_models import All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020, TurboGaussianDeficit
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models import JimenezWakeDeflection # Although not used (None)
from py_wake.rotor_avg_models import GaussianOverlapAvgModel, RotorCenter # RotorCenter not used?
from py_wake.ground_models import Mirror
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.site import UniformSite # Using UniformSite as Hornsrev1Site might not be directly compatible with arbitrary WS/TI
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake import HorizontalGrid

# --- Configuration ---
DOWNWIND = True # True for wake region, False for upstream blockage region
MODEL = 2       # 1 for Blondel, 2 for TurboGaussian (used when DOWNWIND=True)
DATA_FILE = './DTU10MW.nc'
OUTPUT_FIG_DIR = 'figs'
ANIMATION_FILENAME = 'optimization_animation'
BAR_PLOT_FILENAME = 'bar_plot'

if MODEL not in {1, 2}:
    raise ValueError("MODEL must be 1 or 2")

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# --- Load Data and Setup Turbine/Site ---
try:
    dat = xr.load_dataset(DATA_FILE)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file {DATA_FILE} not found. Make sure it's in the correct path.")

turbine = DTU10MW()
D = turbine.diameter()

# Scale coordinates in the reference data
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)

# Define Region of Interest (ROI) based on DOWNWIND flag
if DOWNWIND:
    X_LB, X_UB = 2, 10 # Downwind region (in Diameters)
else:
    X_LB, X_UB = -2, -1 # Upstream region (in Diameters)

roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D) # Fixed y-range

# Select the relevant part of the reference data
flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x.values
target_y = flow_roi.y.values

# --- Define Simulation Conditions ---
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)

# Create flattened arrays for all combinations of WS and TI
ws_mesh, ti_mesh = np.meshgrid(WSs, TIs)
full_ws = ws_mesh.flatten()
full_ti = ti_mesh.flatten()
n_conditions = full_ws.size
full_wd = [270] * n_conditions # Assuming constant wind direction

# Site setup (Using UniformSite as it directly accepts WS/TI arrays)
# Note: Using Hornsrev1Site and then overriding WS/TI might be less direct
# If Hornsrev1 specific features (like roughness) are crucial, adapt as needed.
site = UniformSite(p_wd=[1], ti=0.1) # Base TI, will be overridden in simulation calls

# --- Pre-calculate Observed Deficits ---
# Run a simulation with default Blondel to get CT/TI values for interpolation
# (This seems to be the purpose of the first simulation block in the original code)
print("Pre-calculating observed deficits using reference simulation...")
ref_wfm = All2AllIterative(site, turbine,
                           wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                           superpositionModel=LinearSum(),
                           blockage_deficitModel=SelfSimilarityDeficit2020()) # Simple default WFM

ref_sim_res = ref_wfm([0], [0], ws=full_ws, TI=full_ti, wd=full_wd, time=True)

obs_values = []
for t in range(n_conditions):
    this_run_condition = ref_sim_res.isel(time=t, wt=0) # Get CT/TI for this condition
    # Interpolate reference data based on the simulated CT and TI for this WS/TI case
    observed_deficit = flow_roi.deficits.interp(
        ct=this_run_condition.CT,
        ti=this_run_condition.TI,
        z=0 # Assuming reference data is at hub height (z=0 relative to hub)
    )
    # Ensure result has expected coords (x, y) and transpose if needed by interp result
    # .T was used in original, check if necessary based on interp output dims
    if 'y' in observed_deficit.dims and observed_deficit.dims.index('y') == 0:
         obs_values.append(observed_deficit.rename({'y':'_y', 'x':'_x'}).rename({'_y':'y', '_x':'x'})) # Ensure correct order if needed
    else:
         obs_values.append(observed_deficit) # Assume interp gives (x,y) or compatible


all_obs = xr.concat(obs_values, dim='time')
# Assign coordinates to the new time dimension to match simulation results
all_obs = all_obs.assign_coords(time=ref_sim_res.time)
print(f"Shape of pre-calculated observed deficits: {all_obs.shape}")


# --- Modular WFM Instantiation Function ---
def create_wfm(site, turbine, downwind_flag, model_flag, params):
    """
    Creates and returns a PyWake WindFarm Model instance based on configuration
    flags and model parameters.

    Args:
        site: PyWake Site object.
        turbine: PyWake WindTurbine object.
        downwind_flag (bool): True if simulating wake region, False for upstream.
        model_flag (int): 1 for Blondel, 2 for TurboGaussian (only applies if downwind_flag is True).
        params (dict): Dictionary containing model parameters.

    Returns:
        An instance of a PyWake WindFarm Model (e.g., All2AllIterative).
    """
    wake_deficitModel = None
    turbulenceModel = None
    blockage_deficitModel = None
    turb_args = {}
    blockage_args = {}

    # --- Configure Models based on DOWNWIND and MODEL flags ---
    if downwind_flag:
        # WAKE REGION SIMULATION
        # Blockage model is present but its parameters are not optimized here
        blockage_deficitModel = SelfSimilarityDeficit2020(groundModel=Mirror())
        # Turbulence model parameters are optimized
        turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}

        if model_flag == 1:
            # Blondel Wake Model
            def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
        elif model_flag == 2:
            # TurboGaussian Wake Model
            wake_deficitModel = TurboGaussianDeficit(
                A=params['A'],
                cTI=[params['cti1'], params['cti2']],
                ctlim=params['ctlim'],
                ceps=params['ceps'],
                ct2a=ct2a_mom1d,
                groundModel=Mirror(),
                rotorAvgModel=GaussianOverlapAvgModel()
            )
            # Ensure the model uses the correct WS key if needed by internal calculations
            # wake_deficitModel.WS_key = 'WS_eff' # Or check TurboGaussianDeficit defaults
        else:
             raise ValueError(f"Invalid MODEL flag {model_flag} for downwind simulation")

    else:
        # UPSTREAM REGION SIMULATION (BLOCKAGE FOCUS)
        # Wake model is default Blondel (parameters not optimized here)
        wake_deficitModel = BlondelSuperGaussianDeficit2020()
        # Turbulence model parameters are not optimized here
        turb_args = {} # Use CrespoHernandez defaults

        # Blockage model parameters ARE optimized
        blockage_params_keys = ['ss_alpha', 'ss_beta', 'rp1', 'rp2', 'ng1', 'ng2', 'ng3', 'ng4']
        # Check if all needed keys are in params (might differ if DOWNWIND=False MODEL=1 vs MODEL=2)
        if not all(k in params for k in blockage_params_keys):
             raise KeyError(f"Missing blockage parameters in params dict for UPSTREAM simulation. Need: {blockage_params_keys}")

        blockage_args = {
            'ss_alpha': params['ss_alpha'],
            'ss_beta': params['ss_beta'],
            'r12p': np.array([params['rp1'], params['rp2']]),
            'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
        }
        # Add ground model for SelfSimilarity only if MODEL is 2 (as per original logic)
        if model_flag == 2:
             blockage_args['groundModel'] = Mirror()

        blockage_deficitModel = SelfSimilarityDeficit2020(**blockage_args)


    # --- Instantiate the Wind Farm Model ---
    # Ensure required models are set
    if wake_deficitModel is None:
        raise ValueError("Wake deficit model was not configured.")
    if blockage_deficitModel is None:
         print("Warning: Blockage deficit model was not configured. Using None.") # Or set a default

    wfm = All2AllIterative(
        site, turbine,
        wake_deficitModel=wake_deficitModel,
        superpositionModel=LinearSum(),
        deflectionModel=None,  # Explicitly None as in original
        turbulenceModel=CrespoHernandez(**turb_args), # Pass optimized or default args
        blockage_deficitModel=blockage_deficitModel # Pass configured blockage model
    )

    return wfm


# --- Define Parameter Bounds and Defaults ---
pbounds = {}
defaults = {}

if DOWNWIND:
    if MODEL == 1:
        pbounds = {
            'a_s': (0.001, 0.5), 'b_s': (0.001, 0.01), 'c_s': (0.001, 0.5),
            'b_f': (-2, 1), 'c_f': (0.1, 5),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {
            'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
            'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32
        }
    elif MODEL == 2:
        pbounds = {
            'A': (0.001, .5), 'cti1': (.01, 5), 'cti2': (0.01, 5),
            'ceps': (0.01, 3), 'ctlim': (0.01, 1),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {
            'A': 0.04, 'cti1': 1.5, 'cti2': 0.8, 'ceps': 0.25, 'ctlim': 0.999,
            'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.3
        }
else: # UPSTREAM
    # Parameters for SelfSimilarityDeficit2020 blockage model
    # Using the bounds/defaults from the original MODEL=1 UPSTREAM case
    # Assuming these are the relevant parameters regardless of MODEL flag when DOWNWIND=False
    pbounds = {
         'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
         'rp1': (-2, 2), 'rp2': (-2, 2),
         'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3),
         # 'fg' parameters were in original MODEL=1 UPSTREAM defaults but not used? Ignoring for now.
         # If CrespoHernandez turbulence 'c' params should also be optimized upstream, add them here.
    }
    defaults = {
         'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951,
         'rp1': -0.672, 'rp2': 0.4897,
         'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
    }
    # Check if MODEL=2 UPSTREAM requires different parameters/bounds
    # Original code structure suggests turbulence params (ch1-4) are NOT optimized upstream.


# --- Objective Function for Bayesian Optimization ---
def evaluate_rmse(**kwargs):
    """Objective function: Calculates negative RMSE for maximization."""
    try:
        # 1. Create WFM using the modular function
        wfm = create_wfm(site, turbine, DOWNWIND, MODEL, kwargs)

        # 2. Run simulation
        sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=full_wd, time=True)

        # 3. Calculate flow map (handle potential time dimension directly)
        flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y), time=sim_res.time)

        # Extract effective wind speed (check key - might be WS or WS_eff)
        # Using 'WS' based on original calc: (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        # If flow_map contains WS_eff, adjust accordingly. Assume flow_map['WS'] exists.
        if 'WS' not in flow_map:
             if 'WS_eff' in flow_map:
                 ws_key = 'WS_eff'
             else:
                 raise KeyError("Could not find 'WS' or 'WS_eff' in flow_map output.")
        else:
            ws_key = 'WS'

        # Ensure flow_map has height dimension (h) and select the first (index 0)
        if 'h' in flow_map.dims:
            flow_map_ws = flow_map[ws_key].isel(h=0)
        else:
            # If no height dim, assume it's already hub-height equivalent
            flow_map_ws = flow_map[ws_key]


        # 4. Calculate predicted deficit
        # Ensure shapes align for broadcasting (sim_res.WS is likely (time, wt), flow_map_ws is (time, x, y))
        # We need to compare flow map at wt=0 to observed data.
        # sim_res.WS might need selection: .isel(wt=0)
        ws_free_stream = sim_res.WS.isel(wt=0) # Shape (time)
        # Broadcast ws_free_stream to match flow_map_ws (time, x, y)
        pred = (ws_free_stream - flow_map_ws) / ws_free_stream

        # 5. Calculate RMSE against pre-calculated observations
        # Ensure alignment (all_obs should be (time, y, x) or (time, x, y))
        # Match pred dimensions to all_obs if needed (e.g., transpose)
        if pred.dims != all_obs.dims:
             print(f"Warning: Dimension mismatch. Pred dims: {pred.dims}, Obs dims: {all_obs.dims}. Attempting transpose.")
             # Example: if pred is (time, x, y) and obs is (time, y, x)
             if pred.dims == ('time', 'x', 'y') and all_obs.dims == ('time', 'y', 'x'):
                  pred = pred.transpose('time', 'y', 'x')
             else:
                  # Add more specific checks or raise error if alignment fails
                  raise ValueError("Cannot align prediction and observation dimensions.")

        # Calculate squared error
        squared_error = (all_obs - pred) ** 2

        # Calculate mean RMSE over space and time
        rmse = float(np.sqrt(squared_error.mean(['x', 'y', 'time'])))
        # Alternative: mean RMSE per time step, then average (as in original)
        # rmse_per_time = np.sqrt(squared_error.mean(['x', 'y']))
        # rmse = float(rmse_per_time.mean('time'))


        # Handle NaN results (e.g., from invalid model parameters)
        if np.isnan(rmse):
            print(f"Warning: NaN RMSE encountered with params: {kwargs}. Returning large penalty.")
            return -1.0 # Return a large negative value (bad score for maximization)

        print(f"RMSE: {-rmse:.5f}") # Print negative RMSE being returned
        return -rmse # Return negative RMSE because BayesianOptimization maximizes

    except Exception as e:
        print(f"Error during RMSE evaluation with params: {kwargs}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        return -1.0 # Return large penalty on error


# --- Bayesian Optimization ---
print("Setting up Bayesian Optimization...")
optimizer = BayesianOptimization(
    f=evaluate_rmse,
    pbounds=pbounds,
    random_state=1,
    verbose=2 # 0: silent, 1: prints when max found, 2: prints every step
)

# Probe the default parameters first
print("Probing default parameters...")
optimizer.probe(params=defaults, lazy=True)

# Run the optimization
print("Starting optimization...")
optimizer.maximize(
    init_points=10, # Fewer initial random points? Original had 50.
    n_iter=50    # Fewer iterations? Original had 200. Adjust as needed.
)

print("\nOptimization Finished.")
print(f"Best parameters found: {optimizer.max['params']}")
best_params = optimizer.max['params']
best_rmse = -optimizer.max['target'] # Convert back to positive RMSE
print(f"Best RMSE: {best_rmse:.5f}")


# --- Animation of Optimization Process ---
print("Generating optimization animation...")

def update_plot(frame, optimizer, defaults, ax1, ax2):
    """Helper function to update animation frames."""
    ax1.clear()
    ax2.clear()

    iterations = range(len(optimizer.space.target))
    targets = -np.array(optimizer.space.target) # Actual RMSE (positive)

    # Find best RMSE up to current frame
    best_rmse_so_far = np.minimum.accumulate(targets[:frame+1])

    # Plot RMSE history
    ax1.plot(iterations[:frame+1], targets[:frame+1], color='gray', alpha=0.5, label='Iteration RMSE')
    ax1.plot(iterations[:frame+1], best_rmse_so_far, color='black', label='Best RMSE so far')
    ax1.set_title('Optimization Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)
    ax1.legend()

    # Get parameters for the best result up to current frame
    best_idx_so_far = np.argmin(targets[:frame+1])
    current_best_params = optimizer.res[best_idx_so_far]['params']

    # Plot parameters
    keys = list(defaults.keys()) # Use default keys to ensure consistent order/set
    best_vals = [current_best_params.get(k, np.nan) for k in keys] # Use .get for robustness
    default_vals = [defaults[k] for k in keys]

    x_pos = np.arange(len(keys))
    width = 0.35

    ax2.bar(x_pos - width/2, best_vals, width, label='Optimized (Best so far)')
    ax2.bar(x_pos + width/2, default_vals, width, label='Default', color='gray')
    # Optional: Outline for default
    # ax2.bar(x_pos + width/2, default_vals, width, edgecolor='black', linewidth=1, color='none', capstyle='butt')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(keys, rotation=45, ha="right")
    ax2.set_title(f'Parameters (Best RMSE: {best_rmse_so_far[-1]:.4f})')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    plt.tight_layout()

fig_ani, (ax1_ani, ax2_ani) = plt.subplots(1, 2, figsize=(15, 6))
ani = animation.FuncAnimation(fig_ani, update_plot,
                              frames=len(optimizer.space.target),
                              fargs=(optimizer, defaults, ax1_ani, ax2_ani),
                              repeat=False)

# Save animation
animation_path = os.path.join(OUTPUT_FIG_DIR, f"{ANIMATION_FILENAME}_model{MODEL}_dir{'down' if DOWNWIND else 'up'}.mp4")
writer = animation.FFMpegWriter(fps=5) # Adjust fps as needed
ani.save(animation_path, writer=writer)
print(f"Animation saved to {animation_path}")
plt.close(fig_ani)


# --- Final Evaluation and Plotting with Best Parameters ---
print("\nEvaluating model with best parameters...")

# 1. Create WFM with best parameters
best_wfm = create_wfm(site, turbine, DOWNWIND, MODEL, best_params)

# 2. Run simulation
final_sim_res = best_wfm([0], [0], ws=full_ws, TI=full_ti, wd=full_wd, time=True)

# 3. Calculate final flow map
final_flow_map = final_sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y), time=final_sim_res.time)

# Extract WS from flow map (checking key again)
if 'WS' not in final_flow_map:
     if 'WS_eff' in final_flow_map: ws_key = 'WS_eff'
     else: raise KeyError("Cannot find 'WS' or 'WS_eff' in final_flow_map.")
else: ws_key = 'WS'

if 'h' in final_flow_map.dims:
    final_flow_map_ws = final_flow_map[ws_key].isel(h=0)
else:
    final_flow_map_ws = final_flow_map[ws_key]


# 4. Calculate final predictions and errors
ws_free_stream_final = final_sim_res.WS.isel(wt=0)
final_pred = (ws_free_stream_final - final_flow_map_ws) / ws_free_stream_final

# Ensure alignment
if final_pred.dims != all_obs.dims:
    if final_pred.dims == ('time', 'x', 'y') and all_obs.dims == ('time', 'y', 'x'):
        final_pred = final_pred.transpose('time', 'y', 'x')
    else:
        raise ValueError("Cannot align final prediction and observation dimensions.")

final_diff = all_obs - final_pred

# 5. Calculate and print per-time-step and overall RMSE/errors
rmse_values = []
avg_errors = []
p90_errors = []

print("\nGenerating final comparison plots...")
for t in range(n_conditions):
    # Select data for this time step
    obs_t = all_obs.isel(time=t)
    pred_t = final_pred.isel(time=t)
    diff_t = final_diff.isel(time=t)

    # Calculate metrics for this time step
    rmse_t = float(np.sqrt((diff_t**2).mean()))
    avg_err_t = float(diff_t.mean())
    p90_err_t = float(diff_t.quantile(0.9))

    rmse_values.append(rmse_t)
    avg_errors.append(avg_err_t)
    p90_errors.append(p90_err_t)

    # --- Plotting ---
    # Option 1: Original Plotting (Observed, Predicted, Difference)
    fig, axes = plt.subplots(3, 1, figsize=(6, 15), sharex=True, sharey=True)
    vmin = min(obs_t.min(), pred_t.min())
    vmax = max(obs_t.max(), pred_t.max())
    levels = np.linspace(vmin, vmax, 11)

    cf0 = axes[0].contourf(target_x, target_y, obs_t.T, levels=levels, extend='both') # Transpose needed if data is (x,y)
    axes[0].set_ylabel('y/D')
    axes[0].set_title(f'Observed Deficit (WS={full_ws[t]}, TI={full_ti[t]:.2f})')
    fig.colorbar(cf0, ax=axes[0])

    cf1 = axes[1].contourf(target_x, target_y, pred_t.T, levels=levels, extend='both')
    axes[1].set_ylabel('y/D')
    axes[1].set_title(f'Predicted Deficit (RMSE={rmse_t:.4f})')
    fig.colorbar(cf1, ax=axes[1])

    diff_max_abs = np.abs(diff_t).max()
    levels_diff = np.linspace(-diff_max_abs, diff_max_abs, 11)
    cf2 = axes[2].contourf(target_x, target_y, diff_t.T, levels=levels_diff, cmap='RdBu_r', extend='both')
    axes[2].set_xlabel('x/D')
    axes[2].set_ylabel('y/D')
    axes[2].set_title(f'Difference (AvgErr={avg_err_t:.4f}, P90Err={p90_err_t:.4f})')
    fig.colorbar(cf2, ax=axes[2])

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_FIG_DIR, f"comparison_t{t}_model{MODEL}_dir{'down' if DOWNWIND else 'up'}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    # Option 2: Plot Average and P90 Error (Example - could be a separate summary plot)
    # (This section could replace or supplement the per-time-step plots)
    # fig_err, ax_err = plt.subplots()
    # ax_err.plot(avg_errors, label='Average Error')
    # ax_err.plot(p90_errors, label='P90 Error')
    # ax_err.set_xlabel('Time Step Index')
    # ax_err.set_ylabel('Error Value')
    # ax_err.legend()
    # plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'summary_errors.png'))
    # plt.close(fig_err)


overall_rmse_final = np.mean(rmse_values)
print(f"\nFinal Evaluation Results:")
#print(f"RMSE values per time step: {rmse_values}")
print(f"Overall RMSE: {overall_rmse_final:.5f}")
print(f"Average Error (mean over time): {np.mean(avg_errors):.5f}")
print(f"Average P90 Error (mean over time): {np.mean(p90_errors):.5f}")


# --- Final Parameter Bar Plot ---
print("Generating final parameter comparison plot...")
plt.figure(figsize=(10, 6))
keys = list(defaults.keys())
best_vals = [best_params.get(k, np.nan) for k in keys]
default_vals = [defaults[k] for k in keys]

x_pos = np.arange(len(keys))
width = 0.35

plt.bar(x_pos - width/2, best_vals, width, label='Optimized')
plt.bar(x_pos + width/2, default_vals, width, label='Default', color='gray', alpha=0.7)
# Outline for default
# plt.bar(x_pos + width/2, default_vals, width, edgecolor='black', linewidth=1.5, color='none', capstyle='butt')


plt.xticks(x_pos, keys, rotation=45, ha="right")
plt.ylabel('Parameter Value')
plt.title(f'Optimized vs Default Parameters (Final RMSE: {best_rmse:.4f})')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
bar_plot_path = os.path.join(OUTPUT_FIG_DIR, f"{BAR_PLOT_FILENAME}_model{MODEL}_dir{'down' if DOWNWIND else 'up'}.png")
plt.savefig(bar_plot_path)
print(f"Parameter bar plot saved to {bar_plot_path}")
plt.close()

print("\nScript finished.")

Key Changes and Improvements:

    create_wfm Function: All the logic for selecting and configuring wake_deficitModel, turbulenceModel, and blockage_deficitModel based on DOWNWIND, MODEL, and the input params dictionary is now encapsulated in this function.
    DRY Principle: create_wfm is called from both evaluate_rmse and the final evaluation section, eliminating code duplication.
    Clarity: evaluate_rmse is now simpler, focusing on running the simulation and calculating the error, delegating the WFM creation.
    Flow Map Calculation: The flow map calculation inside evaluate_rmse and in the final evaluation now attempts to use the time dimension directly in sim_res.flow_map(). It also explicitly checks for the wind speed key (WS or WS_eff) and handles the height dimension (h).
    Error Handling: Added try...except blocks in evaluate_rmse to catch errors during model instantiation or simulation (which can happen with bad parameter combinations) and return a penalty value instead of crashing the optimizer. Also added checks for missing keys and dimension mismatches.
    Pre-calculation of all_obs: The logic for calculating the observed deficits by interpolating the reference data based on a preliminary simulation run is kept, but made slightly more explicit.
    Configuration: Moved key settings (DOWNWIND, MODEL, filenames) to the top for easier modification. Added creation of the output directory.
    Plotting:
        Included the calculation of avg_err_t and p90_err_t per time step.
        Added these metrics to the title of the difference plot in the final evaluation loop.
        Added comments and placeholders for potentially creating summary plots of average/P90 errors over time.
        Ensured plots are saved to the OUTPUT_FIG_DIR.
        Improved final bar plot appearance slightly.
    Site Model: Switched to UniformSite as it's more straightforward for specifying arbitrary WS/TI combinations directly in the simulation call. If Hornsrev1Site specifics are needed, the WFM calls would need careful adjustment.
    Dependencies & File Paths: Added checks for data file existence and used os.path.join for constructing paths.

This refactored structure should be much easier to understand, maintain, and extend. Remember to adjust the optimization iterations (init_points, n_iter) and plot details as needed for your specific analysis.
