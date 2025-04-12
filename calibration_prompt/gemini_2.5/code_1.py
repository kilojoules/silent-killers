import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020
from py_wake import HorizontalGrid
from py_wake.deflection_models import JimenezWakeDeflection # Although not used? Keep import?
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter # Although not used? Keep import?
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative # PropagateDownwind not used
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
from py_wake.deficit_models.utils import ct2a_mom1d
import os # For creating output directories

# --- Configuration ---
MODEL = 2  # Choose model configuration (1 or 2)
DOWNWIND = True # True for downstream wake region, False for upstream blockage region
OUTPUT_DIR = f'results_model_{MODEL}_downwind_{DOWNWIND}'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figs')
ANIMATION_FILE = os.path.join(OUTPUT_DIR, f'optimization_animation_model_{MODEL}_downwind_{DOWNWIND}.mp4')
BARCHART_FILE = os.path.join(OUTPUT_DIR, f'bar_chart_model_{MODEL}_downwind_{DOWNWIND}.png')
AVG_ERROR_PLOT_FILE = os.path.join(OUTPUT_DIR, f'avg_error_field_model_{MODEL}_downwind_{DOWNWIND}.png')
P90_ERROR_PLOT_FILE = os.path.join(OUTPUT_DIR, f'p90_abs_error_field_model_{MODEL}_downwind_{DOWNWIND}.png')
INDIVIDUAL_PLOTS = False # Set to True to save plots for each time step

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(FIG_DIR) and INDIVIDUAL_PLOTS:
    os.makedirs(FIG_DIR)

if MODEL not in {1, 2}:
    raise ValueError("MODEL must be 1 or 2")

# --- Load Data and Setup Turbine/Site ---
try:
    dat = xr.load_dataset('./DTU10MW.nc')
except FileNotFoundError:
    raise FileNotFoundError("Could not find './DTU10MW.nc'. Make sure the dataset is in the correct path.")

turbine = DTU10MW()
D = turbine.diameter()
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D) # Scale coordinates by diameter

site = Hornsrev1Site()

# --- Define Region of Interest (ROI) ---
if DOWNWIND:
    X_LB, X_UB = 2, 10
else:
    X_LB, X_UB = -2, -1
roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D)

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x.values # Use .values for HorizontalGrid
target_y = flow_roi.y.values # Use .values for HorizontalGrid
print(f"Using ROI: x={roi_x}, y={roi_y}")
print(f"Target grid shape: x={target_x.shape}, y={target_y.shape}")

# --- Define Simulation Cases (Wind Speed, Turbulence Intensity) ---
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)
full_ws, full_ti = np.meshgrid(WSs, TIs)
full_ws = full_ws.flatten()
full_ti = full_ti.flatten()
full_wd = np.full_like(full_ws, 270.0) # Constant wind direction
n_cases = full_ws.size
print(f"Total simulation cases: {n_cases}")

# --- Pre-calculate Observed Deficits ---
# We need CT and TI for each case to interpolate the reference data
# Run a quick simulation with default models just to get CT/TI values
# (Or get them directly from the turbine object if possible and appropriate)
# Note: This assumes CT/TI are primarily functions of WS/TI and not significantly affected
# by the wake model parameters themselves during interpolation lookup.
temp_wfm = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                            superpositionModel=LinearSum(),
                            turbulenceModel=CrespoHernandez(),
                            blockage_deficitModel=SelfSimilarityDeficit2020())
sim_res_for_interp = temp_wfm([0], [0], ws=full_ws, ti=full_ti, wd=full_wd)

obs_values = []
for i in range(n_cases):
    # Ensure single values are passed for interpolation
    ct_val = sim_res_for_interp.CT.isel(wt=0, time=i).item()
    ti_val = sim_res_for_interp.TI.isel(wt=0, time=i).item()
    # Interpolate - Use fill_value=None to raise error if outside bounds, or a specific value like 0 or np.nan
    try:
        observed_deficit = flow_roi.deficits.interp(ct=ct_val, ti=ti_val, z=0, method='linear', kwargs={"fill_value": None})
        obs_values.append(observed_deficit.rename({'y': 'y_obs', 'x': 'x_obs'})) # Rename coords to avoid conflict later
    except ValueError as e:
        print(f"Warning: Interpolation failed for case {i} (WS={full_ws[i]}, TI={full_ti[i]}, CT={ct_val:.3f}). Skipping. Error: {e}")
        # Optionally append NaNs or handle differently
        # For now, we might exclude this case if interpolation fails.
        # This requires adjusting the simulation inputs later.
        # Let's create an array full of NaNs with the correct shape
        nan_array = xr.DataArray(np.full((len(target_y), len(target_x)), np.nan),
                                 coords=[('y_obs', target_y), ('x_obs', target_x)])
        obs_values.append(nan_array)


# Concatenate along a new 'time' dimension, matching the simulation output structure
# Ensure coordinates match the target grid AFTER interpolation
all_obs = xr.concat(obs_values, dim='time').rename({'x_obs':'x', 'y_obs':'y'})
all_obs['time'] = np.arange(n_cases) # Assign time coordinate explicitly
all_obs = all_obs.transpose('time', 'y', 'x') # Match typical simulation output order
print(f"Shape of observed deficits (all_obs): {all_obs.shape}")

# Check for cases completely filled with NaNs due to interpolation issues
nan_cases_mask = all_obs.isnull().all(dim=['x', 'y'])
valid_cases_mask = ~nan_cases_mask
valid_indices = np.where(valid_cases_mask)[0]

if len(valid_indices) != n_cases:
    print(f"Warning: {n_cases - len(valid_indices)} cases resulted in NaN due to interpolation issues. Excluding these from optimization.")
    # Filter simulation inputs and observed data to only include valid cases
    full_ws = full_ws[valid_indices]
    full_ti = full_ti[valid_indices]
    full_wd = full_wd[valid_indices]
    all_obs = all_obs.isel(time=valid_indices)
    all_obs['time'] = np.arange(len(valid_indices)) # Re-index time coord
    n_cases = len(valid_indices)
    print(f"Using {n_cases} valid simulation cases.")

if n_cases == 0:
    raise ValueError("No valid simulation cases remaining after checking interpolation. Check reference data bounds (CT, TI) and simulation inputs.")

# --- Modular Wind Farm Model Creation ---
def create_wind_farm_model(site, turbine, MODEL, DOWNWIND, params):
    """
    Instantiates and configures the PyWake WindFarmModel based on MODEL,
    DOWNWIND flag, and provided parameters.
    """
    wake_deficitModel = None
    blockage_deficitModel = None
    turbulenceModel = None
    def_args = {}
    turb_args = {}
    blockage_args = {}

    # --- Configure based on DOWNWIND (Wake) or UPSTREAM (Blockage) focus ---
    if DOWNWIND:
        # --- Configure WAKE model based on MODEL ---
        if MODEL == 1:
            # Blondel Wake Model + Crespo Turbulence
            def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            blockage_deficitModel = None # No blockage optimization in this path
        elif MODEL == 2:
            # TurboGaussian Wake Model + Crespo Turbulence
            wake_deficitModel = TurboGaussianDeficit(
                A=params['A'],
                cTI=[params['cti1'], params['cti2']],
                ctlim=params['ctlim'],
                ceps=params['ceps'],
                ct2a=ct2a_mom1d,
                groundModel=Mirror(), # Include ground model
                rotorAvgModel=GaussianOverlapAvgModel() # Include rotor avg model
            )
            # TurboGaussianDeficit might use a different WS key internally
            # wake_deficitModel.WS_key = 'WS_jlk' # Check if still needed/correct for current py_wake version
            blockage_deficitModel = None # No blockage optimization in this path
        else:
             raise ValueError(f"Invalid MODEL {MODEL} specified for DOWNWIND case.")

        # Common Turbulence model for DOWNWIND cases
        turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
        turbulenceModel = CrespoHernandez(**turb_args)

    else: # UPSTREAM (Blockage focus)
        # --- Configure BLOCKAGE model ---
        # Common Wake model (Blondel default) when focusing on blockage
        wake_deficitModel = BlondelSuperGaussianDeficit2020() # Use default params or allow tuning? Currently default.

        # Blockage model configuration depends on params provided
        blockage_args = {
            'ss_alpha': params['ss_alpha'],
            'ss_beta': params['ss_beta'],
            'r12p': np.array([params['rp1'], params['rp2']]),
            'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
            # Note: The 'fg' parameters from the original defaults seem to be for Frandsen Deficit, not SelfSimilarity
        }
        # Add ground model to blockage if MODEL == 2 (as per original logic)
        if MODEL == 2:
             blockage_args['groundModel'] = Mirror()

        blockage_deficitModel = SelfSimilarityDeficit2020(**blockage_args)
        turbulenceModel = None # No turbulence optimization in this path

    # --- Instantiate the Wind Farm Model ---
    wfm = All2AllIterative(
        site=site,
        windTurbines=turbine,
        wake_deficitModel=wake_deficitModel,
        superpositionModel=LinearSum(),
        deflectionModel=None, # Explicitly None as in original
        turbulenceModel=turbulenceModel,
        blockage_deficitModel=blockage_deficitModel
    )
    return wfm

# --- Objective Function for Bayesian Optimization ---
def evaluate_rmse(**kwargs):
    """
    Objective function: Calculates the negative Root Mean Square Error (RMSE)
    between observed and predicted wake deficits for the given parameters.
    """
    try:
        # 1. Create the wind farm model with current parameters
        wfm = create_wind_farm_model(site, turbine, MODEL, DOWNWIND, kwargs)

        # 2. Run the simulation for all cases
        sim_res = wfm(x=[0], y=[0], # Single turbine at origin
                      ws=full_ws,
                      ti=full_ti,
                      wd=full_wd,
                      time=True) # Use time dimension

        # 3. Calculate the flow map on the target grid
        flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))

        # 4. Calculate predicted deficit
        # Ensure WS has time dimension for broadcasting (it should from time=True)
        ws_free_stream = sim_res.WS.expand_dims(dim={'x': target_x, 'y': target_y}, axis=[2,3]) # Add spatial dims for broadcast

        # Check if flow_map has 'h' dimension, select if necessary
        if 'h' in flow_map.dims:
            ws_effective = flow_map.WS_eff.isel(h=0)
        else:
             ws_effective = flow_map.WS_eff # Assume only one height level if 'h' is missing

        # Ensure dimensions match for subtraction (time, y, x)
        ws_effective = ws_effective.transpose('time', 'y', 'x')
        ws_free_stream = ws_free_stream.transpose('time', 'y', 'x')


        # Add a small epsilon to avoid division by zero if WS is ever zero (unlikely for WS > 4)
        pred_deficit = (ws_free_stream - ws_effective) / (ws_free_stream + 1e-9)

        # 5. Calculate RMSE
        # Ensure dimensions match before subtraction
        if pred_deficit.shape != all_obs.shape:
             print(f"Shape mismatch: pred_deficit={pred_deficit.shape}, all_obs={all_obs.shape}")
             # Attempt to align if it's just coordinate names or order?
             # This indicates a potential issue in data preparation or simulation output handling.
             # For now, return a large error penalty.
             return -1.0 # Large penalty

        squared_errors = (all_obs - pred_deficit) ** 2
        rmse = float(np.sqrt(squared_errors.mean())) # Mean over all dimensions (time, x, y)

        # Handle potential NaNs from calculation or failed interpolation
        if np.isnan(rmse):
            print("Warning: RMSE calculation resulted in NaN. Returning large penalty.")
            return -0.5 # Return a penalty value as in original code

        # BayesianOptimization maximizes, so return negative RMSE
        return -rmse

    except Exception as e:
        # Catch potential errors during model instantiation or simulation
        print(f"Error during evaluation with params {kwargs}: {e}")
        import traceback
        traceback.print_exc()
        return -1.0 # Return a large penalty value

# --- Define Parameter Bounds and Defaults ---
if DOWNWIND:
    if MODEL == 1: # Blondel + Crespo
        pbounds = {
            'a_s': (0.001, 0.5), 'b_s': (0.001, 0.02), 'c_s': (0.001, 0.5), # Blondel wake
            'b_f': (-2, 1), 'c_f': (0.1, 5),                              # Blondel wake (Frandsen part)
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2), # Crespo turbulence
        }
        defaults = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
                    'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32}
    elif MODEL == 2: # TurboGaussian + Crespo
         pbounds = {
            'A': (0.001, .5), 'cti1': (.01, 5), 'cti2': (0.01, 5), # TurboGaussian wake
            'ceps': (0.01, 3), 'ctlim': (0.9, 1), # TurboGaussian wake (ctlim adjusted bounds)
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2), # Crespo turbulence
        }
         defaults = { 'A': 0.04, 'cti1': 1.5, 'cti2': 0.8, 'ceps': 0.25, 'ctlim': 0.999,
                     'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32 } # Combined defaults
else: # UPSTREAM (SelfSimilarity Blockage)
    # Assuming Blondel wake with defaults, only tuning blockage + potentially ground model effect
     pbounds = {
        'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),         # SelfSimilarity shape params
        'rp1': (-2, 2), 'rp2': (-2, 2),                      # SelfSimilarity radial power params
        'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3) # SelfSimilarity near-field Gaussian params
        # 'fg' parameters removed as they seem related to Frandsen, not SelfSimilarity
    }
     defaults = { # Defaults from original code (check if appropriate for SelfSimilarity)
        'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951,
        'rp1': -0.672, 'rp2': 0.4897,
        'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
    }
     # Note: MODEL==2 adds groundModel=Mirror() in create_wind_farm_model for blockage
     # No tunable parameters for Mirror() itself here.


# --- Run Bayesian Optimization ---
print("Starting Bayesian Optimization...")
optimizer = BayesianOptimization(
    f=evaluate_rmse,
    pbounds=pbounds,
    random_state=1,
    verbose=2 # Print progress
)

# Probe default parameters first
print("Probing default parameters...")
optimizer.probe(params=defaults, lazy=True)

# Run the optimization
optimizer.maximize(
    init_points=50, # Number of random exploration steps
    n_iter=200,     # Number of Bayesian optimization steps
)

print("Optimization Finished.")
print(f"Best parameters found: {optimizer.max['params']}")
best_params = optimizer.max['params']
best_rmse = -optimizer.max['target'] # RMSE is positive
print(f"Best RMSE achieved: {best_rmse:.5f}")

# --- Animation of Optimization Process ---
print("Generating optimization animation...")
fig_anim, (ax_conv, ax_params) = plt.subplots(1, 2, figsize=(15, 6))

def update_plot(frame):
    ax_conv.clear()
    ax_params.clear()

    # Get the best parameters and corresponding RMSE up to the current frame
    best_iter_params = {}
    best_iter_rmse = float('inf')
    history_rmse = -np.array([res['target'] for res in optimizer.res[:frame+1]])
    best_history_rmse = np.minimum.accumulate(history_rmse)

    # Find the parameters corresponding to the best RMSE up to this frame
    best_idx_so_far = np.argmin(history_rmse) # Index of best in current frame's history
    best_iter_params = optimizer.res[best_idx_so_far]['params']
    best_iter_rmse = history_rmse[best_idx_so_far]


    # Plot convergence
    ax_conv.plot(history_rmse, color='gray', alpha=0.5, label='Iteration RMSE')
    ax_conv.plot(best_history_rmse, color='black', label='Best RMSE so far')
    ax_conv.set_title('Optimization Convergence')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('RMSE')
    ax_conv.legend()
    ax_conv.grid(True)

    # Plot parameters (best found up to this frame vs default)
    keys = list(defaults.keys()) # Use default keys to ensure consistent ordering
    current_best_vals = [best_iter_params.get(k, np.nan) for k in keys] # Use .get for safety
    default_vals = [defaults[k] for k in keys]

    x_pos = np.arange(len(keys))
    width = 0.35

    rects1 = ax_params.bar(x_pos - width/2, current_best_vals, width, label='Optimized (Best so far)')
    rects2 = ax_params.bar(x_pos + width/2, default_vals, width, label='Default', alpha=0.7)

    ax_params.set_ylabel('Parameter Value')
    ax_params.set_title(f'Params at Best RMSE: {best_iter_rmse:.4f}')
    ax_params.set_xticks(x_pos)
    ax_params.set_xticklabels(keys, rotation=45, ha="right")
    ax_params.legend()
    ax_params.grid(axis='y', linestyle='--')

    fig_anim.tight_layout()
    return ax_conv, ax_params

# Create and save animation
ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(optimizer.res), repeat=False)
writer = animation.FFMpegWriter(fps=15)
ani.save(ANIMATION_FILE, writer=writer)
print(f"Saved optimization animation to {ANIMATION_FILE}")
plt.close(fig_anim)

# --- Final Evaluation and Plotting with Best Parameters ---
print("Evaluating final model with best parameters...")

# 1. Create the final optimized wind farm model
wfm_final = create_wind_farm_model(site, turbine, MODEL, DOWNWIND, best_params)

# 2. Run simulation with the final model
sim_res_final = wfm_final(x=[0], y=[0], ws=full_ws, ti=full_ti, wd=full_wd, time=True)

# 3. Calculate final flow map
flow_map_final = sim_res_final.flow_map(HorizontalGrid(x=target_x, y=target_y))

# 4. Calculate final predicted deficit
ws_free_stream_final = sim_res_final.WS.expand_dims(dim={'x': target_x, 'y': target_y}, axis=[2,3])
if 'h' in flow_map_final.dims:
    ws_effective_final = flow_map_final.WS_eff.isel(h=0)
else:
    ws_effective_final = flow_map_final.WS_eff
ws_effective_final = ws_effective_final.transpose('time', 'y', 'x')
ws_free_stream_final = ws_free_stream_final.transpose('time', 'y', 'x')

pred_deficit_final = (ws_free_stream_final - ws_effective_final) / (ws_free_stream_final + 1e-9)
pred_deficit_final = pred_deficit_final.transpose('time', 'y', 'x') # Ensure same order as all_obs

# 5. Calculate Errors
errors = all_obs - pred_deficit_final # Shape (time, y, x)
abs_errors = np.abs(errors)

# 6. Calculate Summary Error Fields
avg_error = errors.mean(dim='time')
p90_abs_error = abs_errors.quantile(0.9, dim='time', interpolation='linear') # Use linear interpolation for quantile

# 7. Calculate Overall Error Statistics
final_overall_rmse = float(np.sqrt(((errors) ** 2).mean()))
mean_abs_error = float(abs_errors.mean())
mean_avg_error = float(avg_error.mean()) # Should be close to 0 if unbiased
mean_p90_abs_error = float(p90_abs_error.mean())

print("\n--- Final Model Performance ---")
print(f"Overall RMSE: {final_overall_rmse:.5f}")
print(f"Mean Absolute Error (MAE): {mean_abs_error:.5f}")
print(f"Mean of Average Error Field: {mean_avg_error:.5f}")
print(f"Mean of P90 Absolute Error Field: {mean_p90_abs_error:.5f}")

# 8. Plot Summary Error Fields
print("Generating final summary plots...")

# Plot Average Error
fig_avg, ax_avg = plt.subplots(figsize=(8, 6))
levels = np.linspace(avg_error.min(), avg_error.max(), 11)
cf_avg = ax_avg.contourf(target_x / D, target_y / D, avg_error.T, levels=levels, cmap='RdBu_r') # Transpose if necessary
cb_avg = fig_avg.colorbar(cf_avg, ax=ax_avg)
ax_avg.set_xlabel('x/D')
ax_avg.set_ylabel('y/D')
ax_avg.set_title(f'Average Error (Observed - Predicted Deficit)\nMean: {mean_avg_error:.3f}')
ax_avg.set_aspect('equal', adjustable='box')
fig_avg.tight_layout()
fig_avg.savefig(AVG_ERROR_PLOT_FILE)
print(f"Saved average error plot to {AVG_ERROR_PLOT_FILE}")
plt.close(fig_avg)

# Plot P90 Absolute Error
fig_p90, ax_p90 = plt.subplots(figsize=(8, 6))
levels = np.linspace(0, p90_abs_error.max(), 11) # P90 of abs error starts from 0
cf_p90 = ax_p90.contourf(target_x / D, target_y / D, p90_abs_error.T, levels=levels, cmap='viridis') # Transpose if necessary
cb_p90 = fig_p90.colorbar(cf_p90, ax=ax_p90)
ax_p90.set_xlabel('x/D')
ax_p90.set_ylabel('y/D')
ax_p90.set_title(f'P90 Absolute Error |Observed - Predicted Deficit|\nMean: {mean_p90_abs_error:.3f}')
ax_p90.set_aspect('equal', adjustable='box')
fig_p90.tight_layout()
fig_p90.savefig(P90_ERROR_PLOT_FILE)
print(f"Saved P90 absolute error plot to {P90_ERROR_PLOT_FILE}")
plt.close(fig_p90)

# 9. (Optional) Plot individual time step comparisons
if INDIVIDUAL_PLOTS:
    print("Generating individual time step plots...")
    for t in range(n_cases):
        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

        # Determine common color scale for observed and predicted
        vmin = min(all_obs.isel(time=t).min(), pred_deficit_final.isel(time=t).min())
        vmax = max(all_obs.isel(time=t).max(), pred_deficit_final.isel(time=t).max())
        levels_op = np.linspace(vmin, vmax, 11)

        # Determine color scale for difference
        diff_max_abs = abs_errors.isel(time=t).max()
        levels_d = np.linspace(-diff_max_abs, diff_max_abs, 11)

        # Observed
        cf0 = axes_ind[0].contourf(target_x / D, target_y / D, all_obs.isel(time=t).T, levels=levels_op, cmap='viridis')
        fig_ind.colorbar(cf0, ax=axes_ind[0])
        axes_ind[0].set_title(f'Observed Deficit (t={t})')
        axes_ind[0].set_ylabel('y/D')

        # Predicted
        cf1 = axes_ind[1].contourf(target_x / D, target_y / D, pred_deficit_final.isel(time=t).T, levels=levels_op, cmap='viridis')
        fig_ind.colorbar(cf1, ax=axes_ind[1])
        axes_ind[1].set_title(f'Predicted Deficit (t={t})')
        axes_ind[1].set_xlabel('x/D')


        # Difference
        cf2 = axes_ind[2].contourf(target_x / D, target_y / D, errors.isel(time=t).T, levels=levels_d, cmap='RdBu_r')
        fig_ind.colorbar(cf2, ax=axes_ind[2])
        axes_ind[2].set_title(f'Error (Obs - Pred) (t={t})')

        for ax in axes_ind:
             ax.set_aspect('equal', adjustable='box')

        fig_ind.suptitle(f'WS={full_ws[t]:.1f} m/s, TI={full_ti[t]:.2f}')
        fig_ind.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        fig_ind.savefig(os.path.join(FIG_DIR, f'comparison_t_{t:03d}.png'))
        plt.close(fig_ind)
    print(f"Saved individual plots to {FIG_DIR}")


# 10. Plot Final Parameter Comparison Bar Chart
print("Generating final parameter bar chart...")
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
keys = list(defaults.keys())
best_vals = [best_params.get(k, np.nan) for k in keys]
default_vals = [defaults[k] for k in keys]
x_pos = np.arange(len(keys))
width = 0.35

rects1 = ax_bar.bar(x_pos - width/2, best_vals, width, label=f'Optimized (RMSE={best_rmse:.4f})')
rects2 = ax_bar.bar(x_pos + width/2, default_vals, width, label='Default', alpha=0.7)

ax_bar.set_ylabel('Parameter Value')
ax_bar.set_title('Optimized vs Default Parameters')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(keys, rotation=45, ha="right")
ax_bar.legend()
ax_bar.grid(axis='y', linestyle='--')
fig_bar.tight_layout()
fig_bar.savefig(BARCHART_FILE)
print(f"Saved final parameter bar chart to {BARCHART_FILE}")
plt.close(fig_bar)

print("\nScript finished.")
