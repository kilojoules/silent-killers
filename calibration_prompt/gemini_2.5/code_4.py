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
# from py_wake.deflection_models import JimenezWakeDeflection # Not used in the optimized models
from py_wake.turbulence_models import CrespoHernandez
# from py_wake.rotor_avg_models import RotorCenter # Not used in the optimized models
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import All2AllIterative # PropagateDownwind not used
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
from py_wake.deficit_models.utils import ct2a_mom1d
import warnings
import matplotlib # Use a non-interactive backend for saving figures
matplotlib.use('Agg')

# Suppress specific warnings if needed (e.g., RuntimeWarning from divide by zero if WS can be 0)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress xarray future warnings if any

# --- Configuration ---
DOWNWIND = True
MODEL = 2 # 1 for Blondel-based, 2 for TurboGaussian-based (downwind) or SS+Mirror (upwind)
DATA_FILE = './DTU10MW.nc'
OUTPUT_DIR = 'optimization_results'
ANIMATION_FILENAME = f'{OUTPUT_DIR}/optimization_animation_model{MODEL}_range' # Range added later
BAR_PLOT_FILENAME = f'{OUTPUT_DIR}/bar_plot_model{MODEL}_range' # Range added later
FINAL_PLOTS_FILENAME = f'{OUTPUT_DIR}/final_error_maps_model{MODEL}_range' # Range added later

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

if MODEL not in {1, 2}:
    raise ValueError("MODEL must be 1 or 2")

# --- Load Data and Setup Turbine/Site ---
try:
    dat = xr.load_dataset(DATA_FILE)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}. Please ensure it's in the correct path.")

turbine = DTU10MW()
D = turbine.diameter()
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)

# --- Define Region of Interest (ROI) ---
if DOWNWIND:
    X_LB, X_UB = 2, 10
    region_label = f"{X_LB}D_to_{X_UB}D_downwind"
else:
    X_LB, X_UB = -2, -1
    region_label = f"{X_LB}D_to_{X_UB}D_upwind"

# Update filenames with range info
ANIMATION_FILENAME = ANIMATION_FILENAME.replace('_range', f'_{region_label}') + '.mp4'
BAR_PLOT_FILENAME = BAR_PLOT_FILENAME.replace('_range', f'_{region_label}') + '.png'
FINAL_PLOTS_FILENAME = FINAL_PLOTS_FILENAME.replace('_range', f'_{region_label}') + '.png'


roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D)

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x.values # Use .values for efficiency in grid creation later
target_y = flow_roi.y.values

# --- Define Simulation Conditions ---
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)
full_ti = np.repeat(TIs, len(WSs))
full_ws = np.tile(WSs, len(TIs))
assert full_ws.size == full_ti.size

site = Hornsrev1Site()

# --- Pre-calculate Observed Deficits ---
# Use a default WFM setup just to get plausible CT/TI values for interpolation
# Note: This assumes the optimized parameters won't drastically change CT/TI.
# If they do, the comparison might be less accurate.
temp_wfm = All2AllIterative(
    site, turbine,
    wake_deficitModel=BlondelSuperGaussianDeficit2020(),
    superpositionModel=LinearSum(),
    blockage_deficitModel=SelfSimilarityDeficit2020() # Include blockage for consistency
)
initial_sim_res = temp_wfm(
    [0], [0], # Single turbine at origin
    ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, # Wind from West
    time=True # Ensure time dimension exists
)

obs_values = []
for t in range(initial_sim_res.time.size):
    # Ensure we select the turbine's operating point (wt=0)
    ct_interp = initial_sim_res.CT.isel(time=t, wt=0).item() # Use .item() for scalar
    ti_interp = initial_sim_res.TI.isel(time=t, wt=0).item()

    # Interpolate. Use fill_value=np.nan if outside bounds, handle later.
    observed_deficit = flow_roi.deficits.interp(
        ct=ct_interp, ti=ti_interp, z=0, # Assuming reference data is at hub height (z=0)
        method="linear", # Explicitly state method
        kwargs={"fill_value": np.nan} # Handle extrapolation
    )
    obs_values.append(observed_deficit.rename({'y': 'y_obs', 'x': 'x_obs'})) # Rename to avoid clashes

# Concatenate along the 'time' dimension, renaming coords back if necessary
all_obs = xr.concat(obs_values, dim='time').rename({'y_obs': 'y', 'x_obs': 'x'})
# Ensure coordinates match the target grid - interpolation might introduce small differences
all_obs['x'] = target_x
all_obs['y'] = target_y

print(f"Shape of pre-calculated observed deficits (all_obs): {all_obs.shape}")
# Check for NaNs introduced by interpolation, potentially filter them later if problematic
print(f"Number of NaN values in all_obs: {np.isnan(all_obs.values).sum()}")


# --- Modular WFM Creation Function ---
def create_wfm(site, turbine, params, model_type, is_downwind):
    """Instantiates the PyWake wind farm model based on configuration."""

    wake_deficitModel = None
    blockage_deficitModel = None
    turbulenceModel = None
    deflectionModel = None # Keeping deflection off as per original code
    superpositionModel = LinearSum()

    # --- Configure Wake Deficit Model ---
    if is_downwind:
        if model_type == 1:
            def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
        elif model_type == 2:
            # Ensure necessary keys exist in params
            required_keys = ['A', 'cti1', 'cti2', 'ctlim', 'ceps']
            if not all(k in params for k in required_keys):
                raise KeyError(f"Missing required parameters for TurboGaussianDeficit: {required_keys}")

            wake_deficitModel = TurboGaussianDeficit(
                A=params['A'],
                cTI=[params['cti1'], params['cti2']],
                ctlim=params['ctlim'],
                ceps=params['ceps'],
                ct2a=ct2a_mom1d,
                groundModel=Mirror(), # As per original conditional logic
                rotorAvgModel=GaussianOverlapAvgModel() # As per original
            )
            # wake_deficitModel.WS_key = 'WS_jlk' # PyWake usually handles this internally now
        else:
            raise ValueError("Invalid model_type for downwind")

    else: # Upwind (Blockage focus)
        # For upwind, use a simple wake model (or None if purely blockage)
        # The original code still used Blondel here, let's keep that consistency.
        wake_deficitModel = BlondelSuperGaussianDeficit2020() # Using defaults
        # Configure Blockage Model (SelfSimilarityDeficit2020)
        required_keys = ['ss_alpha', 'ss_beta', 'rp1', 'rp2', 'ng1', 'ng2', 'ng3', 'ng4']
        if not all(k in params for k in required_keys):
            raise KeyError(f"Missing required parameters for SelfSimilarityDeficit2020: {required_keys}")

        blockage_args = {
            'ss_alpha': params['ss_alpha'],
            'ss_beta': params['ss_beta'],
            'r12p': np.array([params['rp1'], params['rp2']]),
            'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
        }
        if model_type == 2: # Add Mirror ground model for SS blockage only if MODEL == 2
             blockage_args['groundModel'] = Mirror()
        # Add fg parameters if they are part of the optimization space for UPWIND MODEL 1
        if model_type == 1 and 'fg1' in params:
             blockage_args['fgp'] = np.array([params['fg1'], params['fg2'], params['fg3'], params['fg4']])

        blockage_deficitModel = SelfSimilarityDeficit2020(**blockage_args)


    # --- Configure Turbulence Model ---
    # CrespoHernandez is used in both downwind cases and optionally for upwind
    if 'ch1' in params: # Check if turbulence params are being optimized
        turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
        turbulenceModel = CrespoHernandez(**turb_args)
    else:
        turbulenceModel = CrespoHernandez() # Use defaults if not optimizing


    # --- Instantiate WFM ---
    wfm = All2AllIterative(
        site, turbine,
        wake_deficitModel=wake_deficitModel,
        superpositionModel=superpositionModel,
        deflectionModel=deflectionModel,
        turbulenceModel=turbulenceModel,
        blockage_deficitModel=blockage_deficitModel
    )
    return wfm

# --- Objective Function for Bayesian Optimization ---
simulation_grid = HorizontalGrid(x=target_x, y=target_y) # Define grid once

def evaluate_rmse(**kwargs):
    """Evaluates the RMSE for a given set of parameters."""
    try:
        # Instantiate WFM using the modular function
        wfm = create_wfm(site, turbine, kwargs, MODEL, DOWNWIND)

        # Run simulation
        sim_res = wfm(
            [0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True
        )

        # Calculate flow map efficiently for all time steps
        flow_map = sim_res.flow_map(simulation_grid) # No need to loop

        # Calculate predicted deficit
        # Ensure broadcasting works correctly: sim_res.WS is (time, wt), flow_map is (time, h, y, x)
        # We need WS corresponding to the free stream for each time step.
        # Assuming single turbine (wt=0) and hub height (h=0)
        ws_free_stream = sim_res.WS.isel(wt=0) # Shape (time)
        ws_effective = flow_map.WS_eff.isel(h=0) # Shape (time, y, x)

        # Calculate deficit: (FreeStream - Effective) / FreeStream
        # Need to align dimensions for broadcasting: ws_free_stream[:, np.newaxis, np.newaxis]
        pred = (ws_free_stream[:, np.newaxis, np.newaxis] - ws_effective) / ws_free_stream[:, np.newaxis, np.newaxis]
        pred = pred.rename({'x': 'x', 'y': 'y'}) # Ensure coords are named correctly if needed


        # Calculate RMSE
        # Make sure dimensions and coordinates align between all_obs and pred
        if not all(np.allclose(all_obs[c].values, pred[c].values) for c in ['x', 'y']):
             print("Warning: Mismatch in x/y coordinates between observations and predictions.")
             # Attempt to reindex prediction to match observation coordinates
             pred = pred.reindex_like(all_obs, method='nearest', tolerance=1e-6)


        squared_error = (all_obs - pred) ** 2
        mean_sq_error_spatial = squared_error.mean(dim=['x', 'y']) # Mean over spatial dimensions first
        rmse_per_time = np.sqrt(mean_sq_error_spatial)
        mean_rmse = float(rmse_per_time.mean(dim='time')) # Mean over time


        # Handle potential NaN results (e.g., from division by zero if WS=0, or interpolation issues)
        if np.isnan(mean_rmse):
            print(f"Warning: NaN RMSE encountered for parameters: {kwargs}")
            return -1.0 # Return a very poor score instead of NaN

        return -mean_rmse # Maximize negative RMSE

    except Exception as e:
        print(f"Error during evaluation with params {kwargs}: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging
        return -1.0 # Return a very poor score on error


# --- Define Parameter Bounds and Defaults ---
if MODEL == 1:
    if DOWNWIND:
        pbounds = {
            'a_s': (0.001, 0.5), 'b_s': (0.001, 0.01), 'c_s': (0.001, 0.5),
            'b_f': (-2, 1), 'c_f': (0.1, 5),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
                    'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32}
    else: # UPWIND MODEL 1 (Focus on SelfSimilarity + Blondel Wake + CH Turb)
         pbounds = {
            'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
            'rp1': (-2, 2), 'rp2': (-2, 2),
            'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3),
            # Include fg parameters for SelfSimilarity as per original definition
            'fg1': (-2, 2), 'fg2': (-2, 2), 'fg3': (-2, 2), 'fg4': (-2, 2),
            # Optionally include turbulence params if desired for upwind
            # 'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
         defaults = {
            'ss_alpha': 0.8889, 'ss_beta': 1.4142,
            'rp1': -0.672, 'rp2': 0.4897,
            'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
            'fg1': -0.0649, 'fg2': 0.4911, 'fg3': 1.116, 'fg4': -0.1577,
            # 'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32 # Default CH if optimizing
         }
elif MODEL == 2:
    if DOWNWIND: # MODEL 2 DOWNWIND (TurboGaussian + CH Turb)
        pbounds = {
            'A': (0.001, .5), 'cti1': (.01, 5), 'cti2': (0.01, 5),
            'ceps': (0.01, 3), 'ctlim': (0.01, 1),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {
            'A': 0.04, 'cti1': 1.5, 'cti2': 0.8, 'ceps': 0.25, 'ctlim': 0.999,
            'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.3
        }
    else: # UPWIND MODEL 2 (Focus on SelfSimilarity with Mirror + Blondel Wake + CH Turb)
        # Same SS params as UPWIND MODEL 1, but groundModel=Mirror is added in create_wfm
         pbounds = {
            'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
            'rp1': (-2, 2), 'rp2': (-2, 2),
            'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3),
            # fg parameters are part of SS definition, but maybe not tuned in this specific original setup?
            # Let's keep them consistent with UPWIND MODEL 1 for now. Add/remove as needed.
            'fg1': (-2, 2), 'fg2': (-2, 2), 'fg3': (-2, 2), 'fg4': (-2, 2),
            # Optionally include turbulence params if desired for upwind
            # 'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
         defaults = { # Same defaults as UPWIND MODEL 1 for SS params
            'ss_alpha': 0.8889, 'ss_beta': 1.4142,
            'rp1': -0.672, 'rp2': 0.4897,
            'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
            'fg1': -0.0649, 'fg2': 0.4911, 'fg3': 1.116, 'fg4': -0.1577,
            # 'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32 # Default CH if optimizing
         }

# --- Bayesian Optimization ---
print("Setting up Bayesian Optimization...")
optimizer = BayesianOptimization(f=evaluate_rmse, pbounds=pbounds, random_state=1, verbose=2)

# Probe default parameters first
print("Probing default parameters...")
try:
    optimizer.probe(params=defaults, lazy=True)
except KeyError as e:
    print(f"Error probing default parameters: Missing key {e}. Check defaults match pbounds for MODEL={MODEL}, DOWNWIND={DOWNWIND}")
    # Handle error, maybe exit or use different defaults

print("Starting optimization...")
# Consider reducing init_points/n_iter for faster testing if needed
optimizer.maximize(init_points=10, n_iter=40) # Reduced for quicker run example
# optimizer.maximize(init_points=50, n_iter=200) # Original values

best_params = optimizer.max['params']
best_rmse = -optimizer.max['target']

print("\nOptimization Finished.")
print(f"Best parameters found: {best_params}")
print(f"Best RMSE: {best_rmse}")


# --- Animation of Optimization Process ---
print("Generating optimization animation...")

fig_ani, (ax_ani1, ax_ani2) = plt.subplots(1, 2, figsize=(15, 6))

def update_plot(frame):
    ax_ani1.clear()
    ax_ani2.clear()

    iterations = np.arange(frame + 1)
    all_targets = -np.array([res['target'] for res in optimizer.res[:frame+1]]) # Get actual RMSEs

    # Calculate best RMSE up to current frame
    best_rmse_history = np.minimum.accumulate(all_targets)
    current_best_rmse = best_rmse_history[-1]
    best_params_idx = np.argmin(all_targets) # Index of the best params so far
    current_best_params = optimizer.res[best_params_idx]['params']


    # Plot RMSE history
    ax_ani1.plot(iterations, all_targets, color='gray', alpha=0.5, label='RMSE per iteration')
    ax_ani1.plot(iterations, best_rmse_history, color='black', label='Best RMSE so far')
    ax_ani1.set_title('Optimization Convergence')
    ax_ani1.set_xlabel('Iteration')
    ax_ani1.set_ylabel('RMSE')
    ax_ani1.legend()
    ax_ani1.grid(True)
    ax_ani1.set_ylim(bottom=0) # RMSE shouldn't be negative

    # Plot parameters
    keys = list(current_best_params.keys()) # Use keys from the best params found so far
    best_vals = [current_best_params.get(k, np.nan) for k in keys] # Use .get for safety
    default_vals = [defaults.get(k, np.nan) for k in keys] # Use .get for safety

    x_pos = np.arange(len(keys))
    width = 0.35

    ax_ani2.bar(x_pos - width/2, best_vals, width, label='Optimized (Best So Far)')
    # Ensure default comparison only uses keys present in the current optimization
    ax_ani2.bar(x_pos + width/2, default_vals, width, label='Default', edgecolor='black', linewidth=1, color='none')

    ax_ani2.set_title(f'Params @ Best RMSE: {current_best_rmse:.4f}')
    ax_ani2.set_xticks(x_pos)
    ax_ani2.set_xticklabels(keys, rotation=45, ha='right')
    ax_ani2.legend()
    ax_ani2.grid(axis='y')
    fig_ani.tight_layout() # Apply tight layout to the figure

    return ax_ani1, ax_ani2

# Check if optimizer ran and has results
if optimizer.res:
    ani = animation.FuncAnimation(fig_ani, update_plot, frames=len(optimizer.res), repeat=False)
    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=10) # Reduced fps slightly
        ani.save(ANIMATION_FILENAME, writer=writer)
        print(f"Animation saved to {ANIMATION_FILENAME}")
    except FileNotFoundError:
        print("FFMpeg writer not found. Cannot save animation. Please install ffmpeg.")
    except Exception as e:
        print(f"Error saving animation: {e}")
else:
    print("No optimization results found, skipping animation.")

plt.close(fig_ani) # Close the animation figure


# --- Final Analysis with Best Parameters ---
print("Running final simulation with best parameters...")
final_wfm = create_wfm(site, turbine, best_params, MODEL, DOWNWIND)
final_sim_res = final_wfm(
    [0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True
)

# Calculate final flow map and prediction correctly
final_flow_map = final_sim_res.flow_map(simulation_grid)

ws_free_stream_final = final_sim_res.WS.isel(wt=0)
ws_effective_final = final_flow_map.WS_eff.isel(h=0)
final_pred = (ws_free_stream_final[:, np.newaxis, np.newaxis] - ws_effective_final) / ws_free_stream_final[:, np.newaxis, np.newaxis]
final_pred = final_pred.rename({'x': 'x', 'y': 'y'}) # Ensure coords match

# Reindex final prediction just in case coords differ slightly after calculation
final_pred = final_pred.reindex_like(all_obs, method='nearest', tolerance=1e-6)

# Calculate difference (Error = Observed - Predicted)
diff = all_obs - final_pred

# Calculate summary statistics for error maps
avg_error = diff.mean(dim='time')
p90_error = diff.quantile(0.9, dim='time', interpolation='linear') # Specify interpolation for quantile

# --- Plotting Final Error Maps ---
print("Generating final error maps...")
fig_final, axes_final = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Determine common color scale limits based on errors
max_abs_avg_err = np.nanmax(np.abs(avg_error.values))
max_p90_err = np.nanmax(p90_error.values)
min_p90_err = np.nanmin(p90_error.values) # P90 can be negative if model overpredicts deficit
vmin_p90 = min(min_p90_err, -max_p90_err) # Symmetrical around 0 if possible, else fit range
vmax_p90 = max(max_p90_err, -min_p90_err)

# Plot Average Error
cont1 = axes_final[0].contourf(target_x / D, target_y / D, avg_error.T, cmap='coolwarm', levels=np.linspace(-max_abs_avg_err, max_abs_avg_err, 11))
fig_final.colorbar(cont1, ax=axes_final[0], label='Average Error (Obs - Pred)')
axes_final[0].set_title('Average Deficit Error')
axes_final[0].set_xlabel('Downwind Distance (x/D)')
axes_final[0].set_ylabel('Crosswind Distance (y/D)')
axes_final[0].set_aspect('equal', adjustable='box')


# Plot P90 Error
cont2 = axes_final[1].contourf(target_x / D, target_y / D, p90_error.T, cmap='coolwarm', levels=np.linspace(vmin_p90, vmax_p90, 11))
fig_final.colorbar(cont2, ax=axes_final[1], label='P90 Error (Obs - Pred)')
axes_final[1].set_title('90th Percentile Deficit Error')
axes_final[1].set_xlabel('Downwind Distance (x/D)')
# axes_final[1].set_ylabel('Crosswind Distance (y/D)') # Shared Y axis
axes_final[1].set_aspect('equal', adjustable='box')


fig_final.suptitle(f'Deficit Error Maps (Model {MODEL}, {region_label}) - Best RMSE: {best_rmse:.4f}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig(FINAL_PLOTS_FILENAME)
print(f"Final error maps saved to {FINAL_PLOTS_FILENAME}")
plt.close(fig_final)

# --- Final Parameter Bar Plot ---
print("Generating final parameter comparison bar plot...")
fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

keys = list(best_params.keys())
best_vals = [best_params.get(k, np.nan) for k in keys]
default_vals = [defaults.get(k, np.nan) for k in keys] # Use defaults corresponding to the specific model/case

x_pos = np.arange(len(keys))
width = 0.35

ax_bar.bar(x_pos - width/2, best_vals, width, label='Optimized')
ax_bar.bar(x_pos + width/2, default_vals, width, label='Default', edgecolor='black', linewidth=1, color='none')

ax_bar.set_ylabel('Parameter Value')
ax_bar.set_title(f'Optimized vs Default Parameters (Model {MODEL}, {region_label})\nOptimal RMSE: {best_rmse:.4f}')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(keys, rotation=45, ha='right')
ax_bar.legend()
ax_bar.grid(axis='y')
plt.tight_layout()
plt.savefig(BAR_PLOT_FILENAME)
print(f"Final parameter bar plot saved to {BAR_PLOT_FILENAME}")
plt.close(fig_bar)

print("\nScript finished.")
