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
from py_wake.deflection_models import JimenezWakeDeflection # Keep for potential future use
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter # Keep for potential future use
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
from py_wake.deficit_models.utils import ct2a_mom1d
import os
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", message=".*Creating an ndarray from ragged nested sequences.*") # Suppress numpy warning from bayes_opt

MODEL = 2 # 1: Blondel-based, 2: TurboGaussian-based (Wake)
DOWNWIND = True # True: Optimize Downwind Wake, False: Optimize Upstream Blockage

OUTPUT_DIR = f'results_model_{MODEL}_downwind_{DOWNWIND}'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)

# --- Load Data and Turbine ---
dat = xr.load_dataset('./DTU10MW.nc')
turbine = DTU10MW()
D = turbine.diameter()
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D) # Scale coordinates

# --- Define Region of Interest (ROI) ---
if DOWNWIND:
    X_LB, X_UB = 2, 10
    REGION_NAME = "Downwind"
else:
    X_LB, X_UB = -2, -1
    REGION_NAME = "Upstream"

roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D)

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x.values # Use .values to avoid potential xarray coord issues later
target_y = flow_roi.y.values
grid_for_flowmap = HorizontalGrid(x=target_x, y=target_y)

# --- Define Simulation Conditions ---
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)
full_ws, full_ti = np.meshgrid(WSs, TIs)
full_ws = full_ws.flatten()
full_ti = full_ti.flatten()
n_conditions = full_ti.size

# --- Site ---
site = Hornsrev1Site()

# --- Pre-calculate Observed Deficits ---
# Run a preliminary simulation just to get CT and TI values corresponding to WS/TI inputs
# Use default parameters for this, the specific model doesn't matter as much as getting the CT/TI
prelim_wfm = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                            superpositionModel=LinearSum(),
                            blockage_deficitModel=SelfSimilarityDeficit2020(),
                            turbulenceModel=CrespoHernandez())

sim_res_prelim = prelim_wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * n_conditions, time=True)

# Interpolate observed data onto the CT/TI values from the simulation conditions
obs_values = []
for t in range(n_conditions):
    this_sim_state = sim_res_prelim.isel(time=t, wt=0)
    # Interpolate deficits based on the calculated CT and TI for this condition
    observed_deficit = flow_roi.deficits.interp(ct=this_sim_state.CT, ti=this_sim_state.TI, z=0)
    obs_values.append(observed_deficit.T) # Transpose to match (y, x) convention if needed

all_obs = xr.concat(obs_values, dim='time').rename({'time': 'condition'}) # Rename dim for clarity
all_obs = all_obs.assign_coords(condition=np.arange(n_conditions)) # Ensure condition coord is set

# --- Modular WFM Creation Function ---
def create_wfm(params, downwind_flag, model_type_flag):
    """
    Creates and configures the WindFarmModel based on flags and parameters.

    Args:
        params (dict): Dictionary of parameters for sub-models.
        downwind_flag (bool): True for downwind wake focus, False for upstream blockage.
        model_type_flag (int): 1 for Blondel-based, 2 for TurboGaussian-based wake.

    Returns:
        All2AllIterative: Configured wind farm model instance.
    """
    wake_deficitModel = None
    blockage_deficitModel = None
    turb_args = {}
    wake_args = {}
    blockage_args = {}

    # --- Configure Turbulence Model (Common for Downwind) ---
    if downwind_flag:
         turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}

    # --- Configure Wake/Blockage Models based on flags ---
    if downwind_flag:
        # Downwind Optimization: Tune Wake and Turbulence
        blockage_deficitModel = None # Or a default non-tuned SelfSimilarity? Let's assume None for now.
                                     # If you want to include default blockage:
                                     # blockage_deficitModel=SelfSimilarityDeficit2020()

        if model_type_flag == 1: # Blondel Wake
            wake_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**wake_args)
        elif model_type_flag == 2: # TurboGaussian Wake
            wake_args = {
                'A': params['A'],
                'cTI': [params['cti1'], params['cti2']],
                'ctlim': params['ctlim'],
                'ceps': params['ceps'],
                'ct2a': ct2a_mom1d, # Kept from original
                'groundModel': Mirror(), # Kept from original
                'rotorAvgModel': GaussianOverlapAvgModel() # Kept from original
            }
            wake_deficitModel = TurboGaussianDeficit(**wake_args)
            wake_deficitModel.WS_key = 'WS_jlk' # Kept from original
        else:
            raise ValueError("Invalid model_type_flag for downwind optimization")

    else: # Upstream Optimization: Tune Blockage (SelfSimilarity)
        wake_deficitModel = None # Or a default non-tuned Wake model? Let's assume None.
                                 # If you want to include default wake:
                                 # wake_deficitModel=BlondelSuperGaussianDeficit2020()
        turb_args = {} # Turbulence model not typically tuned for blockage

        blockage_args = {
            'ss_alpha': params['ss_alpha'],
            'ss_beta': params['ss_beta'],
            'r12p': np.array([params['rp1'], params['rp2']]),
            'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
        }
        # Ground model for blockage depends on MODEL flag (as per original logic)
        if model_type_flag == 2: # This condition seemed arbitrary in the original, but preserved here
             blockage_args['groundModel'] = Mirror()

        blockage_deficitModel = SelfSimilarityDeficit2020(**blockage_args)


    # --- Instantiate WFM ---
    wfm = All2AllIterative(
        site=site,
        windTurbines=turbine,
        wake_deficitModel=wake_deficitModel,
        superpositionModel=LinearSum(),
        deflectionModel=None, # Assuming no deflection tuning
        turbulenceModel=CrespoHernandez(**turb_args) if turb_args else None, # Only add if params exist
        blockage_deficitModel=blockage_deficitModel
    )
    return wfm

# --- Define Parameter Bounds and Defaults ---
pbounds = {}
defaults = {}

if DOWNWIND:
    defaults_turb = {'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32}
    pbounds_turb = {'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2)}
    defaults.update(defaults_turb)
    pbounds.update(pbounds_turb)

    if MODEL == 1: # Blondel Wake
        defaults_wake = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41}
        pbounds_wake = {'a_s': (0.001, 0.5), 'b_s': (0.001, 0.01), 'c_s': (0.001, 0.5), 'b_f': (-2, 1), 'c_f': (0.1, 5)}
        defaults.update(defaults_wake)
        pbounds.update(pbounds_wake)
    elif MODEL == 2: # TurboGaussian Wake
        defaults_wake = {'A': 0.04, 'cti1': 1.5, 'cti2': 0.8, 'ceps': 0.25, 'ctlim': 0.999}
        pbounds_wake = {'A': (0.001, .5), 'cti1': (.01, 5), 'cti2': (0.01, 5), 'ceps': (0.01, 3), 'ctlim': (0.01, 1)}
        defaults.update(defaults_wake)
        pbounds.update(pbounds_wake)
    else:
        raise ValueError("Invalid MODEL number for DOWNWIND")

else: # UPSTREAM (Blockage)
    defaults_blockage = {
        'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951,
        'rp1': -0.672, 'rp2': 0.4897,
        'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336
    }
    # Note: Original code had 'fg1'..'fg4' in defaults but not pbounds - assumed typo and ignored here. If needed, add them.
    pbounds_blockage = {
        'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
        'rp1': (-2, 2), 'rp2': (-2, 2),
        'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3)
    }
    defaults.update(defaults_blockage)
    pbounds.update(pbounds_blockage)
    # Note: The original UPSTREAM code also had bounds for fg1-fg4, but didn't use them in SelfSimilarityDeficit2020
    # If the 'fg' parameters *are* needed for SelfSimilarityDeficit2020 (check PyWake docs for your version), add them here.

# --- Objective Function for Bayesian Optimization ---
def evaluate_rmse(**kwargs):
    """
    Evaluates the RMSE between simulated and observed deficits for given parameters.
    Designed to be maximized by BayesianOptimization (returns -RMSE).
    """
    try:
        # 1. Create WFM instance with current parameters
        wfm = create_wfm(kwargs, DOWNWIND, MODEL)

        # 2. Run simulation
        sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * n_conditions, time=True)

        # 3. Calculate flow map (once for all conditions)
        # Ensure grid has no conflicting coordinates with sim_res
        flow_map = sim_res.flow_map(grid=grid_for_flowmap) # Use pre-defined grid

        # 4. Calculate predicted deficit
        # Ensure shapes align: sim_res.WS is (condition, wt), flow_map['WS_eff'] is (condition, h, y, x)
        # Need to broadcast WS correctly. sim_res.WS is likely identical for all wt=[0] and h=[0] here.
        ws_free_stream = sim_res.WS.isel(wt=0) # Shape (condition,)
        ws_eff = flow_map['WS_eff'].isel(h=0) # Shape (condition, y, x)
        # Expand ws_free_stream to match ws_eff for broadcasting
        ws_free_stream_expanded = ws_free_stream.expand_dims(dim={'y': target_y, 'x': target_x}, axis=[1,2])

        pred = (ws_free_stream_expanded - ws_eff) / ws_free_stream_expanded
        pred = pred.rename({'condition': 'condition'}) # Ensure dim name matches all_obs

        # Realign coordinates just in case floating point differences caused issues
        pred = pred.reindex_like(all_obs, method='nearest', tolerance=1e-6)
        obs_reindexed = all_obs.reindex_like(pred, method='nearest', tolerance=1e-6)


        # 5. Calculate overall RMSE (mean over space and conditions)
        rmse = float(np.sqrt(((obs_reindexed - pred) ** 2).mean(['x', 'y', 'condition'])))

        # Handle potential NaNs from simulation/interpolation issues
        if np.isnan(rmse):
            print("Warning: NaN RMSE encountered. Returning large penalty.")
            return -10.0 # Return a large negative number (bad score)

        # BayesianOptimization maximizes, so return negative RMSE
        return -rmse

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(f"Parameters causing error: {kwargs}")
        return -10.0 # Penalize errors heavily


# --- Bayesian Optimization ---
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

# Run optimization
print("Running optimization iterations...")
optimizer.maximize(
    init_points=50, # Number of random exploration steps
    n_iter=200,     # Number of Bayesian optimization steps
    # Add other args like acquisition function if needed
)

print("Optimization Complete.")
print(f"Best parameters found: {optimizer.max['params']}")
print(f"Best RMSE: {-optimizer.max['target']}")

best_params = optimizer.max['params']
best_rmse = -optimizer.max['target']

# --- Save Optimization Results ---
results_df = optimizer.res_to_df() # Get results as pandas DataFrame
results_df.to_csv(os.path.join(OUTPUT_DIR, f'optimization_history_{REGION_NAME}_model_{MODEL}.csv'))

# --- Generate Optimization Animation ---
print("Generating optimization animation...")
fig_anim, (ax_anim1, ax_anim2) = plt.subplots(1, 2, figsize=(15, 6))

def update_plot(frame):
    ax_anim1.clear()
    ax_anim2.clear()

    iterations = np.arange(frame + 1)
    targets = -optimizer.space.target[:frame+1] # RMSE values

    # Find best RMSE up to current frame
    best_history = np.minimum.accumulate(targets)
    current_best_rmse = best_history[-1]
    best_idx = np.argmin(targets) # Index of the best params *within this frame's subset*
    # Need the actual best params up to this frame
    best_overall_idx_so_far = np.argmin(optimizer.space.target[:frame+1])
    current_best_params = optimizer.res[best_overall_idx_so_far]['params']


    # Plot RMSE history
    ax_anim1.plot(iterations, targets, color='gray', alpha=0.5, label='RMSE per iteration')
    ax_anim1.plot(iterations, best_history, color='black', label='Best RMSE so far')
    ax_anim1.set_title('Optimization Convergence')
    ax_anim1.set_xlabel('Iteration')
    ax_anim1.set_ylabel('RMSE')
    ax_anim1.legend()
    ax_anim1.grid(True)

    # Plot parameters
    keys = list(defaults.keys()) # Use defaults keys to ensure consistent order
    best_vals = [current_best_params.get(k, np.nan) for k in keys] # Use .get for safety
    default_vals = [defaults[k] for k in keys]

    x_pos = np.arange(len(keys))
    width = 0.35

    rects1 = ax_anim2.bar(x_pos - width/2, best_vals, width, label='Optimized (Best so far)')
    rects2 = ax_anim2.bar(x_pos + width/2, default_vals, width, label='Default', alpha=0.6)
    # Or use outline style from original code:
    # ax_anim2.bar(x_pos + width/2, default_vals, width, edgecolor='black', linewidth=1.5, color='none', label='Default')


    ax_anim2.set_ylabel('Parameter Value')
    ax_anim2.set_title(f'Parameters (Best RMSE: {current_best_rmse:.4f})')
    ax_anim2.set_xticks(x_pos)
    ax_anim2.set_xticklabels(keys, rotation=45, ha="right")
    ax_anim2.legend()
    ax_anim2.grid(axis='y', linestyle='--')
    fig_anim.tight_layout() # Use fig level tight_layout

    return fig_anim, # Return the figure object

# Create and save animation
ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(optimizer.space.target), blit=False, repeat=False) # blit=False often more reliable
writer = animation.FFMpegWriter(fps=10)
anim_filename = os.path.join(OUTPUT_DIR, f'optimization_animation_{REGION_NAME}_model_{MODEL}.mp4')
ani.save(anim_filename, writer=writer)
print(f"Animation saved to {anim_filename}")
plt.close(fig_anim) # Close the animation figure


# --- Final Analysis with Best Parameters ---
print("Running final analysis with best parameters...")

# 1. Create WFM with best parameters
wfm_best = create_wfm(best_params, DOWNWIND, MODEL)

# 2. Run simulation
sim_res_best = wfm_best([0], [0], ws=full_ws, TI=full_ti, wd=[270] * n_conditions, time=True)

# 3. Calculate flow map
flow_map_best = sim_res_best.flow_map(grid=grid_for_flowmap)

# 4. Calculate predicted deficit
ws_free_stream_best = sim_res_best.WS.isel(wt=0)
ws_eff_best = flow_map_best['WS_eff'].isel(h=0)
ws_free_stream_best_expanded = ws_free_stream_best.expand_dims(dim={'y': target_y, 'x': target_x}, axis=[1,2])
pred_best = (ws_free_stream_best_expanded - ws_eff_best) / ws_free_stream_best_expanded
pred_best = pred_best.rename({'condition': 'condition'})

# Realign coordinates
pred_best = pred_best.reindex_like(all_obs, method='nearest', tolerance=1e-6)
obs_reindexed_final = all_obs.reindex_like(pred_best, method='nearest', tolerance=1e-6)

# 5. Calculate Error Metrics per Condition
error = pred_best - obs_reindexed_final
abs_error = np.abs(error)

mae_per_condition = abs_error.mean(dim=['x', 'y'])
mean_error_per_condition = error.mean(dim=['x', 'y']) # Bias
rmse_per_condition = np.sqrt((error**2).mean(dim=['x', 'y']))
p90_abs_error_per_condition = abs_error.quantile(0.9, dim=['x', 'y'])

# Create a DataFrame for easier plotting/analysis of metrics
metrics_df = pd.DataFrame({
    'WS': full_ws,
    'TI': full_ti,
    'Condition': np.arange(n_conditions),
    'RMSE': rmse_per_condition.values,
    'MAE': mae_per_condition.values,
    'Bias': mean_error_per_condition.values,
    'P90_Abs_Error': p90_abs_error_per_condition.values
})

# Verify overall RMSE calculated matches the optimizer's result
overall_rmse_check = np.sqrt((error**2).mean())
print(f"Overall RMSE from final calc: {overall_rmse_check:.5f} (Optimizer best: {best_rmse:.5f})")

# --- Plotting Final Results ---

# 1. Bar plot of parameters
print("Generating final parameter comparison plot...")
plt.figure(figsize=(10, 6))
keys = list(defaults.keys())
best_vals = [best_params.get(k, np.nan) for k in keys]
default_vals = [defaults[k] for k in keys]
x_pos = np.arange(len(keys))
width = 0.35

plt.bar(x_pos - width/2, best_vals, width, label='Optimized')
plt.bar(x_pos + width/2, default_vals, width, label='Default', alpha=0.6)
# Or outline:
# plt.bar(x_pos + width/2, default_vals, width, edgecolor='black', linewidth=1.5, color='none', label='Default')

plt.ylabel('Parameter Value')
plt.title(f'Optimized vs Default Parameters (Final RMSE: {best_rmse:.4f})')
plt.xticks(x_pos, keys, rotation=45, ha="right")
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'param_comparison_{REGION_NAME}_model_{MODEL}.png'))
plt.close()

# 2. Plot error metrics vs WS/TI
print("Generating error metric plots...")
fig_metrics, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

metrics_to_plot = ['RMSE', 'MAE', 'Bias', 'P90_Abs_Error']
colors = plt.cm.viridis(np.linspace(0, 1, len(TIs))) # Color by TI

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    for j, ti_val in enumerate(TIs):
        subset = metrics_df[metrics_df['TI'] == ti_val]
        ax.plot(subset['WS'], subset[metric], marker='o', linestyle='-', color=colors[j], label=f'TI={ti_val:.2f}')
    ax.set_title(f'{metric} vs Wind Speed')
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=':')
    if i >= 2: # Only xlabel on bottom row
       ax.set_xlabel('Wind Speed (m/s)')
    if i == 0: # Add legend to first plot
        ax.legend(title="Turbulence Intensity", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle(f'Error Metrics vs Conditions ({REGION_NAME}, Model {MODEL})', y=1.02)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.savefig(os.path.join(OUTPUT_DIR, f'error_metrics_{REGION_NAME}_model_{MODEL}.png'))
plt.close()

# 3. Example Contour Plot (Optional: e.g., for median RMSE case)
print("Generating example contour plot...")
median_rmse_condition = metrics_df.iloc[(metrics_df['RMSE'] - metrics_df['RMSE'].median()).abs().argsort()[:1]]
median_idx = median_condition['Condition'].iloc[0]
median_ws = median_condition['WS'].iloc[0]
median_ti = median_condition['TI'].iloc[0]

obs_median = obs_reindexed_final.sel(condition=median_idx)
pred_median = pred_best.sel(condition=median_idx)
error_median = error.sel(condition=median_idx)

vmin = min(obs_median.min(), pred_median.min()) * 0.9 # Shared scale for deficits
vmax = max(obs_median.max(), pred_median.max()) * 1.1
err_max_abs = np.abs(error_median).max() # Symmetric scale for error

fig_contour, axc = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

c0 = axc[0].contourf(target_x / D, target_y / D, obs_median, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
axc[0].set_title(f'Observed Deficit (WS={median_ws} m/s, TI={median_ti:.2f})')
fig_contour.colorbar(c0, ax=axc[0], label='Velocity Deficit')

c1 = axc[1].contourf(target_x / D, target_y / D, pred_median, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
axc[1].set_title(f'Predicted Deficit (RMSE={metrics_df.loc[median_idx, "RMSE"]:.4f})')
fig_contour.colorbar(c1, ax=axc[1], label='Velocity Deficit')

c2 = axc[2].contourf(target_x / D, target_y / D, error_median, levels=50, cmap='coolwarm', vmin=-err_max_abs, vmax=err_max_abs)
axc[2].set_title('Error (Prediction - Observation)')
fig_contour.colorbar(c2, ax=axc[2], label='Error')

for ax in axc:
    ax.set_xlabel('Downwind Distance (x/D)')
    ax.set_ylabel('Crosswind Distance (y/D)')
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f'example_contours_{REGION_NAME}_model_{MODEL}_cond_{median_idx}.png'))
plt.close()

print(f"--- Analysis Complete. Results saved in {OUTPUT_DIR} ---")
