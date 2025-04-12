import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import wake and deficit models
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.deficit_models.gaussian import TurboGaussianDeficit, BlondelSuperGaussianDeficit2020
from py_wake import HorizontalGrid
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.deficit_models.utils import ct2a_mom1d

from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import Hornsrev1Site

from bayes_opt import BayesianOptimization

# Global model settings
MODEL = 2           # Either 1 or 2 (see below branches)
DOWNWIND = True     # True: downwind, False: upstream

# ============================================================================
# DATA & DOMAIN SETUP FUNCTIONS
# ============================================================================
def load_flow_data(file_path, turbine):
    """
    Load the Xarray dataset and rescale x, y coordinates based on the turbine diameter.
    """
    dat = xr.load_dataset(file_path)
    D = turbine.diameter()
    dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)
    return dat, D

def setup_domain(dat, D, downwind=True):
    """
    Define region of interest (ROI) for the flow field.
    Returns: flow_roi, target_x, target_y, X_LB, X_UB
    """
    if downwind:
        X_LB = 2
        X_UB = 10
    else:
        X_LB = -2
        X_UB = -1
    roi_x = slice(X_LB * D, X_UB * D)
    roi_y = slice(-2 * D, 2 * D)
    flow_roi = dat.sel(x=roi_x, y=roi_y)
    return flow_roi, flow_roi.x, flow_roi.y, X_LB, X_UB

def get_conditions(WS_range=np.arange(4, 11), TI_range=np.arange(0.05, 0.45, 0.05)):
    """
    Generate flattened arrays for wind speed and turbulence intensity combinations.
    """
    full_ti = np.array([TI_range for _ in range(WS_range.size)]).flatten()
    full_ws = np.array([[WS_range[ii]] * TI_range.size for ii in range(WS_range.size)]).flatten()
    assert full_ws.size == full_ti.size
    return full_ws, full_ti

# ============================================================================
# WAKE MODEL & SIMULATION FUNCTIONS
# ============================================================================
def create_wake_model(site, turbine, params, model=MODEL, downwind=DOWNWIND, defaults=None):
    """
    Create and return a wake model instance based on given parameters.
    This encapsulates all the logic for creating the wake deficit and turbine/turbulence settings.
    """
    if downwind:
        if model == 1:
            # Use Blondel Super Gaussian model
            def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            blockage_args = {}
        else:  # MODEL == 2
            turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
            wake_deficitModel = TurboGaussianDeficit(A=params['A'], 
                                                     cTI=[params['cti1'], params['cti2']],
                                                     ctlim=params['ctlim'], ceps=params['ceps'],
                                                     ct2a=ct2a_mom1d,
                                                     groundModel=Mirror(),
                                                     rotorAvgModel=GaussianOverlapAvgModel())
            # Flag if alternate velocity key is needed (see your data/velocity deficit differences)
            wake_deficitModel.WS_key = 'WS_jlk'
            blockage_args = {}
    else:
        # For upstream conditions use different settings:
        turb_args = {}
        wake_deficitModel = BlondelSuperGaussianDeficit2020()
        blockage_args = {'ss_alpha': params['ss_alpha'], 
                         'ss_beta': params['ss_beta'], 
                         'r12p': np.array([params['rp1'], params['rp2']]), 
                         'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])}
        if model == 2:
            blockage_args['groundModel'] = Mirror()
    # Instantiate the wind farm model
    wfm = All2AllIterative(site, turbine,
                           wake_deficitModel=wake_deficitModel,
                           superpositionModel=LinearSum(), deflectionModel=None,
                           turbulenceModel=CrespoHernandez(**turb_args),
                           blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args))
    return wfm

def run_simulation(wfm, full_ws, full_ti, target_x, target_y):
    """
    Run the simulation over all time steps and concatenate the flow map results.
    Returns the simulation result and the aggregated flow map.
    """
    sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ws.size, time=True)
    # Build full flow map over time
    flow_map = None
    for tt in range(full_ws.size):
        fm = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y), time=[tt])['WS_eff']
        flow_map = fm if flow_map is None else xr.concat([flow_map, fm], dim='time')
    return sim_res, flow_map

def compute_rmse(sim_res, flow_map, flow_roi, target_x, target_y):
    """
    Compute RMSE between observed deficits (interpolated onto simulation conditions)
    and the relative deficit prediction from the simulation.
    
    The predicted deficit is computed as:
         (simulated free-stream WS - effective WS) / simulated free-stream WS
         
    Returns the RMSE over the grid and across time.
    """
    rmse_values = []
    for t in range(flow_map.time.size):
        this_pred_sim = sim_res.isel(time=t)
        observed_deficit = flow_roi.deficits.interp(ct=this_pred_sim.CT, ti=this_pred_sim.TI, z=0).isel(wt=0)
        pred = (this_pred_sim.WS - flow_map.isel(h=0, time=t)) / this_pred_sim.WS
        diff = observed_deficit.T - pred
        rmse = np.sqrt(np.mean(diff**2))
        rmse_values.append(rmse)
    overall_rmse = np.mean(rmse_values)
    return overall_rmse, rmse_values

# ============================================================================
# OPTIMIZATION EVALUATION & PLOTTING
# ============================================================================
def evaluate_rmse(**kwargs):
    """
    This function is the objective for Bayesian Optimization.
    Given keyword parameters, it instantiates the wake model, runs the simulation, 
    and returns the negative RMSE as the objective function.
    """
    # Create a wake model from current parameters in kwargs
    site = Hornsrev1Site()
    turbine = DTU10MW()
    wfm = create_wake_model(site, turbine, kwargs, model=MODEL, downwind=DOWNWIND)
    # Re-use the conditions and grid from the main scope
    sim_res, flow_map = run_simulation(wfm, full_ws, full_ti, target_x, target_y)
    # Compute relative deficit prediction based on WS_eff versus free-stream WS
    pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
    rmse = float(np.sqrt(((all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
    # Return negative rmse (since BayesianOptimization maximizes)
    return -rmse if not np.isnan(rmse) else -0.5

def update_plot(frame, optimizer, defaults, ax1, ax2):
    """
    Callback for matplotlib animation to show optimization progress. 
    Reports both the best RMSE so far and the optimized parameter values compared to their defaults.
    """
    ax1.clear()
    ax2.clear()
    best_so_far_params = {}
    best_so_far_rmse = float('inf')
    best_so_far_rmses = []
    for i in range(frame + 1):
        current_rmse = -optimizer.space.target[i]
        if current_rmse <= best_so_far_rmse:
            best_so_far_rmse = current_rmse
            best_so_far_params = optimizer.res[i]['params']
        best_so_far_rmses.append(best_so_far_rmse)

    ax1.plot(-np.array(optimizer.space.target), color='gray', alpha=0.5, label='All RMSE')
    ax1.plot(np.array(best_so_far_rmses), color='black', label='Best RMSE so far')
    ax1.set_title('Optimization Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)
    ax1.legend()

    keys = list(best_so_far_params.keys())
    best_vals = [best_so_far_params[key] for key in keys]
    default_vals = [defaults[key] for key in keys]

    ax2.bar(keys, best_vals, label='Optimized')
    ax2.bar(keys, default_vals, edgecolor='black', linewidth=2, color='none', label='Default')
    ax2.set_title(f'Best RMSE: {best_so_far_rmse:.4f}')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    plt.tight_layout()
    return ax1, ax2

def plot_flow_field_comparison(sim_res, flow_map, flow_roi, target_x, target_y, save_prefix='figs/downsream_err'):
    """
    For every time step, generate a triple subplot figure to display:
      - The observed deficit field
      - The predicted deficit (computed as (WS - WS_eff) / WS)
      - The difference (error) field along with reporting average and p90 error metrics.
    Saves each figure using the provided prefix.
    """
    for t in range(flow_map.time.size):
        this_pred_sim = sim_res.isel(time=t)
        observed_deficit = flow_roi.deficits.interp(ct=this_pred_sim.CT, ti=this_pred_sim.TI, z=0).isel(wt=0)
        pred = (this_pred_sim.WS - flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
        diff = observed_deficit.T - pred

        avg_error = np.mean(np.abs(diff))
        p90_error = np.percentile(np.abs(diff), 90)

        fig, ax = plt.subplots(3, 1, figsize=(5, 15))
        cf0 = ax[0].contourf(target_x, target_y, observed_deficit.T)
        cf1 = ax[1].contourf(target_x, target_y, pred)
        cf2 = ax[2].contourf(target_x, target_y, diff)
        plt.colorbar(cf0, ax=ax[0])
        plt.colorbar(cf1, ax=ax[1])
        plt.colorbar(cf2, ax=ax[2])
        ax[0].set_ylabel('Observed')
        ax[1].set_ylabel('Prediction')
        ax[2].set_ylabel('Diff')
        ax[2].set_title(f'Avg error: {avg_error:.4f}, P90: {p90_error:.4f}')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{t}.png')
        plt.clf()
    plt.close('all')

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Instantiate turbine and site
    turbine = DTU10MW()
    site = Hornsrev1Site()

    # Load dataset and set up domain
    dat, D = load_flow_data('./DTU10MW.nc', turbine)
    flow_roi, target_x, target_y, X_LB, X_UB = setup_domain(dat, D, downwind=DOWNWIND)

    # Generate condition arrays
    full_ws, full_ti = get_conditions()

    # Run initial simulation to collect reference (observed) deficits
    sim_res_init = All2AllIterative(site, turbine,
                                    wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                                    superpositionModel=LinearSum(), deflectionModel=None,
                                    turbulenceModel=CrespoHernandez(),
                                    blockage_deficitModel=SelfSimilarityDeficit2020())([0], [0],
                                                                                         ws=full_ws, TI=full_ti,
                                                                                         wd=[270] * full_ws.size,
                                                                                         time=True)
    flow_map_init = sim_res_init.flow_map(HorizontalGrid(x=target_x, y=target_y))
    # The reference observations: note that flow_roi.deficits is used.
    obs_values = []
    for t in range(flow_map_init.time.size):
        this_sim = sim_res_init.isel(time=t, wt=0)
        obs_val = flow_roi.deficits.interp(ct=this_sim.CT, ti=this_sim.TI, z=0)
        obs_values.append(obs_val.T)
    all_obs = xr.concat(obs_values, dim='time')

    # Define parameter bounds and defaults for the optimizer based on MODEL and DOWNWIND:
    if MODEL == 1:
        if DOWNWIND:
            pbounds = {
                'a_s': (0.001, 0.5),
                'b_s': (0.001, 0.01),
                'c_s': (0.001, 0.5),
                'b_f': (-2, 1),
                'c_f': (0.1, 5),
                'ch1': (-1, 2),
                'ch2': (-1, 2),
                'ch3': (-1, 2),
                'ch4': (-1, 2),
            }
            defaults = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
                        'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32}
        else:
            pbounds = {
                'ss_alpha': (0.05, 3),
                'ss_beta': (0.05, 3),
                'rp1': (-2, 2),
                'rp2': (-2, 2),
                'ng1': (-3, 3),
                'ng2': (-3, 3),
                'ng3': (-3, 3),
                'ng4': (-3, 3),
                'fg1': (-2, 2),
                'fg2': (-2, 2),
                'fg3': (-2, 2),
                'fg4': (-2, 2)
            }
            defaults = {
                'ss_alpha': 0.8888888888888888,
                'ss_beta': 1.4142135623730951,
                'rp1': -0.672,
                'rp2': 0.4897,
                'ng1': -1.381,
                'ng2': 2.627,
                'ng3': -1.524,
                'ng4': 1.336,
                'fg1': -0.06489,
                'fg2': 0.4911,
                'fg3': 1.116,
                'fg4': -0.1577
            }
    else:
        # MODEL == 2
        pbounds = {
            'A': (0.001, .5),
            'cti1': (0.01, 5),
            'cti2': (0.01, 5),
            'ceps': (0.01, 3),
            'ctlim': (0.01, 1),
            'ch1': (-1, 2),
            'ch2': (-1, 2),
            'ch3': (-1, 2),
            'ch4': (-1, 2),
        }
        defaults = {
            'A': 0.04,
            'cti1': 1.5,
            'cti2': 0.8,
            'ceps': 0.25,
            'ctlim': 0.999,
            'ch1': 0.73,
            'ch2': 0.8325,
            'ch3': -0.0325,
            'ch4': -0.3
        }

    # Run Bayesian Optimization
    optimizer = BayesianOptimization(f=evaluate_rmse, pbounds=pbounds, random_state=1)
    optimizer.probe(params=defaults, lazy=True)
    optimizer.maximize(init_points=50, n_iter=200)

    best_params = optimizer.max['params']
    best_rmse = -optimizer.max['target']

    # Create an animation for optimization convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ani = animation.FuncAnimation(
        fig, 
        lambda frame: update_plot(frame, optimizer, defaults, ax1, ax2),
        frames=len(optimizer.space.target),
        repeat=False
    )
    writer = animation.FFMpegWriter(fps=15)
    ani.save('optimization_animation_%i_%i.mp4' % (X_LB, X_UB), writer=writer)
    plt.close('all')

    # Use best_params for a final simulation
    if DOWNWIND:
        if MODEL == 1:
            # Use Blondel deficit model
            def_params = {k: best_params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_params)
        else:
            wake_deficitModel = TurboGaussianDeficit(A=best_params['A'], 
                                                     cTI=[best_params['cti1'], best_params['cti2']],
                                                     ctlim=best_params['ctlim'],
                                                     ceps=best_params['ceps'],
                                                     ct2a=ct2a_mom1d,
                                                     groundModel=Mirror(),
                                                     rotorAvgModel=GaussianOverlapAvgModel())
            wake_deficitModel.WS_key = 'WS_jlk'
        turb_args = {'c': np.array([best_params['ch1'], best_params['ch2'], best_params['ch3'], best_params['ch4']])}
        blockage_args = {}
    else:
        turb_args = {}
        blockage_args = {'ss_alpha': best_params['ss_alpha'], 
                         'ss_beta': best_params['ss_beta'], 
                         'r12p': np.array([best_params['rp1'], best_params['rp2']]), 
                         'ngp': np.array([best_params['ng1'], best_params['ng2'], best_params['ng3'], best_params['ng4']])}
        if MODEL == 2:
            blockage_args['groundModel'] = Mirror()

    # Finally, build the final wfm using the optimized parameters:
    wfm_final = All2AllIterative(site, turbine,
                                 wake_deficitModel=wake_deficitModel,
                                 superpositionModel=LinearSum(), deflectionModel=None,
                                 turbulenceModel=CrespoHernandez(**turb_args),
                                 blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args))
    sim_res_final, flow_map_final = run_simulation(wfm_final, full_ws, full_ti, target_x, target_y)
    overall_rmse, rmse_values = compute_rmse(sim_res_final, flow_map_final, flow_roi, target_x, target_y)
    print(f"RMSE values per time step: {rmse_values}")
    print(f"Overall RMSE: {overall_rmse}")

    # Save final bar plot showing parameter comparison (optimized vs. default)
    best_vals = [best_params[key] for key in best_params.keys()]
    default_vals = [defaults[key] for key in best_params.keys()]
    plt.bar(list(best_params.keys()), best_vals, label='Optimized')
    plt.bar(list(best_params.keys()), default_vals, edgecolor='black', linewidth=2,
            color='none', label='Default')
    plt.title('Optimal RMSE: %.4f' % best_rmse)
    plt.tight_layout()
    plt.savefig('bar_LB_%i_UP_%i.png' % (X_LB, X_UB))
    plt.clf()

    # Plot the flow field comparisons with enhanced error reporting (average & p90)
    plot_flow_field_comparison(sim_res_final, flow_map_final, flow_roi, target_x, target_y, 
                               save_prefix='figs/downsream_err')

