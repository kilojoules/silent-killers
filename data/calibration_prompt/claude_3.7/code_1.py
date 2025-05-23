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
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
from py_wake.deficit_models.utils import ct2a_mom1d


class WindFarmModel:
    """A modular wind farm model for wake and blockage analysis."""
    
    def __init__(self, data_path='./DTU10MW.nc', model=2, downwind=True):
        """
        Initialize the wind farm model.
        
        Args:
            data_path: Path to the dataset file
            model: Model type (1 or 2)
            downwind: If True, analyze wake effects. If False, analyze blockage effects.
        """
        self.MODEL = model
        self.DOWNWIND = downwind
        self.validate_model()
        
        # Initialize turbine and site
        self.turbine = DTU10MW()
        self.D = self.turbine.diameter()
        self.site = Hornsrev1Site()
        
        # Load dataset
        self.dat = self.load_dataset(data_path)
        
        # Set region of interest
        self.set_roi()
        
        # Setup test conditions
        self.setup_test_conditions()
        
        # Set default parameters and bounds
        self.set_defaults_and_bounds()
        
        # Initialize best parameters and RMSE
        self.best_params = None
        self.best_rmse = None
        self.optimizer = None
        
    def validate_model(self):
        """Validate model selection."""
        if self.MODEL not in {1, 2}:
            raise ValueError(f"Model must be 1 or 2, got {self.MODEL}")
    
    def load_dataset(self, data_path):
        """Load and prepare dataset."""
        dat = xr.load_dataset(data_path)
        # Normalize coordinates by turbine diameter
        return dat.assign_coords(x=dat.x * self.D, y=dat.y * self.D)
    
    def set_roi(self):
        """Set region of interest based on model type."""
        if self.DOWNWIND:
            self.X_LB = 2
            self.X_UB = 10
        else:
            self.X_LB = -2
            self.X_UB = -1
            
        roi_x = slice(self.X_LB * self.D, self.X_UB * self.D)
        roi_y = slice(-2 * self.D, 2 * self.D)
        
        self.flow_roi = self.dat.sel(x=roi_x, y=roi_y)
        self.target_x = self.flow_roi.x
        self.target_y = self.flow_roi.y
    
    def setup_test_conditions(self):
        """Set up test wind speeds and turbulence intensities."""
        self.TIs = np.arange(0.05, 0.45, 0.05)
        self.WSs = np.arange(4, 11)
        
        # Create full TI and WS arrays
        full_ti = [self.TIs for _ in range(self.WSs.size)]
        self.full_ti = np.array(full_ti).flatten()
        
        full_ws = [[self.WSs[ii]] * self.TIs.size for ii in range(self.WSs.size)]
        self.full_ws = np.array(full_ws).flatten()
        
        assert self.full_ws.size == self.full_ti.size
        
        # Get observed values
        self.get_observed_values()
    
    def get_observed_values(self):
        """Get observed deficit values from reference model."""
        # Create reference simulation
        sim_res = All2AllIterative(
            self.site, self.turbine,
            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
            superpositionModel=LinearSum(), 
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(),
            blockage_deficitModel=SelfSimilarityDeficit2020()
        )([0], [0], ws=self.full_ws, TI=self.full_ti, wd=[270] * self.full_ti.size, time=True)
        
        # Create flow map
        self.flow_map = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y))
        
        # Collect observed values
        obs_values = []
        for t in range(self.flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t, wt=0)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            )
            obs_values.append(observed_deficit.T)
        
        self.all_obs = xr.concat(obs_values, dim='time')
    
    def set_defaults_and_bounds(self):
        """Set default parameters and bounds based on model type."""
        if self.MODEL == 1:
            if self.DOWNWIND:
                self.pbounds = {
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
                self.defaults = {
                    'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
                    'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32
                }
            else:
                self.pbounds = {
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
                self.defaults = {
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
        else:  # MODEL == 2
            self.defaults = {
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
            self.pbounds = {
                'A': (0.001, .5),
                'cti1': (.01, 5),
                'cti2': (0.01, 5),
                'ceps': (0.01, 3),
                'ctlim': (0.01, 1),
                'ch1': (-1, 2),
                'ch2': (-1, 2),
                'ch3': (-1, 2),
                'ch4': (-1, 2),
            }
    
    def create_wind_farm_model(self, params):
        """
        Create wind farm model with specific parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            The configured wind farm model
        """
        if self.DOWNWIND:
            if self.MODEL == 1:
                def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
                turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
                blockage_args = {}
                wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            else:  # MODEL == 2
                turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
                wake_deficitModel = TurboGaussianDeficit(
                    A=params['A'], 
                    cTI=[params['cti1'], params['cti2']],
                    ctlim=params['ctlim'], 
                    ceps=params['ceps'],
                    ct2a=ct2a_mom1d,
                    groundModel=Mirror(),
                    rotorAvgModel=GaussianOverlapAvgModel()
                )
                wake_deficitModel.WS_key = 'WS_jlk'
                blockage_args = {}
        else:  # not DOWNWIND
            def_args = {}
            turb_args = {}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            blockage_args = {
                'ss_alpha': params['ss_alpha'], 
                'ss_beta': params['ss_beta'], 
                'r12p': np.array([params['rp1'], params['rp2']]), 
                'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
            }
            if self.MODEL == 2:
                blockage_args['groundModel'] = Mirror()
        
        # Create and return the wind farm model
        return All2AllIterative(
            self.site, 
            self.turbine,
            wake_deficitModel=wake_deficitModel,
            superpositionModel=LinearSum(), 
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(**turb_args),
            blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args)
        )
    
    def evaluate_rmse(self, **kwargs):
        """
        Evaluate RMSE for given parameters.
        
        Returns:
            Negative RMSE (for maximization)
        """
        # Create wind farm model with parameters
        wfm = self.create_wind_farm_model(kwargs)
        
        # Run simulation
        sim_res = wfm([0], [0], ws=self.full_ws, TI=self.full_ti, wd=[270] * self.full_ti.size, time=True)
        
        # Calculate flow map
        flow_map = None
        for tt in range(self.full_ws.size):
            fm = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y), time=[tt])['WS_eff']
            if flow_map is None:
                flow_map = fm
            else:
                flow_map = xr.concat([flow_map, fm], dim='time')
        
        # Calculate deficit and RMSE
        pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        rmse = float(np.sqrt(((self.all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
        
        if np.isnan(rmse):
            return -0.5
        
        return -rmse  # Negative because we're maximizing
    
    def optimize(self, init_points=50, n_iter=200, random_state=1):
        """
        Run Bayesian optimization to find optimal parameters.
        
        Args:
            init_points: Number of initial random points
            n_iter: Number of optimization iterations
            random_state: Random seed for reproducibility
        """
        self.optimizer = BayesianOptimization(
            f=self.evaluate_rmse, 
            pbounds=self.pbounds, 
            random_state=random_state
        )
        
        # Probe with defaults
        self.optimizer.probe(params=self.defaults, lazy=True)
        
        # Run optimization
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Store best parameters and RMSE
        self.best_params = self.optimizer.max['params']
        self.best_rmse = -self.optimizer.max['target']
        
        return self.best_params, self.best_rmse
    
    def create_optimization_animation(self, output_filename=None):
        """
        Create animation of optimization progress.
        
        Args:
            output_filename: Filename for the output animation
        """
        if self.optimizer is None:
            raise ValueError("Must run optimize() before creating animation")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def update_plot(frame):
            ax1.clear()
            ax2.clear()
            
            # Get the best parameters and RMSE up to the current frame
            best_so_far_params = {}
            best_so_far_rmse = float('inf')
            best_so_far_rmses = []
            
            for i in range(frame + 1):
                if -self.optimizer.space.target[i] <= best_so_far_rmse:
                    best_so_far_rmse = -self.optimizer.space.target[i]
                    best_so_far_params = self.optimizer.res[i]['params']
                best_so_far_rmses.append(best_so_far_rmse)
            
            # Plot the entire history in gray
            ax1.plot(-np.array(self.optimizer.space.target), color='gray', alpha=0.5)
            # Plot the best RMSE so far in black
            ax1.plot(np.array(best_so_far_rmses), color='black')
            ax1.set_title('Optimization Convergence')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('RMSE')
            ax1.grid(True)
            
            # Use the best parameters so far for the bar plot
            keys = list(best_so_far_params.keys())
            best_vals = []
            default_vals = []
            
            for key in keys:
                best_vals.append(best_so_far_params[key])
                default_vals.append(self.defaults[key])
            
            ax2.bar(keys, best_vals, label='Optimized')
            ax2.bar(keys, default_vals, edgecolor='black', linewidth=2, color='none', capstyle='butt', label='Default')
            ax2.set_title(f'Best RMSE: {best_so_far_rmse:.4f}')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            plt.tight_layout()
            return ax1, ax2
        
        ani = animation.FuncAnimation(
            fig, 
            update_plot, 
            frames=len(self.optimizer.space.target), 
            repeat=False
        )
        
        if output_filename is None:
            output_filename = f'optimization_animation_{self.X_LB}_{self.X_UB}.mp4'
            
        # Save animation
        writer = animation.FFMpegWriter(fps=15)
        ani.save(output_filename, writer=writer)
        plt.close('all')
        
        return output_filename
    
    def create_parameter_comparison_plot(self, output_filename=None):
        """
        Create bar plot comparing default and optimized parameters.
        
        Args:
            output_filename: Filename for the output plot
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before creating comparison plot")
            
        best_vals = []
        default_vals = []
        keys = self.best_params.keys()
        
        for key in keys:
            best_vals.append(self.best_params[key])
            default_vals.append(self.defaults[key])
        
        plt.figure(figsize=(12, 6))
        plt.bar(keys, best_vals, label='Optimized')
        plt.bar(
            keys, 
            default_vals,
            edgecolor='black',
            linewidth=2,
            color='none',
            capstyle='butt',
            label='Default'
        )
        
        plt.title(f'Optimal RMSE: {self.best_rmse:.4f}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if output_filename is None:
            output_filename = f'bar_LB_{self.X_LB}_UP_{self.X_UB}.png'
            
        plt.savefig(output_filename)
        plt.close()
        
        return output_filename
    
    def evaluate_model_performance(self, output_dir='figs'):
        """
        Evaluate model performance using the best parameters.
        
        Args:
            output_dir: Directory to save output figures
        
        Returns:
            Overall RMSE and per-time-step RMSE values
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before evaluating model performance")
        
        # Create model with best parameters
        wfm = self.create_wind_farm_model(self.best_params)
        
        # Run simulation
        sim_res = wfm([0], [0], ws=self.full_ws, TI=self.full_ti, wd=[270] * self.full_ti.size, time=True)
        
        # Calculate RMSE for each time step
        rmse_values = []
        errors = []
        
        for t in range(self.flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            ).isel(wt=0)
            
            pred = (this_pred_sim.WS - self.flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
            diff = observed_deficit.T - pred
            
            # Store all errors for calculating statistics
            errors.append(diff.values.flatten())
            
            # Calculate RMSE for this time step
            rmse = np.sqrt(np.mean(diff**2))
            rmse_values.append(rmse)
            
            # Create visualization
            fig, ax = plt.subplots(3, 1, figsize=(5, 15))
            co = ax[0].contourf(self.target_x, self.target_y, observed_deficit.T)
            cp = ax[1].contourf(self.target_x, self.target_y, pred)
            cd = ax[2].contourf(self.target_x, self.target_y, diff)
            
            for jj, c in enumerate([co, cp, cd]):
                fig.colorbar(c, ax=ax[jj])
                
            ax[0].set_ylabel('Observed')
            ax[1].set_ylabel('Prediction')
            ax[2].set_ylabel('Diff')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/downsream_err_{t}.png')
            plt.close()
        
        # Calculate overall RMSE
        overall_rmse = np.mean(rmse_values)
        
        # Calculate p90 of errors
        all_errors = np.concatenate(errors)
        p90_error = np.percentile(np.abs(all_errors), 90)
        
        print(f"RMSE values per time step: {rmse_values}")
        print(f"Overall RMSE: {overall_rmse}")
        print(f"P90 of absolute errors: {p90_error}")
        
        # Create additional error statistics plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(errors, labels=[f'Time {i}' for i in range(len(errors))])
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Error Distribution by Time Step')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_distributions.png')
        plt.close()
        
        return overall_rmse, rmse_values, p90_error


# Example usage
def main():
    # Create model
    wfm = WindFarmModel(data_path='./DTU10MW.nc', model=2, downwind=True)
    
    # Run optimization
    best_params, best_rmse = wfm.optimize(init_points=50, n_iter=200)
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")
    
    # Create optimization animation
    animation_file = wfm.create_optimization_animation()
    print(f"Created animation: {animation_file}")
    
    # Create parameter comparison plot
    comparison_plot = wfm.create_parameter_comparison_plot()
    print(f"Created parameter comparison plot: {comparison_plot}")
    
    # Evaluate model performance
    overall_rmse, rmse_values, p90_error = wfm.evaluate_model_performance()
    print(f"Overall RMSE: {overall_rmse}")
    print(f"P90 of absolute errors: {p90_error}")


if __name__ == "__main__":
    main()
