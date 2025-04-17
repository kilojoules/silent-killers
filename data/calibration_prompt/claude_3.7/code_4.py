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


class WindFarmOptimizer:
    """Class to handle wind farm wake model optimization"""
    
    def __init__(self, data_path='./DTU10MW.nc', model_type=2, is_downwind=True):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        data_path : str
            Path to the reference data
        model_type : int
            Model type (1 or 2)
        is_downwind : bool
            Whether to analyze downwind (True) or upstream (False) effects
        """
        # Validate model type
        if model_type not in {1, 2}:
            raise ValueError(f"Model type must be 1 or 2, got {model_type}")
        
        self.model_type = model_type
        self.is_downwind = is_downwind
        
        # Load turbine and data
        self.turbine = DTU10MW()
        self.D = self.turbine.diameter()
        self.load_data(data_path)
        
        # Set up the site
        self.site = Hornsrev1Site()
        
        # Define wind speed and TI settings
        self.setup_wind_conditions()
        
        # Define parameter bounds and defaults
        self.setup_parameter_space()
    
    def load_data(self, data_path):
        """Load and prepare reference data"""
        dat = xr.load_dataset(data_path)
        dat = dat.assign_coords(x=dat.x * self.D, y=dat.y * self.D)
        
        # Define region of interest based on downwind/upwind
        if self.is_downwind:
            X_LB, X_UB = 2, 10
        else:
            X_LB, X_UB = -2, -1
        
        self.X_LB, self.X_UB = X_LB, X_UB
        roi_x = slice(X_LB * self.D, X_UB * self.D)
        roi_y = slice(-2 * self.D, 2 * self.D)
        
        # Extract region of interest
        self.flow_roi = dat.sel(x=roi_x, y=roi_y)
        self.target_x = self.flow_roi.x
        self.target_y = self.flow_roi.y
    
    def setup_wind_conditions(self):
        """Set up the wind speeds and turbulence intensities to test"""
        TIs = np.arange(0.05, 0.45, 0.05)
        WSs = np.arange(4, 11)
        
        full_ti = [TIs for _ in range(WSs.size)]
        self.full_ti = np.array(full_ti).flatten()
        
        full_ws = [[WSs[ii]] * TIs.size for ii in range(WSs.size)]
        self.full_ws = np.array(full_ws).flatten()
        
        assert (self.full_ws.size == self.full_ti.size)
    
    def setup_parameter_space(self):
        """Define parameter bounds and defaults based on model type"""
        if self.model_type == 1:
            if self.is_downwind:
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
                    'a_s': 0.17, 
                    'b_s': 0.005, 
                    'c_s': 0.2, 
                    'b_f': -0.68, 
                    'c_f': 2.41,
                    'ch1': 0.73, 
                    'ch2': 0.8325, 
                    'ch3': -0.0325, 
                    'ch4': -0.32
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
        else:  # model_type == 2
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
        Create wind farm model based on model type and parameters
        
        Parameters:
        -----------
        params : dict
            Model parameters
            
        Returns:
        --------
        wfm : PyWake wind farm model
        """
        # Initialize empty args dictionaries
        def_args = {}
        turb_args = {}
        blockage_args = {}
        
        # Configure wake deficit model based on model type and downwind/upwind
        if self.is_downwind:
            if self.model_type == 1:
                def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
                turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
                wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            else:  # model_type == 2
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
        else:  # upwind (blockage)
            def_args = {}
            turb_args = {}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            blockage_args = {
                'ss_alpha': params['ss_alpha'], 
                'ss_beta': params['ss_beta'], 
                'r12p': np.array([params['rp1'], params['rp2']]), 
                'ngp': np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']])
            }
            if self.model_type == 2:
                blockage_args['groundModel'] = Mirror()
        
        # Create and return the wind farm model
        wfm = All2AllIterative(
            self.site, 
            self.turbine,
            wake_deficitModel=wake_deficitModel,
            superpositionModel=LinearSum(), 
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(**turb_args),
            blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args)
        )
        
        return wfm
    
    def compute_reference_deficits(self, sim_res):
        """
        Compute reference observed deficits for each time step
        
        Parameters:
        -----------
        sim_res : PyWake simulation result
            Initial simulation result to get CT and TI values
            
        Returns:
        --------
        all_obs : xarray DataArray
            Observed deficits for all time steps
        """
        obs_values = []
        
        for t in range(sim_res.time.size):
            this_pred_sim = sim_res.isel(time=t, wt=0)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            )
            obs_values.append(observed_deficit.T)
        
        return xr.concat(obs_values, dim='time')
    
    def evaluate_rmse(self, **kwargs):
        """
        Evaluate RMSE for given parameters
        
        Parameters:
        -----------
        **kwargs : dict
            Model parameters
            
        Returns:
        --------
        neg_rmse : float
            Negative RMSE (for maximization in BayesOpt)
        """
        # Create wind farm model with given parameters
        wfm = self.create_wind_farm_model(kwargs)
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.full_ws, 
            TI=self.full_ti, 
            wd=[270] * self.full_ti.size, 
            time=True
        )
        
        # Get flow map for target grid
        flow_map = None
        for tt in range(self.full_ws.size):
            fm = sim_res.flow_map(
                HorizontalGrid(x=self.target_x, y=self.target_y), 
                time=[tt]
            )['WS_eff']
            
            if flow_map is None:
                flow_map = fm
            else:
                flow_map = xr.concat([flow_map, fm], dim='time')
        
        # Get reference deficits if not already computed
        if not hasattr(self, 'all_obs'):
            self.all_obs = self.compute_reference_deficits(sim_res)
        
        # Calculate velocity deficits from model predictions
        pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        
        # Calculate RMSE
        rmse = float(np.sqrt(((self.all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
        
        # Return negative RMSE for maximization
        if np.isnan(rmse):
            return -0.5
        return -rmse
    
    def optimize(self, init_points=50, n_iter=200, random_state=1):
        """
        Perform Bayesian optimization
        
        Parameters:
        -----------
        init_points : int
            Number of initial random points
        n_iter : int
            Number of optimization iterations
        random_state : int
            Random seed
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        best_rmse : float
            Best RMSE value
        optimizer : BayesianOptimization
            The optimizer object
        """
        # Initialize optimizer
        self.optimizer = BayesianOptimization(
            f=self.evaluate_rmse, 
            pbounds=self.pbounds, 
            random_state=random_state
        )
        
        # Probe with default parameters
        self.optimizer.probe(params=self.defaults, lazy=True)
        
        # Run optimization
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Extract best results
        best_params = self.optimizer.max['params']
        best_rmse = -self.optimizer.max['target']
        
        return best_params, best_rmse, self.optimizer
    
    def create_animation(self, output_filename=None):
        """
        Create animation of optimization progress
        
        Parameters:
        -----------
        output_filename : str, optional
            Output filename for the animation
            
        Returns:
        --------
        ani : matplotlib.animation.FuncAnimation
            Animation object
        """
        if not hasattr(self, 'optimizer'):
            raise ValueError("Optimization not performed yet. Call optimize() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def update_plot(frame):
            ax1.clear()
            ax2.clear()
            
            # Get the best parameters and corresponding RMSE up to the current frame
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
            ax2.bar(keys, default_vals, edgecolor='black', linewidth=2, 
                   color='none', capstyle='butt', label='Default')
            ax2.set_title(f'Best RMSE: {best_so_far_rmse:.4f}')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            plt.tight_layout()
            return ax1, ax2
        
        ani = animation.FuncAnimation(
            fig, update_plot, 
            frames=len(self.optimizer.space.target), 
            repeat=False
        )
        
        if output_filename is not None:
            if output_filename == 'auto':
                output_filename = f'optimization_animation_{self.X_LB}_{self.X_UB}.mp4'
            
            # Save as MP4
            writer = animation.FFMpegWriter(fps=15)
            ani.save(output_filename, writer=writer)
            plt.close('all')
        
        return ani
    
    def evaluate_best_model(self, best_params=None, visualize=True, output_dir='figs'):
        """
        Evaluate the best model and plot results
        
        Parameters:
        -----------
        best_params : dict, optional
            Best parameters (if None, use optimizer.max)
        visualize : bool
            Whether to create visualizations
        output_dir : str
            Directory for output figures
            
        Returns:
        --------
        overall_rmse : float
            Overall RMSE
        rmse_values : list
            RMSE values for each time step
        """
        import os
        if visualize and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if best_params is None:
            if not hasattr(self, 'optimizer'):
                raise ValueError("Optimization not performed yet. Call optimize() first.")
            best_params = self.optimizer.max['params']
        
        # Create model with best parameters
        wfm = self.create_wind_farm_model(best_params)
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.full_ws, 
            TI=self.full_ti, 
            wd=[270] * self.full_ti.size, 
            time=True
        )
        
        # Get flow map
        flow_map = sim_res.flow_map(
            HorizontalGrid(x=self.target_x, y=self.target_y)
        )
        
        # Calculate errors for each time step
        rmse_values = []
        error_data = []
        
        for t in range(flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            ).isel(wt=0)
            
            # Calculate prediction deficit
            pred = (this_pred_sim.WS - flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
            
            # Calculate difference and RMSE
            diff = observed_deficit.T - pred
            rmse = np.sqrt(np.mean(diff**2))
            rmse_values.append(rmse)
            
            # Store error data for statistical analysis
            error_data.append(diff.values.flatten())
            
            # Create visualizations
            if visualize:
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
                direction = "downstream" if self.is_downwind else "upstream"
                plt.savefig(f'{output_dir}/{direction}_err_{t}.png')
                plt.close()
        
        # Calculate overall RMSE
        overall_rmse = np.mean(rmse_values)
        
        # Calculate error statistics
        all_errors = np.concatenate(error_data)
        error_mean = np.mean(all_errors)
        error_std = np.std(all_errors)
        error_p90 = np.percentile(all_errors, 90)
        
        print(f"RMSE values per time step: {rmse_values}")
        print(f"Overall RMSE: {overall_rmse}")
        print(f"Error statistics - Mean: {error_mean}, Std: {error_std}, P90: {error_p90}")
        
        # Plot error histogram
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.hist(all_errors, bins=50, alpha=0.75)
            plt.axvline(error_mean, color='r', linestyle='--', label=f'Mean: {error_mean:.4f}')
            plt.axvline(error_p90, color='g', linestyle='--', label=f'P90: {error_p90:.4f}')
            plt.xlabel('Error (Observed - Predicted)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            direction = "downstream" if self.is_downwind else "upstream"
            plt.savefig(f'{output_dir}/{direction}_error_histogram.png')
            plt.close()
        
        # Plot parameter comparison
        if visualize:
            keys = best_params.keys()
            best_vals = []
            default_vals = []
            
            for key in keys:
                best_vals.append(best_params[key])
                default_vals.append(self.defaults[key])
            
            plt.figure(figsize=(12, 6))
            plt.bar(keys, best_vals)
            plt.bar(keys, default_vals,
                  edgecolor='black',
                  linewidth=2,
                  color='none',
                  capstyle='butt')
            plt.title(f'Optimal RMSE: {overall_rmse:.4f}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'bar_LB_{self.X_LB}_UP_{self.X_UB}.png')
            plt.close()
        
        return overall_rmse, rmse_values


# Example usage
if __name__ == "__main__":
    # Create optimizer with selected model and direction
    optimizer = WindFarmOptimizer(
        data_path='./DTU10MW.nc',
        model_type=2,  # Use model type 2
        is_downwind=True  # Analyze downwind effects
    )
    
    # Run optimization
    best_params, best_rmse, _ = optimizer.optimize(
        init_points=50, 
        n_iter=200, 
        random_state=1
    )
    
    # Create animation of optimization progress
    optimizer.create_animation(output_filename='auto')
    
    # Evaluate best model and generate visualizations
    optimizer.evaluate_best_model(visualize=True)
    
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")
