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


class WindFarmModelOptimizer:
    """Main class for wind farm model optimization"""
    
    def __init__(self, data_path, model_type=2, downwind=True):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset
        model_type : int
            Model type (1 or 2)
        downwind : bool
            Whether to analyze downwind region
        """
        self.data_path = data_path
        self.MODEL = model_type
        self.DOWNWIND = downwind
        
        if self.MODEL not in {1, 2}:
            raise ValueError("Model type must be 1 or 2")
            
        # Load data and setup configurations
        self.load_data()
        self.setup_configurations()
        self.setup_optimization_bounds()
        
    def load_data(self):
        """Load and prepare the dataset"""
        # Load turbine data
        self.turbine = DTU10MW()
        self.D = self.turbine.diameter()
        
        # Load dataset
        self.dat = xr.load_dataset(self.data_path)
        self.dat = self.dat.assign_coords(x=self.dat.x * self.D, y=self.dat.y * self.D)
        
        # Setup region of interest based on downwind configuration
        if self.DOWNWIND:
            self.X_LB = 2
            self.X_UB = 10
        else:
            self.X_LB = -2
            self.X_UB = -1
            
        roi_x = slice(self.X_LB * self.D, self.X_UB * self.D)
        roi_y = slice(-2 * self.D, 2 * self.D)
        
        # Select flow region of interest
        self.flow_roi = self.dat.sel(x=roi_x, y=roi_y)
        self.target_x = self.flow_roi.x
        self.target_y = self.flow_roi.y
        
    def setup_configurations(self):
        """Setup wind and turbulence intensity configurations"""
        # Define wind speeds and turbulence intensities
        TIs = np.arange(0.05, 0.45, 0.05)
        WSs = np.arange(4, 11)
        
        # Create full arrays of wind speeds and TIs
        full_ti = [TIs for _ in range(WSs.size)]
        self.full_ti = np.array(full_ti).flatten()
        
        full_ws = [[WSs[ii]] * TIs.size for ii in range(WSs.size)]
        self.full_ws = np.array(full_ws).flatten()
        
        assert self.full_ws.size == self.full_ti.size, "Wind speed and TI arrays must have the same size"
        
        # Setup site
        self.site = Hornsrev1Site()
        
    def setup_optimization_bounds(self):
        """Setup parameter bounds and defaults for optimization"""
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
        else:  # Model 2
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
        Create a wind farm model with the given parameters
        
        Parameters:
        -----------
        params : dict
            Model parameters
            
        Returns:
        --------
        wfm : Wind Farm Model
            Configured wind farm model
        """
        # Initialize empty args dictionaries
        def_args = {}
        turb_args = {}
        blockage_args = {}
        
        # Configure models based on downwind setting and model type
        if self.DOWNWIND:
            if self.MODEL == 1:
                def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
                turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
                wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            else:
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
        else:
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
        
    def evaluate_rmse(self, **kwargs):
        """
        Evaluate RMSE for the given parameters
        
        Parameters:
        -----------
        **kwargs : dict
            Model parameters
            
        Returns:
        --------
        float
            Negative RMSE (for maximization in Bayesian Optimization)
        """
        # Create wind farm model with the parameters
        wfm = self.create_wind_farm_model(kwargs)
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.full_ws, 
            TI=self.full_ti, 
            wd=[270] * self.full_ti.size, 
            time=True
        )
        
        # Calculate flow map
        flow_map = None
        for tt in range(self.full_ws.size):
            fm = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y), time=[tt])['WS_eff']
            if flow_map is None:
                flow_map = fm
            else:
                flow_map = xr.concat([flow_map, fm], dim='time')
                
        # Calculate predictions and observed values
        obs_values = []
        for t in range(flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t, wt=0)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            )
            obs_values.append(observed_deficit.T)
            
        all_obs = xr.concat(obs_values, dim='time')
        
        # Calculate normalized deficit predictions
        pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        
        # Calculate RMSE
        rmse = float(np.sqrt(((all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
        
        # Return negative RMSE for maximization
        if np.isnan(rmse):
            return -0.5
        return -rmse
    
    def run_optimization(self, init_points=50, n_iter=200):
        """
        Run Bayesian optimization
        
        Parameters:
        -----------
        init_points : int
            Number of initial random points
        n_iter : int
            Number of iterations
            
        Returns:
        --------
        dict
            Best parameters
        float
            Best RMSE
        """
        # Create optimizer
        self.optimizer = BayesianOptimization(
            f=self.evaluate_rmse, 
            pbounds=self.pbounds, 
            random_state=1
        )
        
        # Probe default parameters
        self.optimizer.probe(params=self.defaults, lazy=True)
        
        # Run optimization
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Get best parameters and RMSE
        self.best_params = self.optimizer.max['params']
        self.best_rmse = -self.optimizer.max['target']
        
        return self.best_params, self.best_rmse
    
    def create_convergence_animation(self, output_filename):
        """
        Create animation of optimization convergence
        
        Parameters:
        -----------
        output_filename : str
            Output filename for the animation
        """
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
            
            # Bar plot for parameter comparison
            keys = list(best_so_far_params.keys())
            best_vals = []
            default_vals = []
            
            for key in keys:
                best_vals.append(best_so_far_params[key])
                default_vals.append(self.defaults[key])
                
            ax2.bar(keys, best_vals, label='Optimized')
            ax2.bar(
                keys, 
                default_vals, 
                edgecolor='black', 
                linewidth=2, 
                color='none', 
                capstyle='butt', 
                label='Default'
            )
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
        
        # Save as MP4
        writer = animation.FFMpegWriter(fps=15)
        ani.save(output_filename, writer=writer)
        plt.close('all')
        
    def evaluate_final_model(self, output_dir='figs'):
        """
        Evaluate the final model with optimized parameters
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output figures
            
        Returns:
        --------
        float
            Overall RMSE
        list
            RMSE values for each time step
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create wind farm model with optimized parameters
        wfm = self.create_wind_farm_model(self.best_params)
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.full_ws, 
            TI=self.full_ti, 
            wd=[270] * self.full_ti.size, 
            time=True
        )
        
        # Calculate flow map
        flow_map = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y))
        
        # Calculate RMSE for each time step
        rmse_values = []
        rmse_p90s = []
        errors = []
        
        for t in range(flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            ).isel(wt=0)
            
            # Calculate normalized deficit prediction
            pred = (this_pred_sim.WS - flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
            
            # Calculate difference and RMSE
            diff = observed_deficit.T - pred
            errors.append(diff)
            rmse = np.sqrt(np.mean(diff**2))
            rmse_values.append(rmse)
            
            # Calculate p90 of absolute errors
            abs_errors = np.abs(diff.values.flatten())
            p90 = np.percentile(abs_errors, 90)
            rmse_p90s.append(p90)
            
            # Create visualization
            fig, ax = plt.subplots(3, 1, figsize=(8, 18))
            
            co = ax[0].contourf(self.target_x, self.target_y, observed_deficit.T)
            cp = ax[1].contourf(self.target_x, self.target_y, pred)
            cd = ax[2].contourf(self.target_x, self.target_y, diff)
            
            for jj, c in enumerate([co, cp, cd]):
                fig.colorbar(c, ax=ax[jj])
                
            ax[0].set_ylabel('Observed')
            ax[1].set_ylabel('Prediction')
            ax[2].set_ylabel('Difference')
            
            # Add titles with statistics
            ax[0].set_title(f'Observed Deficit - Time step {t}')
            ax[1].set_title(f'Predicted Deficit - Time step {t}')
            ax[2].set_title(f'Difference - RMSE: {rmse:.4f}, P90: {p90:.4f}')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/flow_field_comparison_{t}.png')
            plt.close()
            
        # Calculate overall error statistics    
        overall_rmse = np.mean(rmse_values)
        overall_p90 = np.mean(rmse_p90s)
        
        # Create error distribution plot
        all_errors = np.concatenate([e.values.flatten() for e in errors])
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=50, alpha=0.7)
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f'Error Distribution - Overall RMSE: {overall_rmse:.4f}, P90: {overall_p90:.4f}')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/error_distribution.png')
        plt.close()
        
        # Create parameter comparison bar plot
        keys = self.best_params.keys()
        best_vals = []
        default_vals = []
        
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
        plt.title(f'Parameter Comparison - Optimal RMSE: {self.best_rmse:.4f}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/parameter_comparison.png')
        plt.close()
        
        # Create RMSE by time step plot
        plt.figure(figsize=(10, 6))
        plt.plot(rmse_values, 'o-', label='RMSE')
        plt.plot(rmse_p90s, 's-', label='P90 Error')
        plt.axhline(overall_rmse, color='red', linestyle='--', label=f'Mean RMSE: {overall_rmse:.4f}')
        plt.axhline(overall_p90, color='green', linestyle='--', label=f'Mean P90: {overall_p90:.4f}')
        plt.title('Error Metrics by Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Error Metric')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_by_timestep.png')
        plt.close()
        
        return overall_rmse, rmse_values, overall_p90, rmse_p90s


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = WindFarmModelOptimizer(
        data_path='./DTU10MW.nc',
        model_type=2,
        downwind=True
    )
    
    # Run optimization
    best_params, best_rmse = optimizer.run_optimization(init_points=50, n_iter=200)
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")
    
    # Create animation of optimization convergence
    optimizer.create_convergence_animation(
        f'optimization_animation_{optimizer.X_LB}_{optimizer.X_UB}.mp4'
    )
    
    # Evaluate final model
    overall_rmse, rmse_values, overall_p90, rmse_p90s = optimizer.evaluate_final_model()
    print(f"RMSE values per time step: {rmse_values}")
    print(f"Overall RMSE: {overall_rmse}")
    print(f"P90 values per time step: {rmse_p90s}")
    print(f"Overall P90: {overall_p90}")
