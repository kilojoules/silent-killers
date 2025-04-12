import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any

# PyWake imports
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.deficit_models.gaussian import TurboGaussianDeficit, BlondelSuperGaussianDeficit2020
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake import HorizontalGrid
from py_wake.deficit_models.utils import ct2a_mom1d

from bayes_opt import BayesianOptimization


class WakeModelConfig:
    """Configuration for wake models based on the chosen model type and flow direction."""
    
    def __init__(self, model_type: int = 2, downwind: bool = True):
        """
        Initialize wake model configuration.
        
        Args:
            model_type: 1 for BlondelSuperGaussianDeficit2020, 2 for TurboGaussianDeficit
            downwind: True for downwind analysis, False for upstream (blockage) analysis
        """
        if model_type not in {1, 2}:
            raise ValueError(f"Model type must be 1 or 2, got {model_type}")
        
        self.model_type = model_type
        self.downwind = downwind
        self.bounds, self.defaults = self._get_parameter_settings()
    
    def _get_parameter_settings(self) -> Tuple[Dict, Dict]:
        """Get parameter bounds and defaults based on configuration."""
        if self.model_type == 1:
            if self.downwind:
                # Blondel model for wake
                bounds = {
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
                defaults = {
                    'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
                    'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32
                }
            else:
                # Blondel model for blockage
                bounds = {
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
        else:  # model_type == 2
            # TurboGaussian model
            bounds = {
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
        
        return bounds, defaults
    
    def create_wind_farm_model(self, site, turbine, params: Dict) -> All2AllIterative:
        """
        Create wind farm model based on configuration and parameters.
        
        Args:
            site: Site object
            turbine: Turbine object
            params: Dictionary of parameters for the model
            
        Returns:
            Configured wind farm model
        """
        if self.downwind:
            if self.model_type == 1:
                # Blondel model for wake
                def_args = {k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']}
                turb_args = {'c': np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])}
                blockage_args = {}
                wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
            else:  # model_type == 2
                # TurboGaussian model
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
        else:  # upstream / blockage
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
        return All2AllIterative(
            site, 
            turbine,
            wake_deficitModel=wake_deficitModel,
            superpositionModel=LinearSum(), 
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(**turb_args),
            blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args)
        )


class WindFarmOptimizer:
    """Optimize wind farm wake model parameters using Bayesian optimization."""
    
    def __init__(self, config: WakeModelConfig, reference_data: xr.Dataset, turbine, site):
        """
        Initialize the optimizer.
        
        Args:
            config: Wake model configuration
            reference_data: Reference dataset with observed deficits
            turbine: Turbine object
            site: Site object
        """
        self.config = config
        self.reference_data = reference_data
        self.turbine = turbine
        self.site = site
        
        # Extract target coordinates
        D = turbine.diameter()
        if config.downwind:
            x_lb, x_ub = 2, 10
        else:
            x_lb, x_ub = -2, -1
            
        roi_x = slice(x_lb * D, x_ub * D)
        roi_y = slice(-2 * D, 2 * D)
        
        self.flow_roi = reference_data.sel(x=roi_x, y=roi_y)
        self.target_x = self.flow_roi.x
        self.target_y = self.flow_roi.y
        
        # Setup wind speeds and turbulence intensities
        self.TIs = np.arange(0.05, 0.45, 0.05)
        self.WSs = np.arange(4, 11)
        self.full_ti = np.array([[ti for ti in self.TIs] for _ in range(len(self.WSs))]).flatten()
        self.full_ws = np.array([[ws] * len(self.TIs) for ws in self.WSs]).flatten()
        
        # Initialize optimizer
        self.optimizer = BayesianOptimization(
            f=self._evaluate_rmse, 
            pbounds=config.bounds, 
            random_state=1
        )
        
        # Run reference simulation to get observed values
        sim_res = All2AllIterative(
            site, turbine,
            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
            superpositionModel=LinearSum(), 
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(),
            blockage_deficitModel=SelfSimilarityDeficit2020()
        )([0], [0], ws=self.full_ws, TI=self.full_ti, wd=[270] * len(self.full_ti), time=True)
        
        self.flow_map = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y))
        
        # Extract observed deficit values
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
    
    def _evaluate_rmse(self, **params) -> float:
        """
        Evaluate RMSE for given parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Negative RMSE (for maximization)
        """
        # Create wind farm model with given parameters
        wfm = self.config.create_wind_farm_model(self.site, self.turbine, params)
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.full_ws, 
            TI=self.full_ti, 
            wd=[270] * len(self.full_ti), 
            time=True
        )
        
        # Calculate flow map
        flow_map = None
        for tt in range(len(self.full_ws)):
            fm = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y), time=[tt])['WS_eff']
            if flow_map is None:
                flow_map = fm
            else:
                flow_map = xr.concat([flow_map, fm], dim='time')
        
        # Calculate deficit
        pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        
        # Calculate RMSE
        rmse = float(np.sqrt(((self.all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
        
        # Handle NaN values
        if np.isnan(rmse):
            return -0.5
        
        return -rmse  # Negative because we want to maximize
    
    def optimize(self, init_points: int = 50, n_iter: int = 200) -> Dict:
        """
        Run optimization.
        
        Args:
            init_points: Number of initial random points
            n_iter: Number of iterations
            
        Returns:
            Best parameters
        """
        # Probe with default parameters
        self.optimizer.probe(params=self.config.defaults, lazy=True)
        
        # Run optimization
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Return best parameters
        return self.optimizer.max['params']


class ResultsAnalyzer:
    """Analyze and visualize optimization results."""
    
    def __init__(self, optimizer: WindFarmOptimizer, best_params: Dict):
        """
        Initialize analyzer.
        
        Args:
            optimizer: WindFarmOptimizer object
            best_params: Dictionary of best parameters
        """
        self.optimizer = optimizer
        self.best_params = best_params
        self.best_rmse = -optimizer.optimizer.max['target']
        self.config = optimizer.config
        self.flow_roi = optimizer.flow_roi
        self.target_x = optimizer.target_x
        self.target_y = optimizer.target_y
        
        # Create output directory
        self.output_dir = Path('./figs')
        self.output_dir.mkdir(exist_ok=True)
    
    def create_optimization_animation(self, output_filename: str = None) -> None:
        """
        Create animation showing optimization progress.
        
        Args:
            output_filename: Name of output file
        """
        if output_filename is None:
            D = self.optimizer.turbine.diameter()
            if self.config.downwind:
                x_lb, x_ub = 2, 10
            else:
                x_lb, x_ub = -2, -1
            output_filename = f'optimization_animation_{x_lb}_{x_ub}.mp4'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def update_plot(frame):
            ax1.clear()
            ax2.clear()
            
            # Get the best parameters and corresponding RMSE up to the current frame
            best_so_far_params = {}
            best_so_far_rmse = float('inf')
            best_so_far_rmses = []
            
            for i in range(frame + 1):
                if -self.optimizer.optimizer.space.target[i] <= best_so_far_rmse:
                    best_so_far_rmse = -self.optimizer.optimizer.space.target[i]
                    best_so_far_params = self.optimizer.optimizer.res[i]['params']
                best_so_far_rmses.append(best_so_far_rmse)
            
            # Plot the entire history in gray
            ax1.plot(-np.array(self.optimizer.optimizer.space.target), color='gray', alpha=0.5)
            # Plot the best RMSE so far in black
            ax1.plot(np.array(best_so_far_rmses), color='black')
            ax1.set_title('Optimization Convergence')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('RMSE')
            ax1.grid(True)
            
            # Use the best parameters so far for the bar plot
            keys = list(best_so_far_params.keys())
            best_vals = [best_so_far_params[key] for key in keys]
            default_vals = [self.config.defaults[key] for key in keys]
            
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
            frames=len(self.optimizer.optimizer.space.target), 
            repeat=False
        )
        
        # Save as MP4
        writer = animation.FFMpegWriter(fps=15)
        ani.save(output_filename, writer=writer)
        plt.close(fig)
    
    def create_parameter_bar_chart(self, output_filename: str = None) -> None:
        """
        Create bar chart comparing default and optimized parameters.
        
        Args:
            output_filename: Name of output file
        """
        if output_filename is None:
            D = self.optimizer.turbine.diameter()
            if self.config.downwind:
                x_lb, x_ub = 2, 10
            else:
                x_lb, x_ub = -2, -1
            output_filename = f'bar_LB_{x_lb}_UP_{x_ub}.png'
        
        keys = list(self.best_params.keys())
        best_vals = [self.best_params[key] for key in keys]
        default_vals = [self.config.defaults[key] for key in keys]
        
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
        plt.savefig(output_filename)
        plt.close()
    
    def analyze_flow_fields(self) -> dict:
        """
        Analyze flow fields using optimized parameters.
        
        Returns:
            Dictionary with error statistics
        """
        # Create wind farm model with best parameters
        wfm = self.config.create_wind_farm_model(
            self.optimizer.site, 
            self.optimizer.turbine, 
            self.best_params
        )
        
        # Run simulation
        sim_res = wfm(
            [0], [0], 
            ws=self.optimizer.full_ws, 
            TI=self.optimizer.full_ti, 
            wd=[270] * len(self.optimizer.full_ti), 
            time=True
        )
        
        # Calculate flow map
        flow_map = sim_res.flow_map(HorizontalGrid(x=self.target_x, y=self.target_y))
        
        # Analyze errors
        rmse_values = []
        abs_errors = []
        p90_errors = []
        
        for t in range(flow_map.time.size):
            this_pred_sim = sim_res.isel(time=t)
            observed_deficit = self.flow_roi.deficits.interp(
                ct=this_pred_sim.CT, 
                ti=this_pred_sim.TI, 
                z=0
            ).isel(wt=0)
            
            # Calculate predicted deficit
            pred = (this_pred_sim.WS - flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
            
            # Calculate difference
            diff = observed_deficit.T - pred
            abs_diff = np.abs(diff)
            
            # Calculate error metrics
            rmse = np.sqrt(np.mean(diff**2))
            mean_abs_err = np.mean(abs_diff)
            p90_err = np.percentile(abs_diff, 90)
            
            rmse_values.append(float(rmse))
            abs_errors.append(float(mean_abs_err))
            p90_errors.append(float(p90_err))
            
            # Create plots
            fig, ax = plt.subplots(3, 1, figsize=(5, 15))
            
            co = ax[0].contourf(self.target_x, self.target_y, observed_deficit.T)
            cp = ax[1].contourf(self.target_x, self.target_y, pred)
            cd = ax[2].contourf(self.target_x, self.target_y, diff)
            
            for jj, c in enumerate([co, cp, cd]):
                fig.colorbar(c, ax=ax[jj])
            
            ax[0].set_ylabel('Observed')
            ax[1].set_ylabel('Prediction')
            ax[2].set_ylabel('Diff')
            
            # Add error metrics to the plot
            ax[2].set_title(f'RMSE: {rmse:.4f}, MAE: {mean_abs_err:.4f}, P90: {p90_err:.4f}')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'flow_field_case_{t}.png')
            plt.close(fig)
        
        # Calculate overall metrics
        overall_metrics = {
            'rmse_values': rmse_values,
            'overall_rmse': np.mean(rmse_values),
            'abs_errors': abs_errors,
            'overall_mae': np.mean(abs_errors),
            'p90_errors': p90_errors,
            'overall_p90': np.mean(p90_errors)
        }
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        case_indices = range(len(rmse_values))
        
        plt.plot(case_indices, rmse_values, 'o-', label='RMSE')
        plt.plot(case_indices, abs_errors, 's-', label='MAE')
        plt.plot(case_indices, p90_errors, '^-', label='P90')
        
        plt.axhline(overall_metrics['overall_rmse'], linestyle='--', color='C0', alpha=0.7, 
                    label=f'Avg RMSE: {overall_metrics["overall_rmse"]:.4f}')
        plt.axhline(overall_metrics['overall_mae'], linestyle='--', color='C1', alpha=0.7,
                    label=f'Avg MAE: {overall_metrics["overall_mae"]:.4f}')
        plt.axhline(overall_metrics['overall_p90'], linestyle='--', color='C2', alpha=0.7,
                    label=f'Avg P90: {overall_metrics["overall_p90"]:.4f}')
        
        plt.title('Error Metrics Across Test Cases')
        plt.xlabel('Case Index')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_metrics_summary.png')
        plt.close()
        
        return overall_metrics


def main():
    """Main function to run the optimization and analysis."""
    # Load reference data
    dat = xr.load_dataset('./DTU10MW.nc')
    turbine = DTU10MW()
    D = turbine.diameter()
    dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)
    
    # Define model configuration
    MODEL = 2  # 1 for BlondelSuperGaussianDeficit2020, 2 for TurboGaussianDeficit
    DOWNWIND = True  # True for downwind analysis, False for upstream (blockage) analysis
    
    config = WakeModelConfig(model_type=MODEL, downwind=DOWNWIND)
    site = Hornsrev1Site()
    
    # Initialize optimizer
    optimizer = WindFarmOptimizer(config, dat, turbine, site)
    
    # Run optimization
    best_params = optimizer.optimize(init_points=50, n_iter=200)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(optimizer, best_params)
    
    # Create outputs
    analyzer.create_optimization_animation()
    analyzer.create_parameter_bar_chart()
    metrics = analyzer.analyze_flow_fields()
    
    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
    print(f"Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"Overall P90: {metrics['overall_p90']:.4f}")


if __name__ == "__main__":
    main()
