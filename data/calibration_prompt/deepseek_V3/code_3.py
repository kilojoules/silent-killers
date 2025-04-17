import numpy as np
import xarray as xr
from py_wake import ... # all your imports

class WakeModelOptimizer:
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.load_data()
        self.setup_defaults()
        
    def load_data(self):
        """Load and prepare reference data"""
        self.dat = xr.load_dataset('./DTU10MW.nc')
        self.turbine = DTU10MW()
        D = self.turbine.diameter()
        self.dat = self.dat.assign_coords(x=self.dat.x * D, y=self.dat.y * D)
        
        # Set region of interest based on config
        roi_x = slice(self.config['X_LB'] * D, self.config['X_UB'] * D)
        roi_y = slice(-2 * D, 2 * D)
        self.flow_roi = self.dat.sel(x=roi_x, y=roi_y)
        
    def setup_defaults(self):
        """Set default parameters and bounds"""
        self.defaults = {...}  # Your existing defaults
        self.pbounds = {...}   # Your existing pbounds
        
    def create_model(self, params):
        """Instantiate the wake model with given parameters"""
        if self.config['DOWNWIND']:
            if self.config['MODEL'] == 1:
                # Model 1 downwind configuration
                wake_model = BlondelSuperGaussianDeficit2020(
                    a_s=params['a_s'],
                    b_s=params['b_s'],
                    c_s=params['c_s'],
                    b_f=params['b_f'],
                    c_f=params['c_f']
                )
            else:
                # Model 2 downwind configuration
                wake_model = TurboGaussianDeficit(
                    A=params['A'],
                    cTI=[params['cti1'], params['cti2']],
                    ctlim=params['ctlim'],
                    ceps=params['ceps'],
                    ct2a=ct2a_mom1d,
                    groundModel=Mirror(),
                    rotorAvgModel=GaussianOverlapAvgModel()
                )
                wake_model.WS_key = 'WS_jlk'
        else:
            # Upwind configurations
            wake_model = BlondelSuperGaussianDeficit2020()
            
        return All2AllIterative(
            site=Hornsrev1Site(),
            windTurbines=self.turbine,
            wake_deficitModel=wake_model,
            superpositionModel=LinearSum(),
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(
                c=np.array([params['ch1'], params['ch2'], 
                          params['ch3'], params['ch4']])
            ),
            blockage_deficitModel=SelfSimilarityDeficit2020(
                **self.get_blockage_args(params)
            )
        )
    
    def get_blockage_args(self, params):
        """Get blockage model arguments based on configuration"""
        if not self.config['DOWNWIND']:
            return {
                'ss_alpha': params['ss_alpha'],
                'ss_beta': params['ss_beta'],
                'r12p': np.array([params['rp1'], params['rp2']]),
                'ngp': np.array([params['ng1'], params['ng2'], 
                               params['ng3'], params['ng4']]),
                'groundModel': Mirror() if self.config['MODEL'] == 2 else None
            }
        return {}
    
    def evaluate_model(self, params):
        """Evaluate model performance with given parameters"""
        wfm = self.create_model(params)
        
        # Setup simulation conditions
        TIs = np.arange(0.05, 0.45, 0.05)
        WSs = np.arange(4, 11)
        full_ti = np.tile(TIs, len(WSs))
        full_ws = np.repeat(WSs, len(TIs))
        
        # Run simulation
        sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, 
                     wd=[270]*len(full_ws), time=True)
        
        # Calculate velocity deficit
        flow_map = sim_res.flow_map(HorizontalGrid(
            x=self.flow_roi.x, 
            y=self.flow_roi.y
        ))
        pred_deficit = (sim_res.WS - flow_map.WS_eff.isel(h=0)) / sim_res.WS
        
        # Compare with reference data
        errors = []
        for t in range(flow_map.time.size):
            obs_deficit = self.flow_roi.deficits.interp(
                ct=sim_res.isel(time=t).CT,
                ti=sim_res.isel(time=t).TI,
                z=0
            ).T
            errors.append(obs_deficit - pred_deficit.isel(time=t))
        
        # Calculate metrics
        error_da = xr.concat(errors, dim='time')
        rmse = float(np.sqrt((error_da**2).mean(['x', 'y'])).mean('time'))
        avg_error = float(error_da.mean())
        p90_error = float(error_da.quantile(0.9))
        
        return {
            'rmse': rmse if not np.isnan(rmse) else 1e6,
            'avg_error': avg_error,
            'p90_error': p90_error,
            'pred_deficit': pred_deficit,
            'errors': error_da
        }
    
    def optimize(self, n_iter=200, init_points=50):
        """Run Bayesian optimization"""
        optimizer = BayesianOptimization(
            f=lambda **params: -self.evaluate_model(params)['rmse'],
            pbounds=self.pbounds,
            random_state=1
        )
        optimizer.probe(params=self.defaults, lazy=True)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        best_params = optimizer.max['params']
        best_rmse = -optimizer.max['target']
        
        return best_params, best_rmse, optimizer
    
    def visualize_results(self, params, output_path='results'):
        """Generate comprehensive visualizations"""
        results = self.evaluate_model(params)
        
        # 1. Optimization convergence plot
        self.plot_convergence(optimizer, output_path)
        
        # 2. Parameter comparison plot
        self.plot_parameter_comparison(params, output_path)
        
        # 3. Flow field comparisons
        self.plot_flow_fields(results, output_path)
        
        # 4. Error distribution plots
        self.plot_error_distribution(results['errors'], output_path)
    
    def plot_flow_fields(self, results, output_path):
        """Plot observed vs predicted flow fields with error metrics"""
        pred_deficit = results['pred_deficit']
        
        for t in range(pred_deficit.time.size):
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # Observed
            obs = self.flow_roi.deficits.interp(
                ct=sim_res.isel(time=t).CT,
                ti=sim_res.isel(time=t).TI,
                z=0
            ).T
            c0 = axes[0].contourf(self.flow_roi.x, self.flow_roi.y, obs)
            axes[0].set_title('Observed Deficit')
            fig.colorbar(c0, ax=axes[0])
            
            # Predicted
            c1 = axes[1].contourf(self.flow_roi.x, self.flow_roi.y, 
                                 pred_deficit.isel(time=t))
            axes[1].set_title('Predicted Deficit')
            fig.colorbar(c1, ax=axes[1])
            
            # Error
            error = obs - pred_deficit.isel(time=t)
            c2 = axes[2].contourf(self.flow_roi.x, self.flow_roi.y, error)
            axes[2].set_title(f'Error (RMSE: {np.sqrt(np.mean(error**2)):.4f})')
            fig.colorbar(c2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(f'{output_path}/flow_comparison_t{t}.png')
            plt.close()
            
    # Additional visualization methods would go here...

# Example usage
if __name__ == "__main__":
    config = {
        'DOWNWIND': True,
        'MODEL': 2,
        'X_LB': 2,
        'X_UB': 10
    }
    
    optimizer = WakeModelOptimizer(config)
    best_params, best_rmse, opt = optimizer.optimize()
    optimizer.visualize_results(best_params)
