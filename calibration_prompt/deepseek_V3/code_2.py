import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from py_wake import ...  # Keep all your existing imports

class WakeModelFactory:
    def __init__(self, model_type, downstream=True):
        self.model_type = model_type
        self.downstream = downstream
        self.default_params = self._get_default_params()
        
    def _get_default_params(self):
        # Returns default parameters based on model_type and downstream
        if self.model_type == 1:
            if self.downstream:
                return {...}  # Your MODEL 1 downstream defaults
            else:
                return {...}  # Your MODEL 1 upstream defaults
        else:
            return {...}  # Your MODEL 2 defaults
    
    def create_model(self, params=None):
        """Create WFM instance with given parameters"""
        params = params or self.default_params
        
        if self.model_type == 1:
            if self.downstream:
                wake_deficit = BlondelSuperGaussianDeficit2020(
                    a_s=params['a_s'],
                    b_s=params['b_s'],
                    c_s=params['c_s'],
                    b_f=params['b_f'],
                    c_f=params['c_f']
                )
            else:
                wake_deficit = BlondelSuperGaussianDeficit2020()
                # Upstream parameters
        else:
            wake_deficit = TurboGaussianDeficit(
                A=params['A'],
                cTI=[params['cti1'], params['cti2']],
                ctlim=params['ctlim'],
                ceps=params['ceps'],
                ct2a=ct2a_mom1d,
                groundModel=Mirror(),
                rotorAvgModel=GaussianOverlapAvgModel()
            )
            wake_deficit.WS_key = 'WS_jlk'
        
        # Common model components
        turbulence_model = CrespoHernandez(
            c=np.array([params['ch1'], params['ch2'], params['ch3'], params['ch4']])
        )
        
        blockage_model = None
        if not self.downstream:
            blockage_model = SelfSimilarityDeficit2020(
                ss_alpha=params['ss_alpha'],
                ss_beta=params['ss_beta'],
                r12p=np.array([params['rp1'], params['rp2']]),
                ngp=np.array([params['ng1'], params['ng2'], params['ng3'], params['ng4']]),
                groundModel=Mirror() if self.model_type == 2 else None
            )
        
        return All2AllIterative(
            site=Hornsrev1Site(),
            windTurbines=DTU10MW(),
            wake_deficitModel=wake_deficit,
            superpositionModel=LinearSum(),
            deflectionModel=None,
            turbulenceModel=turbulence_model,
            blockage_deficitModel=blockage_model
        )

class FlowAnalyzer:
    def __init__(self, flow_map, reference_data):
        self.flow_map = flow_map
        self.reference = reference_data
        
    def calculate_metrics(self):
        """Calculate various error metrics"""
        # Calculate velocity deficit from flow_map
        pred_deficit = (self.flow_map.WS_eff - self.flow_map.WS) / self.flow_map.WS
        
        # Compare with reference data
        errors = self.reference - pred_deficit
        
        metrics = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'mae': np.mean(np.abs(errors)),
            'p90': np.percentile(np.abs(errors), 90),
            'avg_error': np.mean(errors),
            'spatial_rmse': np.sqrt(np.mean(errors**2, axis=(1, 2))  # Per timestep
        }
        return metrics
    
    def plot_comparison(self, time_idx=0):
        """Generate comparison plots for a specific timestep"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Observed
        self.reference.isel(time=time_idx).plot(ax=axes[0])
        axes[0].set_title('Observed Velocity Deficit')
        
        # Predicted
        pred_deficit = (self.flow_map.isel(time=time_idx).WS_eff - 
                       self.flow_map.isel(time=time_idx).WS) / self.flow_map.isel(time=time_idx).WS
        pred_deficit.plot(ax=axes[1])
        axes[1].set_title('Predicted Velocity Deficit')
        
        # Difference
        diff = self.reference.isel(time=time_idx) - pred_deficit
        diff.plot(ax=axes[2])
        axes[2].set_title('Difference')
        
        plt.tight_layout()
        return fig

def main():
    # Load data
    dat = xr.load_dataset('./DTU10MW.nc')
    turbine = DTU10MW()
    D = turbine.diameter()
    dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)
    
    # Configuration
    DOWNSTREAM = True
    MODEL_TYPE = 2
    
    # Region of interest
    roi_x = slice(2 * D if DOWNSTREAM else -2 * D, 
                 10 * D if DOWNSTREAM else -1 * D)
    roi_y = slice(-2 * D, 2 * D)
    flow_roi = dat.sel(x=roi_x, y=roi_y)
    
    # Create model factory
    model_factory = WakeModelFactory(model_type=MODEL_TYPE, downstream=DOWNSTREAM)
    
    # Optimization setup
    def evaluate_rmse(**kwargs):
        wfm = model_factory.create_model(kwargs)
        sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True)
        flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))
        
        # Calculate metrics
        analyzer = FlowAnalyzer(flow_map, flow_roi.deficits)
        metrics = analyzer.calculate_metrics()
        return -metrics['rmse']  # Negative for maximization
    
    # Run optimization
    optimizer = BayesianOptimization(
        f=evaluate_rmse,
        pbounds=model_factory.get_parameter_bounds(),
        random_state=1
    )
    optimizer.probe(params=model_factory.default_params, lazy=True)
    optimizer.maximize(init_points=50, n_iter=200)
    
    # Analysis of best model
    best_params = optimizer.max['params']
    best_model = model_factory.create_model(best_params)
    sim_res = best_model([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True)
    flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))
    
    analyzer = FlowAnalyzer(flow_map, flow_roi.deficits)
    metrics = analyzer.calculate_metrics()
    
    print(f"Overall RMSE: {metrics['rmse']}")
    print(f"P90 Error: {metrics['p90']}")
    print(f"Average Error: {metrics['avg_error']}")
    
    # Generate and save plots
    for t in range(flow_map.time.size):
        fig = analyzer.plot_comparison(time_idx=t)
        fig.savefig(f'figs/downstream_err_{t}.png')
        plt.close(fig)

if __name__ == "__main__":
    main()
