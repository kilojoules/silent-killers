import numpy as np
import xarray as xr
from py_wake import HorizontalGrid, All2AllIterative
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020, TurboGaussianDeficit
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.superposition_models import LinearSum
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------
# Configuration and Constants
# --------------------------
D = 178.3  # DTU 10MW diameter
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)
MODEL = 2
DOWNWIND = True
X_LB = 2 if DOWNWIND else -2
X_UB = 10 if DOWNWIND else -1

# --------------------------
# Data Loading and Preparation
# --------------------------
def load_reference_data():
    dat = xr.load_dataset('./DTU10MW.nc').assign_coords(x=lambda ds: ds.x * D, 
                                                      y=lambda ds: ds.y * D)
    roi_x = slice(X_LB * D, X_UB * D)
    roi_y = slice(-2 * D, 2 * D)
    return dat.sel(x=roi_x, y=roi_y)

# --------------------------
# Model Configuration
# --------------------------
class ModelConfigurator:
    def __init__(self, model_type, downwind):
        self.model_type = model_type
        self.downwind = downwind
        
    def get_parameter_bounds(self):
        if self.model_type == 1:
            return {
                'a_s': (0.001, 0.5), 'b_s': (0.001, 0.01), 'c_s': (0.001, 0.5),
                'b_f': (-2, 1), 'c_f': (0.1, 5), 'ch1': (-1, 2), 'ch2': (-1, 2),
                'ch3': (-1, 2), 'ch4': (-1, 2)
            } if self.downwind else {
                'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
                'rp1': (-2, 2), 'rp2': (-2, 2), 'ng1': (-3, 3), 'ng2': (-3, 3),
                'ng3': (-3, 3), 'ng4': (-3, 3)
            }
        return {
            'A': (0.001, 0.5), 'cti1': (0.01, 5), 'cti2': (0.01, 5),
            'ceps': (0.01, 3), 'ctlim': (0.01, 1), 'ch1': (-1, 2),
            'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2)
        }

    def create_wake_model(self, params):
        if self.model_type == 1:
            return BlondelSuperGaussianDeficit2020(**{
                k: params[k] for k in ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']
            }) if self.downwind else SelfSimilarityDeficit2020(**self._blockage_params(params))
        return TurboGaussianDeficit(
            A=params['A'], cTI=[params['cti1'], params['cti2']],
            ctlim=params['ctlim'], ceps=params['ceps'],
            ct2a=ct2a_mom1d, groundModel=Mirror(),
            rotorAvgModel=GaussianOverlapAvgModel()
        )

    def _blockage_params(self, params):
        return {
            'ss_alpha': params.get('ss_alpha', 0.89),
            'ss_beta': params.get('ss_beta', 1.41),
            'r12p': np.array([params.get('rp1', -0.67), params.get('rp2', 0.49)]),
            'ngp': np.array([params.get('ng1', -1.38), params.get('ng2', 2.63),
                           params.get('ng3', -1.52), params.get('ng4', 1.34)])
        }

# --------------------------
# Simulation Setup
# --------------------------
def create_wind_farm_model(wake_model, params):
    return All2AllIterative(
        site=Hornsrev1Site(),
        windTurbines=DTU10MW(),
        wake_deficitModel=wake_model,
        superpositionModel=LinearSum(),
        turbulenceModel=CrespoHernandez(
            c=np.array([params[f'ch{i}'] for i in range(1,5)])
        ),
        deflectionModel=None,
        blockage_deficitModel=SelfSimilarityDeficit2020(groundModel=Mirror())
    )

# --------------------------
# Error Calculation
# --------------------------
def calculate_error_metrics(pred, obs):
    errors = obs - pred
    return {
        'rmse': np.sqrt((errors**2).mean()),
        'mae': np.abs(errors).mean(),
        'p90': np.percentile(np.abs(errors), 90)
    }

# --------------------------
# Optimization Core
# --------------------------
class WakeOptimizer:
    def __init__(self, configurator):
        self.configurator = configurator
        self.flow_roi = load_reference_data()
        self.target_grid = HorizontalGrid(
            x=self.flow_roi.x, y=self.flow_roi.y)
        
        # Create full parameter grid
        self.full_ws, self.full_ti = np.meshgrid(WSs, TIs, indexing='ij')
        self.full_ws = self.full_ws.flatten()
        self.full_ti = self.full_ti.flatten()

    def evaluate_model(self, **params):
        try:
            wake_model = self.configurator.create_wake_model(params)
            wfm = create_wind_farm_model(wake_model, params)
            
            sim_res = wfm([0], [0], ws=self.full_ws, TI=self.full_ti, 
                        wd=[270]*len(self.full_ws))
            
            flow_map = sim_res.flow_map(self.target_grid)
            pred = (sim_res.WS - flow_map.WS_eff.isel(h=0)) / sim_res.WS
            
            errors = []
            for t in range(flow_map.time.size):
                obs = self.flow_roi.deficits.interp(
                    ct=sim_res.isel(time=t).CT,
                    ti=sim_res.isel(time=t).TI
                ).T
                errors.append(calculate_error_metrics(pred.isel(time=t), obs))
            
            return -pd.DataFrame(errors).mean().rmse
        except:
            return -0.5  # Handle invalid parameters

# --------------------------
# Visualization
# --------------------------
class ResultVisualizer:
    @staticmethod
    def create_animation(optimizer, file_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def update(frame):
            # ... (similar animation logic as before)
            return ax1, ax2
        
        ani = animation.FuncAnimation(fig, update, frames=len(optimizer.res))
        ani.save(file_name, writer=animation.FFMpegWriter(fps=15))
        
    @staticmethod
    def plot_error_distribution(errors):
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(np.mean(errors), color='k', linestyle='dashed', linewidth=1)
        plt.title(f"Error Distribution (Mean: {np.mean(errors):.3f}, P90: {np.percentile(errors, 90):.3f})")
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    configurator = ModelConfigurator(MODEL, DOWNWIND)
    optimizer = BayesianOptimization(
        f=WakeOptimizer(configurator).evaluate_model,
        pbounds=configurator.get_parameter_bounds(),
        random_state=1
    )
    
    # Run optimization
    optimizer.maximize(init_points=50, n_iter=200)
    
    # Generate visualizations
    ResultVisualizer.create_animation(optimizer, 'optimization.mp4')
    best_params = optimizer.max['params']
    
    # Final model evaluation
    final_wfm = create_wind_farm_model(
        configurator.create_wake_model(best_params), best_params)
    final_res = final_wfm([0], [0], ws=WSs, TI=TIs, wd=270)
    
    # Generate final plots
    ResultVisualizer.plot_error_distribution(
        (final_res.WS - final_res.flow_map().WS_eff) / final_res.WS)
