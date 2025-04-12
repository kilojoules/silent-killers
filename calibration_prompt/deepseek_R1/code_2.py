import numpy as np
import xarray as xr
from py_wake import HorizontalGrid, All2AllIterative
from py_wake.deficit_models import GaussianDeficitModel, BlockageDeficitModel
from py_wake.turbulence_models import TurbulenceModel
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class WakeModelFactory:
    @staticmethod
    def create_model(config, params):
        if config['downwind']:
            return DownstreamModel(config, params)
        else:
            return UpstreamModel(config, params)

class DownstreamModel:
    def __init__(self, config, params):
        self.config = config
        self.params = params
        
    def build(self):
        if self.config['model'] == 1:
            deficit_model = BlondelSuperGaussianDeficit2020(
                a_s=self.params['a_s'],
                b_s=self.params['b_s'],
                c_s=self.params['c_s'],
                b_f=self.params['b_f'],
                c_f=self.params['c_f']
            )
        else:
            deficit_model = TurboGaussianDeficit(
                A=self.params['A'],
                cTI=[self.params['cti1'], self.params['cti2']],
                ctlim=self.params['ctlim'],
                ceps=self.params['ceps'],
                ct2a=ct2a_mom1d,
                groundModel=Mirror(),
                rotorAvgModel=GaussianOverlapAvgModel()
            )
            deficit_model.WS_key = 'WS_jlk'

        return All2AllIterative(
            site=Hornsrev1Site(),
            windTurbines=DTU10MW(),
            wake_deficitModel=deficit_model,
            superpositionModel=LinearSum(),
            deflectionModel=None,
            turbulenceModel=CrespoHernandez(
                c=np.array([self.params['ch1'], self.params['ch2'], 
                           self.params['ch3'], self.params['ch4']])
            ),
            blockage_deficitModel=SelfSimilarityDeficit2020(groundModel=Mirror())
        )

class ModelEvaluator:
    def __init__(self, config, reference_data):
        self.config = config
        self.reference_data = reference_data
        self.flow_map_cache = None

    def calculate_metrics(self, pred, obs):
        return {
            'rmse': np.sqrt(((obs - pred)**2).mean(['x', 'y'])).mean('time'),
            'mae': np.abs(obs - pred).mean(['x', 'y'])).mean('time'),
            'p90': np.quantile(np.abs(obs - pred), 0.9)
        }

    def evaluate(self, params):
        model = WakeModelFactory.create_model(self.config, params).build()
        sim_res = model([0], [0], ws=full_ws, TI=full_ti, wd=[270]*full_ti.size)
        
        flow_map = self._compute_flow_map(sim_res)
        pred = (sim_res.WS - flow_map.isel(h=0)) / sim_res.WS
        return self.calculate_metrics(pred, self.reference_data)

    def _compute_flow_map(self, sim_res):
        if self.flow_map_cache is None:
            self.flow_map_cache = xr.concat([
                sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y), time=[tt])['WS_eff']
                for tt in range(full_ws.size)
            ], dim='time')
        return self.flow_map_cache

class OptimizationVisualizer:
    @staticmethod
    def create_animation(optimizer, config):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ani = animation.FuncAnimation(fig, lambda f: OptimizationVisualizer._update_frame(f, ax1, ax2, optimizer), 
                                    frames=len(optimizer.space.target), repeat=False)
        return ani

    @staticmethod
    def _update_frame(frame, ax1, ax2, optimizer):
        ax1.clear()
        ax2.clear()
        
        best_so_far_rmse = float('inf')
        best_so_far_params = {}
        best_rmses = []
        
        for i in range(frame + 1):
            current_rmse = -optimizer.space.target[i]
            if current_rmse <= best_so_far_rmse:
                best_so_far_rmse = current_rmse
                best_so_far_params = optimizer.res[i]['params']
            best_rmses.append(best_so_far_rmse)
        
        ax1.plot(-np.array(optimizer.space.target), color='gray', alpha=0.5)
        ax1.plot(best_rmses, color='black')
        ax1.set_title('Optimization Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        
        keys = list(best_so_far_params.keys())
        best_vals = [best_so_far_params[k] for k in keys]
        default_vals = [defaults[k] for k in keys]
        
        ax2.bar(keys, best_vals, label='Optimized')
        ax2.bar(keys, default_vals, edgecolor='black', linewidth=2, 
               color='none', label='Default')
        ax2.set_title(f'Best RMSE: {best_so_far_rmse:.4f}')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        return ax1, ax2

# Main execution flow
if __name__ == "__main__":
    config = {
        'model': MODEL,
        'downwind': DOWNWIND,
        'x_limits': (X_LB, X_UB),
        'D': D
    }

    # Load and prepare reference data
    reference_data = xr.load_dataset('./DTU10MW.nc').pipe(prepare_reference_data)
    
    # Initialize components
    evaluator = ModelEvaluator(config, reference_data)
    visualizer = OptimizationVisualizer()
    
    # Bayesian Optimization setup
    optimizer = BayesianOptimization(
        f=lambda **p: -evaluator.evaluate(p)['rmse'],
        pbounds=get_parameter_bounds(config),
        random_state=1
    )
    optimizer.probe(params=get_default_parameters(config), lazy=True)
    optimizer.maximize(init_points=50, n_iter=200)
    
    # Visualization
    ani = visualizer.create_animation(optimizer, config)
    ani.save('optimization_animation.mp4', writer=animation.FFMpegWriter(fps=15))
    
    # Final evaluation with best parameters
    best_params = optimizer.max['params']
    final_metrics = evaluator.evaluate(best_params)
    generate_final_plots(final_metrics, best_params)
