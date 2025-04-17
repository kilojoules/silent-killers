import numpy as np
import matplotlib.pyplot as plt
from SALib import ProblemSpec
from SALib.analyze import sobol
from joblib import Parallel, delayed
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid

# 1. Define uncertainty parameters and ranges
problem = {
    'num_vars': 3,
    'names': ['ws', 'wd', 'TI'],
    'bounds': [
        [8.0, 12.0],       # Wind speed (m/s)
        [260.0, 280.0],    # Wind direction (degrees)
        [0.08, 0.12]       # Turbulence intensity
    ]
}

# 2. Generate Sobol samples
n_samples = 500  # Reduced for demonstration; increase for accuracy
sp = ProblemSpec(problem)
sp.sample_saltelli(n_samples, calc_second_order=False)
param_values = sp.samples

# 3. Initialize PyWake model
site = Hornsrev1Site()
x, y = site.initial_position.T
wind_turbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)
grid = HorizontalGrid(resolution=50, extend=0.2)  # Coarse grid for speed

# 4. Parallel simulation function
def simulate(sample):
    ws, wd, ti = sample
    sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
    flow_map = sim_res.flow_map(grid=grid)
    return flow_map.WS_eff.values.flatten()

# 5. Run simulations in parallel
Y = Parallel(n_jobs=-1)(delayed(simulate)(sample) for sample in param_values)
Y = np.array(Y)

# 6. Compute Sobol indices for each grid point
nx, ny = grid.shape()
S1 = np.zeros((nx, ny, problem['num_vars']))
ST = np.zeros((nx, ny, problem['num_vars']))

for i in range(nx):
    for j in range(ny):
        idx = i * ny + j
        y_ij = Y[:, idx]
        if np.all(np.isnan(y_ij)):
            continue
        sp.set_results(y_ij)
        analysis = sp.analyze(sobol, calc_second_order=False)
        S1[i,j] = analysis['S1']
        ST[i,j] = analysis['ST']

# 7. Plot sensitivity maps
x_coords, y_coords = grid.x, grid.y
param_names = problem['names']

for param_idx, param_name in enumerate(param_names):
    # First-order indices
    plt.figure(figsize=(10, 6))
    plt.contourf(x_coords, y_coords, S1[:,:,param_idx], levels=20, cmap='viridis')
    plt.colorbar(label='S1')
    plt.title(f'First-order Sobol Index: {param_name}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(f'sobol_S1_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Total-order indices
    plt.figure(figsize=(10, 6))
    plt.contourf(x_coords, y_coords, ST[:,:,param_idx], levels=20, cmap='viridis')
    plt.colorbar(label='ST')
    plt.title(f'Total-order Sobol Index: {param_name}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(f'sobol_ST_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
