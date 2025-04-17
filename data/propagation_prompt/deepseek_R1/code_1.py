import numpy as np
import matplotlib.pyplot as plt
from SALib import ProblemSpec
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake import HorizontalGrid
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import xarray as xr

# 1. Define parameters and their uncertainties
problem = {
    'num_vars': 3,
    'names': ['ws', 'wd', 'TI'],
    'bounds': [[9.5, 10.5],   # Wind speed ±0.5 m/s
              [265, 275],     # Wind direction ±5°
              [0.08, 0.12]]   # Turbulence intensity ±0.02
}

# Generate Sobol samples
N = 500  # Base samples (total = N*(2D+2) = 500*8=4000)
sp = ProblemSpec(problem)
sp.sample_saltelli(N)
samples = sp.samples

# 2. Setup PyWake model
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# Define grid for flow maps
grid = HorizontalGrid(resolution=100, extend=500)
flow_map_example = wf_model(x, y, wd=270, ws=10).flow_map(grid=grid)
X, Y = flow_map_example.X, flow_map_example.Y
grid_shape = X.shape
num_points = X.size

# 3. Parallel simulation for all samples
def process_sample(params):
    ws, wd, ti = params
    sim_res = wf_model(x, y, wd=wd, ws=ws, TI=ti)
    flow_map = sim_res.flow_map(grid=grid)
    return flow_map.WS_eff.values.reshape(-1)

# Split samples into chunks for parallel processing
chunks = np.array_split(samples, 8)  # Adjust based on available CPUs
results = []

with ProcessPoolExecutor() as executor:
    for chunk in chunks:
        chunk_results = list(executor.map(process_sample, chunk))
        results.extend(chunk_results)

ws_eff_matrix = np.array(results)

# 4. Compute Sobol indices
sp.set_results(ws_eff_matrix)
sp.analyze_sobol()

# 5. Plot sensitivity maps for each parameter
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Wind Speed (WS)', 'Wind Direction (WD)', 'Turbulence Intensity (TI)']
for i, ax in enumerate(axes):
    S1 = sp.analysis['S1'][:, i].reshape(grid_shape)
    contour = ax.contourf(X, Y, S1, levels=15, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='First-order Sobol Index')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Sensitivity to {titles[i]}')

plt.tight_layout()
plt.savefig('sobol_sensitivity_maps.png')
plt.show()

# Save results to NetCDF for further analysis
ds = xr.Dataset(
    {
        "S1_ws": (["x", "y"], sp.analysis['S1'][:, 0].reshape(grid_shape)),
        "S1_wd": (["x", "y"], sp.analysis['S1'][:, 1].reshape(grid_shape)),
        "S1_ti": (["x", "y"], sp.analysis['S1'][:, 2].reshape(grid_shape)),
    },
    coords={"x": X[0,:], "y": Y[:,0]},
)

ds.to_netcdf('sobol_sensitivity.nc')
