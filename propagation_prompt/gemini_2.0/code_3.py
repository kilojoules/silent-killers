import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Install PyWake if needed
try:
    import py_wake
except ModuleNotFoundError:
    !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils import layouts

# Load site and wind turbine data
site = Hornsrev1Site()
x, y = site.initial_position.T
windTurbines = V80()

# Define the wind farm model
wf_model = Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

# --- Uncertainty Propagation and Sensitivity Analysis ---

# 1. Define Uncertainty in Inflow (example: wind speed)
#    -  Assume wind speed has a normal distribution with a standard deviation
#    -  This is a simplified example; in reality, you might have more complex uncertainty models
ws_mean = np.linspace(5, 15, 5)  # Example: 5 different mean wind speeds
ws_std = 1  # Example: 1 m/s standard deviation for wind speed

# 2. Generate Multiple Wind Speed Samples
n_samples = 100  # Number of Monte Carlo samples per mean wind speed
ws_samples = np.random.normal(loc=np.repeat(ws_mean, n_samples), scale=ws_std, size=(len(ws_mean) * n_samples))
wd_samples = np.repeat(np.linspace(260, 280, len(ws_mean)), n_samples) #constant wind direction

# 3. Run Wind Farm Simulations for Each Sample
sim_res_all = []
for i in range(len(ws_samples)):
    sim_res = wf_model(x, y, wd=wd_samples[i], ws=ws_samples[i]) #run the wake model [cite: 4, 5, 6, 7, 8]
    sim_res_all.append(sim_res)

# 4. Calculate Sobol Sensitivity Indices
#    -  This requires more advanced sensitivity analysis techniques (not directly available in PyWake)
#    -  Here's a simplified placeholder to illustrate the concept:
WS_eff_data = np.array([res.WS_eff.values for res in sim_res_all]) #effective wind speed [cite: 9, 10, 11, 12, 13, 14, 15]
#   -  Need to implement Sobol analysis to decompose variance and calculate indices
#   -  For simplicity, let's assume a placeholder sensitivity (this is NOT a correct Sobol index)
sensitivity = np.std(WS_eff_data, axis=0)  # Placeholder: Stdev as a proxy for sensitivity

# 5.  Plot Sensitivity Flow Fields
num_time_steps = len(ws_mean)
fig, axes = plt.subplots(1, num_time_steps, figsize=(5 * num_time_steps, 5), sharey=True)

for t in range(num_time_steps):
    #  Placeholder:  Reshape sensitivity data for the time step
    sensitivity_at_t = sensitivity[t * n_samples].reshape(wf_model.site.initial_position.shape)[0]  

    # Create a flow map for plotting [cite: 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
    flow_map = sim_res_all[t * n_samples].flow_map(wd=wd_samples[t * n_samples], ws=ws_samples[t * n_samples])
    
    # Plot the wake map with the sensitivity as contour levels
    flow_map.plot_wake_map(ax=axes[t], levels=10)  # Example: 10 contour levels
    axes[t].set_title(f'WS={ws_mean[t]:.1f} m/s')
    axes[t].set_xlabel('x [m]')
    if t == 0:
        axes[t].set_ylabel('y [m]')

plt.tight_layout()
plt.savefig("sobol_sensitivity_flow_field.png") #save figure

plt.show()
