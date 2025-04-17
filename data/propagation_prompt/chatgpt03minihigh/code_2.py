# Define the nominal values
ws_nom = inflow['ws_nom']
wd_nom = inflow['wd_nom']

# Create the uncertainty problem definition for SALib
# Bounds are defined relative to the nominal value
problem = {
    'num_vars': num_vars,
    'names': ['ws', 'wd'],
    'bounds': [
        [ws_nom - delta_ws, ws_nom + delta_ws],
        [wd_nom - delta_wd, wd_nom + delta_wd]
    ]
}

# Generate Saltelli samples (the returned array has shape [N*(2+num_vars), num_vars])
param_values = saltelli.sample(problem, N, calc_second_order=False)
n_samples = param_values.shape[0]
print(f"  Generated {n_samples} samples for Sobol analysis.")

# Preallocate an array to store the flow-field output for each sample.
# The shape will be (n_samples, Ny, Nx) where (Ny, Nx) are the grid dimensions.
Ny, Nx = grid_x.size, grid_y.size  # note: grid object is built with x and y vectors.
flow_fields = np.empty((n_samples, grid.y.size, grid.x.size))

# Run the simulation for each sample
print("  Running simulations for each sample:")
for i in tqdm(range(n_samples)):
    sample_ws, sample_wd = param_values[i]
    # Compute the flow field (WS_eff) for this sample.
    flow_fields[i, :, :] = simulate_flow(sample_ws, sample_wd)

# -----------------------------
# Compute Sobol sensitivity indices at each grid point
# We will compute the first order (S1) and total order (ST) sensitivity indices.
# For each grid cell, the model output is the collection of simulated WS_eff values.
# -----------------------------
S1_field = np.zeros((num_vars, grid.y.size, grid.x.size))
ST_field = np.zeros((num_vars, grid.y.size, grid.x.size))

print("  Computing Sobol sensitivity indices at each grid cell...")
# Loop over every grid point (nested loop over y and x indices)
for iy in range(grid.y.size):
    for ix in range(grid.x.size):
        Y = flow_fields[:, iy, ix]  # model outputs for current grid point, shape (n_samples,)
        # Run Sobol analysis for the scalar output Y.
        # Note: setting print_to_console=False to avoid clutter.
        Si = sobol.analyze(problem, Y, print_to_console=False, conf_level=0.95)
        # Store first order and total order indices for each parameter.
        for ip in range(num_vars):
            S1_field[ip, iy, ix] = Si['S1'][ip]
            ST_field[ip, iy, ix] = Si['ST'][ip]

# -----------------------------
# Save sensitivity maps (as images) for each uncertain parameter
# We create contour plots for both the first order and total order indices.
# The x and y coordinates on the grid are taken from grid.x and grid.y.
# -----------------------------
for ip, param_name in enumerate(problem['names']):
    # Plot first order sensitivity index
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(grid.X, grid.Y, S1_field[ip, :, :], levels=10)
    plt.colorbar(cp, label=f'First order Sobol S1 for {param_name}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'{snap_label}: First Order Sensitivity of WS_eff to {param_name}')
    fname_S1 = f'sobol_S1_{snap_label}_{param_name}.png'
    plt.savefig(fname_S1, dpi=150)
    plt.close()
    print(f"    Saved {fname_S1}")
    
    # Plot total order sensitivity index
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(grid.X, grid.Y, ST_field[ip, :, :], levels=10)
    plt.colorbar(cp, label=f'Total Sobol ST for {param_name}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'{snap_label}: Total Sensitivity of WS_eff to {param_name}')
    fname_ST = f'sobol_ST_{snap_label}_{param_name}.png'
    plt.savefig(fname_ST, dpi=150)
    plt.close()
    print(f"    Saved {fname_ST}")

