# Define a grid for the flow map based on the site extent (with added margin)
x_min, x_max = x_pos.min() - 500, x_pos.max() + 500
y_min, y_max = y_pos.min() - 500, y_pos.max() + 500
# Use a coarser grid for manageable computational cost in sensitivity analysis
grid_x = np.linspace(x_min, x_max, 20)
grid_y = np.linspace(y_min, y_max, 20)
grid = HorizontalGrid(x=grid_x, y=grid_y)
grid_shape = (len(grid_y), len(grid_x))  # (ny, nx)

# --------------------------
# Define inflow scenarios (which here represent different times)
# Each scenario has its own nominal inflow conditions.
# --------------------------
scenarios = [
   {'label': 'Scenario1', 'ws_nom': 9.8, 'wd_nom': 270, 'ti_nom': 0.10},
   {'label': 'Scenario2', 'ws_nom': 8.5, 'wd_nom': 260, 'ti_nom': 0.12},
   {'label': 'Scenario3', 'ws_nom': 10.0, 'wd_nom': 275, 'ti_nom': 0.09},
]

# For each scenario, propagate uncertainties and compute sensitivity maps
for scen in scenarios:
    label = scen['label']
    ws_nom = scen['ws_nom']
    wd_nom = scen['wd_nom']
    ti_nom = scen['ti_nom']
    
    # --------------------------
    # Define the uncertain problem.
    # We assume realistic measurement uncertainties:
    #    ws ±0.5 m/s, wd ±2°, ti ±0.02.
    # --------------------------
    problem = {
        'num_vars': 3,
        'names': ['ws', 'wd', 'ti'],
        'bounds': [[ws_nom - 0.5, ws_nom + 0.5],
                   [wd_nom - 2, wd_nom + 2],
                   [ti_nom - 0.02, ti_nom + 0.02]]
    }
    
    # --------------------------
    # Generate samples using the Saltelli sampler.
    # The total number of simulations will be N*(2*d + 2) where d=3.
    # Here we set N to a moderate value for demonstration purposes.
    # --------------------------
    N = 32  # base sample size; adjust as needed
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    num_samples = param_values.shape[0]
    
    # Prepare an array to store the flow field output for each simulation.
    # Each simulation returns a 2D flow field (WS_eff) which is flattened.
    M = grid_shape[0] * grid_shape[1]
    Y = np.zeros((num_samples, M))
    
    print(f"Running simulations for {label} with {num_samples} samples...")
    for i, params in enumerate(param_values):
        sample_ws, sample_wd, sample_ti = params
        try:
            # Run the simulation for this sample
            field = run_flow_map(wf_model, x_pos, y_pos, sample_ws, sample_wd, sample_ti, grid)
            Y[i, :] = field.flatten()
        except Exception as e:
            print(f"Simulation failed for sample {i} with parameters {params}: {e}")
            Y[i, :] = np.nan
    
    # Remove any failed simulations (rows containing NaNs)
    valid_indices = ~np.isnan(Y).any(axis=1)
    Y_valid = Y[valid_indices, :]
    param_values_valid = param_values[valid_indices, :]
    print(f"{Y_valid.shape[0]} successful simulations out of {num_samples}.")
    
    # --------------------------
    # Compute Sobol sensitivity indices (first-order S1) for every grid point.
    # For each grid cell, we have a vector of outputs (one per successful simulation).
    # --------------------------
    # Pre-allocate arrays to hold S1 for each parameter at each grid cell.
    S1_ws = np.zeros(M)
    S1_wd = np.zeros(M)
    S1_ti = np.zeros(M)
    
    print("Performing Sobol sensitivity analysis for each grid point...")
    for j in range(M):
        # Y_j is the vector of WS_eff values at grid cell j.
        Y_j = Y_valid[:, j]
        # Run Sobol analysis. Here we only compute first-order indices.
        Si = sobol.analyze(problem, Y_j, calc_second_order=False, print_to_console=False)
        # Si['S1'] is an array of S1 indices for [ws, wd, ti]
        S1_ws[j] = Si['S1'][0]
        S1_wd[j] = Si['S1'][1]
        S1_ti[j] = Si['S1'][2]
    
    # Reshape the sensitivity indices to the grid shape.
    S1_ws_map = S1_ws.reshape(grid_shape)
    S1_wd_map = S1_wd.reshape(grid_shape)
    S1_ti_map = S1_ti.reshape(grid_shape)
    
    # --------------------------
    # Plot and save the sensitivity maps
    # --------------------------
    # Sensitivity with respect to wind speed (ws)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(grid_x, grid_y, S1_ws_map, levels=10)
    ax.set_title(f"{label} - S1 Sensitivity for Wind Speed (ws)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(cf, ax=ax)
    plt.savefig(f"{label}_S1_ws.png")
    plt.close(fig)
    
    # Sensitivity with respect to wind direction (wd)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(grid_x, grid_y, S1_wd_map, levels=10)
    ax.set_title(f"{label} - S1 Sensitivity for Wind Direction (wd)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(cf, ax=ax)
    plt.savefig(f"{label}_S1_wd.png")
    plt.close(fig)
    
    # Sensitivity with respect to turbulence intensity (ti)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(grid_x, grid_y, S1_ti_map, levels=10)
    ax.set_title(f"{label} - S1 Sensitivity for Turbulence Intensity (ti)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(cf, ax=ax)
    plt.savefig(f"{label}_S1_ti.png")
    plt.close(fig)
    
    print(f"Sensitivity maps for {label} saved to disk as '{label}_S1_ws.png', '{label}_S1_wd.png', and '{label}_S1_ti.png'.\n")

