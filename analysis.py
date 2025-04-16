import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches # Import for legend patches
import seaborn as sns
import os
import glob
import numpy as np

# --- Configuration ---
prompt_config = {
    'propagation_prompt': 'Easy',
    'calibration_prompt': 'Medium',
    'optimization_prompt': 'Hard',
    'optimization_prompt2': 'Hard'
}
base_data_dir = '.'
code_metrics_filename = 'llm_code_metrics.csv'
response_metrics_filename = 'llm_response_metrics.csv'
combined_code_metrics_output = 'all_prompts_code_metrics.csv'
combined_response_metrics_output = 'all_prompts_response_metrics.csv'
output_plot_dir = './plots_grid_revised' # Changed output dir name
os.makedirs(output_plot_dir, exist_ok=True)

# --- Data Aggregation ---
# (Assume this section runs successfully as before and creates combined_code_df)
all_code_metrics_dfs = []
# ... (Loading loop as in previous version) ...
print("--- Starting Data Aggregation ---")
for prompt_dir_name, difficulty_label in prompt_config.items():
    print(f"Processing directory: {prompt_dir_name} (Difficulty: {difficulty_label})")
    prompt_path = os.path.join(base_data_dir, prompt_dir_name)
    code_csv_path = os.path.join(prompt_path, code_metrics_filename)
    # Load Code Metrics CSV
    try:
        code_df = pd.read_csv(code_csv_path)
        code_df['prompt_difficulty'] = difficulty_label
        code_df['prompt_dir'] = prompt_dir_name
        all_code_metrics_dfs.append(code_df)
        print(f"  Loaded {len(code_df)} rows from {code_csv_path}")
    except FileNotFoundError:
        print(f"  ⚠️ Warning: Code metrics file not found: {code_csv_path}")
    except Exception as e:
        print(f"  ❌ Error loading {code_csv_path}: {e}")
# (Load response metrics - omitted for brevity)

combined_code_df = pd.DataFrame()
if all_code_metrics_dfs:
    combined_code_df = pd.concat(all_code_metrics_dfs, ignore_index=True)
    print(f"\n✅ Combined {len(combined_code_df)} total rows for code metrics.")
    # (Saving combined CSV - omitted for brevity)
else:
    print("\n⚠️ No code metrics dataframes were loaded. Cannot proceed.")
    exit()
# --- End Data Aggregation ---


# --- Plotting ---
if not combined_code_df.empty:
    print("\n--- Generating Plots ---")

    # --- Plotting Setup ---
    try:
        if 'file' not in combined_code_df.columns:
             raise ValueError("'file' column not found.")
        combined_code_df['seed_number'] = combined_code_df['file'].str.extract(r'_(\d+)\.', expand=False)
        combined_code_df['seed_number'] = pd.to_numeric(combined_code_df['seed_number'], errors='coerce')
        combined_code_df.dropna(subset=['seed_number'], inplace=True)
        combined_code_df['seed_number'] = combined_code_df['seed_number'].astype(int)

        difficulty_order = ['Easy', 'Medium', 'Hard']
        available_difficulties = [d for d in difficulty_order if d in combined_code_df['prompt_difficulty'].unique()]
        if not available_difficulties: raise ValueError("No valid difficulties found.")
        model_order = sorted(combined_code_df['model'].unique())
        max_seed = int(combined_code_df['seed_number'].max())
        seed_numbers = np.arange(1, max_seed + 1)

        full_index = pd.MultiIndex.from_product(
            [available_difficulties, seed_numbers], names=['prompt_difficulty', 'seed_number']
        )
        plot_df_metrics = combined_code_df[combined_code_df['parsing_error'].isnull()].copy()
        print("Plotting setup complete.")
    except Exception as e:
         print(f"❌ Error during initial data preparation for plotting: {e}")
         exit()

    # --- Plot 1: Code File Existence (Revised Colors & Legend) ---
    print("\n--- Generating Plot 1: Code File Existence ---")
    try:
        fig1, axes1 = plt.subplots(1, len(available_difficulties),
                                  figsize=(7 * len(available_difficulties), len(model_order) * 0.8),
                                  sharey=True, squeeze=False)
        axes1 = axes1.flatten()

        existence_grouped = combined_code_df.groupby(
            ['model', 'prompt_difficulty', 'seed_number']
        ).size()
        existence_pivot = existence_grouped.unstack(
            ['prompt_difficulty', 'seed_number'], fill_value=0
        )
        existence_pivot = (existence_pivot > 0).astype(int) # 0 = Missing, 1 = Present
        existence_pivot = existence_pivot.reindex(index=model_order, columns=full_index, fill_value=0)

        # Define colors: 0=Black (Missing), 1=White (Present) <<< REVISED
        cmap_existence = mcolors.ListedColormap(['black', 'white'])
        bounds_existence = [-0.5, 0.5, 1.5]
        norm_existence = mcolors.BoundaryNorm(bounds_existence, cmap_existence.N)

        for i, difficulty in enumerate(available_difficulties):
            ax = axes1[i]
            try:
                 data_subset = existence_pivot.loc[:, difficulty]
            except KeyError: continue
            data_subset = data_subset.reindex(columns=seed_numbers, fill_value=0)
            x_coords = np.arange(data_subset.shape[1] + 1)
            y_coords = np.arange(data_subset.shape[0] + 1)
            mesh = ax.pcolormesh(x_coords, y_coords, data_subset.values,
                                 cmap=cmap_existence, norm=norm_existence,
                                 edgecolors='darkgray', linewidth=0.5) # Changed edge color
            ax.set_xticks(seed_numbers - 0.5); ax.set_xticklabels(seed_numbers)
            if i == 0:
                 ax.set_yticks(np.arange(len(model_order)) + 0.5); ax.set_yticklabels(model_order)
                 ax.set_ylabel("Model")
            else: ax.tick_params(axis='y', which='both', left=False)
            ax.set_xlabel("Seed Number (Trial)"); ax.set_title(f"Difficulty: {difficulty}")
            ax.invert_yaxis()

        #fig1.suptitle("Code File Existence", y=1.04) # Simplified title

        # --- Add Manual Legend for Existence ---
        legend_patches = [mpatches.Patch(color='white', label='Present / Valid Code'),
                          mpatches.Patch(color='black', label='Missing / Parsing Error')]
        fig1.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.99, 1.0))
        # --- End Legend ---

        fig1.tight_layout(rect=[0, 0, 0.9, 0.98]) # Adjust layout slightly for legend

        plot_filename = 'grid_existence_revised.png'
        plot_path = os.path.join(output_plot_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"  ✅ Saved plot: {plot_path}")
        plt.close(fig1)

    except Exception as e:
        print(f"  ❌ Error generating Existence plot: {e}")
        if 'fig1' in locals(): plt.close(fig1)

# --- Plot 2: Lines of Code (LOC) (Revised Layout & Continuous Colorbar) ---
    print("\n--- Generating Plot 2: Lines of Code (LOC) ---")
    try:
        # Use constrained_layout=True for better automatic spacing
        fig2, axes2 = plt.subplots(1, len(available_difficulties),
                                  figsize=(5.5 * len(available_difficulties), len(model_order) * 0.6), # Kept slightly wider
                                  sharey=True, squeeze=False, constrained_layout=True) # Use constrained_layout
        axes2 = axes2.flatten()

        # Pivot data for LOC using plot_df_metrics (only valid code)
        loc_pivot = pd.pivot_table(plot_df_metrics,
                                   values='loc', index='model',
                                   columns=['prompt_difficulty', 'seed_number'],
                                   aggfunc='mean', fill_value=np.nan)

        # Use full_index defined outside the try block
        loc_pivot = loc_pivot.reindex(index=model_order, columns=full_index, fill_value=np.nan)

        # --- Use Continuous Colormap ---
        cmap_loc = plt.get_cmap('coolwarm').copy() # Get a copy
        cmap_loc.set_bad('black') # Set color for NaN values (missing/parsing error)
        # --- End Continuous Colormap ---

        # Find min/max LOC for consistent color scaling, ignore NaN
        vmin = np.nanmin(loc_pivot.values) if not np.all(np.isnan(loc_pivot.values)) else 0
        vmax = np.nanmax(loc_pivot.values) if not np.all(np.isnan(loc_pivot.values)) else 1

        mesh = None # Initialize mesh to None
        for i, difficulty in enumerate(available_difficulties):
            ax = axes2[i]
            try:
                 data_subset = loc_pivot.loc[:, difficulty]
            except KeyError: continue
            data_subset = data_subset.reindex(columns=seed_numbers, fill_value=np.nan)

            x_coords = np.arange(data_subset.shape[1] + 1)
            y_coords = np.arange(data_subset.shape[0] + 1)

            # Plot using pcolormesh with continuous map
            mesh = ax.pcolormesh(x_coords, y_coords, data_subset.values,
                                 cmap=cmap_loc, vmin=vmin, vmax=vmax, # Use continuous map
                                 edgecolors='lightgray', linewidth=0.5)

            ax.set_xticks(seed_numbers - 0.5); ax.set_xticklabels(seed_numbers)
            if i == 0:
                 ax.set_yticks(np.arange(len(model_order)) + 0.5); ax.set_yticklabels(model_order)
                 ax.set_ylabel("Model")
            else: ax.tick_params(axis='y', which='both', left=False)
            ax.set_xlabel("Seed Number (Trial)"); ax.set_title(f"Difficulty: {difficulty}")
            ax.invert_yaxis()

        if mesh is None: # Check if any plotting actually happened
             raise ValueError("No data plotted for LOC.")

        #fig2.suptitle("Lines of Code (LOC) (Black=Missing/Parsing Error)", y=1.02)

        # Add a color bar - constrained_layout should handle spacing better
        fig2.colorbar(mesh, ax=axes2.ravel().tolist(), shrink=0.6, label='Lines of Code')

        # NOTE: Removed tight_layout() call - use constrained_layout from fig creation

        plot_filename = 'grid_loc_continuous.png' # Changed filename
        plot_path = os.path.join(output_plot_dir, plot_filename)
        #plt.tight_layout()
        plt.savefig(plot_path) # Save before potential layout issues manifest display
        print(f"  ✅ Saved plot: {plot_path}")
        plt.close(fig2)

    except Exception as e:
        print(f"  ❌ Error generating LOC plot: {e}")
        if 'fig2' in locals(): plt.close(fig2)


# --- Plot 3: Bad Exception Rate (BER) ---
    print("\n--- Generating Plot 3: Bad Exception Rate (BER) ---")
    try:
        # Use constrained_layout=True for better automatic spacing
        fig3, axes3 = plt.subplots(1, len(available_difficulties),
                                   figsize=(5.5 * len(available_difficulties), len(model_order) * 0.6), # Adjust size as needed
                                   sharey=True, squeeze=False, constrained_layout=True) # Use constrained_layout
        axes3 = axes3.flatten()

        # Pivot data for Bad Exception Rate using plot_df_metrics (only valid code)
        ber_pivot = pd.pivot_table(plot_df_metrics,
                                   values='bad_exception_rate', # <<< CHANGED VALUE
                                   index='model',
                                   columns=['prompt_difficulty', 'seed_number'],
                                   aggfunc='mean', # Use mean (or first, etc.) as there should only be one value
                                   fill_value=np.nan)

        # Use full_index defined outside the try block
        ber_pivot = ber_pivot.reindex(index=model_order, columns=full_index, fill_value=np.nan)

        # --- Use Sequential Colormap (e.g., Reds) ---
        # Higher rate = "more red" might be intuitive for "bad" rate
        cmap_ber = plt.get_cmap('Reds').copy() # <<< CHANGED COLORMAP
        cmap_ber.set_bad('black') # Set color for NaN values (missing/parsing error)
        # --- End Colormap ---

        # Set vmin/vmax for rate (0.0 to 1.0)
        vmin_ber = 0.0 # <<< SET vmin
        vmax_ber = 1.0 # <<< SET vmax

        mesh3 = None # Initialize mesh to None
        for i, difficulty in enumerate(available_difficulties):
            ax = axes3[i]
            try:
                # Check if the difficulty level exists in the pivoted data columns
                if difficulty in ber_pivot.columns.get_level_values('prompt_difficulty'):
                   data_subset = ber_pivot.loc[:, difficulty]
                else:
                   # Create an empty DataFrame with NaN if difficulty is missing
                   data_subset = pd.DataFrame(np.nan, index=model_order, columns=seed_numbers)

                # Ensure columns match the expected seed numbers, filling missing ones with NaN
                data_subset = data_subset.reindex(columns=seed_numbers, fill_value=np.nan)

            except KeyError: # Should be less likely with the check above, but keep for safety
                print(f"  ⚠️ Warning: KeyError accessing difficulty '{difficulty}' for BER plot. Skipping subplot.")
                # Create an empty DataFrame with NaN if key error occurs during slicing
                data_subset = pd.DataFrame(np.nan, index=model_order, columns=seed_numbers)


            x_coords = np.arange(data_subset.shape[1] + 1)
            y_coords = np.arange(data_subset.shape[0] + 1)

            # Plot using pcolormesh with the new colormap and fixed range
            mesh3 = ax.pcolormesh(x_coords, y_coords, data_subset.values,
                                  cmap=cmap_ber, vmin=vmin_ber, vmax=vmax_ber, # <<< Use BER map and vmin/vmax
                                  edgecolors='lightgray', linewidth=0.5)

            ax.set_xticks(seed_numbers - 0.5); ax.set_xticklabels(seed_numbers)
            if i == 0:
                ax.set_yticks(np.arange(len(model_order)) + 0.5); ax.set_yticklabels(model_order)
                ax.set_ylabel("Model")
            else: ax.tick_params(axis='y', which='both', left=False)
            ax.set_xlabel("Seed Number (Trial)"); ax.set_title(f"Difficulty: {difficulty}")
            ax.invert_yaxis()

        if mesh3 is None: # Check if any plotting actually happened
            raise ValueError("No data plotted for Bad Exception Rate.")

        #fig3.suptitle("Bad Exception Rate (Black=Missing/Parsing Error)", y=1.02) # <<< UPDATED TITLE

        # Add a color bar
        fig3.colorbar(mesh3, ax=axes3.ravel().tolist(), shrink=0.6, label='Bad Exception Rate (0.0 to 1.0)') # <<< UPDATED LABEL

        plot_filename = 'grid_bad_exception_rate.png' # <<< CHANGED FILENAME
        plot_path = os.path.join(output_plot_dir, plot_filename)
        plt.savefig(plot_path) # Save before potential layout issues manifest display
        print(f"  ✅ Saved plot: {plot_path}")
        plt.close(fig3) # <<< Close fig3

    except Exception as e:
        print(f"  ❌ Error generating Bad Exception Rate plot: {e}")
        if 'fig3' in locals(): plt.close(fig3) # <<< Close fig3 if error

# --- Continue with any other plots or end of script ---

# --- Plot 4: Bad Exception Blocks (Count) ---
    print("\n--- Generating Plot 4: Bad Exception Blocks (Count) ---")
    try:
        # Use constrained_layout=True for better automatic spacing
        fig4, axes4 = plt.subplots(1, len(available_difficulties),
                                   figsize=(5.5 * len(available_difficulties), len(model_order) * 0.6), # Adjust size as needed
                                   sharey=True, squeeze=False, constrained_layout=True) # Use constrained_layout
        axes4 = axes4.flatten()

        # Pivot data for Bad Exception Blocks count using plot_df_metrics (only valid code)
        bec_pivot = pd.pivot_table(plot_df_metrics,
                                   values='bad_exception_blocks', # <<< CHANGED VALUE
                                   index='model',
                                   columns=['prompt_difficulty', 'seed_number'],
                                   aggfunc='mean', # Use mean (or first, etc.) as there should only be one value
                                   fill_value=np.nan)

        # Use full_index defined outside the try block
        bec_pivot = bec_pivot.reindex(index=model_order, columns=full_index, fill_value=np.nan)

        # --- Use Sequential Colormap (e.g., Reds) ---
        cmap_bec = plt.get_cmap('Reds').copy() # Reusing 'Reds', could choose another sequential map
        cmap_bec.set_bad('black') # Set color for NaN values (missing/parsing error)
        # --- End Colormap ---

        # Find min/max count for consistent color scaling, ignore NaN
        vmin_bec = 0 # Minimum count is 0
        vmax_bec = np.nanmax(bec_pivot.values) if not np.all(np.isnan(bec_pivot.values)) else 0
        # Ensure vmax is at least 1 if max is 0, to avoid vmin=vmax issues if all counts are 0
        vmax_bec = max(vmax_bec, 1.0)
        # --- End Color Scaling ---


        mesh4 = None # Initialize mesh to None
        for i, difficulty in enumerate(available_difficulties):
            ax = axes4[i]
            try:
                 # Check if the difficulty level exists in the pivoted data columns
                if difficulty in bec_pivot.columns.get_level_values('prompt_difficulty'):
                   data_subset = bec_pivot.loc[:, difficulty]
                else:
                   # Create an empty DataFrame with NaN if difficulty is missing
                   data_subset = pd.DataFrame(np.nan, index=model_order, columns=seed_numbers)

                # Ensure columns match the expected seed numbers, filling missing ones with NaN
                data_subset = data_subset.reindex(columns=seed_numbers, fill_value=np.nan)

            except KeyError: # Should be less likely with the check above, but keep for safety
                print(f"  ⚠️ Warning: KeyError accessing difficulty '{difficulty}' for BEC plot. Skipping subplot.")
                # Create an empty DataFrame with NaN if key error occurs during slicing
                data_subset = pd.DataFrame(np.nan, index=model_order, columns=seed_numbers)

            x_coords = np.arange(data_subset.shape[1] + 1)
            y_coords = np.arange(data_subset.shape[0] + 1)

            # Plot using pcolormesh with the count colormap and dynamic range
            mesh4 = ax.pcolormesh(x_coords, y_coords, data_subset.values,
                                  cmap=cmap_bec, vmin=vmin_bec, vmax=vmax_bec, # <<< Use BEC map and vmin/vmax
                                  edgecolors='lightgray', linewidth=0.5)

            ax.set_xticks(seed_numbers - 0.5); ax.set_xticklabels(seed_numbers)
            if i == 0:
                ax.set_yticks(np.arange(len(model_order)) + 0.5); ax.set_yticklabels(model_order)
                ax.set_ylabel("Model")
            else: ax.tick_params(axis='y', which='both', left=False)
            ax.set_xlabel("Seed Number (Trial)"); ax.set_title(f"Difficulty: {difficulty}")
            ax.invert_yaxis()

        if mesh4 is None: # Check if any plotting actually happened
            raise ValueError("No data plotted for Bad Exception Blocks count.")

        #fig4.suptitle("Bad Exception Blocks (Count) (Black=Missing/Parsing Error)", y=1.02) # <<< UPDATED TITLE

        # Add a color bar
        # Use integer formatting if vmax is reasonably small, otherwise default float formatting
        format_str = '%.0f' if vmax_bec < 10 else None
        fig4.colorbar(mesh4, ax=axes4.ravel().tolist(), shrink=0.6, label='Bad Exception Blocks (Count)', format=format_str) # <<< UPDATED LABEL

        plot_filename = 'grid_bad_exception_count.png' # <<< CHANGED FILENAME
        plot_path = os.path.join(output_plot_dir, plot_filename)
        plt.savefig(plot_path) # Save before potential layout issues manifest display
        print(f"  ✅ Saved plot: {plot_path}")
        plt.close(fig4) # <<< Close fig4

    except Exception as e:
        print(f"  ❌ Error generating Bad Exception Blocks count plot: {e}")
        if 'fig4' in locals(): plt.close(fig4) # <<< Close fig4 if error

# --- Continue with any other plots or end of script ---
