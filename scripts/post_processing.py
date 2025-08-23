"""
post_processing.py
==================
Scan every prompt directory inside --root, load llm_code_metrics.csv files
built by `process_files.py`, aggregate, and create the multi‑panel plots
identical to the original notebook implementation.
"""

# ---------- imports ---------------------------------------------------------
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------- configuration ---------------------------------------------------
PROMPT_CONFIG = {
    'propagation_prompt':  'Easy',
    'calibration_prompt':  'Medium',
    'optimization_prompt': 'Hard',
    'optimization_prompt2': 'Hard',
}
CODE_METRICS_FILE   = "llm_code_metrics.csv"
OUTPUT_PLOT_DIR     = Path("plots")
OUTPUT_PLOT_DIR.mkdir(exist_ok=True)

# ---------- data loading ----------------------------------------------------
def _load_and_aggregate(root: Path) -> pd.DataFrame:
    print("--- Starting Data Aggregation ---")
    frames = []
    for prompt_dir, difficulty in PROMPT_CONFIG.items():
        csv_path = root / prompt_dir / CODE_METRICS_FILE
        if not csv_path.exists():
            print(f"  ⚠️  {csv_path} not found – skipping")
            continue
        df = pd.read_csv(csv_path)
        df["prompt_dir"]        = prompt_dir
        df["prompt_difficulty"] = difficulty
        frames.append(df)
        print(f"  Loaded {len(df)} rows from {csv_path}")
    if not frames:
        sys.exit("❌  No CSVs found – aborting")
    combined = pd.concat(frames, ignore_index=True)
    print(f"✅  Combined {len(combined)} total rows\n")
    return combined

# ---------- helper from your original code ----------------------------------
def prepare_plotting_data(df):
    print("--- Preparing Data for Plotting ---")
    df["seed_number"] = (
        df["file"].str.extract(r"_(\d+)\.", expand=False).astype(float)
    )
    df.dropna(subset=["seed_number"], inplace=True)
    df["seed_number"] = df["seed_number"].astype(int)

    model_order  = sorted(df["model"].unique())
    diff_order   = ["Easy", "Medium", "Hard"]
    available    = [d for d in diff_order if d in df["prompt_difficulty"].unique()]
    seed_numbers = np.arange(1, df["seed_number"].max() + 1)

    full_index = pd.MultiIndex.from_product(
        [available, seed_numbers],
        names=["prompt_difficulty", "seed_number"]
    )
    print("Plotting setup complete.")
    return model_order, available, seed_numbers, full_index

def calculate_pivot_tables(df, model_order, full_index):
    print("--- Calculating Pivot Tables ---")
    pivots = {}
    # 1. STATUS
    existence = (
        df.groupby(["model", "prompt_difficulty", "seed_number"])
          .size()
          .unstack(["prompt_difficulty", "seed_number"], fill_value=0)
    )
    existence = existence.reindex(index=model_order, columns=full_index, fill_value=0) > 0

    errors = (
        pd.pivot_table(df, values="parsing_error", index="model",
                       columns=["prompt_difficulty", "seed_number"],
                       aggfunc="first", fill_value=np.nan)
          .reindex(index=model_order, columns=full_index)
    )
    state = pd.DataFrame(0, index=model_order, columns=full_index)         # Missing
    state[existence &  errors.notna()]  = 1                                # Error
    state[existence &  errors.isna()]   = 2                                # OK
    pivots["status"] = state

    # 2. other metrics (only for OK rows)
    ok = df[df["parsing_error"].isna()].copy()
    pivots["loc"] = (
        ok.pivot_table(values="loc", index="model",
                       columns=["prompt_difficulty", "seed_number"])
          .reindex(index=model_order, columns=full_index)
    )
    pivots["ber"] = (
        ok.pivot_table(values="bad_exception_rate", index="model",
                       columns=["prompt_difficulty", "seed_number"])
          .reindex(index=model_order, columns=full_index)
    )
    pivots["bec"] = (
        ok.pivot_table(values="bad_exception_blocks", index="model",
                       columns=["prompt_difficulty", "seed_number"])
          .reindex(index=model_order, columns=full_index)
    )
    print("Pivot tables ready.")
    return pivots

def plot_metric_grid(pivot, *, title, cmap, filename_suffix,
                     model_order, diff, seeds,
                     norm=None, vmin=None, vmax=None,
                     legend_handles=None, cbar_label=None,
                     nan_color="black"):
    n_models, n_diffs = len(model_order), len(diff)
    fig, axes = plt.subplots(
        1, n_diffs,
        figsize=(6 * n_diffs, 0.7 * n_models),
        sharey=True,
        constrained_layout=True
    )
    axes = axes.flatten()
    is_continuous = norm is None
    cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    if is_continuous:
        cmap.set_bad(nan_color)

    for i, d in enumerate(diff):
        ax = axes[i]
        subset = pivot[d] if d in pivot.columns.get_level_values(0) else \
                 pd.DataFrame(np.nan, index=model_order, columns=seeds)
        subset = subset.reindex(columns=seeds)
        mesh = ax.pcolormesh(
            np.arange(subset.shape[1] + 1),
            np.arange(subset.shape[0] + 1),
            subset.values,
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
            edgecolors="darkgray", linewidth=0.5
        )
        ax.set_xticks(np.arange(len(seeds)) + 0.5)
        ax.set_xticklabels(seeds)
        if i == 0:
            ax.set_yticks(np.arange(len(model_order)) + 0.5)
            ax.set_yticklabels(model_order)
            ax.set_ylabel("Model")
        else:
            ax.tick_params(axis="y", left=False)
        ax.set_xlabel("Seed")
        ax.set_title(f"Difficulty: {d}")
        ax.invert_yaxis()

    fig.suptitle(title, y=1.05)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right")
    elif cbar_label and is_continuous:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label=cbar_label, shrink=0.7)

    out = OUTPUT_PLOT_DIR / f"grid_{filename_suffix}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ saved {out}")

def plot_statistical_analysis(df: pd.DataFrame):
    """
    Generates box plots to show the distribution of bad_exception_rate
    for each model, faceted by prompt difficulty.
    
    Also overlays the mean and standard error of the mean (SEM).
    """
    print("\n--- Generating Statistical Box Plots ---")
    
    # Use only successfully parsed runs that contained at least one try/except block
    # This makes the "bad rate" statistic more meaningful.
    plot_df = df#[(df["has_try"] == True)].copy()

    # Get the order of difficulties
    difficulty_order = ["Easy", "Medium", "Hard"]
    
    for difficulty in difficulty_order:
        if difficulty not in plot_df["prompt_difficulty"].unique():
            continue

        subset_df = plot_df[plot_df["prompt_difficulty"] == difficulty]

        # --- Create the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Create the box plots
        sns.boxplot(
            data=subset_df,
            x="model",
            y="bad_exception_rate",
            ax=ax,
            boxprops={'alpha': 0.6}
        )
        
        # 2. Overlay a stripplot to show individual data points
        sns.stripplot(
            data=subset_df,
            x="model",
            y="bad_exception_rate",
            ax=ax,
            jitter=0.2,
            alpha=0.7,
            color='dodgerblue'
        )

        # 3. Calculate and overlay the mean + standard error bars
        stats = subset_df.groupby('model')['bad_exception_rate'].agg(['mean', 'sem']).reset_index()
        
        # Get model order from the plot to ensure stats align with boxes
        model_order = [tick.get_text() for tick in ax.get_xticklabels()]
        stats = stats.set_index('model').reindex(model_order).reset_index()
        
        ax.errorbar(
            x=stats.index, 
            y=stats['mean'], 
            yerr=stats['sem'],
            fmt='o',         # Format for the mean point
            color='black',
            capsize=5,       # Width of the error bar caps
            label='Mean ± SEM',
            markersize=8,
            elinewidth=2
        )

        # --- Final Touches ---
        ax.set_title(f"Bad Exception Rate Distribution (Difficulty: {difficulty})", fontsize=16, pad=20)
        ax.set_ylabel("Bad Exception Rate", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.set_ylim(-0.05, 1.05) # Rate is between 0 and 1
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"boxplot_bad_rate_{difficulty.lower()}.png"
        out_path = OUTPUT_PLOT_DIR / filename
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✅ saved {out_path}")

# ---------- main ------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Folder that holds the prompt dirs")
    args = ap.parse_args(argv)

    root      = Path(args.root).resolve()
    code_df   = _load_and_aggregate(root)
    models, diffs, seeds, full_idx = prepare_plotting_data(code_df)
    pivots    = calculate_pivot_tables(code_df, models, full_idx)

    # ---------- plot 1: STATUS ---------------------------------------------
    legend = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Parsed OK"),
        mpatches.Patch(facecolor="grey",  label="Parsing Error"),
        mpatches.Patch(facecolor="black", label="Missing File"),
    ]
    plot_metric_grid(
        pivots["status"], title="Code File Status",
        cmap=mcolors.ListedColormap(["black", "grey", "white"]),
        norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
        legend_handles=legend, filename_suffix="status_3color",
        model_order=models, diff=diffs, seeds=seeds
    )

    # ---------- plot 2: LOC -------------------------------------------------
    loc = pivots["loc"]
    plot_metric_grid(
        loc, title="Lines of Code (LOC)",
        cmap="coolwarm", filename_suffix="loc_continuous",
        vmin=np.nanmin(loc.values), vmax=np.nanmax(loc.values),
        cbar_label="Lines of Code",
        model_order=models, diff=diffs, seeds=seeds
    )

    # ---------- plot 3: BER -------------------------------------------------
    plot_metric_grid(
        pivots["ber"], title="",
        cmap="Reds", vmin=0.0, vmax=1.0, cbar_label="Bad exception Rate",
        filename_suffix="bad_exception_rate",
        model_order=models, diff=diffs, seeds=seeds
    )

    # ---------- plot 4: BEC -------------------------------------------------
    bec = pivots["bec"]
    plot_metric_grid(
        bec, title="Bad Exception Blocks (Count)",
        cmap="Reds", vmin=0, vmax=max(1, np.nanmax(bec.values)),
        cbar_label="Count", filename_suffix="bad_exception_count",
        model_order=models, diff=diffs, seeds=seeds
    )

    # ---------- plot 5: simple bar -----------------------------------------
    ok_counts = code_df[code_df["parsing_error"].isna()] \
                    .groupby("prompt_difficulty")["file"].count() \
                    .reindex(["Easy", "Medium", "Hard"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(ok_counts.index, ok_counts.values, color=plt.cm.viridis([0.3,0.6,0.9]))
    ax.bar_label(bars, padding=3)
    ax.set_title("Successfully Parsed Files per Difficulty")
    ax.set_ylim(0, ok_counts.max()*1.15 or 1)
    out = OUTPUT_PLOT_DIR / "bar_parsed_ok_by_difficulty.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✅ saved {out}")


    # -----------------------------------------------------------------
    # 1.  Build file‑level helper columns
    ok_df = code_df[code_df["parsing_error"].isna()].copy()
    ok_df["bad_rate_file"] = ok_df["bad_exception_rate"].fillna(0)      # 0 if no try blocks
    ok_df["has_try"]       = ok_df["exception_handling_blocks"] > 0

    plot_statistical_analysis(ok_df)
    print("🎉  All plots generated")
    
    # -----------------------------------------------------------------
    # 2.  Aggregate per model × difficulty
    summary = (
        ok_df
        .groupby(["model", "prompt_difficulty"])
        .agg(
            seeds               = ("file", "size"),
            seeds_with_try      = ("has_try", "sum"),
            mean_bad_rate_all   = ("bad_rate_file", "mean"),   # includes zero‑try seeds
            mean_bad_rate_non0  = (
                "bad_rate_file",
                lambda s: s.loc[ok_df.loc[s.index, "has_try"]].mean()    # 👈 only seeds_with_try
            ),
            total_excepts       = ("exception_handling_blocks", "sum"),
            bad_excepts         = ("bad_exception_blocks",      "sum"),
            mean_loc            = ("loc",                       "mean"),
        )
        .round({"mean_bad_rate_all": 2, "mean_bad_rate_non0": 2})
        .reset_index()
    )
    
    summary.to_csv(OUTPUT_PLOT_DIR / "summary_by_model.csv", index=False)
    print("📝  wrote summary_by_model.csv")
    
    # -----------------------------------------------------------------
    # 3.  Collapse across models to get difficulty‑level view
    diff_summary = (
        summary
        .groupby("prompt_difficulty")
        .agg(
            seeds               = ("seeds",          "sum"),
            seeds_with_try      = ("seeds_with_try", "sum"),
            mean_bad_rate_all   = ("mean_bad_rate_all",  "mean"),
            mean_bad_rate_non0  = ("mean_bad_rate_non0", "mean"),
            total_excepts       = ("total_excepts",   "sum"),
            bad_excepts         = ("bad_excepts",     "sum"),
            mean_loc            = ("mean_loc",        "mean"),
        )
        .round(2)
        .reset_index()
    )
    
    diff_summary.to_csv(OUTPUT_PLOT_DIR / "summary_by_difficulty.csv", index=False)
    print("📝  wrote summary_by_difficulty.csv")
    
    # Optional console preview
    print("\n=== Mean bad‑rate (incl. zero‑try files) ===")
    print(summary.pivot(index="model", columns="prompt_difficulty", values="mean_bad_rate_all").fillna("—"))
    
    print("\n=== Mean bad‑rate (only seeds with try/except) ===")
    print(summary.pivot(index="model", columns="prompt_difficulty", values="mean_bad_rate_non0").fillna("—"))
    
        
    

if __name__ == "__main__":
    main()

