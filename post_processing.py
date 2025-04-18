"""
post_processing.py
==================
Scan every prompt directory inside --root, load llm_code_metrics.csv files
built by `process_files.py`, aggregate, and create the multi‑panel plots
identical to the original notebook implementation.
"""

# ---------- imports ---------------------------------------------------------
import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
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
OUTPUT_PLOT_DIR     = Path("plots_grid_refactored")
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
        pivots["ber"], title="Bad Exception Rate (BER)",
        cmap="Reds", vmin=0.0, vmax=1.0, cbar_label="Rate (0–1)",
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

    print("🎉  All plots generated")

if __name__ == "__main__":
    main()

