"""
process_files.py
================
Walk the directory tree for ONE prompt folder and write
   llm_response_metrics.csv
   llm_code_metrics.csv
in that folder.

> python -m llm_metrics.process_files --base-dir ./propagation_prompt
"""

import argparse
import csv
import sys
from pathlib import Path

import silent_killers.metrics_definitions as md  # local import

MODELS = [
    'gemini_2.0', 'gemini_2.5', "chatgpt40", "claude_3.7",
    "claude_3.7_thinking", "deepseek_R1", "chatgpt03minihigh", "deepseek_V3"
]

def _collect_metrics(path: Path, pattern: str, fn):
    """
    Helper that yields dict(model, file, path, **metric_values)
    """
    for model in MODELS:
        model_dir = path / model
        if not model_dir.is_dir():
            continue
        for fname in sorted(model_dir.glob(pattern)):
            raw = fname.read_text(encoding="utf-8", errors="ignore")
            metrics = fn(raw)
            row = {"model": model, "file": fname.name, "path": str(fname)}
            row |= {m.name: m.value for m in metrics}
            yield row

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Prompt directory that contains the model sub‑folders")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    if not base.is_dir():
        sys.exit(f"❌  {base} is not a directory")

    # --- responses -----------------------------------------------------------
    resp_rows = list(_collect_metrics(base, "response_*.txt", md.response_metrics))
    if resp_rows:
        _write_csv(base / "llm_response_metrics.csv", resp_rows)

    # --- code ---------------------------------------------------------------
    code_rows = list(_collect_metrics(base, "code_*.py",       md.code_metrics))
    if code_rows:
        _write_csv(base / "llm_code_metrics.csv", code_rows)

    print("✅  Metrics written")

def _write_csv(path: Path, rows: list[dict]):
    # Collect every key that appears in *any* row
    field_set = {k for row in rows for k in row.keys()}
    fieldnames = sorted(field_set)          # stable order is nice but not required

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
                                fieldnames=fieldnames,
                                extrasaction="ignore")  # ignore truly unexpected keys
        writer.writeheader()
        writer.writerows(rows)               # all rows now conform

if __name__ == "__main__":
    main()

