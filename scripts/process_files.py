# scripts/process_files.py
"""
process_files.py
================
Walk the directory tree for ONE prompt folder, find all 'response_*.txt'
files, and generate two corresponding metric CSVs:
    llm_response_metrics.csv  (naive regex metrics on raw text)
    llm_code_metrics.csv      (AST metrics on all found code blocks)
"""

import argparse
import csv
import sys
import re
from pathlib import Path
from collections import defaultdict

import silent_killers.metrics_definitions as md

# Only match explicitly-tagged Python code blocks (```python or ```py).
# Bare ``` blocks are excluded to avoid capturing shell, YAML, tracebacks, etc.
RE_CODE_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.DOTALL)


def process_response_file(path: Path, strict: bool = False):
    """
    Processes a single response file, returning a dictionary for response
    metrics and another for aggregated code metrics.

    Always returns a code_metrics dict (even when no code blocks are found),
    so that every response file gets a row in the CSV.
    """
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    # 1. Calculate response-level metrics (naive text search)
    response_metrics = {m.name: m.value for m in md.response_metrics(raw_text)}

    # 2. Find all Python code blocks and calculate aggregated AST metrics
    code_blocks = RE_CODE_BLOCK.findall(raw_text)

    # If no code blocks are found, return a row with zeros so the file
    # appears in the CSV (distinguishable from "API call never made").
    if not code_blocks:
        return response_metrics, {
            "code_block_count": 0,
            "loc": 0,
            "exception_handling_blocks": 0,
            "bad_exception_blocks": 0,
            "bad_exception_rate": 0.0,
            "pass_exception_blocks": 0,
            "total_pass_statements": 0,
            "uses_traceback": False,
            "parsing_error": "",
        }

    # Use defaultdict to easily sum up metrics from multiple code blocks
    aggregated = defaultdict(float)
    parsing_errors = []
    parsed_block_count = 0

    for i, code in enumerate(code_blocks):
        # Use the public code_metrics() API so strict mode is respected
        block_metrics = {m.name: m.value for m in md.code_metrics(code, strict=strict)}

        if block_metrics.get("parsing_error"):
            parsing_errors.append(f"Block {i+1}: {block_metrics['parsing_error']}")
            continue

        parsed_block_count += 1
        aggregated["loc"] += block_metrics["loc"]
        aggregated["exception_handling_blocks"] += block_metrics["exception_handling_blocks"]
        aggregated["bad_exception_blocks"] += block_metrics["bad_exception_blocks"]
        aggregated["pass_exception_blocks"] += block_metrics["pass_exception_blocks"]
        aggregated["total_pass_statements"] += block_metrics["total_pass_statements"]
        if block_metrics.get("uses_traceback"):
            aggregated["uses_traceback"] = 1

    # Finalize aggregated metrics
    total_excepts = aggregated["exception_handling_blocks"]
    bad_excepts = aggregated["bad_exception_blocks"]

    code_result = dict(aggregated)
    code_result["parsing_error"] = "; ".join(parsing_errors) if parsing_errors else ""
    code_result["bad_exception_rate"] = round(bad_excepts / total_excepts, 2) if total_excepts else 0.0
    code_result["code_block_count"] = len(code_blocks)

    return response_metrics, code_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Prompt directory that contains the model sub-folders")
    ap.add_argument("--strict", action="store_true",
                    help="Flag ANY exception handler without re-raise as bad")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    if not base.is_dir():
        sys.exit(f"❌  {base} is not a directory")

    response_rows = []
    code_rows = []

    # Iterate through all models and their response files
    model_dirs = [d for d in base.iterdir() if d.is_dir()]
    for model_dir in model_dirs:
        for fname in sorted(model_dir.glob("response_*.txt")):
            resp_metrics, code_metrics_dict = process_response_file(
                fname, strict=args.strict
            )

            # Create the row structure with metadata
            base_row = {"model": model_dir.name, "file": fname.name, "path": str(fname)}

            if resp_metrics:
                response_rows.append(base_row | resp_metrics)
            # Always write a code row (even for zero-block responses)
            code_rows.append(base_row | code_metrics_dict)

    # --- Write CSVs ---
    if response_rows:
        _write_csv(base / "llm_response_metrics.csv", response_rows)
    if code_rows:
        _write_csv(base / "llm_code_metrics.csv", code_rows)

    print(f"✅  Metrics written (strict={args.strict})")

def _write_csv(path: Path, rows: list[dict]):
    # Helper to write list of dictionaries to a CSV file
    field_set = {k for row in rows for k in row.keys()}
    fieldnames = sorted(list(field_set))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
