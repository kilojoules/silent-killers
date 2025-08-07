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

# This regex is used to find Python code blocks in the response text.
RE_CODE_BLOCK = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)

def process_response_file(path: Path):
    """
    Processes a single response file, returning a dictionary for response
    metrics and another for aggregated code metrics.
    """
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    
    # 1. Calculate response-level metrics (naive text search)
    response_metrics = {m.name: m.value for m in md.response_metrics(raw_text)}

    # 2. Find all Python code blocks and calculate aggregated AST metrics
    code_blocks = RE_CODE_BLOCK.findall(raw_text)
    
    # If no code blocks are found, return empty code metrics
    if not code_blocks:
        return response_metrics, {}

    # Use defaultdict to easily sum up metrics from multiple code blocks
    aggregated_metrics = defaultdict(float)
    parsing_errors = []

    for i, code in enumerate(code_blocks):
        try:
            # Run the robust AST analysis on the code block
            tree = md.ast.parse(code)
            visitor = md._CodeMetricsVisitor()
            visitor.visit(tree)
            
            # Aggregate the key metrics by summing them
            aggregated_metrics["loc"] += len(code.splitlines())
            aggregated_metrics["exception_handling_blocks"] += visitor.total_excepts
            aggregated_metrics["bad_exception_blocks"] += visitor.bad_excepts
            aggregated_metrics["pass_exception_blocks"] += visitor.pass_exception_blocks
            aggregated_metrics["total_pass_statements"] += visitor.total_pass_statements
            if visitor.uses_traceback:
                 aggregated_metrics["uses_traceback"] = 1 # Mark as true if used in any block
        
        except SyntaxError as e:
            parsing_errors.append(f"Block {i+1}: {e}")

    # Finalize aggregated metrics
    total_excepts = aggregated_metrics["exception_handling_blocks"]
    bad_excepts = aggregated_metrics["bad_exception_blocks"]
    
    aggregated_metrics["parsing_error"] = "; ".join(parsing_errors) if parsing_errors else ""
    aggregated_metrics["bad_exception_rate"] = round(bad_excepts / total_excepts, 2) if total_excepts else 0.0
    aggregated_metrics['code_block_count'] = len(code_blocks)

    return response_metrics, dict(aggregated_metrics)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Prompt directory that contains the model sub-folders")
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
            resp_metrics, code_metrics = process_response_file(fname)
            
            # Create the row structure with metadata
            base_row = {"model": model_dir.name, "file": fname.name, "path": str(fname)}
            
            if resp_metrics:
                response_rows.append(base_row | resp_metrics)
            if code_metrics:
                code_rows.append(base_row | code_metrics)

    # --- Write CSVs ---
    if response_rows:
        _write_csv(base / "llm_response_metrics.csv", response_rows)
    if code_rows:
        _write_csv(base / "llm_code_metrics.csv", code_rows)

    print("✅  Metrics written")

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
