#!/usr/bin/env bash
# vast_run_llama.sh — runs the Silent Killers Llama benchmark on Vast AI
#
# This script is executed ON the remote GPU instance. It:
#   1. Clones the repo (llama-benchmark branch)
#   2. Installs dependencies
#   3. Runs all 3 Llama models × 3 prompts × 20 seeds
#   4. Processes metrics
#   5. Pushes results back to the branch
set -euo pipefail

echo "=== Silent Killers Llama Benchmark ==="
echo "Started: $(date)"

# --- Setup ---
export HF_TOKEN="${HF_TOKEN}"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1

pip install -q torch transformers accelerate bitsandbytes \
    pandas numpy matplotlib pyyaml python-dotenv

cd /workspace
git clone --branch llama-benchmark https://x-access-token:${GH_TOKEN}@github.com/kilojoules/silent-killers.git
cd silent-killers
pip install -e . -q

# --- Run experiments for each model sequentially ---
# (models are loaded one at a time to manage VRAM)

echo ""
echo "=== Running Llama-3.2-1B-Instruct ==="
python scripts/run_experiments.py \
    --models-config models_llama.yaml \
    --model-alias llama_1b \
    --seeds 20

echo ""
echo "=== Running Llama-3.2-3B-Instruct ==="
python scripts/run_experiments.py \
    --models-config models_llama.yaml \
    --model-alias llama_3b \
    --seeds 20

echo ""
echo "=== Running Llama-3.1-8B-Instruct ==="
python scripts/run_experiments.py \
    --models-config models_llama.yaml \
    --model-alias llama_8b \
    --seeds 20

# --- Process metrics ---
echo ""
echo "=== Processing metrics ==="
for prompt_dir in data/calibration_prompt data/propagation_prompt data/optimization_prompt; do
    echo "Processing $prompt_dir ..."
    python scripts/process_files.py --base-dir "$prompt_dir"
done

# --- Print summary ---
echo ""
echo "=== Results Summary ==="
python -c "
import csv
from pathlib import Path

for prompt_dir in ['calibration_prompt', 'propagation_prompt', 'optimization_prompt']:
    csv_path = Path('data') / prompt_dir / 'llm_code_metrics.csv'
    if not csv_path.exists():
        print(f'{prompt_dir}: no metrics file')
        continue
    print(f'\n--- {prompt_dir} ---')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    llama_rows = [r for r in rows if r['model'].startswith('llama_')]
    for model in ['llama_1b', 'llama_3b', 'llama_8b']:
        model_rows = [r for r in llama_rows if r['model'] == model]
        if not model_rows:
            print(f'  {model}: no data')
            continue
        n = len(model_rows)
        bad_rates = [float(r.get('bad_exception_rate', 0)) for r in model_rows]
        n_with_exceptions = sum(1 for r in model_rows if float(r.get('exception_handling_blocks', 0)) > 0)
        n_bad = sum(1 for r in model_rows if float(r.get('bad_exception_rate', 0)) > 0)
        mean_bad = sum(bad_rates) / len(bad_rates) if bad_rates else 0
        print(f'  {model}: {n} responses, {n_with_exceptions} with exceptions, {n_bad} bad, mean_bad_rate={mean_bad:.2f}')
"

# --- Push results ---
echo ""
echo "=== Pushing results ==="
git config user.email "vast-ai@runner.local"
git config user.name "Vast AI Runner"
git add -A data/
git commit -m "Add Llama 1B/3B/8B benchmark results (20 seeds x 3 prompts)

Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct
Seeds: 20 per model per prompt
Prompts: calibration (easy), propagation (medium), optimization (hard)" || echo "Nothing to commit"
git push origin llama-benchmark

echo ""
echo "=== Done ==="
echo "Finished: $(date)"
