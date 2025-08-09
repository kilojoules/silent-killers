python scripts/process_files.py --base-dir data/calibration_prompt
python scripts/process_files.py --base-dir data/optimization_prompt
python scripts/process_files.py --base-dir data/propagation_prompt

python scripts/post_processing.py --root data
