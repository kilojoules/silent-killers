import h5py
import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = "layouts.h5_temp"
OUTPUT_FILE = "layouts.h5"

def safe_merge(temp_dir, output_file):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with h5py.File(output_file, mode) as h5_out:
        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
            try:
                if os.path.getsize(worker_file) < 1024:
                    logger.warning(f"Skipping invalid or empty file: {worker_file}")
                    continue
                with h5py.File(worker_file, 'r') as h5_in:
                    for key in h5_in:
                        if key != 'init' and key not in h5_out:
                            h5_out.copy(h5_in[key], key)
            except (OSError, IOError) as e:
                logger.warning(f"Could not read {worker_file}: {e}")
            finally:
                try:
                    os.remove(worker_file)
                except Exception as e:
                    logger.warning(f"Failed to remove {worker_file}: {e}")
    try:
        os.rmdir(temp_dir)
    except OSError as e:
        logger.warning(f"Could not remove temp dir {temp_dir}: {e}")
    logger.info(f"Safe merge complete. Output: {output_file}")

if __name__ == "__main__":
    safe_merge(TEMP_DIR, OUTPUT_FILE)

