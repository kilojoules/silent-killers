import os
import h5py
import glob
import logging
from multiprocessing import Pool

# ... (Existing code)

def is_valid_hdf5_file(filepath, min_size=1024):  # Example minimal size
    """Checks if a file is a valid HDF5 file."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < min_size:
        return False
    try:
        with h5py.File(filepath, 'r') as f:
            # Optionally, try to read a small piece of data
            # If the file is corrupted, this might raise an OSError
            if 'init' in f:
                f['init'][()]  
        return True
    except OSError:
        return False

def optimize_layout(config, farm_boundaries, grid_size=18, random_pct=30):
    # ... (Existing code)
    try:
        # ... (Optimization code)
        with h5py.File(worker_file, 'a') as f:
            # ... (Write data)
            pass
    except Exception as e:
        logger.error(f"Worker {current_process().pid}: Error writing to file: {e}")
        return False
    finally:
        gc.collect()
    return True

def main():
    # ... (Existing code)

    with Pool(processes=args.processes, initializer=worker_init, initargs=(temp_dir,)) as pool:
        # ... (Optimization execution)
        results = []
        for result in pool.imap_unordered(optimize_partial, configs):
            results.append(result)

    # Validate and merge
    with h5py.File(args.output, mode) as h5_out:
        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
            if not is_valid_hdf5_file(worker_file):
                logger.error(f"Invalid worker file: {worker_file}, aborting merge")
                pool.terminate()  # Terminate workers
                raise Exception(f"Invalid worker file: {worker_file}")
            try:
                with h5py.File(worker_file, 'r') as h5_in:
                    for key in h5_in:
                        if key != 'init' and key not in h5_out:
                            h5_out.copy(h5_in[key], key)
                os.remove(worker_file)
            except Exception as e:
                logger.error(f"Error merging worker file {worker_file}: {e}")
                pool.terminate()
                raise Exception(f"Error merging worker file {worker_file}: {e}")

    # ... (Cleanup)

if __name__ == "__main__":
    main()
