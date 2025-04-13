# Modify the worker_init function to add validation
def worker_init(output_dir):
    """Initialize per-worker temporary HDF5 file"""
    import os
    import logging
    import h5py
    import numpy as np
    from multiprocessing import current_process
    
    logger = logging.getLogger(__name__)
    worker_id = current_process().pid
    worker_file = os.path.join(output_dir, f"worker_{worker_id}.h5")
    
    # Store file path on worker process object
    current_process().worker_file = worker_file
    
    # Initialize empty HDF5 file with a test dataset to ensure it's valid
    try:
        with h5py.File(worker_file, 'w') as f:
            f.create_dataset('init', data=np.array([1]))
            # Flush to ensure data is written to disk
            f.flush()
            logger.info(f"Worker {worker_id}: Successfully initialized HDF5 file")
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to create HDF5 file: {str(e)}")
        # Re-raise to prevent worker from continuing with invalid file
        raise

    # Force process affinity to separate cores (Linux only)
    try:
        import psutil
        process = psutil.Process()
        worker_idx = int(current_process().name.split('-')[-1]) if '-' in current_process().name else 0
        process.cpu_affinity([worker_idx % os.cpu_count()])
        logger.info(f"Worker {worker_id} assigned to CPU {worker_idx % os.cpu_count()}")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: CPU affinity setup failed: {str(e)}")


# Add a function to validate HDF5 files
def is_valid_hdf5(filepath, min_size_bytes=800):
    """
    Check if a file is a valid HDF5 file and meets minimum size requirements
    
    Args:
        filepath: Path to the HDF5 file
        min_size_bytes: Minimum expected file size in bytes
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    import os
    import h5py
    
    # Check if file exists
    if not os.path.exists(filepath):
        return False
    
    # Check minimum file size
    if os.path.getsize(filepath) < min_size_bytes:
        return False
    
    # Try to open as HDF5
    try:
        with h5py.File(filepath, 'r') as _:
            pass
        return True
    except Exception:
        return False


# Modify the merging part of the main function
def main():
    # ... (rest of the code remains the same)
    
    # Merge worker files into final output
    if not args.no_hot_start and os.path.exists(args.output):
        # Use append mode to preserve existing results
        mode = 'a'
    else:
        # Use write mode for new file or when hot start is disabled
        mode = 'w'
    
    successful_merges = 0
    failed_merges = 0
    
    with h5py.File(args.output, mode) as h5_out:
        # Add metadata
        h5_out.attrs['grid_size'] = args.grid_size
        h5_out.attrs['random_pct'] = args.random_pct
        h5_out.attrs['total_configs'] = len(configs) + len(completed_configs)
        h5_out.attrs['successful_configs'] = success_count + len(completed_configs)
        
        # Copy data from worker files
        worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
        logger.info(f"Found {len(worker_files)} worker files to merge")
        
        for worker_file in worker_files:
            # Validate file before attempting to open
            if not is_valid_hdf5(worker_file):
                logger.warning(f"Skipping invalid or empty worker file: {worker_file}")
                failed_merges += 1
                # Remove invalid file
                try:
                    os.remove(worker_file)
                except Exception as e:
                    logger.error(f"Failed to remove invalid file {worker_file}: {str(e)}")
                continue
                
            try:
                with h5py.File(worker_file, 'r') as h5_in:
                    keys_copied = 0
                    for key in h5_in:
                        if key != 'init' and key not in h5_out:  # Skip 'init' dataset and avoid duplicates
                            h5_out.copy(h5_in[key], key)
                            keys_copied += 1
                    
                    logger.info(f"Merged {keys_copied} configurations from {worker_file}")
                    successful_merges += 1
                    
                # Clean up worker file after successful merge
                os.remove(worker_file)
            except Exception as e:
                logger.error(f"Error merging {worker_file}: {str(e)}")
                failed_merges += 1
                # Don't delete file on error - keep for potential recovery
    
    logger.info(f"Merging complete: {successful_merges} successful, {failed_merges} failed")
    
    # Report if there were failures
    if failed_merges > 0:
        print(f"WARNING: {failed_merges} worker files could not be merged. See logs for details.", 
              file=sys.stderr, flush=True)
        print(f"Temporary files preserved in {temp_dir} for recovery.", file=sys.stderr, flush=True)
    else:
        # Only remove temp directory if all merges were successful
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to remove temp directory: {str(e)}")
    
    logger.info(f"Results saved to {args.output}")
