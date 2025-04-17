def worker_init(output_dir):
    """Initialize per-worker temporary HDF5 file with robust error handling"""
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
    
    # Initialize empty HDF5 file with verification
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Create file and immediately verify it
            with h5py.File(worker_file, 'w') as f:
                f.create_dataset('init', data=np.array([1]))
                f.flush()  # Force write to disk
                
            # Verify the file can be reopened
            with h5py.File(worker_file, 'r') as f:
                if 'init' not in f:
                    raise RuntimeError("File verification failed")
            
            logger.info(f"Worker {worker_id}: Successfully initialized HDF5 file")
            return
            
        except Exception as e:
            logger.error(f"Worker {worker_id}: Attempt {attempt + 1} failed to create HDF5 file: {str(e)}")
            if os.path.exists(worker_file):
                try:
                    os.remove(worker_file)
                except:
                    pass
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed to initialize worker file after {max_attempts} attempts")
            time.sleep(1)  # Wait before retrying

    # Force process affinity to separate cores (Linux only)
    try:
        import psutil
        process = psutil.Process()
        worker_idx = int(current_process().name.split('-')[-1]) if '-' in current_process().name else 0
        process.cpu_affinity([worker_idx % os.cpu_count()])
        logger.info(f"Worker {worker_id} assigned to CPU {worker_idx % os.cpu_count()}")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: CPU affinity setup failed: {str(e)}")


def merge_worker_files(output_file, temp_dir, completed_configs):
    """Safely merge worker files into final output with validation"""
    import h5py
    import os
    import glob
    
    worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
    if not worker_files:
        logger.warning("No worker files found to merge")
        return
    
    # First pass: validate all worker files
    valid_worker_files = []
    corrupted_files = []
    
    for worker_file in worker_files:
        try:
            # Skip empty files
            if os.path.getsize(worker_file) == 0:
                corrupted_files.append(worker_file)
                continue
                
            # Test opening the file
            with h5py.File(worker_file, 'r') as h5_in:
                # Verify basic structure
                if 'init' not in h5_in:
                    corrupted_files.append(worker_file)
                    continue
                    
                # Count valid configurations
                valid_configs = 0
                for key in h5_in:
                    if key != 'init' and isinstance(h5_in[key], h5py.Group):
                        if 'layout' in h5_in[key] and 'aep' in h5_in[key]:
                            valid_configs += 1
                
                if valid_configs > 0:
                    valid_worker_files.append(worker_file)
                else:
                    corrupted_files.append(worker_file)
                    
        except Exception as e:
            logger.error(f"Corrupted worker file detected: {worker_file} - {str(e)}")
            corrupted_files.append(worker_file)
    
    # Log validation results
    logger.info(f"Found {len(valid_worker_files)} valid worker files with data")
    if corrupted_files:
        logger.warning(f"Found {len(corrupted_files)} corrupted/empty worker files")
    
    # Second pass: merge valid files
    mode = 'a' if os.path.exists(output_file) else 'w'
    success_count = len(completed_configs)
    
    with h5py.File(output_file, mode) as h5_out:
        # Add/update metadata
        h5_out.attrs['grid_size'] = args.grid_size
        h5_out.attrs['random_pct'] = args.random_pct
        
        for worker_file in valid_worker_files:
            try:
                with h5py.File(worker_file, 'r') as h5_in:
                    for key in h5_in:
                        if key != 'init' and key not in h5_out:
                            try:
                                h5_out.copy(h5_in[key], key)
                                success_count += 1
                            except Exception as e:
                                logger.error(f"Failed to copy group {key} from {worker_file}: {str(e)}")
                
                # Clean up worker file after successful merge
                os.remove(worker_file)
                
            except Exception as e:
                logger.error(f"Failed to merge {worker_file}: {str(e)}")
                # Leave corrupted files for manual inspection
        
        # Update success counts
        h5_out.attrs['total_configs'] = len(configs) + len(completed_configs)
        h5_out.attrs['successful_configs'] = success_count
    
    # Clean up corrupted files
    for corrupted_file in corrupted_files:
        try:
            os.remove(corrupted_file)
        except:
            pass
    
    logger.info(f"Merged {success_count - len(completed_configs)} new configurations into {output_file}")
