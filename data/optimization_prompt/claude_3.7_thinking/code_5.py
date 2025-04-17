def validate_hdf5_file(file_path):
    """
    Validate if a file is a valid HDF5 file
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        (bool, str): Tuple of (is_valid, error_message)
    """
    import h5py
    import os
    
    # Basic file checks
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if os.path.getsize(file_path) == 0:
        return False, f"File is empty (0 bytes): {file_path}"
        
    # Try to open the file to verify HDF5 format
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if it has any keys or just the 'init' dataset
            keys = list(f.keys())
            if not keys:
                return False, f"File has no datasets or groups: {file_path}"
            if len(keys) == 1 and keys[0] == 'init':
                return False, f"File only contains initialization data: {file_path}"
                
            # Verify that at least one farm configuration exists
            has_valid_data = False
            for key in keys:
                if key != 'init' and isinstance(f[key], h5py.Group):
                    if 'layout' in f[key] and 'aep' in f[key]:
                        has_valid_data = True
                        break
            
            if not has_valid_data:
                return False, f"File has no valid farm configurations: {file_path}"
                
        return True, ""
    except Exception as e:
        return False, f"Error validating HDF5 file: {str(e)}"


def merge_worker_files(output_file, temp_dir, grid_size, random_pct, configs, 
                       completed_configs, success_count, mode='a'):
    """
    Safely merge worker files into the final output file, handling corrupted files
    
    Args:
        output_file: Path to the output HDF5 file
        temp_dir: Directory containing worker files
        grid_size: Grid size parameter for metadata
        random_pct: Random percentage parameter for metadata
        configs: List of configurations processed
        completed_configs: Dictionary of previously completed configurations
        success_count: Count of successful optimizations
        mode: File open mode ('a' for append, 'w' for write)
        
    Returns:
        (int, int): Tuple of (merged_files_count, skipped_files_count)
    """
    import h5py
    import os
    import glob
    import logging
    
    logger = logging.getLogger(__name__)
    
    merged_count = 0
    skipped_count = 0
    failed_files = []
    
    # Create backup of existing output file if it exists and we're in write mode
    if mode == 'w' and os.path.exists(output_file):
        import shutil
        backup_file = f"{output_file}.backup"
        logger.info(f"Creating backup of existing output file: {backup_file}")
        try:
            shutil.copy2(output_file, backup_file)
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")
    
    try:
        with h5py.File(output_file, mode) as h5_out:
            # Add metadata
            h5_out.attrs['grid_size'] = grid_size
            h5_out.attrs['random_pct'] = random_pct
            h5_out.attrs['total_configs'] = len(configs) + len(completed_configs)
            h5_out.attrs['successful_configs'] = success_count + len(completed_configs)
            
            # Get list of worker files
            worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
            logger.info(f"Found {len(worker_files)} worker files to merge")
            
            # Track configurations that have been merged
            merged_configs = set()
            
            # Process each worker file
            for worker_file in worker_files:
                # Validate the file before attempting to open it
                is_valid, error_msg = validate_hdf5_file(worker_file)
                
                if not is_valid:
                    logger.warning(f"Skipping invalid worker file: {worker_file}. Reason: {error_msg}")
                    skipped_count += 1
                    failed_files.append((worker_file, error_msg))
                    continue
                
                try:
                    # Try to merge this worker's data
                    with h5py.File(worker_file, 'r') as h5_in:
                        # Copy each farm configuration if it doesn't already exist in output
                        for key in h5_in:
                            if key != 'init':  # Skip initialization dataset
                                if key not in h5_out and key not in merged_configs:
                                    try:
                                        h5_out.copy(h5_in[key], key)
                                        merged_configs.add(key)
                                        merged_count += 1
                                    except Exception as e:
                                        logger.warning(f"Error copying group {key} from {worker_file}: {str(e)}")
                                else:
                                    logger.debug(f"Skipping duplicate config {key} from {worker_file}")
                    
                    # Only remove files that were successfully processed
                    try:
                        os.remove(worker_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove worker file {worker_file}: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error processing worker file {worker_file}: {str(e)}")
                    skipped_count += 1
                    failed_files.append((worker_file, str(e)))
            
            logger.info(f"Merged {merged_count} configurations from worker files")
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} worker files due to errors")
                for failed_file, reason in failed_files:
                    logger.warning(f"  - {failed_file}: {reason}")
            
            return merged_count, skipped_count
            
    except Exception as e:
        logger.critical(f"Critical error during merge: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        return 0, len(worker_files)
