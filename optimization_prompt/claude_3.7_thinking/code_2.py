def main():
    """Main function to run the optimization"""
    # ... existing code ...

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
        
        # Copy data from worker files with error handling
        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
            # First check if file is valid (non-empty and readable)
            if os.path.getsize(worker_file) == 0:
                logger.warning(f"Skipping empty worker file: {worker_file}")
                failed_merges += 1
                # Remove the empty file
                try:
                    os.remove(worker_file)
                except OSError as e:
                    logger.error(f"Failed to remove empty file {worker_file}: {e}")
                continue
                
            try:
                with h5py.File(worker_file, 'r') as h5_in:
                    # Check if file has any valid data besides 'init'
                    has_valid_data = False
                    for key in h5_in:
                        if key != 'init':
                            has_valid_data = True
                            if key not in h5_out:  # Avoid duplicates
                                h5_out.copy(h5_in[key], key)
                                successful_merges += 1
                    
                    if not has_valid_data:
                        logger.warning(f"Worker file has no valid data: {worker_file}")
                        failed_merges += 1
                
                # Clean up worker file only if successfully processed
                os.remove(worker_file)
                
            except (OSError, IOError, RuntimeError) as e:
                # Handle corrupted HDF5 files
                logger.error(f"Failed to open worker file {worker_file}: {e}")
                failed_merges += 1
                
                # Rename the problematic file instead of removing it
                # This allows for later investigation if needed
                error_file = f"{worker_file}.error"
                try:
                    os.rename(worker_file, error_file)
                    logger.info(f"Renamed corrupt file to: {error_file}")
                except OSError as rename_err:
                    logger.error(f"Failed to rename corrupt file {worker_file}: {rename_err}")
    
    # Log merge statistics
    logger.info(f"Merged {successful_merges} datasets from worker files")
    if failed_merges > 0:
        logger.warning(f"Failed to merge {failed_merges} worker files")
    
    # Try to clean up temp directory, but don't fail if there are still files
    try:
        remaining_files = glob.glob(os.path.join(temp_dir, "*"))
        if not remaining_files:
            os.rmdir(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        else:
            logger.warning(f"Temporary directory not removed, contains {len(remaining_files)} files: {temp_dir}")
    except OSError as e:
        logger.error(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    logger.info(f"Results saved to {args.output}")
