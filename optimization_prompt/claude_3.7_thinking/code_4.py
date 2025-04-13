def merge_worker_files(output_file, temp_dir, metadata=None, force=False):
    """
    Merge worker files into the main output file
    
    Args:
        output_file: Path to output HDF5 file
        temp_dir: Directory containing worker files
        metadata: Optional dictionary of metadata to add to the output file
        force: Whether to force creation of a new file (overwrite existing)
        
    Returns:
        Tuple of (success_count, skipped_count, failed_count)
    """
    logger.info(f"Merging worker files from {temp_dir} into {output_file}")
    
    # Decide file mode
    if force or not os.path.exists(output_file):
        mode = 'w'
        logger.info("Creating new output file")
    else:
        mode = 'a'
        logger.info("Appending to existing output file")
        
    # Prepare worker file list
    worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
    logger.info(f"Found {len(worker_files)} worker files")
    
    # Track statistics
    success_count = 0
    skipped_count = 0
    failed_count = 0
    invalid_files = []
    
    # Create output file and merge
    try:
        with h5py.File(output_file, mode) as h5_out:
            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    h5_out.attrs[key] = value
                    
            # Process each worker file
            for worker_file in worker_files:
                # Skip invalid files
                if not validate_worker_file(worker_file):
                    invalid_files.append(worker_file)
                    failed_count += 1
                    continue
                    
                try:
                    with h5py.File(worker_file, 'r') as h5_in:
                        keys_copied = 0
                        for key in h5_in:
                            if key != 'init':  # Skip init dataset
                                if key in h5_out:
                                    skipped_count += 1
                                    continue
                                    
                                h5_out.copy(h5_in[key], key)
                                keys_copied += 1
                                success_count += 1
                                
                        logger.info(f"Copied {keys_copied} datasets from {worker_file}")
                        
                    # Remove successfully processed file
                    os.remove(worker_file)
                except Exception as e:
                    logger.error(f"Error processing worker file {worker_file}: {str(e)}")
                    invalid_files.append(worker_file)
                    failed_count += 1
    except Exception as e:
        logger.error(f"Error creating or writing to output file: {str(e)}")
        return 0, 0, len(worker_files)
        
    # Handle invalid files
    if invalid_files:
        invalid_dir = os.path.join(temp_dir, "invalid")
        os.makedirs(invalid_dir, exist_ok=True)
        
        for file in invalid_files:
            try:
                import shutil
                shutil.move(file, os.path.join(invalid_dir, os.path.basename(file)))
                logger.warning(f"Moved invalid file to {invalid_dir}: {os.path.basename(file)}")
            except Exception as e:
                logger.error(f"Failed to move invalid file {file}: {str(e)}")
                
    logger.info(f"Merge complete: {success_count} datasets copied, {skipped_count} skipped, {failed_count} failed")
    return success_count, skipped_count, failed_count


def recover_from_previous_run(output_file, temp_dir):
    """
    Attempt to recover data from a previous run that was interrupted
    
    Args:
        output_file: Path to output HDF5 file
        temp_dir: Directory containing temporary worker files
        
    Returns:
        Boolean indicating if recovery was attempted
    """
    worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
    
    if not worker_files:
        logger.info("No worker files found for recovery")
        return False
        
    logger.warning(f"Found {len(worker_files)} worker files from previous run")
    logger.warning("Attempting to recover data...")
    
    # Get existing configurations from output file
    completed_configs = scan_existing_results(output_file)
    
    # Create recovery metadata
    metadata = {
        'recovery_timestamp': time.time(),
        'recovery_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'recovered_from': len(worker_files),
        'existing_configs': len(completed_configs)
    }
    
    # Merge the worker files
    success, skipped, failed = merge_worker_files(output_file, temp_dir, metadata)
    
    logger.warning(f"Recovery summary: {success} datasets recovered, {skipped} skipped, {failed} failed")
    
    return True


def graceful_shutdown(pool, temp_dir, output_file, configs, completed_configs, results, args):
    """Handle keyboard interrupt by saving partial results"""
    logger.warning("Keyboard interrupt detected. Attempting to save partial results...")
    
    try:
        # Close the pool without waiting for ongoing tasks
        pool.terminate()
        pool.join()
        
        # Mark the remaining tasks as failures in our results list
        remaining = len(configs) - len(results)
        results.extend([False] * remaining)
        
        # Calculate successful optimizations
        success_count = sum(1 for r in results if r)
        logger.info(f"Successfully completed {success_count}/{len(configs)} optimizations")
        
        # Merge the results we have so far
        metadata = {
            'grid_size': args.grid_size,
            'random_pct': args.random_pct,
            'total_configs': len(configs) + len(completed_configs),
            'successful_configs': success_count + len(completed_configs),
            'interrupted': True,
            'interrupt_timestamp': time.time()
        }
        
        merge_worker_files(output_file, temp_dir, metadata)
        
        logger.info("Partial results saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {str(e)}")
        return False


def main():
    """Main function to run the optimization"""
    parser = argparse.ArgumentParser(description="Parallel Wind Farm Layout Optimizer")
    parser.add_argument("--seeds", type=int, required=True, help="Number of random seeds")
    parser.add_argument("--output", type=str, default="layouts.h5", help="Output file")
    parser.add_argument("--processes", type=int, default=os.cpu_count(), help="Parallel workers")
    parser.add_argument("--grid-size", type=int, default=18, help="Smart start grid size")
    parser.add_argument("--random-pct", type=int, default=30, help="Smart start random percentage")
    parser.add_argument("--chunk-size", type=int, default=1, help="Chunk size for parallel processing")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Progress bar update interval in seconds")
    parser.add_argument("--no-hot-start", action="store_true", help="Disable hot start (ignore existing results)")
    parser.add_argument("--recover", action="store_true", help="Attempt to recover from previous interrupted run")
    args = parser.parse_args()

    # Setup temporary storage
    temp_dir = f"{args.output}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Check for recovery scenario
    if args.recover and os.path.exists(temp_dir):
        if recover_from_previous_run(args.output, temp_dir):
            logger.info("Recovery complete")
            if len(glob.glob(os.path.join(temp_dir, "worker_*.h5"))) == 0:
                try:
                    os.rmdir(temp_dir)
                    logger.info(f"Removed empty temporary directory: {temp_dir}")
                except:
                    pass
            return

    # Load farm boundaries and proceed with normal execution
    # ... [rest of the main function as before until the Pool creation] ...

    # Create pool with worker-specific files
    pool = None
    try:
        pool = Pool(
            processes=args.processes,
            initializer=worker_init,
            initargs=(temp_dir,),
            maxtasksperchild=10
        )
        
        # ... [remainder of the function with pool operations] ...
    
    except KeyboardInterrupt:
        if pool:
            graceful_shutdown(pool, temp_dir, args.output, configs, completed_configs, results, args)
        sys.exit(1)
    finally:
        if pool:
            pool.close()
            pool.join()
    
    # Final merge of worker files
    success, skipped, failed = merge_worker_files(
        args.output, 
        temp_dir, 
        {
            'grid_size': args.grid_size,
            'random_pct': args.random_pct,
            'total_configs': len(configs) + len(completed_configs),
            'successful_configs': success_count + len(completed_configs)
        }
    )
    
    # Clean up temp directory if empty
    try:
        os.rmdir(temp_dir)
        logger.info(f"Removed temporary directory: {temp_dir}")
    except OSError:
        logger.info(f"Temporary directory {temp_dir} is not empty and was preserved")
    
    logger.info(f"Results saved to {args.output}")
