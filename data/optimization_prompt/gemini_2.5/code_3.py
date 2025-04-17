# In main() function
# ... (previous code) ...

success_count = sum(1 for r in results if r) # From worker return values
logger.info(f"Reported successes from workers: {success_count}/{len(configs)}")

actually_merged_count = 0
failed_files = []
worker_files_found = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
logger.info(f"Found {len(worker_files_found)} worker files to attempt merging.")

# Determine the correct mode ('w' or 'a') based on args.no_hot_start and file existence
mode = 'w'
if not args.no_hot_start and os.path.exists(args.output):
    mode = 'a'
    logger.info(f"Output file {args.output} exists. Using append mode.")
else:
     logger.info(f"Creating new output file {args.output} or overwriting.")

try: # Wrap the entire output file opening for safety
    with h5py.File(args.output, mode) as h5_out:
        # Add or update metadata - do this first
        if mode == 'w' or 'total_configs' not in h5_out.attrs: # Initialize if new file or attribute missing
            h5_out.attrs['grid_size'] = args.grid_size
            h5_out.attrs['random_pct'] = args.random_pct
            h5_out.attrs['total_configs'] = len(configs) + len(completed_configs) # Expected total
            h5_out.attrs['successful_merged_configs'] = len(completed_configs) # Start with hot-start count
            h5_out.attrs['failed_worker_files_count'] = 0
        # We will update counts after the loop

        # Copy data from worker files
        for worker_file in worker_files_found:
            try:
                # Check 1: Skip zero-byte files
                if os.path.getsize(worker_file) == 0:
                    logger.warning(f"Skipping 0-byte worker file (invalid): {worker_file}")
                    failed_files.append(worker_file + " (0 bytes)")
                    continue # Move to the next file

                # Check 2: Try opening the file
                with h5py.File(worker_file, 'r') as h5_in:
                    copied_keys_count = 0
                    for key in h5_in:
                        if key == 'init': # Skip the initialization dataset
                            continue
                        if key in h5_out:
                            logger.warning(f"Key '{key}' from {worker_file} already exists in {args.output}. Skipping.")
                            continue

                        # Attempt the copy
                        h5_out.copy(h5_in[key], key)
                        copied_keys_count += 1

                    # Update overall count if we copied something meaningful
                    if copied_keys_count > 0:
                       # Assuming one group per config/worker file run
                       actually_merged_count += 1 
                    else:
                        logger.warning(f"Worker file {worker_file} was opened but contained no new data groups to copy.")
                        failed_files.append(worker_file + " (no data)")


                # If open and copy succeeded (or skipped existing), remove the worker file
                try:
                    os.remove(worker_file)
                except OSError as e:
                     logger.error(f"Could not remove processed worker file {worker_file}: {e}")

            except (OSError, h5py.H5Error) as e:
                # Catch errors specifically from h5py or OS during file open/read
                logger.error(f"Failed to open or read potentially invalid worker file {worker_file}: {e}")
                failed_files.append(worker_file + f" ({type(e).__name__})")
                # DO NOT remove the failed file, leave it for inspection
            except Exception as e: # Catch any other unexpected errors during processing
                logger.error(f"An unexpected error occurred while processing {worker_file}: {e}")
                failed_files.append(worker_file + f" (Unexpected {type(e).__name__})")
                # DO NOT remove the failed file

        # Update final counts in attributes after loop finishes
        final_merged_count = len(completed_configs) + actually_merged_count
        h5_out.attrs['successful_merged_configs'] = final_merged_count
        h5_out.attrs['failed_worker_files_count'] = len(failed_files)
        logger.info(f"Merging complete. Total configurations successfully merged into {args.output}: {final_merged_count}")

except Exception as e:
     logger.error(f"FATAL: Could not open or write to the main output file {args.output}: {e}")
     # Decide how to handle this - maybe exit?
     sys.exit(f"FATAL error accessing output file {args.output}. Exiting.")


# Report and handle failures
if failed_files:
    logger.error(f"Merge completed, but {len(failed_files)} worker files encountered errors and were skipped:")
    for f in failed_files:
        logger.error(f"  - {f}")
    logger.warning("The main output file may be missing data from these failed workers.")
    # Consider exiting with a non-zero status code if any failures occurred
    # sys.exit(1) # Uncomment to signal failure
else:
    logger.info(f"All {len(worker_files_found)} found worker files were processed successfully.")

# Clean up temp directory (optional: only remove if empty or just contains failed files)
try:
    remaining_files = os.listdir(temp_dir)
    if not remaining_files:
        logger.info(f"Removing empty temporary directory: {temp_dir}")
        os.rmdir(temp_dir)
    else:
        logger.warning(f"Temporary directory {temp_dir} not removed as it still contains files (likely failed ones): {remaining_files}")
except OSError as e:
    logger.error(f"Error during temporary directory cleanup {temp_dir}: {e}")

logger.info(f"Results saved to {args.output}")

# ... (rest of main) ...

# Optional: Exit with error code if files failed
if failed_files:
   sys.exit(f"Process finished, but {len(failed_files)} worker files failed to merge.")
