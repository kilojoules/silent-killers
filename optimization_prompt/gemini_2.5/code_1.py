import h5py
import os
import glob
import logging # Make sure logger is available

logger = logging.getLogger(__name__) # Assuming logger is set up as in the script

# ... (inside main function)

    with h5py.File(args.output, mode) as h5_out:
        # ... (add metadata attributes) ...

        # Copy data from worker files
        worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
        invalid_files_found = False # Flag to track if any invalid files are encountered

        for worker_file in worker_files:
            # --- Start Modification ---
            is_valid = False
            try:
                # Basic check: Ensure file is not empty
                if os.path.getsize(worker_file) == 0:
                    logger.error(f"Worker file {worker_file} is empty (0 bytes). Skipping.")
                    invalid_files_found = True
                    continue # Skip this file

                # Optional but recommended: Check if HDF5 library recognizes it
                if not h5py.is_hdf5(worker_file):
                   logger.error(f"Worker file {worker_file} is not a valid HDF5 file according to h5py.is_hdf5(). Skipping.")
                   invalid_files_found = True
                   continue # Skip this file

                # Now, try opening it (still wrap in try-except for safety)
                with h5py.File(worker_file, 'r') as h5_in:
                    # Check for the 'init' dataset added during worker initialization
                    if 'init' not in h5_in:
                         logger.warning(f"Worker file {worker_file} seems valid but lacks the 'init' dataset. Proceeding cautiously.")
                         # Decide if you want to treat this as an error or just a warning

                    # Copy data (excluding 'init' and avoiding duplicates)
                    for key in h5_in:
                        if key != 'init' and key not in h5_out:
                            h5_out.copy(h5_in[key], key)
                    is_valid = True # Successfully processed

            except OSError as e:
                 logger.error(f"OSError opening worker file {worker_file}: {e}. Skipping.")
                 invalid_files_found = True
            except Exception as e: # Catch other potential h5py errors
                 logger.error(f"Unexpected error processing worker file {worker_file}: {e}. Skipping.")
                 invalid_files_found = True
            finally:
                 # Clean up worker file only if it was successfully processed or deemed invalid
                 if is_valid or invalid_files_found: # Modify this condition based on desired cleanup
                    try:
                        os.remove(worker_file)
                    except OSError as remove_err:
                         logger.error(f"Failed to remove worker file {worker_file}: {remove_err}")
            # --- End Modification ---

        # --- Add Check after Loop ---
        if invalid_files_found:
             logger.error("One or more worker files were invalid or could not be processed.")
             # Option 1: Exit with an error status if any failure is critical
             print("Error: Failed to merge results due to invalid worker files. Check logs.", file=sys.stderr)
             sys.exit(1)
             # Option 2: Continue if partial results are acceptable, maybe adjust final metadata
             # h5_out.attrs['merge_status'] = 'Incomplete due to worker file errors'


    # Clean up temp directory (only if no errors caused an early exit)
    try:
        # Check if directory is empty before removing
        if not os.listdir(temp_dir):
             os.rmdir(temp_dir)
        else:
             logger.warning(f"Temporary directory {temp_dir} not empty after processing. Manual cleanup might be needed.")
    except OSError as rmdir_err:
        logger.error(f"Failed to remove temporary directory {temp_dir}: {rmdir_err}")

    logger.info(f"Results saved to {args.output}")

# ... (rest of the script)
