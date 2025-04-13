# ... (previous code in main function) ...

    failed_worker_files = []
    with h5py.File(args.output, mode) as h5_out:
        # Add metadata
        h5_out.attrs['grid_size'] = args.grid_size
        h5_out.attrs['random_pct'] = args.random_pct
        # Adjust total/successful counts later if files fail
        h5_out.attrs['total_configs'] = len(configs) + len(completed_configs)
        # Initial successful count, might decrease if worker files are bad
        h5_out.attrs['successful_configs'] = success_count + len(completed_configs)

        # Copy data from worker files
        worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
        logger.info(f"Attempting to merge {len(worker_files)} worker files...")

        for worker_file in worker_files:
            try:
                # Check file size first, h5py might hang on empty files
                if os.path.getsize(worker_file) == 0:
                    logger.warning(f"Skipping empty worker file: {worker_file}")
                    failed_worker_files.append(worker_file)
                    continue # Skip to the next file

                with h5py.File(worker_file, 'r') as h5_in:
                    # Optional: Add a check for the 'init' dataset if needed
                    # if 'init' not in h5_in:
                    #     logger.warning(f"Worker file {worker_file} missing 'init' dataset, might be incomplete. Skipping.")
                    #     failed_worker_files.append(worker_file)
                    #     continue

                    copied_keys = 0
                    for key in h5_in:
                        if key != 'init': # Skip 'init' dataset
                           if key not in h5_out:
                                h5_out.copy(h5_in[key], key)
                                copied_keys += 1
                           else:
                               # This case might happen if hot-start merges results already present
                               logger.debug(f"Key {key} from {worker_file} already exists in {args.output}. Skipping copy.")
                    logger.debug(f"Copied {copied_keys} keys from {worker_file}")

            except OSError as e:
                # Catch the specific error or broader h5py errors
                logger.error(f"Failed to open or read worker file {worker_file}: {e}. Skipping this file.")
                failed_worker_files.append(worker_file)
            except Exception as e:
                # Catch any other unexpected errors during processing
                logger.error(f"Unexpected error processing worker file {worker_file}: {e}. Skipping this file.")
                failed_worker_files.append(worker_file)
            finally:
                 # Attempt to remove the worker file regardless of success/failure
                 try:
                     if os.path.exists(worker_file):
                        os.remove(worker_file)
                 except Exception as e:
                     logger.warning(f"Could not remove worker file {worker_file}: {e}")

        # Update final count if any worker files failed
        if failed_worker_files:
             # A more accurate success count requires knowing how many results were *expected*
             # from the failed files. This is complex. We'll adjust based on the initial 'success_count'
             # assuming each failed file corresponds to at least one failed task reported earlier.
             # For simplicity, we just log the number of failed files.
             logger.warning(f"Failed to merge data from {len(failed_worker_files)} worker files: {failed_worker_files}")
             # You could potentially adjust h5_out.attrs['successful_configs'] here if you
             # can determine how many configurations were lost.


    # Clean up temp directory (only if it's empty)
    try:
        if not os.listdir(temp_dir): # Check if directory is empty
             os.rmdir(temp_dir)
        else:
             logger.warning(f"Temp directory {temp_dir} not empty after merge attempt. Manual cleanup might be needed.")
    except Exception as e:
        logger.warning(f"Could not remove temp directory {temp_dir}: {e}")

    logger.info(f"Results saved to {args.output}. {len(failed_worker_files)} worker files could not be merged.")

    # Optionally, exit with a non-zero status code if files failed
    if failed_worker_files:
        sys.exit(1) # Signal that the process finished with errors


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
