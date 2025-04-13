# Original code [cite: 45]
#        # Copy data from worker files
#        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
#            with h5py.File(worker_file, 'r') as h5_in:
#                for key in h5_in:
#                    if key != 'init' and key not in h5_out:  # Skip 'init' dataset and avoid duplicates
#                        h5_out.copy(h5_in[key], key)
#            # Clean up worker file
#            os.remove(worker_file)

# Modified code for robustness
        successfully_merged_count = 0
        failed_files = []
        worker_files = glob.glob(os.path.join(temp_dir, "worker_*.h5"))
        logger.info(f"Found {len(worker_files)} worker files to merge.")

        for worker_file in worker_files:
            try:
                with h5py.File(worker_file, 'r') as h5_in:
                    copied_keys = 0
                    # Ensure the file is not empty and has groups other than 'init'
                    keys_to_copy = [k for k in h5_in if k != 'init']
                    if not keys_to_copy:
                         logger.warning(f"Worker file {worker_file} is empty or contains only 'init'. Skipping.")
                         failed_files.append(worker_file)
                         continue # Skip to the next file

                    for key in keys_to_copy:
                        if key not in h5_out:  # Avoid duplicates if merging into existing file
                            h5_out.copy(h5_in[key], key)
                            copied_keys += 1
                        else:
                            logger.warning(f"Key {key} from {worker_file} already exists in output. Skipping.")
                # Only increment if keys were actually copied
                if copied_keys > 0:
                     successfully_merged_count += 1
                logger.debug(f"Successfully processed {copied_keys} keys from {worker_file}")
                # Clean up successful worker file immediately
                os.remove(worker_file)

            except OSError as e:
                logger.error(f"Failed to open or read HDF5 file {worker_file}: {e}. Skipping this file.")
                failed_files.append(worker_file)
                # Do NOT remove the failed file yet, maybe keep for inspection
            except Exception as e:
                logger.error(f"An unexpected error occurred processing {worker_file}: {e}. Skipping this file.")
                failed_files.append(worker_file)
                # Do NOT remove the failed file yet

        # Update final metadata
        # Note: This assumes total_configs was correctly set initially.
        # Adjust successful_configs based on actual merges.
        final_successful_count = len(h5_out.keys()) # Count groups in the output file
        h5_out.attrs['successful_configs'] = final_successful_count # Update based on actual merged groups
        logger.info(f"Merged data from {successfully_merged_count} worker files successfully.")
        if failed_files:
            logger.warning(f"Failed to process or skipped {len(failed_files)} worker files: {failed_files}")
        else:
             logger.info("All worker files processed successfully.")


    # Clean up temp directory (only if it's empty or only contains failed files)
    remaining_items = os.listdir(temp_dir)
    if not remaining_items or all(os.path.join(temp_dir, item) in failed_files for item in remaining_items):
        try:
            # If keeping failed files, remove them first if desired, otherwise clean dir
            # for f in failed_files: os.remove(f)
            os.rmdir(temp_dir)
            logger.info(f"Cleaned up temporary directory {temp_dir}")
        except OSError as e:
             logger.error(f"Could not remove temporary directory {temp_dir}: {e}")
    else:
         logger.warning(f"Temporary directory {temp_dir} not removed as it contains unexpected files: {remaining_items}")
