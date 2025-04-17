# Original code section in main() [cite: 1, 45]
#        # Copy data from worker files
#        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
#            with h5py.File(worker_file, 'r') as h5_in:
#                for key in h5_in:
#                    if key != 'init' and key not in h5_out:  # Skip 'init' dataset and avoid duplicates
#                        h5_out.copy(h5_in[key], key)
#            # Clean up worker file
#            os.remove(worker_file)

# Modified code section in main()
#        # Copy data from worker files
        merged_count = 0
        failed_files = []
        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
            try:
                # Attempt to open the worker file
                with h5py.File(worker_file, 'r') as h5_in:
                    # Check if it's basically valid (has some keys other than 'init')
                    keys_in_file = list(h5_in.keys())
                    if not keys_in_file or keys_in_file == ['init']:
                        logger.warning(f"Worker file {worker_file} is empty or contains no results. Skipping.")
                        failed_files.append(worker_file)
                        continue # Skip to the next file

                    # Copy data if valid
                    for key in keys_in_file:
                        if key != 'init':
                            if key not in h5_out:
                                h5_out.copy(h5_in[key], key)
                                merged_count += 1 # Count successful merges based on groups copied
                            else:
                                logger.warning(f"Group {key} from {worker_file} already exists in output file. Skipping duplicate.")
                                
            except OSError as e: # Catch the specific error
                logger.error(f"Failed to open or read worker file {worker_file}: {e}. Skipping.")
                failed_files.append(worker_file)
            except Exception as e: # Catch other potential errors during processing
                logger.error(f"An unexpected error occurred while processing {worker_file}: {e}. Skipping.")
                failed_files.append(worker_file)
            finally:
                 # Still attempt to clean up the worker file regardless of success/failure
                 try:
                     os.remove(worker_file)
                 except OSError as e:
                     logger.error(f"Failed to remove worker file {worker_file}: {e}")


        # Update the success count based on actual merges + previously completed ones
        final_successful_configs = merged_count + len(completed_configs)
        h5_out.attrs['successful_configs'] = final_successful_configs
        
        logger.info(f"Merged data from valid worker files. Total successful configurations in output: {final_successful_configs}")
        if failed_files:
            logger.warning(f"Could not process the following worker files (data may be lost): {failed_files}")
