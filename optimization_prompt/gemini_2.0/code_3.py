def optimize_layout(config, farm_boundaries, ...):
     try:
         ... # Optimization code
         with h5py.File(worker_file, 'a') as f:
             ... # Write data
         return True  # Or return a result object if needed
     except Exception as e:
         logger.error(f"Worker failed for {config}: {e}", exc_info=True)
         raise  # Re-raise the exception to be caught by the main process
 

 def main():
     ...
     with Pool(...) as pool:
         results = []
         for i in range(len(configs)):  # Iterate with a loop to handle exceptions
             try:
                 result = pool.apply_async(optimize_layout, (configs[i], farm_boundaries, ...))
                 results.append(result)
             except Exception as e:
                 logger.error(f"Failed to start worker for config {configs[i]}: {e}")
                 # Handle start-up failure, possibly exit or set a flag
         
         successes = []
         failures = []
         for result in tqdm(results, desc="Collecting results"):
             try:
                 success = result.get(timeout=600)  # Add timeout
                 successes.append(success)
             except Exception as e:
                 logger.error(f"Worker returned an error: {e}")
                 failures.append(e)  # Or some error indicator
 

     if failures:
         logger.error("Some workers failed. Aborting merge.")
         # Clean up temp files
         for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
             os.remove(worker_file)
         os.rmdir(temp_dir)
         return
     
     valid_worker_files = []
     for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
         try:
             with h5py.File(worker_file, 'r') as f:
                 _ = f['init'][()]  # Try reading a known dataset
             valid_worker_files.append(worker_file)
         except Exception as e:
             logger.error(f"Invalid worker file: {worker_file} - {e}")
             os.remove(worker_file)  # Remove invalid file
     
     if not valid_worker_files:
         logger.error("No valid worker files to merge.")
         os.rmdir(temp_dir)
         return
 

     with h5py.File(args.output, mode) as h5_out:
         ... # Merge valid_worker_files
     
     # Clean up temp directory
     for worker_file in valid_worker_files:
         os.remove(worker_file)
     os.rmdir(temp_dir)
