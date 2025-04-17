import h5py
 import os
 import glob
 import multiprocessing

 # ... (Existing code)

 def is_valid_hdf5_file(filepath):
  """Check if an HDF5 file is valid."""
  try:
  with h5py.File(filepath, 'r') as f:
  return True  #  Can read the file
  except OSError:
  return False

 def main():
  # ... (Existing code)

  with Pool(
  processes=args.processes,
  initializer=worker_init,
  initargs=(temp_dir,),
  maxtasksperchild=10
  ) as pool:
  # ... (Optimization)
  results = []
  # ...
  
  #  Add error handling
  
  # ...
  
  # Merge worker files with validation
  valid_worker_files = []
  for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
  if is_valid_hdf5_file(worker_file):
  valid_worker_files.append(worker_file)
  else:
  logger.error(f"Invalid worker file: {worker_file}, skipping")
  
  if not valid_worker_files:
  logger.error("No valid worker files found. Aborting merge.")
  return

  with h5py.File(args.output, mode) as h5_out:
  # ... (Metadata)
  for worker_file in valid_worker_files:
  # ... (Copy data)
  
  # ... (Cleanup)

 if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', force=True)
  main()
