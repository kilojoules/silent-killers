import os
 import glob
 import h5py
 import logging
 

 def is_valid_h5_file(filepath):
  """
  Check if an HDF5 file is valid.
  """
  if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:  # Arbitrary minimum size
  return False
  try:
  with h5py.File(filepath, 'r') as f:
  # Check for the 'init' dataset or any other known dataset/attribute
  if 'init' in f:  
  return True
  else:
  return False
  except OSError:
  return False
 

 def merge_worker_files(output_file, temp_dir, args):
  """
  Merge worker files into the final output file, with validation.
  """
  invalid_file_count = 0
  with h5py.File(output_file, 'w') as h5_out:  # Or 'a' for append mode if needed
  h5_out.attrs['grid_size'] = args.grid_size
  h5_out.attrs['random_pct'] = args.random_pct
  
  for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
  if not is_valid_h5_file(worker_file):
  logging.error(f"Invalid worker file: {worker_file}, skipping.")
  invalid_file_count += 1
  continue
  
  try:
  with h5py.File(worker_file, 'r') as h5_in:
  for key in h5_in:
  if key != 'init' and key not in h5_out:
  h5_out.copy(h5_in[key], key)
  except OSError as e:
  logging.error(f"Error processing worker file {worker_file}: {e}")
  invalid_file_count += 1
  
  if invalid_file_count > 0:
  logging.critical(f"Detected {invalid_file_count} invalid worker files. Merge aborted.")
  # Optionally, clean up or retry
  return False
  return True
 

 def main():
  # ... [Your existing main function code]
  
  if merge_worker_files(args.output, temp_dir, args):
  logging.info(f"Successfully merged worker files into {args.output}")
  else:
  logging.error("Failed to merge worker files.")
  # ... [Rest of your main function]
