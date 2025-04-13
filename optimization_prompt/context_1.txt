import numpy as np
import h5py
import argparse
from tqdm import tqdm
from multiprocessing import Pool, current_process, Event, Value
from functools import partial
import gc
import logging
import os
import glob
import time
import sys

from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformSite
from py_wake.site.xrsite import XRSite
from py_wake import Nygaard_2022
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular, SimpleYawModel
from py_wake.utils.gradients import autograd

from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config_key(config):
    """Generate a unique key for a configuration to identify it in the output file"""
    return f"farm{config['farm_idx']}_t{config['type_idx']}_s{config['ss_seed']}"


def scan_existing_results(output_file):
    """Scan existing output file to identify already completed configurations
    
    Args:
        output_file: Path to the HDF5 output file
        
    Returns:
        Dictionary of completed configuration keys
    """
    completed_configs = {}
    
    if not os.path.exists(output_file):
        return completed_configs
    
    try:
        with h5py.File(output_file, 'r') as f:
            # Skip top-level attributes and only process groups
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # Read the attributes to reconstruct the configuration
                    config = {}
                    for attr_name, attr_value in f[key].attrs.items():
                        config[attr_name] = attr_value
                    
                    # Check if this configuration has all required data
                    if 'layout' in f[key] and 'aep' in f[key]:
                        config_key = key  # Use the group name as the key directly
                        completed_configs[config_key] = config
                    
        logger.info(f"Found {len(completed_configs)} completed configurations in {output_file}")
    except Exception as e:
        logger.warning(f"Error scanning existing results: {e}")
    
    return completed_configs


def filter_completed_configs(configs, completed_configs):
    """Filter out configurations that have already been completed
    
    Args:
        configs: List of configuration dictionaries
        completed_configs: Dictionary of completed configuration keys
        
    Returns:
        List of configurations that still need to be run
    """
    remaining_configs = []
    
    for config in configs:
        config_key = get_config_key(config)
        if config_key not in completed_configs:
            remaining_configs.append(config)
    
    return remaining_configs

def load_farm_boundaries():
    """Load pre-defined boundary coordinates for all farms in the study"""
    dk1d_tender_9 = np.array([
        695987.1049296035, 6201357.423000679,
        693424.6641628657, 6200908.39284379,
        684555.8480807217, 6205958.057693831,
        683198.5005206821, 6228795.090001539,
        696364.6957258906, 6223960.959626805,
        697335.3172284428, 6204550.027416158,]).reshape((-1,2)).T.tolist()
    
    dk0z_tender_5 = np.array([
        696839.729308316, 6227384.493837256,
        683172.617280614, 6237928.363392656,
        681883.784179578, 6259395.212250201,
        695696.2991147212, 6254559.15746051,]).reshape((-1,2)).T.tolist()
    
    dk0w_tender_3 = np.array([
        706694.3923283464, 6224158.532895836,
        703972.0844905999, 6226906.597455995,
        702624.6334635273, 6253853.5386425415,
        712771.6248419734, 6257704.934445341,
        715639.3355871611, 6260664.6846508905,
        721593.2420745814, 6257906.998015941,]).reshape((-1,2)).T.tolist()
    
    dk0v_tender_1 = np.array([
        695387.9840492046, 6260724.982986244,
        690623.9453331482, 6265534.095966523,
        689790.3527486034, 6282204.661276843,
        706966.9276208277, 6287633.435873629,
        708324.2751808674, 6264796.40356592,
        696034.3037791394, 6260723.0585712865,]).reshape((-1,2)).T.tolist()
    
    dk0Y_tender_4 = np.array([
        688971.224327626, 6289970.317104408,
        699859.6944068453, 6313455.877252994,
        706084.6136432136, 6313894.002391787,
        711981.4247481308, 6312278.1352986405,
        712492.2381035917, 6310678.304996811,
        705728.3384563944, 6295172.010736138,
        703484.1092881403, 6292667.063932351,
        695423.0025504732, 6290179.436863188,]).reshape((-1,2)).T.tolist()
    
    dk0x_tender_2 = np.array([
        715522.0997350881, 6271624.869308893,
        714470.0221534981, 6296972.621665262,
        735902.8674733848, 6290515.568009201,
        726238.5223950314, 6268396.3424808625,]).reshape((-1,2)).T.tolist()
    
    dk1a_tender_6 = np.array([
        741993.8028788125, 6285017.514473925,
        754479.4211245844, 6280870.400239231,
        755733.9969961186, 6260088.643106768,
        753546.8632103675, 6256441.8767611785,
        738552.854493294, 6267674.686871577,
        738130.3486627713, 6276124.151480917,]).reshape((-1,2)).T.tolist()
    
    dk1b_tender7 = np.array([
        730392.0211541882, 6258565.789403263,
        741435.0294020491, 6261729.52759437,
        743007.816872067, 6238891.853815009,
        741806.5300242025, 6237068.79137804,
        729493.7204694732, 6233452.174200128,
        729032.3897788484, 6255601.548896144,]).reshape((-1,2)).T.tolist()
    
    dk1c_tender_8 = np.array([
        719322.3683944925, 6234395.779001247,
        730063.1517509705, 6226372.251569298,
        720738.3338805687, 6206078.654364537,
        712391.7502303863, 6209300.125004388,
        709646.6042396385, 6212504.917381268,]).reshape((-1,2)).T.tolist()
    
    dk1e_tender_10 = np.array([
        705363.6892801415, 6203384.473423205,
        716169.9420085563, 6202667.308115489,
        705315.7291588389, 6178496.656241821,
        693580.7248750407, 6176248.298099114,]).reshape((-1,2)).T.tolist()
    
    return [
        dk0w_tender_3,  # target farm
        dk1d_tender_9,
        dk0z_tender_5,
        dk0v_tender_1,
        dk0Y_tender_4,
        dk0x_tender_2,
        dk1a_tender_6,
        dk1b_tender7,
        dk1c_tender_8,
        dk1e_tender_10
    ]


def get_turbine_types():
    """Generate turbine types with gradually increasing size and power"""
    RPs = np.arange(10, 16).astype(int)  # Rated power range from 10 to 15 MW
    Ds = (240 * np.sqrt(RPs / 15)).astype(int)  # Scale diameter based on sqrt of power ratio
    hub_heights = (Ds / 240 * 150).astype(int)  # Scale hub height proportionally
    
    # Create list of turbine types
    turbines = []
    for i, (rp, d, h) in enumerate(zip(RPs, Ds, hub_heights)):
        turbines.append(GenericWindTurbine(f'WT_{i}', d, h, rp * 1e6))
    
    return turbines


def get_configs(n_seeds, n_turbines_default=None):
    """Generate optimization configurations with scaling of turbine count
    
    Args:
        n_seeds: Number of random seeds per configuration
        n_turbines_default: Optional dictionary of farm_idx to turbine count
                           If None, use default of 66 for farm 0, 50 for others
    """
    if n_turbines_default is None:
        n_turbines_default = {0: 66}  # Default is 66 for farm 0
    
    turbine_types = get_turbine_types()
    
    configs = []
    for farm_idx in range(10):
        for type_idx in range(len(turbine_types)):
            for seed in range(n_seeds):
                # Get number of turbines (default to 50 if not specified)
                n_turbines = n_turbines_default.get(farm_idx, 50)
                
                # Get diameter from the actual turbine type
                diameter = turbine_types[type_idx].diameter()
                
                configs.append({
                    'farm_idx': farm_idx,
                    'type_idx': type_idx,
                    'n_turbines': n_turbines,
                    'diameter': diameter,
                    'ss_seed': seed
                })
    
    return configs


def worker_init(output_dir):
    """Initialize per-worker temporary HDF5 file"""
    import os  # Import at the beginning of the function
    import logging
    import h5py
    import numpy as np
    from multiprocessing import current_process
    
    logger = logging.getLogger(__name__)
    worker_id = current_process().pid
    worker_file = os.path.join(output_dir, f"worker_{worker_id}.h5")
    
    # Store file path on worker process object
    current_process().worker_file = worker_file
    
    # Initialize empty HDF5 file with a test dataset to ensure it's valid
    try:
        with h5py.File(worker_file, 'w') as f:
            f.create_dataset('init', data=np.array([1]))
            logger.info(f"Worker {worker_id}: Successfully initialized HDF5 file")
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to create HDF5 file: {str(e)}")

    # Force process affinity to separate cores (Linux only)
    try:
        import psutil
        process = psutil.Process()
        # Get worker index from Pool (not the pid)
        worker_idx = int(current_process().name.split('-')[-1]) if '-' in current_process().name else 0
        # Set affinity to a specific CPU core
        process.cpu_affinity([worker_idx % os.cpu_count()])
        logger.info(f"Worker {worker_id} assigned to CPU {worker_idx % os.cpu_count()}")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: CPU affinity setup failed: {str(e)}")


def optimize_layout(config, farm_boundaries, grid_size=18, random_pct=30, update_interval=None):
    """Run optimization and save to worker-specific temp file
    
    Args:
        config: Configuration dictionary with farm_idx, type_idx, etc.
        farm_boundaries: List of farm boundary coordinates
        grid_size: Number of grid points in each dimension for smart start
        random_pct: Percentage of random positions to use in smart start
    """
    try:
        worker_file = current_process().worker_file
        farm_idx = config['farm_idx']
        boundary = np.array(farm_boundaries[farm_idx]).T.tolist()
        
        # Get turbine type
        turbine_types = get_turbine_types()
        turbine = turbine_types[config['type_idx']]

        # Initialize site - try to load from cache first
        site_cache_file = f"site_cache_farm_{farm_idx}.nc"
        if os.path.exists(site_cache_file):
            try:
                site = XRSite.load(site_cache_file)
                logger.info(f"Loaded site from cache: {site_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load site from cache: {e}")
                site = UniformSite(ti=0.1)
        else:
            site = UniformSite(ti=0.1)
        
        # Initialize random layout positions
        x_min, x_max = min(b[0] for b in boundary), max(b[0] for b in boundary)
        y_min, y_max = min(b[1] for b in boundary), max(b[1] for b in boundary)
        x = np.random.uniform(x_min, x_max, config['n_turbines'])
        y = np.random.uniform(y_min, y_max, config['n_turbines'])

        # Create optimization problem
        problem = TopFarmProblem(
            design_vars={'x': x, 'y': y},
            cost_comp=PyWakeAEPCostModelComponent(
                Nygaard_2022(site, turbine), 
                n_wt=config['n_turbines'],
                grad_method=autograd),  # Use autograd for gradients
            constraints=[
                XYBoundaryConstraint(boundary, boundary_type='polygon'),
                SpacingConstraint(config['diameter'] * 1.1)  # 1.1D spacing constraint
            ]
        )

        # Run optimization with smart start
        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        problem.smart_start(*np.meshgrid(xs, ys), 
                          problem.cost_comp.get_aep4smart_start(),
                          random_pct=random_pct,
                          seed=config['ss_seed'],
                          show_progress=False)

        # Save to worker-specific file
        with h5py.File(worker_file, 'a') as f:
            grp = f.create_group(f"farm{farm_idx}_t{config['type_idx']}_s{config['ss_seed']}")
            grp.create_dataset('layout', data=np.vstack([problem['x'], problem['y']]))
            grp.create_dataset('aep', data=problem.cost) 
            for k, v in config.items():
                grp.attrs[k] = v

        logger.info(f"Successfully optimized farm {farm_idx}, type {config['type_idx']}, seed {config['ss_seed']}")
        return True

    except Exception as e:
        logger.error(f"Failed {config}: {str(e)}")
        return False
    finally:
        # Clean up memory
        gc.collect()


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
    args = parser.parse_args()

    # Setup temporary storage
    temp_dir = f"{args.output}_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Load farm boundaries
    farm_boundaries = load_farm_boundaries()
    
    # Generate configuration list
    configs = get_configs(args.seeds)
    logger.info(f"Created {len(configs)} configurations to optimize")

    completed_configs = {}
    if not args.no_hot_start:
        completed_configs = scan_existing_results(args.output)
        if completed_configs:
            original_count = len(configs)
            configs = filter_completed_configs(configs, completed_configs)
            logger.info(f"Hot start enabled: {original_count - len(configs)} configurations already completed")
            logger.info(f"Remaining configurations to run: {len(configs)}")
            print(f"Hot start: {original_count - len(configs)} configurations already completed, {len(configs)} remaining", 
                  file=sys.stderr, flush=True)
    
    # If all configurations are already completed, we're done
    if not configs:
        logger.info("All configurations already completed. Nothing to do.")
        print("All configurations already completed. Nothing to do.", file=sys.stderr, flush=True)
        return

    # Create pool with worker-specific files
    with Pool(
        processes=args.processes,
        initializer=worker_init,
        initargs=(temp_dir,),
        maxtasksperchild=10
    ) as pool:
        optimize_partial = partial(optimize_layout, 
                                 farm_boundaries=farm_boundaries,
                                 grid_size=args.grid_size,
                                 random_pct=args.random_pct)
        
        # We'll use chunksize > 1 to make the progress bar update more frequently
        chunk_size = args.chunk_size
        
        # Try to use a smaller chunksize if we have many configs per process
        configs_per_process = len(configs) / args.processes
        if configs_per_process > 10 and chunk_size == 1:
            suggested_chunk = max(1, min(5, int(configs_per_process / 10)))
            logger.info(f"Using chunk size of {suggested_chunk} for better progress updates")
            chunk_size = suggested_chunk
        
        # Run optimizations with progress bar - use imap_unordered for more frequent updates
        results = []
        iterator = pool.imap_unordered(optimize_partial, configs, chunksize=chunk_size)
        
        # Manual progress tracking with more frequent updates
        with tqdm(total=len(configs), desc="Optimizing layouts", mininterval=args.progress_interval) as pbar:
            while True:
                try:
                    # Set a timeout so the progress bar updates even if no tasks complete
                    result = iterator.__next__()
                    results.append(result)
                    pbar.update(1)
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    logger.warning("Keyboard interrupt detected, trying to save partial results...")
                    # Mark the remaining as failures
                    remaining = len(configs) - len(results)
                    results.extend([False] * remaining)
                    break
    
    # Log statistics
    success_count = sum(1 for r in results if r)
    logger.info(f"Completed {success_count}/{len(configs)} optimizations successfully")

    # Merge worker files into final output
    if not args.no_hot_start and os.path.exists(args.output):
        # Use append mode to preserve existing results
        mode = 'a'
    else:
        # Use write mode for new file or when hot start is disabled
        mode = 'w'
    
    with h5py.File(args.output, mode) as h5_out:
        # Add metadata
        h5_out.attrs['grid_size'] = args.grid_size
        h5_out.attrs['random_pct'] = args.random_pct
        h5_out.attrs['total_configs'] = len(configs) + len(completed_configs)
        h5_out.attrs['successful_configs'] = success_count + len(completed_configs)
        
        # Copy data from worker files
        for worker_file in glob.glob(os.path.join(temp_dir, "worker_*.h5")):
            with h5py.File(worker_file, 'r') as h5_in:
                for key in h5_in:
                    if key != 'init' and key not in h5_out:  # Skip 'init' dataset and avoid duplicates
                        h5_out.copy(h5_in[key], key)
            # Clean up worker file
            os.remove(worker_file)
    
    # Clean up temp directory
    os.rmdir(temp_dir)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
