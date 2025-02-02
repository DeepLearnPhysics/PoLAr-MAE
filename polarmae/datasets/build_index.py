import argparse
import multiprocessing
import os
import sys

import h5py as h5
import numpy as np

try:
    from tqdm import trange
except ImportError:
    def trange(*args, **kwargs):
        proc = multiprocessing.current_process()
        worker_id = proc._identity[0] - 1 if proc._identity else 0
        for i in range(args[0]):
            print(worker_id + f'{kwargs.get("desc", "")} {i}/{args[0]}', end='\r')
            yield i

def process_file(file_path: str) -> None:
    """
    Process a PILArNet h5 file and save the number of points in each event
    for use in PILArNet Dataset.
    """
    # progress bar based on the worker's identity
    proc = multiprocessing.current_process()
    worker_id = proc._identity[0] - 1 if proc._identity else 0

    with h5.File(
        file_path,
        'r',
        libver='latest',
        swmr=True,
    ) as f:
        num_points = []
        for i in trange(
            f['cluster'].shape[0],
            desc=f'[Worker {worker_id}] {os.path.basename(file_path)}',
            ncols=80,
            position=worker_id,
            leave=False,
        ):
            cluster_size = f['cluster'][i].reshape(-1, 5)[:, 0]
            num_points.append(cluster_size.sum())

    output_path = file_path.replace(".h5", "_points.npy")
    np.save(output_path, np.array(num_points).squeeze())

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Process PILArNet event lengths.'
    )
    parser.add_argument(
        'file',
        nargs='+',
        type=str,
        help='Input h5 file(s). Use wildcards (*) to match all files in the directory.'
    )
    parser.add_argument(
        '-j',
        '--num-workers',
        type=int,
        default=1,
        help='Number of workers to use for processing.',
    )
    args = parser.parse_args()

    if not args.file:
        print("No files provided")
        sys.exit(1)

    with multiprocessing.Pool(min(args.num_workers, len(args.file))) as pool:
        pool.map(process_file, args.file)

if __name__ == "__main__":
    main()