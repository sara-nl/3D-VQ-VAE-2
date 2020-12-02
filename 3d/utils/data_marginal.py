from pathlib import Path

import nrrd
import numpy as np
from tqdm import tqdm

from utils import ExtractCenterCylinder

def compute_scan_data_marginal(scan_paths, n_id=5000, ignore_min=-1024, mask=True):
    '''
    Expects data to be integers
    Amount of intensity values (indexes) counted is (-ignore_min + n_id)

    returns an array with the per-pixel-intensity count.
    '''
    hist_array = np.zeros((n_id,), dtype=np.int64)

    if mask:
        masker = ExtractCenterCylinder()

    for i, scan_path in enumerate(tqdm(Path(scan_paths).glob("**/*.nrrd"))):
        scan = nrrd.read(scan_path)[0]

        *_, x, y, _ = scan.shape

        if x != 512:
            continue

        if mask:
            scan = masker(scan)

        if scan.min() != ignore_min:
            print(f"Ignoring scan {scan_path}, min is {scan.min()}")
            continue

        values, counts = np.unique(scan, return_counts=True)
        hist_array[values-ignore_min] += counts
    return hist_array

if __name__ == '__main__':
    hist = compute_scan_data_marginal('/scratch/slurm.6807045.0/scratch/14')
    np.savetxt('/home/robertsc/hist.npy', hist)
