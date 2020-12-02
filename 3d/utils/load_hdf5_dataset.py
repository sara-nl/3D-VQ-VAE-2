from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset

class FastMRIhdf5Dataset(Dataset):
    def __init__(self, root: str, data_key: str = 'reconstruction_rss'):
        super(FastMRIhdf5Dataset, self).__init__()
        self.scans = np.array(list(map(str, Path(root).glob(f'**/*'))))
        self.data_key = data_key
        self.length = len(self.scans)

    def __getitem__(self, scan_index: int) -> Tuple[np.array, Tuple[int, int, int]]:
        '''
        returns:
        - scan
        - scan shape
        '''
        with h5py.File(self.scans[scan_index], mode='r') as f:
            data = f[self.data_key]
            return np.asarry(f), f.shape

    def __len__(self):
        return self.length

def read_hdf5_folder(root: Path):
    for h5file in root.iterdir():
        with h5py.File(h5file, "r") as f:
            data = f['reconstruction_rss']
            breakpoint()


if __name__ == '__main__':
    root = Path('/scratch/slurm.6850262.0/scratch/multicoil_val/')
    # dataset = FastMRIhdf5Dataset()
    read_hdf5_folder(root)