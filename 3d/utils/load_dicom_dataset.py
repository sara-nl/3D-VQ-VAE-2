from pathlib import Path

import pydicom
import numpy as np

from torch.utils.data import Dataset

class FastMRIdicomDataset(Dataset):
    def __init__(self, root: str, ext='.dcm'):
        super(FastMRIdicomDataset, self).__init__()
        self.scans = np.asarray(list(map(str, Path(root).rglob(f'*{ext}'))))
        self.length = len(self.scans)

    def __getitem__(self, scan_index: int):
        return pydicom.dcmread(self.scans[scan_index]).pixel_array
    
    def __len__(self):
        return self.length

if __name__ == '__main__':
    path = '/scratch/slurm.6943997.0/scratch/knee_mri_clinical_seq/'
    dataset = FastMRIdicomDataset(path)
    sample = dataset[0]
    breakpoint()
