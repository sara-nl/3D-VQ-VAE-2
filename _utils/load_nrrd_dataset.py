from functools import lru_cache
from itertools import chain, tee
from random import shuffle, sample
from pathlib import Path
from copy import copy

import nrrd
import numpy as np
from monai import transforms


try:
    import torch
    from torch.utils.data import Sampler, Dataset
except ImportError:
    print("PyTorch not found, some functions will not be available")



class CTScanDataset(Dataset):
    '''Warning: file (name) ordering is not preserved'''
    def __init__(self, root, transform=None, size=(512, 512, None), ext='.nrrd'):
        '''size: any scan not compatible with the specified size will be discarded.
        Insert None for a dim that can be any size'''
        
        root_path = Path(root)

        self.transform = transform

        scans = np.array(list(map(str, Path(root).glob(f'**/*{ext}'))))
        scan_sizes = np.array([nrrd.read_header(str(scan_path))['sizes']
                               for scan_path in scans])
        
        faulty_idx = np.unique(np.where(~(scan_sizes == size).T[np.where(size)[0]])[1])
        if len(faulty_idx):
            print(f"Found {len(faulty_idx)} scans where their size doesn't match the input size {size}. Ignoring scans {scans[faulty_idx]}")

        self.scans = np.delete(scans, faulty_idx)
        self.scan_sizes = np.delete(scan_sizes, faulty_idx, axis=0)

        self.reader = transforms.LoadImage()

    def __len__(self):
        return self.scans.shape[0]

    @lru_cache(maxsize=1)
    def get_scan(self, scan_index):
        data, metadata = self.reader(self.scans[scan_index])
        return {'img': data, 'img_meta_dict': metadata}

    def __getitem__(self, index):
        scan = self.get_scan(index)
        return (scan if self.transform is None else self.transform(scan))['img'], -1


class CTSliceDataset(CTScanDataset):
    def __init__(self, root, transform=None, size=(512, 512, None), ext='.nrrd'):
        '''size: any scan not compatible with the specified size will be discarded.
        Insert None for a dim that can be any size'''
        
        super(CTSliceDataset, self).__init__()

        self.scan_heights = self.scan_sizes.T[-1]

        self.cumsum = np.cumsum(np.insert(self.scan_heights, 0, 0))
        num_slices = self.cumsum[-1]

        self.idx = np.empty((num_slices,), dtype=np.int)
        for i, (start, finish) in enumerate(pairwise(self.cumsum)):
            self.idx[start:finish] = i
        
        self.num_slices = num_slices

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        '''returns:
        - Scan ('img')
        - -1 ('target')
        '''
        scan_index = self.idx[index]
        cumsum_index = self.cumsum[scan_index]

        scan, metadata = self.get_scan(scan_index)['img'][...,index - cumsum_index]

        return scan if self.transform is None else self.transform(scan), -1


class SliceSampler(Sampler):
    '''Sampler meant as a semi-random shuffler of the CTSliceDataset,
    since a true random slice shuffler would incur large I/O penalties.
    - mode: can be 'none', 'inter', 'intra', or 'both'
      - 'none' means no shuffling occurs, and the dataset is returned sequentially
      - 'inter' means shuffling between scans, so overall scan order is shuffled every iteration
      - 'intra' means shuffling within scans, so the slice order itself is shuffled every iteration
      - 'both' means both 'inter' and 'intra' (default)
    '''
    def __init__(self, data_source: CTSliceDataset, mode='both'):
        if not mode in (mode_options := ('none', 'inter', 'intra', 'both')):
            raise ValueError(f"Mode needs to be in {mode_options}, found {mode}")

        self.mode = mode
        self.dataset = data_source
        self.pairwise_idx_ranges = np.array(list(pairwise(self.dataset.cumsum)))


    def __iter__(self):
        num_scans = len(self.dataset.scan_heights)

        inter_slice_order = np.arange(num_scans)
        if self.mode in ('inter', 'both'):
            np.random.shuffle(inter_slice_order)

        intra_slice_order = np.arange(len(self.dataset))
        if self.mode in ('intra', 'both'):
            index_ranges = self.pairwise_idx_ranges[inter_slice_order]
            for start, finish in index_ranges:
                np.random.shuffle(intra_slice_order[start:finish])

        return iter(intra_slice_order)

    def __len__(self):
        return len(self.dataset)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)