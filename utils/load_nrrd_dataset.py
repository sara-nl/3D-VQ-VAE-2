from functools import lru_cache
from itertools import chain, tee
from random import shuffle, sample
from pathlib import Path
from copy import copy
from typing import Union, Sequence, Tuple

import nrrd
import numpy as np
from monai import transforms
import torch
from torch.utils.data import Sampler, Dataset



class CTScanDataset(Dataset):
    '''Warning: file (name) ordering is not preserved'''
    def __init__(
        self,
        root: str,
        transform=None,
        size: Tuple[Union[int, None], Union[int, None], Union[int, None]] = (512, 512, None),
        spacing: Union[Tuple[float, float, float], None] = None,
        ext: str = '.nrrd'
    ):

        '''size: any scan not compatible with the specified size will be discarded.
        Insert None for a dim that can be any size'''

        root_path = Path(root)

        self.transform = transform

        scans = np.array(list(map(str, Path(root).glob(f'**/*{ext}'))))
        scan_sizes = np.array([nrrd.read_header(str(scan_path))['sizes'] for scan_path in scans])
        # I know we're reading the scans twice, but I get some strange Nones when I try
        # to do it in one go.
        # types = np.array([nrrd.read_header(str(scan_path))['type'] for scan_path in scans])

        if spacing is not None:
            spacings = np.array([
                nrrd.read_header(str(scan_path))['space directions'][np.diag_indices(3)]
                for scan_path in scans
            ])
            faulty_spacing = np.where(~np.isclose(spacings, spacing, atol=1e-3).all(axis=1))[0]
        else:
            faulty_spacing = []

        if len(faulty_spacing):
            print(f"Found {len(faulty_spacing)} scans where their spacing doesn't match the input spacing {spacing}. Ignoring scans {scans[faulty_spacing]}")

        faulty_sizes = np.unique(np.where(~(scan_sizes == size).T[np.where(size)[0]])[1])
        if len(faulty_sizes):
            print(f"Found {len(faulty_sizes)} scans where their size doesn't match the input size {size}. Ignoring scans {scans[faulty_sizes]}")

        faulty_idx = np.unique(np.append(faulty_sizes, faulty_spacing))

        self.scans = np.delete(scans, faulty_idx)
        self.scan_sizes = np.delete(scan_sizes, faulty_idx, axis=0)

    def __len__(self) -> int:
        return self.scans.shape[0]

    @lru_cache(maxsize=1)
    def get_scan(self, scan_index: int) -> Tuple[np.array, dict]:
        data, metadata = nrrd.read(self.scans[scan_index])
        return data.astype(np.float32), metadata

    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        data, metadata = self.get_scan(index)

        if self.transform is not None:
            data = self.transform(data)

        return data, -1


class CTSliceDataset(CTScanDataset):
    def __init__(self, root, transform=None, size=(512, 512, None), ext='.nrrd'):
        '''size: any scan not compatible with the specified size will be discarded.
        Insert None for a dim that can be any size'''

        super(CTSliceDataset, self).__init__()

        self.scan_heights = self.scan_sizes.T[1]

        self.cumsum = np.cumsum(np.insert(self.scan_heights, 0, 0))
        num_slices = self.cumsum[-1]

        self.idx = np.empty((num_slices,), dtype=np.int)
        for i, (start, finish) in enumerate(pairwise(self.cumsum)):
            self.idx[start:finish] = i

        self.num_slices = num_slices

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        '''returns:
        - Scan ('img')
        - -1 ('target')
        '''
        scan_index = self.idx[index]
        cumsum_index = self.cumsum[scan_index]

        scan, metadata = self.get_scan(scan_index)
        slice_ = scan[...,index - cumsum_index]

        if self.transform is not None:
            slice_ = self.transform(slice_)

        return slice_, -1


class SliceSampler(Sampler):
    '''Sampler meant as a semi-random shuffler of the CTSliceDataset,
    since a true random slice shuffler would incur large I/O penalties.
    - mode: can be 'none', 'inter', 'intra', or 'both'
      - 'none' means no shuffling occurs, and the dataset is returned sequentially
      - 'inter' means shuffling between scans, so overall scan order is shuffled every iteration
      - 'intra' means shuffling within scans, so the slice order itself is shuffled every iteration
      - 'both' means both 'inter' and 'intra' (default)
    '''
    def __init__(self, data_source : CTSliceDataset, mode='both'):
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


class ExtractCenterCylinder(torch.nn.Module):
    def __init__(self, size: Union[Tuple[int, int], None] = None, cache_mask: bool = True):
        super(ExtractCenterCylinder, self).__init__()
        self.mask = self.create_cylinder_xy_mask(size) if size else None
        self.cache_mask = cache_mask

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        *_, x, y, _ = tensor.shape

        if self.mask is not None:
            mask = self.mask
        else:
            mask = self.create_cylinder_xy_mask((x, y))
            if self.cache_mask:
                self.mask = mask

        return tensor[..., mask, :]

    @staticmethod
    def create_cylinder_xy_mask(size: Tuple[int, int]) -> torch.Tensor:
        x_size, y_size = size

        radius = min(x_size, y_size) / 2
        x_center, y_center = x_size / 2, y_size / 2

        x, y = np.ogrid[:x_size, :y_size]
        dist_from_center = np.sqrt((x - x_center)**2 + (y-y_center)**2)

        mask = dist_from_center <= radius

        return torch.BoolTensor(mask)
