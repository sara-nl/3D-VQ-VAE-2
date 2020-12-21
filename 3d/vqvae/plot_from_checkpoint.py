from argparse import ArgumentParser, Namespace
from pathlib import Path

import nrrd
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai import transforms

from model import VQVAE
from utils import CTScanDataset


def main(args: Namespace):

    min_val, max_val, scale_val = -1500, 3000, 1000

    transform = transforms.Compose([
        transforms.AddChannel(),
        transforms.ThresholdIntensity(threshold=max_val, cval=max_val, above=False),
        transforms.ThresholdIntensity(threshold=min_val, cval=min_val, above=True),
        transforms.ScaleIntensity(minv=None, maxv=None, factor=(-1 + 1/scale_val)),
        transforms.ShiftIntensity(offset=1),
        transforms.SpatialPad(spatial_size=(512, 512, 128), mode='constant'),
        transforms.SpatialCrop(roi_size=(512, 512, 128), roi_center=(256, 256, 64)),
        # transforms.Resize(spatial_size=(256, 256, 128)),
        transforms.ToTensor()
    ])

    print("- Loading dataloader")
    dataset = CTScanDataset(args.dataset_path, transform=transform, spacing=(0.976, 0.976, 3))
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    print("- Loading single CT sample")
    single_sample = next(iter(train_loader))
    single_sample = single_sample.cuda()

    print("- Loading model weights")
    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()

    print("- Performing forward pass")
    with torch.no_grad(), torch.cuda.amp.autocast():
        res, *_ = model(single_sample)
        res = torch.nn.functional.softplus(res)

    res = res.squeeze().detach().cpu().numpy()
    res = res * scale_val - scale_val
    res = np.rint(res).astype(np.int)

    print("- Writing to nrrd")
    nrrd.write(str(args.out_path), res, header={'spacings': (0.976, 0.976, 3)})

    print("- Done")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    main(args)