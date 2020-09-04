from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from monai import transforms

from model import VQVAE
from _utils import CTScanDataset

def main(args):
    key, min_val, max_val = 'img', -1000, 3000

    transform = transforms.Compose([
        transforms.AddChannel(),
        transforms.ThresholdIntensity(threshold=max_val, cval=max_val, above=False),
        transforms.ThresholdIntensity(threshold=min_val, cval=min_val, above=True),
        transforms.ScaleIntensity(minv=None, maxv=None, factor=(-1 - 1/min_val)),
        transforms.ShiftIntensity(offset=1),
        transforms.SpatialPad(spatial_size=(512, 512, 128), mode='constant'),
        transforms.RandSpatialCrop(roi_size=(512, 512, 128), random_size=False),
        # transforms.Resize(spatial_size=(256, 256, 128)),
        transforms.ToTensor()
    ])

    dataset = CTScanDataset(args.dataset_path, transform=transform, spacing=(0.976, 0.976, 3))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=6, pin_memory=True)

    model = VQVAE()

    trainer = pl.Trainer(
        gpus=4,
        auto_select_gpus=True,
        distributed_backend='ddp',
        benchmark=True,

        max_epochs=200,
        terminate_on_nan=True,

        profiler=None,
        row_log_interval=100,
        log_save_interval=1000,
    )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()

    main(args)