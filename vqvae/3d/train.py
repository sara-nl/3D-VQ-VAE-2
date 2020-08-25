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
    threshold_value = 3000
    transform = transforms.Compose([
        transforms.AddChanneld(keys=['img']),
        transforms.Spacingd(keys=['img'], pixdim=(1,1,3)),
        transforms.SpatialPadd(keys=['img'], spatial_size=(512, 512, 128)),
        transforms.RandSpatialCropd(keys=['img'], roi_size=(256, 256, 128), random_size=False),
        transforms.ThresholdIntensityd(keys=['img'], threshold=threshold_value, cval=threshold_value),
        transforms.ToTensord(keys=['img'])
    ])

    dataset = CTScanDataset(args.dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    model = VQVAE()

    trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()

    main(args)