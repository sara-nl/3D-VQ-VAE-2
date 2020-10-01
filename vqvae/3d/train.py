from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from monai import transforms

from model import VQVAE
from utils import CTScanDataset


class CTDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=64, train_frac=0.95, num_workers=6):
        super().__init__()
        assert 0 <= train_frac <= 1

        self.path = path
        self.train_frac = train_frac
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        # transform
        key, min_val, max_val = 'img', -1000, 3000

        transform = transforms.Compose([
            transforms.AddChannel(),
            transforms.ThresholdIntensity(threshold=max_val, cval=max_val, above=False),
            transforms.ThresholdIntensity(threshold=min_val, cval=min_val, above=True),
            transforms.ScaleIntensity(minv=None, maxv=None, factor=(-1 - 1/min_val)),
            transforms.ShiftIntensity(offset=1),
            transforms.SpatialPad(spatial_size=(512, 512, 128), mode='constant'),
            transforms.RandSpatialCrop(roi_size=(512, 512, 128), random_size=False),
            transforms.ToTensor()
        ])

        dataset = CTScanDataset(self.path, transform=transform, spacing=(0.976, 0.976, 3))

        train_len = int(len(dataset) * self.train_frac)
        val_len = len(dataset) - train_len

        # train/val split
        train_split, val_split = random_split(dataset, [train_len, val_len])

        # assign to use in dataloaders
        self.train_dataset = train_split
        self.val_dataset = val_split

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)



def main(args):
    datamodule = CTDataModule(path=args.dataset_path, batch_size=args.batch_size, num_workers=6)

    model = VQVAE(
        metric = args.metric,
        input_channels = args.input_channels,
        base_network_channels = args.base_network_channels,
        n_bottleneck_blocks = args.n_bottleneck_blocks,
        n_blocks_per_bottleneck = args.n_blocks_per_bottleneck,
    )

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(save_last=True, save_top_k=5)
    lr_logger = pl.callbacks.lr_logger.LearningRateLogger(logging_interval='step')

    trainer = pl.Trainer(
        gpus=args.gpus,
        auto_select_gpus=True,
        distributed_backend='ddp',
        benchmark=True,

        max_epochs=200,
        terminate_on_nan=True,

        profiler=None,

        checkpoint_callback=checkpoint_callback,
        row_log_interval=50,
        val_check_interval=50,
        log_save_interval=1000,

        callbacks=[lr_logger],
    )

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("dataset_path", type=Path)

    args = parser.parse_args()


    main(args)