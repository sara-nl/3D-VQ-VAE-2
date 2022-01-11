from argparse import ArgumentParser, Namespace
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn

from vqvae.model import VQVAE
from utils import CTDataModule
from metrics.evaluate import SSIM3DSlices


def main(args: Namespace):
    # Same seed as used in train.py, so that train/val splits are also the same
    pl.trainer.seed_everything(seed=42)

    print("- Loading datamodule")
    datamodule = CTDataModule(path=args.dataset_path, batch_size=5, num_workers=5) # mypy: ignore
    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    print("- Loading model weights")
    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()

    data_min, data_max = -0.24, 4
    data_range = data_max - data_min

    train_ssim = SSIM3DSlices(data_range=data_range)
    val_ssim = SSIM3DSlices(data_range=data_range)

    def batch_ssim(batch, ssim_f):
        batch = batch.cuda()
        out, *_ = model(batch)
        out = F.elu(out)
        return val_ssim(out.float(), batch)

    with torch.no_grad(), torch.cuda.amp.autocast():
        val_ssims = torch.Tensor([
            batch_ssim(batch, ssim_f=val_ssim) for batch, _ in tqdm(val_dl)
        ])
        breakpoint()
        train_ssims = torch.Tensor([
            batch_ssim(batch, ssim_f=train_ssim) for batch, _ in tqdm(train_dl)
        ])


    # breakpoint for manual decision what to do with train_ssims/val_ssims
    # TODO: find some better solution to described above
    breakpoint()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    main(args)