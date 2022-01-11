from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import nrrd
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai import transforms

from vqvae.model import VQVAE
from utils import CTScanDataset, DepthPadAndCrop, CTDataModule


def main(args: Namespace):
    pl.seed_everything(42)

    min_val, max_val, scale_val = -1500, 3000, 1000

    print("- Loading dataloader")
    datamodule = CTDataModule(path=args.dataset_path,  train_frac=1, batch_size=1, num_workers=0, rescale_input=args.rescale_input)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()

    print("- Loading single CT sample")
    single_sample, _ = next(iter(train_loader))
    single_sample = single_sample.cuda()

    print("- Loading model weights")
    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()

    print("- Performing forward pass")
    with torch.no_grad(), torch.cuda.amp.autocast():
        res, *_ = model(single_sample)
        res = torch.nn.functional.elu(res)

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
    parser.add_argument("--rescale-input", default=None, type=int, nargs='+')
    args = parser.parse_args()

    main(args)