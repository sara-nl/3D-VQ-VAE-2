import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, Tuple

import torch
import pytorch_lightning as pl
import numpy as np
from model import VQVAE
from utils import CTDataModule


def main(args):
    torch.cuda.empty_cache()

    pl.trainer.seed_everything(seed=42)

    datamodule = CTDataModule(path=args.dataset_path, batch_size=args.batch_size, num_workers=5)

    model = VQVAE(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, save_last=True, monitor='val_recon_loss_mean')

    trainer = pl.Trainer(
        gpus="-1",
        # auto_select_gpus=True,
        distributed_backend='ddp',

        benchmark=True,
        num_nodes=args.num_nodes,

        num_sanity_val_steps=0,
        precision=16,

        accumulate_grad_batches=args.accumulate_grad_batches,

        terminate_on_nan=True,

        profiler=None,

        resume_from_checkpoint=args.checkpoint_path,
        checkpoint_callback=checkpoint_callback,
        log_every_n_steps=50,
        val_check_interval=0.5,
        flush_logs_every_n_steps=100,
        weights_summary='full',
    )
    trainer.fit(model, datamodule=datamodule)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("dataset_path", type=Path)

    args = parser.parse_args()


    main(args)