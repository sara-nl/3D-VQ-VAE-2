from pathlib import Path
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from pixelcnn import PixelCNN
from pixelsnail import PixelSNAIL
from utils.load_lmdb_dataset import LMDBDataModule

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--use-model", type=str, choices=['pixelcnn', 'pixelsnail'], default='pixelcnn')
    use_model = parser.parse_known_args()[0].use_model

    if use_model == 'pixelcnn':
        parser = PixelCNN.add_model_specific_args(parser)
    elif use_model == 'pixelsnail':
        parser = PixelSNAIL.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("level", type=int,
                        help="Which PixelCNN hierarchy level to train")
    parser.add_argument("--batch-size", type=int)

    parser.set_defaults(
        gpus="-1",
        distributed_backend='ddp',

        benchmark=True,

        num_sanity_val_steps=0,
        precision=16,

        log_every_n_steps=50,
        val_check_interval=0.5,
        flush_logs_every_n_steps=100,
        weights_summary='full',

        max_epochs=int(5e4)

    )

    args = parser.parse_args()
    args.use_model = use_model

    assert args.dataset_path.resolve().exists()
    args.dataset_path = str(args.dataset_path.resolve())

    return args


def main(args):
    torch.cuda.empty_cache()
    pl.trainer.seed_everything(seed=42)

    datamodule = LMDBDataModule(
        path=args.dataset_path,
        embedding_id=args.level,
        batch_size=args.batch_size,
        num_workers=5,
    )

    datamodule.setup()
    args.num_embeddings = datamodule.num_embeddings

    if args.use_model == 'pixelcnn':
        model = PixelCNN(args)
    elif args.use_model == 'pixelsnail':
        model = PixelSNAIL(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, save_last=True, monitor='val_loss_mean')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)