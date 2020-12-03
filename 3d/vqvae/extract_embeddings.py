import pickle
from argparse import ArgumentParser
from pathlib import Path

import lmdb
import torch
from tqdm import tqdm

from model import VQVAE
from utils import CTDataModule


GPU = torch.device('cuda')

def extract_samples(model, dataloader):
    model.eval()
    model.to(GPU)

    with torch.no_grad():
        for sample, _ in dataloader:
            sample = sample.to(GPU)
            *_, encoding_idx = zip(*model.encode(sample))
            yield encoding_idx


def get_output_abspath(checkpoint_path: Path, output_path: Path, output_name: str = '') -> str:
    assert output_path.is_dir()
    return str((output_path / (checkpoint_path.stem + '.lmdb' if output_name == '' else output_name)).resolve())


def main(args):
    if args.checkpoint_path is not None:
        model = VQVAE.load_from_checkpoint(str(args.checkpoint_path))
    else:
        model = VQVAE()

    datamodule = CTDataModule(
        path=args.dataset_path,
        batch_size=1,
        train_frac=1,
        num_workers=5
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    db = lmdb.open(
        get_output_abspath(args.checkpoint_path, args.output_path, args.output_name),
        map_size=int(1e12),
        max_dbs=model.n_bottleneck_blocks
    )

    sub_dbs = [db.open_db(str(i).encode()) for i in range(model.n_bottleneck_blocks)]
    with db.begin(write=True) as txn:
        for i, sample_encodings in tqdm(enumerate(extract_samples(model, dataloader)), total=len(dataloader)):
            for sub_db, encoding in zip(sub_dbs, sample_encodings):
                txn.put(str(i).encode(), pickle.dumps(encoding.cpu().numpy()), db=sub_db)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--output-path", type=Path, default=Path("."))
    parser.add_argument("--output-name", type=str, default='', help="default: takes over the checkpoint name with .lmdb file ext")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)

    args = parser.parse_args()

    main(args)