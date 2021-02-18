from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from random import sample
from uuid import uuid4

import lmdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from filelock import FileLock

from model import PixelSNAIL

GPU = torch.device('cuda')

def parse_arguments():
    parser = ArgumentParser()

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = PixelSNAIL.add_model_specific_args(parser)

    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--level", type=int, default=-1,
                        help=('Which PixelSNAIL hierarchy level to train. '
                              'Defaults to highest hierarchy available'))
    parser.add_argument("--size", type=int, nargs='+', required=True,
                        help=('n-tuple of shape (channel, *dims). '
                              '(channel, *dims) should be the same as the size the model was trained on.'))
    parser.add_argument("--num-samples", default=-1, type=int, help=(
        'Number of samples to sample, defaults to number of conditions in db. '
        'Always samples conditions with least amount of samples first.'
    ))
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--tau", default=1.0, type=float, help="Temperature for softmax sampling")

    args = parser.parse_args()

    assert args.batch_size <= args.num_samples
    assert args.batch_size >= 1
    assert args.tau >= 0
    assert args.level >=0


    return args


def _get_db_lock(db_path) -> FileLock:
    return FileLock(str(db_path) + '.lock')


def create_or_load_db(db_path: Path, level: int):
    # saving/loading everything as .pt is pretty dumb,
    # since everything gets dumped immediately into memory.
    # Next, process-safe read/writes are impossible, except using
    # explicit locking as I use below.
    # But, doing it this way is also the most easy solution (for me).
    # _surely_ this won't bite anyone in the ass at some point...
    # FIXME: rewrite whole db logic to use some lazy-reading db format

    db_lock = _get_db_lock(db_path)
    with db_lock:
        if not db_path.exists():
            print(f"No db found! creating new db at {db_path}!")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({}, db_path)

        db = torch.load(db_path, map_location='cpu')

    if level not in db:
        print(f"Hierarchy {level} not found in db; adding.")
        db[level] = {}

    return db


def save_db(db, db_path, level):
    db_lock = _get_db_lock(db_path)
    with db_lock:
        # maybe the db got updated independently in a different process
        possibly_updated_db = torch.load(db_path, map_location='cpu')
        if level in possibly_updated_db:
            db[level].update(possibly_updated_db[level])

        torch.save(db, db_path)


def get_condition_uuids(db, level, num_conditions=2):
    assert level+1 in db

    if len((options := db[level+1].keys())) < num_conditions:
        options = list(chain.from_iterable(options for _ in range(ceil(num_conditions / len(options)))))

    return sample(options, k=num_conditions)


def get_conditions(db, level, uuids):
    assert level+1 in db

    return torch.stack([db[level+1][uuid]['data'] for uuid in uuids])


def main(model_checkpoint, db_path, level, size, num_samples, batch_size, tau):
    torch.cuda.empty_cache()

    model = PixelSNAIL.load_from_checkpoint(model_checkpoint).to(GPU)
    model.eval()

    db_path = db_path.resolve()

    db = create_or_load_db(db_path, level)

    assert ((model.condition_dim == 0 and level + 1 not in db)
            or (model.condition_dim != 0 and level + 1 in db))


    size = (batch_size, *size)

    def sampling_f(data):
        return F.gumbel_softmax(data, tau=tau, dim=1, hard=True)

    for i in tqdm(range(num_samples // batch_size)):
        if level+1 not in db:
            condition_uuids = [None for _ in range(batch_size)]
            condition = None
        else:
            condition_uuids = get_condition_uuids(db, level, batch_size)
            condition = get_conditions(db, level, condition_uuids).to(GPU)

        with torch.cuda.amp.autocast(), torch.no_grad():
            for data, cond_uuid in zip(
                model.sample(size=size, condition=condition, sampling_f=sampling_f),
                condition_uuids
            ):
                db[level][uuid4()] = {'data': data.cpu(), 'condition': cond_uuid}

    save_db(db, db_path, level)



if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))