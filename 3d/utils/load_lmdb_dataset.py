import pickle
from typing import List, Tuple
from pathlib import Path

import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    def __init__(
        self,
        root: str,
        embedding_id: int = -1,
        transform=None,
    ):

        with lmdb.open(root, readonly=True) as env:
            with env.begin() as txn:
                self.length = int(txn.get(b"length"))
                self.n_enc = int(txn.get(b"num_dbs"))
                self.num_embeddings = txn.get(b"num_embeddings")

        assert self.n_enc >= 1
        self.env = lmdb.open(
            root,
            readonly=True,
            max_dbs=self.n_enc,
            lock=False,
            meminit=False
        )
        self.sub_dbs = [self.env.open_db(f"{i}".encode()) for i in range(self.n_enc)]
        self.transform = transform

        # condition one embedding on the get_embeddings-1 ones, ignored if id == -1
        self.get_embeddings = 2
        self.embedding_id = embedding_id

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> List[np.array]:
        '''returns the last self.get_embeddings embeddings'''
        idx = range(self.n_enc) if self.embedding_id == -1 else range(self.embedding_id, self.n_enc)[:self.get_embeddings]
        with self.env.begin() as txn:
            embeddings = [pickle.loads(txn.get(str(index).encode(), db=self.sub_dbs[i])) for i in idx]

        if self.transform is not None:
            embeddings = [
                transform(embedding)
                for transform, embedding in zip(
                    (self.transform[i] for i in idx),
                    embeddings
                )
            ]

        return embeddings

if __name__ == '__main__':
    path = '../vqvae/codes/version_6991397_last_hotfixed.lmdb'
    dataset = LMDBDataset(path)
    datapoint = dataset[0]
    breakpoint()