import pickle
from typing import List, Tuple
from pathlib import Path

import lmdb
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, DataLoader


class LMDBDataModule(pl.LightningDataModule):
    def __init__(self, path, embedding_id, batch_size=16, train_frac=0.95, num_workers=6):
        super(LMDBDataModule, self).__init__()
        assert 0 <= train_frac <= 1

        self.path = path
        self.train_frac = train_frac
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.embedding_id = embedding_id

    def setup(self, stage=None):
        # transform
        transform = None

        dataset = LMDBDataset(self.path, self.embedding_id, transform=transform)
        self.n_enc = dataset.n_enc
        self.num_embeddings = dataset.num_embeddings

        train_len = int(len(dataset) * self.train_frac)
        val_len = len(dataset) - train_len

        # train/val split
        train_split, val_split = random_split(dataset, [train_len, val_len])

        # assign to use in dataloaders
        self.train_dataset = train_split
        self.train_len = train_len
        self.train_batch_size = self.batch_size

        self.val_dataset = val_split
        self.val_len = val_len
        self.val_batch_size = self.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, drop_last=True)



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
                num_embeddings = pickle.loads(txn.get(b"num_embeddings"))

        assert embedding_id < self.n_enc
        self.embedding_id = embedding_id

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
        # should always be 2
        get_embeddings = 2

        self._idx = range(self.n_enc) if self.embedding_id == -1 else range(embedding_id, self.n_enc)[:get_embeddings]
        self.num_embeddings = [num_embeddings[index] for index in self._idx]
        if len(self.num_embeddings) == 1:
            self.num_embeddings.append(0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> List[np.array]:
        '''returns the last self.get_embeddings embeddings'''
        
        with self.env.begin() as txn:
            embeddings = [pickle.loads(txn.get(str(index).encode(), db=self.sub_dbs[i])) for i in self._idx]

        if self.transform is not None:
            embeddings = [
                transform(embedding)
                for transform, embedding in zip(
                    (self.transform[i] for i in self._idx),
                    embeddings
                )
            ]

        return embeddings

if __name__ == '__main__':
    path = '../vqvae/codes/version_6991397_last.lmdb'
    dataset = LMDBDataset(path)
    datapoint = dataset[0]
