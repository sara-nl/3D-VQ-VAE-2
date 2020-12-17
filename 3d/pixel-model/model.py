# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from utils.logging_helpers import sub_metric_log_dict


def shift_down_3d(input, size=1):
    '''
    If an image is given by (b c h w d)
    Left-Pads the image in the h dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [1,2]],
     [[0,0],
      [5,6]]]
    '''
    b, c, h, w, d = input.shape
    return F.pad(input, (0, 0, 0, 0, size, 0))[..., :h, :, :]


def shift_right_3d(input, size=1):
    ''''
    If an image is given by (b c h w d)
    Left-Pads the image in the w dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,1],
      [0,3]],
     [[0,5],
      [0,7]]]
    '''
    b, c, h, w, d = input.shape
    return F.pad(input, (0, 0, size, 0, 0, 0))[..., :w, :]

def shift_backward_3d(input, size=1):
    '''
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [0,0]],
     [[1,2],
      [3,4]]]
    '''
    b, c, h, w, d = input.shape
    return F.pad(input, (size, 0, 0, 0, 0, 0))[..., :d]


class PixelSNAIL(pl.LightningModule):
    def __init__(self, args):
        super(PixelSNAIL, self).__init__()
        self.save_hyperparameters()
        self._parse_input_args(args)

        self.layers = nn.Sequential(
            nn.Conv3d(self.main_dim, self.main_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.main_dim // 2, self.main_dim, kernel_size=3, padding=1)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val')

        loss, log_dict = self.loss_f(batch, batch_idx)

        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()})

        return loss

    def cross_entropy(self, batch, batch_idx):
        data, condition = batch
        data, condition = data.squeeze(dim=1), condition.squeeze(dim=1)

        logits = self(data, condition=condition)

        unreduced_loss = F.cross_entropy(input=logits, target=data, reduction='none')

        log_dict = {
            **sub_metric_log_dict('loss', unreduced_loss),
        }

        return unreduced_loss.mean(), log_dict

    def _parse_input_args(self, args: Namespace):
        if args.metric == 'cross_entropy':
            self.loss_f = self.cross_entropy
        else:
            raise ValueError

        self.lr = args.lr
        self.main_dim, self.condition_dim = args.num_embeddings

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model specific arguments

        # Loss calculation specific
        parser.add_argument('--metric', choices=['cross_entropy'])

        # Optimizer specific arguments
        parser.add_argument('--lr', default=1e-5, type=float)
        return parser

    def forward(self, data, condition=None, cache=None):
        return self.layers(rearrange(
            F.one_hot(data, num_classes=self.main_dim),
            'b h w d c -> b c h w d'
        ).to(torch.float))