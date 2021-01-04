# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

import math
from argparse import ArgumentParser, Namespace
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from utils.logging_helpers import sub_metric_log_dict


def shift_backwards_3d(input, size=1):
    '''
    If an image is given by (b c d h w)
    Front-Pads the image in the h dimension by `size`
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
    b, c, d, h, w = input.shape
    return F.pad(input, (0, 0, 0, 0, size, 0))[..., :d, :, :]


def shift_down_3d(input, size=1):
    ''''
    If an image is given by (b c d h w)
    Top-Pads the image in the h dimension by `size`
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
    b, c, d, h, w = input.shape
    return F.pad(input, (0, 0, size, 0, 0, 0))[..., :h, :]

def shift_right_3d(input, size=1):
    '''
    If an image is given by (b c d h w)
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
    b, c, d, h, w = input.shape
    return F.pad(input, (size, 0, 0, 0, 0, 0))[..., :w]


class CausalConv3dAdd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        mask: str = 'B'
    ):
        super(CausalConv3dAdd, self).__init__()
        assert mask in ('A', 'B')
        self.mask = mask

        assert kernel_size % 2 == 1, "even kernel sizes are not supported"
        assert kernel_size >= 3
        depth_size = kernel_size-1
        height_size = kernel_size-1
        width_size = kernel_size//2 + (1 if mask == 'B' else 0)

        self.half_kernel = kernel_size // 2

        # Split depth, height, width conv into three, allowing the receptive field to grow without blindspot
        # See https://papers.nips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf figure 1.
        self.depth_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(depth_size, kernel_size, kernel_size), bias=bias)
        self.height_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, height_size, kernel_size), bias=bias)
        self.width_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, width_size), bias=bias)

        # padding is always (*width, *height, *depth), i.e. (left, right, top, bottom, front, back)
        half_kernel = kernel_size // 2
        self.depth_pad = (half_kernel, half_kernel, half_kernel, half_kernel, depth_size-1, 0)
        self.height_pad = (half_kernel, half_kernel, height_size-1, 0, 0, 0)
        self.width_pad = (width_size-1, 0, 0, 0, 0, 0)


    def forward(self, depth, height, width):
        depth  = self.depth_conv(F.pad(depth, pad=self.depth_pad))
        height = self.height_conv(F.pad(height, pad=self.height_pad))
        width = self.width_conv(F.pad(width, pad=self.width_pad))

        # width = shift_backwards_3d(depth) + shift_down_3d(height) + (shift_right_3d(width) if self.mask == 'A' else width)
        width = shift_right_3d(width) if self.mask == 'A' else width

        return depth, height, width

    @staticmethod
    def stacks_to_output(depth, height, width):
        return shift_backwards_3d(depth) + shift_down_3d(height) + width

class PixelSNAIL(pl.LightningModule):
    def __init__(self, args):
        super(PixelSNAIL, self).__init__()
        self.save_hyperparameters()
        self._parse_input_args(args)

        self.mask_a = CausalConv3d(self.main_dim, 32, emulated_kernel_size=5, mask='A')

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
            'b h w d c -> b c d h w'
        ).to(torch.float))

if __name__ == '__main__':
    inp = torch.zeros((1, 3, 20,20,20))

    inp[0, :, 9, 9, 9] = 1

    a_mask = CausalConv3dAdd(1, 1, 3, bias=False, mask='A')
    b_mask = CausalConv3dAdd(1, 1, 3, bias=False, mask='B')


    for layer in (a_mask, b_mask):
        layer.depth_conv.weight[:] = 1
        layer.height_conv.weight[:] = 1
        layer.width_conv.weight[:] = 1

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    depth, height, width = inp[:, 0][None], inp[:, 1][None], inp[:, 2][None]

    layer_index = 10
    imgs = [torch.log(width[..., layer_index, :, :] + 1).detach()]

    depth, height, width = a_mask(depth=depth, height=height, width=height)
    for i in range(1, 10):
        imgs.append(torch.log(CausalConv3dAdd.stacks_to_output(depth, height, width)[..., layer_index, :, :] + 1).detach())
        depth, height, width = b_mask(depth, height, width)

    plot = make_grid(torch.cat(imgs), nrow=5, padding=0)
    plt.imshow(plot[0])
    plt.show()
