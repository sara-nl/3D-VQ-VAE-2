# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

import math
from argparse import ArgumentParser, Namespace
from typing import Union
from operator import add, mul, attrgetter
from itertools import starmap

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from utils.logging_helpers import sub_metric_log_dict

def add_scalar_to_stack(bias, stack):
    return starmap(add, product((bias,), stack))

def multiply_scalar_with_stack(scale, stack):
    return starmap(mul, product((scale,), stack))

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


class FixupCausalResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask='B',
        out=False,
        activation = nn.ELU
    ):
        super(FixupCausalResBlock, self).__init__()

        self.out = out

        self.bias1a, self.bias1b, self.bias2a, self.bias2b = (
            nn.Parameter(torch.zeros(1)) for _ in range(4)
        )
        self.scale = nn.Parameter(torch.ones(1))

        branch_channels = max(in_channels, out_channels)
        self.branch_conv1 = CausalConv3dAdd(
            in_channels=in_channels, out_channels=branch_channels, kernel_size=kernel_size, mask=mask, bias=False
        )

        # second conv in the branch is always ok to be 'B'
        self.branch_conv2 = CausalConv3dAdd(
            in_channels=branch_channels, out_channels=out_channels, kernel_size=kernel_size, mask='B', bias=False
        )

        # 1x1x1 conv for channel dim projection
        self.skip_conv = CausalConv3dAdd(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, mask=mask, bias=True
        ) if (in_channels != out_channels or mask == 'A') else None

        self.activation = activation()

    def forward(self, stack):
        out = self.branch_conv1(stack + self.bias1a)
        out = self.activation(out + self.bias1b)

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out = out + (stack if self.skip_conv is None else self.skip_conv(stack))


        if not self.out:
            out = self.activation(out)

        return out
        # A bit ugly, but computationally probably better
        # than to have to call torch.chunk + torch.cat every forward pass
        # Maybe slicing one big tensor every pass is faster?

        # stack_out = self.branch_conv1(*add_scalar_to_stack(self.bias_1a, stack))
        # stack_out = map(self.activation, add_scalar_to_stack(self.bias1b, stack_out))

        # stack_out = self.branch_conv2(*add_scalar_to_stack(self.bias2a, stack))
        # stack_out = add_scalar_to_stack(self.bias2b, multiply_scalar_with_stack(self.scale, stack))

        # # skip connection
        # stack_out = starmap(add, zip(stack_out, stack))

        # if not self.out:
        #     stack_out = map(self.activation, stack_out)
        # else:
        #     stack_out = tuple(stack_out)

        # return stack_out

    def initialize_weights(self, num_layers):
        m = 2 # number of convs in a branch

        weight_getter = attrgetter('depth_conv.weight', 'height_conv.weight', 'width_conv.weight')

        # branch_conv1
        for weight in weight_getter(self.branch_conv1):
            torch.nn.init.kaiming_normal_(weight) * (num_layers ** (-1 / (2*m - m)))

        # branch_conv2
        for weight in weight_getter(self.branch_conv2):
            torch.nn.init.constant_(weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            for weight in weight_getter(self.skip_conv):
                init_method = torch.nn.init.kaiming_normal_ if not self.out else torch.nn.init.xavier_normal_
                init_method(weight)
            bias_getter = attrgetter('depth_conv.bias', 'height_conv.bias', 'width_conv.bias')
            for bias in bias_getter(self.skip_conv):
                torch.nn.init.constant_(tensor=bias, val=0)


class CausalConv3dAdd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        mask: str = 'B',
    ):
        super(CausalConv3dAdd, self).__init__()
        assert mask in ('A', 'B')
        self.mask = mask

        assert kernel_size > 0
        assert kernel_size % 2 == 1, "even kernel sizes are not supported"

        # Making sure kernel size is at least 1
        depth_size  = max(kernel_size-1, 1)
        height_size = max(kernel_size-1, 1)
        width_size  = max(kernel_size//2 + (1 if mask == 'B' else 0), 1)

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


    def forward(self, stack):
        depth, height, width = stack

        depth  = self.depth_conv(F.pad(depth, pad=self.depth_pad))
        height = self.height_conv(F.pad(height, pad=self.height_pad))
        width = self.width_conv(F.pad(width, pad=self.width_pad))

        width = shift_right_3d(width) if self.mask == 'A' else width

        return rearrange([depth, height, width], 'dim b c d h w -> dim b c d h w')

    @staticmethod
    def input_to_stack(input_):
        return repeat(input_, 'b c d h w -> dim b c d h w', dim=3)

    @staticmethod
    def stack_to_output(stack):
        depth, height, width = stack
        return shift_backwards_3d(depth) + shift_down_3d(height) + width


class PixelSNAIL(pl.LightningModule):
    def __init__(self, args):
        super(PixelSNAIL, self).__init__()
        self.save_hyperparameters()
        self._parse_input_args(args)

        self.parse_input = FixupCausalResBlock(
            in_channels=self.input_dim,
            out_channels=self.model_dim,
            kernel_size=self.kernel_size,
            mask='A'
        )

        self.layers = nn.ModuleList([
            FixupCausalResBlock(
                in_channels=self.model_dim,
                out_channels=self.model_dim,
                kernel_size=self.kernel_size,
                mask='B'
            )
            for _ in range(self.num_resblocks)
        ])

        self.parse_output = FixupCausalResBlock(
            in_channels=self.model_dim,
            out_channels=self.input_dim,
            kernel_size=self.kernel_size,
            mask='B',
            out=True
        )

        num_layers = self.num_resblocks + 2 # plus input/output resblocks
        self.apply(lambda layer: layer.initialize_weights(num_layers=num_layers) if isinstance(layer, FixupCausalResBlock) else None)

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

        if len(batch) == 1: # no condition present
            data = batch[0]
            condition = None
        else:
            data, condition = batch
            condition = condition.squeeze(dim=1)
        data = data.squeeze(dim=1)

        logits = self(data, condition=condition)

        unreduced_loss = F.cross_entropy(input=logits, target=data, reduction='none')

        log_dict = {
            **sub_metric_log_dict('loss', unreduced_loss),
        }

        return unreduced_loss.mean(), log_dict

    def _parse_input_args(self, args: Namespace):

        self.input_dim, self.condition_dim = args.num_embeddings

        # TODO: replace with attrsetter
        self.model_dim = args.model_dim
        self.kernel_size = args.kernel_size
        self.num_resblocks = args.num_resblocks

        if args.metric == 'cross_entropy':
            self.loss_f = self.cross_entropy
        else:
            raise ValueError

        self.lr = args.lr

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model specific arguments
        parser.add_argument('--model-dim', default=32, type=int)
        parser.add_argument('--kernel-size', default=3, type=int)
        parser.add_argument('--num-resblocks', default=5, type=int)

        # Loss calculation specific
        parser.add_argument('--metric', choices=['cross_entropy'])

        # Optimizer specific arguments
        parser.add_argument('--lr', default=1e-5, type=float)
        return parser

    def forward(self, data, condition=None, cache=None):
        stack = CausalConv3dAdd.input_to_stack(
            rearrange(F.one_hot(data, num_classes=self.input_dim),
                      'b d h w c -> b c d h w').to(torch.float)
        )

        stack = self.parse_input(stack)

        for layer in self.layers:
            stack = layer(stack)

        return CausalConv3dAdd.stack_to_output(self.parse_output(stack))
