from typing import Union, Callable
from operator import attrgetter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


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


class FixupCausalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask: str = 'B',  # options are ('A', 'B')
        out: bool = False,
        activation: Callable = nn.ELU,
        use_dropout: bool = True,
        dropout_prob: float = 0.5
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

        self.dropout = nn.Dropout3d(dropout_prob) if use_dropout else None

    def forward(self, stack):
        out = self.branch_conv1(stack + self.bias1a)
        out = self.activation(out + self.bias1b)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out = out + (stack if self.skip_conv is None else self.skip_conv(stack))

        if not self.out:
            out = self.activation(out)

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):
        m = 2 # number of convs in a branch

        weight_getter = attrgetter('depth_conv.weight', 'height_conv.weight', 'width_conv.weight')

        # branch_conv1
        for weight in weight_getter(self.branch_conv1):
            torch.nn.init.normal_(
                weight,
                mean=0,
                std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
            )
            # torch.nn.init.kaiming_normal_(weight).mul_(num_layers ** (-1 / (2*m - m)))

        # branch_conv2
        for weight in weight_getter(self.branch_conv2):
            torch.nn.init.constant_(weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            for weight in weight_getter(self.skip_conv):
                init_method = torch.nn.init.kaiming_normal_ if not self.out else torch.nn.init.xavier_normal_
                # init_method = torch.nn.init.kaiming_normal_
                init_method(weight)
            bias_getter = attrgetter('depth_conv.bias', 'height_conv.bias', 'width_conv.bias')
            for bias in bias_getter(self.skip_conv):
                torch.nn.init.constant_(tensor=bias, val=0)


class PreActivationFixupCausalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask: str = 'B',  # options are ('A', 'B')
        condition_dim: int = 0,
        activation: Callable = nn.SiLU,
        use_dropout: bool = True,
        dropout_prob: float = 0.5,
        out: bool = False, # Doesn't do anything
    ):
        super(PreActivationFixupCausalResBlock, self).__init__()

        self.out = out

        self.bias1a, self.bias1b, self.bias2a, self.bias2b, self.bias3a, self.bias3b, self.bias4 = (
            nn.Parameter(torch.zeros(1)) for _ in range(7)
        )
        self.scale = nn.Parameter(torch.ones(1))

        branch_channels = max(in_channels, out_channels)
        self.branch_conv1 = CausalConv3dAdd(
            in_channels=in_channels, out_channels=branch_channels, kernel_size=1, mask=mask, bias=False
        )

        # second conv in the branch is always ok to be 'B'
        self.branch_conv2 = CausalConv3dAdd(
            in_channels=branch_channels, out_channels=branch_channels, kernel_size=kernel_size, mask='B', bias=False
        )

        # same for third
        self.branch_conv3 = CausalConv3dAdd(
            in_channels=branch_channels, out_channels=out_channels, kernel_size=1, mask='B', bias=False
        )

        # 1x1x1 conv for channel dim projection
        # Also needed if mask == 'A', since otherwise causality is broken
        self.skip_conv = CausalConv3dAdd(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, mask=mask, bias=True
        ) if (in_channels != out_channels or mask == 'A') else None

        if condition_dim > 0:
            condition_kernel_size = 3
            self.condition = nn.Conv3d(
                in_channels=condition_dim,
                out_channels=branch_channels,
                kernel_size=condition_kernel_size,
                padding=condition_kernel_size // 2
            )

        self.activation = activation()

        self.dropout = nn.Dropout3d(dropout_prob) if use_dropout else None

    def forward(self, stack, *, condition=None):
        out = self.activation(stack + self.bias1a)
        out = self.branch_conv1(out + self.bias1b)

        if condition is not None:
            assert self.condition is not None, 'Condition projection matrix not initialised!'
            out = out + self.condition(condition)

        out = self.activation(out + self.bias2a)
        out = self.branch_conv2(out + self.bias2b)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.activation(out + self.bias3a)
        out = self.branch_conv3(out + self.bias3b)

        out = out * self.scale + self.bias4

        out = out + (stack if self.skip_conv is None else self.skip_conv(stack))

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):
        m = 2 # number of convs in a branch

        weight_getter = attrgetter('depth_conv.weight', 'height_conv.weight', 'width_conv.weight')

        # branch_conv1
        for weight in weight_getter(self.branch_conv1):
            torch.nn.init.normal_(
                weight,
                mean=0,
                std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
            )
            # torch.nn.init.kaiming_normal_(weight).mul_(num_layers ** (-1 / (2*m - m)))

        # branch_conv2
        for weight in map(weight_getter, (self.branch_conv2, self.branch_conv3)):
            torch.nn.init.constant_(weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            for weight in weight_getter(self.skip_conv):
                # init_method = torch.nn.init.kaiming_normal_ if not self.out else torch.nn.init.xavier_normal_
                init_method = torch.nn.init.xavier_normal_
                init_method(weight)
            bias_getter = attrgetter('depth_conv.bias', 'height_conv.bias', 'width_conv.bias')
            for bias in bias_getter(self.skip_conv):
                torch.nn.init.constant_(tensor=bias, val=0)

