import math
from argparse import ArgumentParser, Namespace
from typing import Union, Callable, Optional, Type, Dict, Tuple
from operator import add, mul, attrgetter

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from pytorch_lightning.metrics import Accuracy, Precision, Recall

from layers import FixupCausalResBlock, PreActFixupCausalResBlock, input_to_stack, stack_to_output
from utils.logging_helpers import sub_metric_log_dict
from utils.argparse_helpers import booltype


def bits_per_dim(mean_nll: torch.Tensor):
    '''Assumes the nll was calculated with the natural logarithm'''
    return mean_nll / np.log(2)

def idx_to_one_hot(data: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    return rearrange(F.one_hot(data, num_classes=num_classes),
                     'b d h w c -> b c d h w').to(torch.float)


class PixelSNAIL(pl.LightningModule):
    def __init__(self, args):
        super(PixelSNAIL, self).__init__()
        self.save_hyperparameters()
        self._parse_input_args(args)

        self.train_metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
        })
        self.val_metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
        })

        self.parse_input = nn.Conv3d(
                in_channels=self.input_dim + self.condition_dim,
                out_channels=self.model_dim,
                kernel_size=1
        )
        self.to_causal = self.causal_conv(
                in_channels=self.model_dim,
                out_channels=self.model_dim,
                kernel_size=self.kernel_size,
                mask='A',
                dropout_prob=self.dropout_prob,
                condition_dim=self.condition_dim,
                bottleneck_divisor=self.bottleneck_divisor,
        )

        self.layers = nn.ModuleList([
            self.causal_conv(
                in_channels=self.model_dim,
                out_channels=self.model_dim,
                kernel_size=self.kernel_size,
                mask='B',
                dropout_prob=self.dropout_prob,
                condition_dim=self.condition_dim,
                bottleneck_divisor=self.bottleneck_divisor,
            )
            for _ in range(self.num_resblocks)
        ])

        self.parse_output = nn.Conv3d(
            in_channels=self.model_dim,
            out_channels=self.input_dim,
            kernel_size=1,
        )

        num_layers = self.num_resblocks + 2 # plus input/output resblocks
        self.apply(
            lambda layer: layer.initialize_weights(num_layers=num_layers)
                          if isinstance(layer, FixupCausalResBlock)
                          else None
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

        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        loss, log_dict = self.loss_f(batch, batch_idx, metrics)

        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()})

        return loss

    def cross_entropy(self, batch, batch_idx, metrics: dict):

        if len(batch) == 1 or not self.use_conditioning: # no condition present
            data = batch[0]
            condition = None
        else:
            data, condition = batch
            condition = condition.squeeze(dim=1)
        data = data.squeeze(dim=1)

        logits = self(data, condition=condition)

        unreduced_loss = F.cross_entropy(input=logits, target=data, reduction='none')
        loss = unreduced_loss.mean()

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
        log_dict = {
            **sub_metric_log_dict('loss', unreduced_loss),
            'bits_per_dim': bits_per_dim(loss),
            **{metric_name: metric(probs, data) for metric_name, metric in metrics.items()}
        }

        return loss, log_dict

    def _parse_input_args(self, args: Namespace):

        self.input_dim, self.condition_dim = args.num_embeddings

        if not args.use_conditioning:
            self.condition_dim = 0

        # TODO: replace with attrsetter
        self.model_dim = args.model_dim
        self.kernel_size = args.kernel_size
        self.num_resblocks = args.num_resblocks
        self.dropout_prob = args.dropout_prob
        self.use_conditioning = args.use_conditioning
        self.bottleneck_divisor = args.bottleneck_divisor

        self.causal_conv = (
            PreActFixupCausalResBlock
            if args.use_pre_activation
            else FixupCausalResBlock
        )

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
        parser.add_argument('--num-resblocks', default=18, type=int)
        parser.add_argument('--dropout-prob', default=0.5, type=float,
                            help="Set to 0 to disable dropout.")
        parser.add_argument('--use-pre-activation', default=False, type=booltype)
        parser.add_argument('--bottleneck-divisor', default=4, type=int,
                            help=("Ignored if `--use-pre-activation False` is passed. "
                                  "Set to 1 to disable bottlenecking"))
        parser.add_argument('--use-conditioning', default=False, type=booltype)
        parser.add_argument('--mixup-alpha', default=1, type=float,
                             help="Set to 1 to disable mixup")

        # Loss calculation specific
        parser.add_argument('--metric', choices=['cross_entropy'])

        # Optimizer specific arguments
        parser.add_argument('--lr', default=1e-5, type=float)
        return parser

    def forward(self, data: torch.LongTensor, condition: torch.LongTensor = None): # type: ignore
        # will get modified in place in all layers if condition is not None
        # cache: Dict[torch.Size, torch.Tensor] = {}
        data = idx_to_one_hot(data, num_classes=self.input_dim)

        if condition is not None:
            b, c, *size = data.size()
            data = torch.cat([
                data,
                F.interpolate(idx_to_one_hot(condition, num_classes=self.condition_dim),
                              size=size, mode='trilinear')
            ], dim=1)

        stack = self.to_causal(input_to_stack(self.parse_input(data)))

        for layer in self.layers:
            stack = layer(stack)

        return self.parse_output(stack_to_output(stack))
