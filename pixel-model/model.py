import math
from argparse import ArgumentParser, Namespace
from typing import Union, Callable
from operator import add, mul, attrgetter

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from pytorch_lightning.metrics import Accuracy, Precision, Recall

from layers import FixupCausalResBlock, PreActivationFixupCausalResBlock, CausalConv3dAdd
from utils.logging_helpers import sub_metric_log_dict


def bits_per_dim(mean_nll: torch.Tensor):
    '''Assumes the nll was calculated with the natural logarithm'''
    return mean_nll / np.log(2)


class PixelSNAIL(pl.LightningModule):
    def __init__(self, args):
        super(PixelSNAIL, self).__init__()
        self.save_hyperparameters()
        self._parse_input_args(args)

        self.train_metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
            'precision': Precision(num_classes=self.input_dim),
            'recall': Recall(num_classes=self.input_dim),
        })
        self.val_metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
            'precision': Precision(num_classes=self.input_dim),
            'recall': Recall(num_classes=self.input_dim),
        })

        self.parse_input = self.causal_conv(
            in_channels=self.input_dim,
            out_channels=self.model_dim,
            kernel_size=self.kernel_size,
            mask='A',
            use_dropout=self.use_dropout,
            dropout_prob=self.dropout_prob
        )

        self.layers = nn.ModuleList([
            self.causal_conv(
                in_channels=self.model_dim,
                out_channels=self.model_dim,
                kernel_size=self.kernel_size,
                mask='B',
                use_dropout=self.use_dropout,
                dropout_prob=self.dropout_prob
            )
            for _ in range(self.num_resblocks)
        ])

        self.parse_output = self.causal_conv(
            in_channels=self.model_dim,
            out_channels=self.input_dim,
            kernel_size=self.kernel_size,
            mask='B',
            use_dropout=self.use_dropout,
            dropout_prob=self.dropout_prob,
            out=True,
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

        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        loss, log_dict = self.loss_f(batch, batch_idx, metrics)

        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()})

        return loss

    def cross_entropy(self, batch, batch_idx, metrics: dict):

        if len(batch) == 1: # no condition present
            data = batch[0]
            condition = None
        else:
            data, condition = batch
            condition = condition.squeeze(dim=1)
        data = data.squeeze(dim=1)

        logits = self(data, condition=condition)

        unreduced_loss = F.cross_entropy(input=logits, target=data, reduction='none')
        loss = unreduced_loss.mean()

        log_dict = {
            **sub_metric_log_dict('loss', unreduced_loss),
            'bits_per_dim': bits_per_dim(loss),
            **{metric_name: metric(logits, data) for metric_name, metric in metrics.items()}
        }

        return loss, log_dict

    def _parse_input_args(self, args: Namespace):

        self.input_dim, self.condition_dim = args.num_embeddings

        # TODO: replace with attrsetter
        self.model_dim = args.model_dim
        self.kernel_size = args.kernel_size
        self.num_resblocks = args.num_resblocks
        self.use_dropout = args.use_dropout
        self.dropout_prob = args.dropout_prob
        self.use_conditioning = args.use_conditioning

        self.causal_conv = (
            PreActivationFixupCausalResBlock
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
        parser.add_argument('--num-resblocks', default=5, type=int)
        parser.add_argument('--use-dropout', default=True, type=bool)
        parser.add_argument('--dropout-prob', default=0.5, type=float)
        parser.add_argument('--use-pre-activation', default=False, type=bool)
        parser.add_argument('--use-conditioning', default=False, type=bool)

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

        if condition is not None:
            condition = rearrange(
                F.one_hot(data, num_classes=self.input_dim),
                'b d h w c -> b c d h w'
            ).to(torch.float)

        stack = self.parse_input(stack, condition=condition)

        for layer in self.layers:
            stack = layer(stack, condition=condition)

        return CausalConv3dAdd.stack_to_output(self.parse_output(stack))
