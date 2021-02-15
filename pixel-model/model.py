import math
from functools import partial
from itertools import product, chain
from argparse import ArgumentParser, Namespace
from typing import Union, Callable, Optional, Type, Dict, Tuple, Any, List, Sequence
from operator import add, mul, attrgetter
from collections import deque

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from einops import rearrange, repeat
from pytorch_lightning.metrics import Accuracy, Precision, Recall
from tqdm import tqdm

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
                in_channels=self.input_dim,
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
                concat_activation=self.use_concat_activation,
        )

        self.layers = nn.ModuleList([
            self.causal_conv(
                in_channels=self.model_dim,
                out_channels=self.model_dim,
                kernel_size=self.kernel_size,
                mask='B',
                dropout_prob=self.dropout_prob,
                condition_dim=(
                    self.model_dim // self.bottleneck_divisor
                    if self.condition_dim != 0
                    else 0
                ),
                bottleneck_divisor=self.bottleneck_divisor,
                concat_activation=self.use_concat_activation,
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
            b, c, *dim = data.size()

            condition = F.interpolate(
                idx_to_one_hot(condition, num_classes=self.condition_dim),
                size=dim, mode='trilinear'
            ).detach()
        data = data.squeeze(dim=1).detach()

        logits = self(
            data=idx_to_one_hot(data, num_classes=self.input_dim).detach(),
            condition=condition
        )

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
        self.use_concat_activation = args.use_concat_activation
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
        parser.add_argument('--use-concat-activation', default=False, type=booltype)
        parser.add_argument('--mixup-alpha', default=1, type=float,
                             help="Set to 1 to disable mixup")

        # Loss calculation specific
        parser.add_argument('--metric', choices=['cross_entropy'])

        # Optimizer specific arguments
        parser.add_argument('--lr', default=1e-5, type=float)
        return parser

    @torch.no_grad()
    def sample(
        self,
        size: Sequence[int],
        condition: torch.LongTensor,
        sampling_f: Callable[[torch.Tensor], Any] = partial(F.gumbel_softmax, tau=1, dim=1)
    ):
        '''Warning: if applicable, the user is responsible for calling model.eval()!'''
        assert len(size) == 4
        batch, *dims = size
        size = (batch, self.input_dim, *dims)

        result = torch.zeros(size, dtype=torch.half, device=self.device) # mypy: ignore

        if condition is not None:
            condition = F.interpolate(
                idx_to_one_hot(condition, num_classes=self.condition_dim),
                size=dims, mode='trilinear'
            ).to(torch.half)
            condition_cache = self._generate_condition_cache(condition)

         # iterate over all dimensions
        max_dim = [0 for _ in dims]
        for dim in tqdm(product(*tuple(range(dim) for dim in dims)), total=math.prod(dims)):
            for i, (d, max_d) in enumerate(zip(dim, max_dim)):
                max_dim[i] = max(d+1, max_d)

            current_slice = (Ellipsis, *(slice(max_d) for max_d in max_dim))
            current_sample = (Ellipsis, *(d for d in dim))

            out = self.forward(
                data=result[current_slice],
                condition_cache=(
                    condition_cache.copy()
                    if condition is not None
                    else None
                )
            )

            result[current_sample] = sampling_f(out[current_sample])

        return torch.argmax(result, dim=1)

    def _generate_condition_cache(self, condition) -> deque:
        # We could also just iterate over self.children(),
        # but that would be a bit risky considering that then the iteration order
        # would depend on the initialisation order.
        # Using the way below, we explicitely control the calling order of layers.
        return deque([
            (condition := layer.condition(condition))
            for layer in chain((self.to_causal,), self.layers)
        ])


    def forward( # mypy: ignore
        self,
        data: torch.LongTensor,
        condition: torch.LongTensor = None,
        # The args below are purely for performance reasons
        condition_cache: Union[deque, None] = None
    ): # type: ignore
        '''Note: the user is responsible for making sure that condition is an appropriate size'''

        stack = input_to_stack(self.parse_input(data))

        for layer in chain((self.to_causal,), self.layers):
            stack, condition, condition_cache = layer(stack, condition, condition_cache)

        return self.parse_output(stack_to_output(stack))
