import math
from functools import partial, lru_cache
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

from layers import CausalAttentionPixelBlock, PreActFixupCausalResBlock, input_to_stack, stack_to_output, GatedResBlock
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
        super().__init__()
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

        condition_dim = self.model_dim if self.use_conditioning else 0

        self.embed_condition = nn.Conv3d(
            in_channels=self.condition_dim,
            out_channels=condition_dim,
            kernel_size=1
        ) if self.use_conditioning else None

        causal_conv = partial(
            self.causal_conv,
            in_channels=self.model_dim,
            out_channels=self.model_dim,
            kernel_size=self.kernel_size,
            dropout_prob=self.causal_dropout_prob,
            condition_dim=condition_dim,
            condition_kernel_size=1,
            bottleneck_divisor=self.bottleneck_divisor,
        )

        self.to_causal = causal_conv(mask='A')

        self.layers = nn.ModuleList([
            CausalAttentionPixelBlock(
                in_channels=self.model_dim,
                bottleneck_divisor=self.bottleneck_divisor,
                causal_conv=partial(causal_conv, mask='B'),
                num_layers=self.num_layers_per_block
            )
            for _ in range(self.num_blocks)

        ])

        self.parse_output = nn.Conv3d(
            in_channels=self.model_dim,
            out_channels=self.input_dim,
            kernel_size=1,
        )

        num_layers = self.num_blocks*self.num_layers_per_block +1 # plus input resblock
        self.apply(
            lambda layer: layer.initialize_weights(num_layers=num_layers)
                          if isinstance(layer, PreActFixupCausalResBlock)
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
        # no_grad to save some MBs
        with torch.no_grad():

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
                )

            data = data.squeeze(dim=1)
            background = self._generate_background((data.shape[0], *data.shape[-3:]))
            attn_mask = self._generate_attention_mask(data.shape[-3:])

        logits = self(
            data=idx_to_one_hot(data, num_classes=self.input_dim),
            background=background, 
            attn_mask=attn_mask,
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
        args.use_gated_block = False

        self.input_dim, self.condition_dim = args.num_embeddings

        if not args.use_conditioning:
            self.condition_dim = 0

        for arg_name in (
            'model_dim',
            'kernel_size',
            'num_layers_per_block',
            'num_blocks',
            'causal_dropout_prob',
            'attention_dropout_prob',
            'bottleneck_divisor',
            'use_conditioning'
        ):
            setattr(self, arg_name, getattr(args, arg_name))

        # self.model_dim = args.model_dim
        # self.kernel_size = args.kernel_size
        # self.num_resblocks = args.num_resblocks
        # self.causal_dropout_prob = args.causal_dropout_prob
        # self.attention_dropout_prob = args.dropout_prob
        # self.use_gated_block = args.use_gated_block
        # self.use_conditioning = args.use_conditioning
        # self.use_concat_activation = args.use_concat_activation
        # self.bottleneck_divisor = args.bottleneck_divisor

        self.causal_conv = PreActFixupCausalResBlock

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
        parser.add_argument('--num-layers-per-block', default=5, type=int)
        parser.add_argument('--num-blocks', default=5, type=int)
        parser.add_argument('--causal-dropout-prob', default=0.5, type=float)
        parser.add_argument('--attention-dropout-prob', default=0.5, type=float,
                            help="Set to 0 to disable dropout.")
        parser.add_argument('--bottleneck-divisor', default=4, type=int,
                            help="Set to 1 to disable bottlenecking")
        parser.add_argument('--use-conditioning', default=False, type=booltype)

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

        result = torch.full(size, -1, dtype=torch.half, device=self.device) # mypy: ignore

        if condition is not None:
            condition = F.interpolate(
                idx_to_one_hot(condition, num_classes=self.condition_dim),
                size=dims, mode='trilinear'
            ).to(torch.half)
            condition_cache = self._generate_condition_cache(condition)

        max_dim = [0 for _ in dims]
        for dim in tqdm(product(*tuple(range(dim) for dim in dims)), total=math.prod(dims)):
            for i in range(len(max_dim)):
                max_dim[i] = max(dim[i]+1, max_dim[i])

            current_slice = (..., *(slice(max_d) for max_d in max_dim))
            current_sample = (..., *(d for d in dim))

            out = self.forward(
                data=result[current_slice],
                condition_cache=(
                    condition_cache.copy()
                    if condition is not None
                    else None
                )
            )

            sample = sampling_f(out[current_sample])

            result[current_sample] = sample

        # FIXME: remove the argmax
        return torch.argmax(result, dim=1)

    def _generate_condition_cache(self, condition) -> deque:
        # We could also just iterate over self.children(),
        # but that would be a bit risky considering that then the iteration order
        # would depend on the initialisation order.
        # Using the way below, we explicitely control the calling order of layers.
        condition = self.embed_condition(condition)
        return deque([layer.condition(condition) for layer in self.layers])


    @lru_cache(maxsize=1)
    def _generate_background(self, sizes):
        # TODO: refactor
        # Yes, the shapes are that big
        # dim=2 is the channel dim
        b, d, h, w = sizes
        return torch.cat([
            torch.linspace(start=-1, end=1, steps=d).view(1, 1, 1, -1, 1, 1).expand(3, b, 1, d, h, w),
            torch.linspace(start=-1, end=1, steps=h).view(1, 1, 1, 1, -1, 1).expand(3, b, 1, d, h, w),
            torch.linspace(start=-1, end=1, steps=w).view(1, 1, 1, 1, 1, -1).expand(3, b, 1, d, h, w)
        ], dim=2).to(self.device)
    
    @lru_cache(maxsize=1)
    def _generate_attention_mask(self, sizes):
        size = math.prod(sizes)
        return torch.tril(torch.ones((size, size))).to(self.device, torch.bool)


    def forward( # type: ignore
        self,
        data: torch.LongTensor,
        background: torch.Tensor,
        attn_mask: torch.BoolTensor,
        condition: torch.LongTensor = None,
        condition_cache: Union[deque, None] = None
    ): # type: ignore
        '''Note: the user is responsible for making sure that condition is an appropriate size'''

        stack = input_to_stack(self.parse_input(data))

        stack = self.to_causal(stack, condition=condition)
        if self.embed_condition is not None and condition_cache is None:
            condition = self.embed_condition(condition)

        for layer in self.layers:
            stack = layer(stack, background, attn_mask, condition, condition_cache)

        return self.parse_output(stack_to_output(stack))
