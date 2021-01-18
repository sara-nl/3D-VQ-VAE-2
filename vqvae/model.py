"""
This is largely a refactor of https://github.com/danieltudosiu/nmpevqvae
"""

from functools import partial
from typing import Tuple
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from layers import Encoder, Decoder, FixupResBlock
from utils import ExtractCenterCylinder
from metrics.distribution import Logistic, mixture_nll_loss, generic_nll_loss
from metrics.evaluate import nmse, psnr, SSIM3DSlices
from utils import sub_metric_log_dict


@torch.no_grad()
def _eval_metrics_log_dict(orig, pred):
    # FIXME: remove hardcoded data range
    metrics = (
        ('nmse', nmse),
        ('psnr', partial(psnr, data_range=4)),
    )
    return {
        func_name: func(orig, pred)
        for func_name, func in metrics
    }


class VQVAE(pl.LightningModule):

    # first in line is the default
    supported_metrics = ("huber",)

    def __init__(self, args: Namespace):

        super(VQVAE, self).__init__()

        self.save_hyperparameters()
        self._parse_input_args(args)

        self.encoder = Encoder(
            in_channels=self.input_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_down_per_enc=self.n_blocks_per_bottleneck,
            num_embeddings=self.num_embeddings
        )
        self.decoder = Decoder(
            out_channels=self.output_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
        )

        self.train_pre_loss_f_metrics = nn.ModuleDict({
            # 'ssim': SSIM3DSlices(),
        })
        self.val_pre_loss_f_metrics = nn.ModuleDict({
            'ssim': SSIM3DSlices(),
        })

        self.apply(lambda layer: layer.initialize_weights(num_layers=self.num_layers) if isinstance(layer, FixupResBlock) else None)

    def forward(self, data):
        commitment_loss, quantizations, encoding_idx = zip(*self.encode(data))

        decoded = self.decode(quantizations)
        return decoded, (commitment_loss, quantizations, encoding_idx)

    def encode(self, data):
        return self.encoder(data)

    def decode(self, quantizations):
        return self.decoder(quantizations)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val')

        pre_loss_f_metrics = (
            self.train_pre_loss_f_metrics
            if mode == 'train'
            else self.val_pre_loss_f_metrics
        )

        loss, log_dict = self.recon_loss_f(batch, batch_idx, **pre_loss_f_metrics)
        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()}, logger=True)

        return loss

    def loc_metric(self, batch, batch_idx, loss_f, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        x, num_valid_slices = batch

        loc, (commitment_loss, *_) = self(x)
        loc = F.softplus(loc)

        masks = torch.zeros_like(x, dtype=torch.bool, requires_grad=False)
        for idx, mask in zip(num_valid_slices, masks):
            mask[..., idx:] += True

        # Set padded values to 0
        torch.masked_fill(loc, mask=masks, value=0)

        log_dict = {}

        # Pre loss metrics need full 3d
        log_dict.update({
            k: v for d in
            (sub_metric_log_dict(metric_name, metric(loc, x)) for metric_name, metric in pre_loss_f_metrics.items())
            for k, v in d.items()
        })

        if self.pre_loss_f:
            loc, x = map(self.pre_loss_f, (loc, x))

        unreduced_recon_loss = loss_f(loc, x, reduction='none')

        for log_metric in (
            sub_metric_log_dict('recon_loss', unreduced_recon_loss),
            {f'commitment_loss_{i}': commitment_loss[i] for i in range(len(commitment_loss))},
            sub_metric_log_dict('loc', loc),
            _eval_metrics_log_dict(orig=x, pred=loc),
        ):
            log_dict.update(log_metric)

        loss = unreduced_recon_loss.mean() + sum(commitment_loss)

        return loss, log_dict

    def huber(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.smooth_l1_loss, **pre_loss_f_metrics)

    def _parse_input_args(self, args: Namespace):
        assert args.metric in self.supported_metrics

        if args.metric == 'huber':
            self.recon_loss_f = self.huber

        self.metric = args.metric
        self.lr = args.base_lr

        self.input_channels = args.input_channels
        self.output_channels = args.input_channels
        self.base_network_channels = args.base_network_channels
        self.n_bottleneck_blocks = args.n_bottleneck_blocks
        self.n_blocks_per_bottleneck = args.n_blocks_per_bottleneck

        assert len(args.num_embeddings) in (1, args.n_bottleneck_blocks)
        if len(args.num_embeddings) == 1:
            self.num_embeddings = [args.num_embeddings[0] for _ in range(args.n_bottleneck_blocks)]
        else:
            self.num_embeddings = args.num_embeddings

        self.num_layers = (
            (3 * args.n_bottleneck_blocks - 1)
            * args.n_blocks_per_bottleneck
            + (2 * args.n_bottleneck_blocks)
        )

        self.pre_loss_f = ExtractCenterCylinder() if args.extract_center_cylinder else None


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model specific arguments
        parser.add_argument('--input-channels', type=int, default=1)
        parser.add_argument('--base-network_channels', type=int, default=4)
        parser.add_argument('--n-bottleneck-blocks', type=int, default=3)
        parser.add_argument('--n-blocks-per-bottleneck', type=int, default=2)
        parser.add_argument('--num-embeddings', type=int, default=256, nargs='+',
                            help=("Can be either a single int or multiple."
                                  " If multiple, number of args should be equal to n-bottleneck-blocks"))

        # loss calculation specific
        parser.add_argument('--extract-center-cylinder', type=bool, default=True)
        parser.add_argument('--metric', choices=cls.supported_metrics, default=cls.supported_metrics[0])

        # Optimizer specific arguments
        parser.add_argument('--base_lr', default=1e-5, type=float)

        # Loss function specific arguments
        # FIXME: put this is a class or something, at least not here
        parser.add_argument('--n-mix', default=2)

        return parser
