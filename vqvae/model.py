"""
This is largely a refactor of https://github.com/danieltudosiu/nmpevqvae
"""

from functools import partial
from typing import Tuple
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
    supported_metrics = ("mse", "rmse", "rmsle", "mae", "huber", "normal_nll", "logistic_mixture_nll", "normal_mixture_nll")

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
            'ssim': SSIM3DSlices(),
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

    def normal_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        loc, log_scale = torch.split(self(x), self.input_channels, dim=1)
        log_scale = torch.max(log_scale, torch.Tensor([-7]).to(device=self.device)) # lower bound log_scale

        if self.pre_loss_f:
            loc, log_scale, x = map(self.pre_loss_f, (loc, log_scale, x))

        unreduced_loss = generic_nll_loss(x, Normal, loc=loc, scale=log_scale.exp(), reduce_sum=False)

        log_dict = {
            **sub_metric_log_dict('loss', unreduced_loss),
            **sub_metric_log_dict('loc', loc),
            **sub_metric_log_dict('log_scale', log_scale),
        }

        return unreduced_loss.mean(), log_dict

    def loc_metric(self, batch, batch_idx, loss_f, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        x, num_valid_slices = batch

        loc, (commitment_loss, *_) = self(x)
        # loc = F.softplus(loc)

        with torch.no_grad():
            masks = torch.zeros_like(x, dtype=torch.bool)
            for idx, mask in zip(num_valid_slices, masks):
                mask[..., idx:] += True

            # Set padded values to 0
            loc = torch.masked_fill(loc, mask=masks, value=0)

            # Pre loss metrics need full 3d
            log_dict = {
                k: v for d in
                (sub_metric_log_dict(metric_name, metric(loc, x)) for metric_name, metric in pre_loss_f_metrics.items())
                for k, v in d.items()
            }

            if self.pre_loss_f:
                loc, x = map(self.pre_loss_f, (loc, x))

        unreduced_recon_loss = loss_f(loc, x, reduction='none')

        with torch.no_grad():
            log_dict = {
                **log_dict,
                **sub_metric_log_dict('recon_loss', unreduced_recon_loss),
                **{f'commitment_loss_{i}': commitment_loss[i] for i in range(len(commitment_loss))},
                **sub_metric_log_dict('loc', loc),
                **_eval_metrics_log_dict(orig=x, pred=loc),
            }

        loss = unreduced_recon_loss.mean() + sum(commitment_loss)

        return loss, log_dict

    def mse(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.mse_loss, **pre_loss_f_metrics)

    def rmsle(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        def sle_loss(pred, actual, reduction='none'):
            assert reduction == 'none'
            return (((pred + 1).log() - (actual + 1).log()) ** 2)

        msle, log_dict = self.loc_metric(batch, batch_idx, sle_loss, **pre_loss_f_metrics)

        for key, value in log_dict.items():
            if 'loss' in key:
                log_dict[key] = value.sqrt()

        return msle.sqrt(), log_dict

    def rmse(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        mse, log_dict = self.mse(batch, batch_idx, **pre_loss_f_metrics)

        for key, value in log_dict.items():
            if 'loss' in key:
                log_dict[key] = value.sqrt()

        return mse.sqrt(), log_dict

    def mae(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.l1_loss, **pre_loss_f_metrics)

    def huber(self, batch, batch_idx, **pre_loss_f_metrics) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.smooth_l1_loss, **pre_loss_f_metrics)

    def loc_scale_mixture_nll(self, batch, batch_idx, loc_scale_dist) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        log_pi_k, loc, log_scale = torch.split(self(x), self.input_channels*self.n_mix, dim=1)
        loc = F.softplus(loc)
        log_scale = torch.max(log_scale, torch.Tensor([-7]).to(device=self.device)) # lower bound log_scale

        # XXX: these two lines below are untested
        if self.pre_loss_f:
            log_pi_k, loc, log_scale, x = map(self.pre_loss_f, (log_pi_k, loc, log_scale, x))

        unreduced_nll_loss = mixture_nll_loss(x, loc_scale_dist, self.n_mix, log_pi_k, loc=loc, scale=log_scale.exp(), reduce_sum=False)

        log_dict = {
            **sub_metric_log_dict('loss', unreduced_nll_loss),
            **sub_metric_log_dict('log_pi_k', log_pi_k),
            **sub_metric_log_dict('loc', loc),
            **sub_metric_log_dict('log_scale', log_scale),
        }

        return unreduced_nll_loss.mean(), log_dict

    def logistic_mixture_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        return self.loc_scale_mixture_nll(batch, batch_idx, Logistic)

    def normal_mixture_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        return self.loc_scale_mixture_nll(batch, batch_idx, Normal)

    def _parse_input_args(self, args: Namespace):
        assert args.metric in self.supported_metrics

        if args.metric == 'normal_nll':
            self.recon_loss_f = self.normal_nll
            out_multiplier = 2
        elif args.metric == 'mse':
            self.recon_loss_f = self.mse
            out_multiplier = 1
        elif args.metric == 'rmse':
            self.recon_loss_f = self.rmse
            out_multiplier = 1
        elif args.metric == 'mae':
            self.recon_loss_f = self.mae
            out_multiplier = 1
        elif args.metric == 'rmsle':
            self.recon_loss_f = self.rmsle
            out_multiplier = 1
        elif args.metric == 'huber':
            self.recon_loss_f = self.huber
            out_multiplier = 1
        elif args.metric == 'logistic_mixture_nll':
            self.recon_loss_f = self.logistic_mixture_nll
            self.n_mix = args.n_mix
            out_multiplier = args.n_mix * 3
        elif args.metric == 'normal_mixture_nll':
            self.recon_loss_f = self.normal_mixture_nll
            self.n_mix = args.n_mix
            out_multiplier = args.n_mix * 3

        self.metric = args.metric
        self.lr = args.base_lr

        self.input_channels = args.input_channels
        self.output_channels = args.input_channels * out_multiplier
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
