"""
This is largely a refactor of https://github.com/danieltudosiu/nmpevqvae
"""

from itertools import zip_longest, chain
from functools import partial
from typing import Tuple
from argparse import ArgumentParser, Namespace
from math import prod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch import nn

from utils import ExtractCenterCylinder
from metrics.distribution import Logistic, mixture_nll_loss, generic_nll_loss
from metrics.evaluate import nmse, psnr, ssim3d


@torch.no_grad()
def _sub_metric_log_dict(sub_metric_name, sub_metric):
    with torch.no_grad():
        return {
            f'{sub_metric_name}_{func_name}': func(sub_metric)
            for func_name, func in (
                ('min', torch.min),
                ('max', torch.max),
                ('mean', torch.mean),
                ('median', torch.median),
                ('std', torch.std)
            )
        }

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

@torch.no_grad()
def _eval_ssim3d(orig, pred):
    '''SSIM needs to be seperate because of spatial dimensions'''
    return {'ssim3d': ssim3d(orig, pred)}


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
        )
        self.decoder = Decoder(
            out_channels=self.output_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
        )

        self.apply(lambda layer: layer.initialize_weights(num_layers=self.num_layers) if isinstance(layer, FixupResBlock) else None)

    def forward(self, data):
        _, quantizations, _, _, _ = zip(*self.encode(data)) # list transpose of encoder outputs
        decoded = self.decode(quantizations)
        return decoded

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

        loss, log_dict = self.recon_loss_f(batch, batch_idx)

        self.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()})

        return loss

    def normal_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        loc, log_scale = torch.split(self(x), self.input_channels, dim=1)
        log_scale = torch.max(log_scale, torch.Tensor([-7]).to(device=self.device)) # lower bound log_scale

        if self.pre_loss_f:
            loc, log_scale, x = map(self.pre_loss_f, (loc, log_scale, x))

        unreduced_loss = generic_nll_loss(x, Normal, loc=loc, scale=log_scale.exp(), reduce_sum=False)

        log_dict = {
            **_sub_metric_log_dict('loss', unreduced_loss),
            **_sub_metric_log_dict('loc', loc),
            **_sub_metric_log_dict('log_scale', log_scale),
        }

        return unreduced_loss.mean(), log_dict

    def loc_metric(self, batch, batch_idx, loss_f) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        loc = F.softplus(self(x))

        # ssim before cylinder extraction because that flattens xy dimensions
        eval_ssim_dict = _eval_ssim3d(x, loc)

        if self.pre_loss_f:
            loc, x = map(self.pre_loss_f, (loc, x))

        unreduced_loss = loss_f(loc, x, reduction='none')

        log_dict = {
            **_sub_metric_log_dict('loss', unreduced_loss),
            **_sub_metric_log_dict('loc', loc),
            **_eval_metrics_log_dict(orig=x, pred=loc),
            **eval_ssim_dict
        }

        return unreduced_loss.mean(), log_dict

    def mse(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.mse_loss)

    def rmsle(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        def sle_loss(pred, actual, reduction='none'):
            assert reduction == 'none'
            return (((pred + 1).log() - (actual + 1).log()) ** 2)

        msle, log_dict = self.loc_metric(batch, batch_idx, sle_loss)

        for key, value in log_dict.items():
            if 'loss' in key:
                log_dict[key] = value.sqrt()

        return msle.sqrt(), log_dict

    def rmse(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        mse, log_dict = self.mse(batch, batch_idx)

        for key, value in log_dict.items():
            if 'loss' in key:
                log_dict[key] = value.sqrt()

        return mse.sqrt(), log_dict

    def mae(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.l1_loss)

    def huber(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        return self.loc_metric(batch, batch_idx, F.smooth_l1_loss)

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
            **_sub_metric_log_dict('loss', unreduced_nll_loss),
            **_sub_metric_log_dict('log_pi_k', log_pi_k),
            **_sub_metric_log_dict('loc', loc),
            **_sub_metric_log_dict('log_scale', log_scale),
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

        # loss calculation specific
        parser.add_argument('--extract-center-cylinder', type=bool, default=True)
        parser.add_argument('--metric', choices=cls.supported_metrics, default=cls.supported_metrics[0])

        # Optimizer specific arguments
        parser.add_argument('--base_lr', default=1e-5, type=float)

        # Loss function specific arguments
        # FIXME: put this is a class or something, at least not here
        parser.add_argument('--n-mix', default=2)

        return parser



class DownBlock(nn.Module):
    def __init__(self, in_channels, n_down=2):
        super(DownBlock, self).__init__()

        self.layers = nn.Sequential(*(
            FixupResBlock(in_channels*2**i, in_channels*2**(i+1), mode='down')
            for i in range(n_down)
        ))

    def forward(self, data):
        return self.layers(data)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, aux_channels=0, n_up=2, mode='encoder'):
        super(UpBlock, self).__init__()

        assert mode in ('encoder', 'decoder')

        # Some slight channel size shenanigans required for the first layer because of possible concat beforehand
        self.layers = nn.Sequential(*filter(None, chain.from_iterable(
            (FixupResBlock(
                in_channels if i == n_up-1 else out_channels*(2**(i+1)),
                out_channels*(2**i),
                mode='up'
             ),
             FixupResBlock(
                out_channels*(2**i),
                out_channels*(2**i),
                mode='same'
             ) if mode == 'decoder' else None
            )
            for i in range(n_up-1, -1, -1)
        ))) # Like I'm programming lisp or something

    def forward(self, data):
        return self.layers(data)


class PreQuantization(nn.Module):
    def __init__(self, in_channels, out_channels, n_up=2):
        super(PreQuantization, self).__init__()
        self.has_aux = in_channels - out_channels * 8 != 0

        if self.has_aux:
            self.upsample = UpBlock(out_channels * 2 ** n_up, out_channels, n_up=n_up)

        self.pre_q = FixupResBlock(in_channels, out_channels, mode='same')

    def forward(self, data, auxilary=None):
        assert self.has_aux is (auxilary is not None)

        if self.has_aux:
            data = torch.cat([data, self.upsample(auxilary)], dim=1)

        return self.pre_q(data)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_network_channels, n_enc=3, n_down_per_enc=2):
        super(Encoder, self).__init__()

        self.parse_input = FixupResBlock(in_channels, base_network_channels, mode='same')

        before_channels = base_network_channels
        self.down, self.pre_quantize, self.quantize = (nn.ModuleList() for _ in range(3))
        for i in range(n_enc):
            after_channels = before_channels * 2 ** n_down_per_enc

            self.down.append(DownBlock(before_channels, n_down_per_enc))

            assert after_channels % 8 == 0
            embedding_dim = after_channels // 8

            self.pre_quantize.append(PreQuantization(
                in_channels=after_channels + (embedding_dim if i != n_enc-1 else 0),
                out_channels=embedding_dim,
                n_up=n_down_per_enc
            ))
            self.quantize.append(
                Quantizer(num_embeddings=512, embedding_dim=embedding_dim, commitment_cost=7)
            )

            before_channels = after_channels


    def forward(self, data):
        preprocessed = self.parse_input(data)

        down = preprocessed
        downsampled = [(down := downblock(down)) for downblock in self.down]

        aux = None
        quantizations = []
        for down, pre_quantize, quantize in reversed(list(zip(downsampled, self.pre_quantize, self.quantize))):
            pre_quantizion = pre_quantize(down, aux)

            quantization = (None, pre_quantizion, None, None, None)
            # quantization = quantize(pre_quantizion)

            quantizations.append(quantization)
            _, aux, _, _, _ = quantization



        return reversed(quantizations)


class Decoder(nn.Module):
    def __init__(self, out_channels, base_network_channels, n_enc=3, n_up_per_enc=2):
        super(Decoder, self).__init__()

        self.up = nn.ModuleList()
        after_channels = base_network_channels

        for i in range(n_enc):
            before_channels = after_channels * 2 ** n_up_per_enc

            assert before_channels % 8 == 0
            embedding_dim = before_channels // 8

            in_channels = embedding_dim + (before_channels if i != n_enc-1 else 0)

            self.up.append(UpBlock(
                in_channels=in_channels,
                out_channels=after_channels,
                n_up=n_up_per_enc,
                mode='decoder'
            ))

            after_channels = before_channels

        self.out = nn.Sequential(
            FixupResBlock(base_network_channels, base_network_channels, mode='same'),
            FixupResBlock(base_network_channels, out_channels, mode='out'),
        )

    def forward(self, quantizations):
        for i, (quantization, up) in enumerate(reversed(list(zip(quantizations, self.up)))):
            out = quantization if i == 0 else torch.cat([quantization, out], dim=1)
            out = up(out)

        out = self.out(out)

        return out


class ResizeConv3D(nn.Conv3d):
    def __init__(self, *conv_args, **conv_kwargs):
        super(ResizeConv3D, self).__init__(*conv_args, **conv_kwargs)
        self.upsample = nn.Upsample(mode='trilinear', scale_factor=2, align_corners=False)

    def forward(self, input):
        return super(ResizeConv3D, self).forward(self.upsample(input))


class FixupResBlock(torch.nn.Module):
    # Adapted from:
    # https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py#L20

    """
    FIXME: check wether the biases should be a single scalar or a vector
    """

    def __init__(self, in_channels, out_channels, mode, activation=nn.LeakyReLU):
        super(FixupResBlock, self).__init__()

        assert mode in ("down", "same", "up", "out")
        self.mode = mode

        self.bias1a, self.bias1b, self.bias2a, self.bias2b = (
            nn.Parameter(torch.zeros(1)) for _ in range(4)
        )
        self.scale = nn.Parameter(torch.ones(1))

        self.activation = activation()

        if mode == 'down':
            conv = nn.Conv3d
            kernel_size, stride, padding = 4, 2, 1
        elif mode in ('same', 'out'):
            conv = nn.Conv3d
            kernel_size, stride, padding = 3, 1, 1
        elif mode == 'up':
            conv = ResizeConv3D
            kernel_size, stride, padding = 3, 1, 1

        self.branch_conv1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )

        self.skip_conv1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True
        )

        self.branch_conv2 = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


    def forward(self, input):
        out = self.branch_conv1(input + self.bias1a)
        out = self.activation(out + self.bias1b)

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += self.skip_conv1(input) # linear projection of the input onto the output

        if self.mode != 'out':
            out = self.activation(out)

        return out

    def initialize_weights(self, num_layers):

        torch.nn.init.kaiming_normal_(self.branch_conv1.weight, nonlinearity='leaky_relu')

        torch.nn.init.constant_(tensor=self.branch_conv2.weight, val=0)

        torch.nn.init.kaiming_normal_(self.skip_conv1.weight)
        torch.nn.init.constant_(tensor=self.skip_conv1.bias, val=0)


class Quantizer(torch.nn.Module):
    # Code taken from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I

    """
    EMA-updated Vector Quantizer

    FIXME: Ugly-ass indexing of embedding table
    FIXME: usage of torch.matmul instead of '@' operator
    FIXME: Remove either output one-hot(TODO: check actually one-hot) or dense encoding indices
    """
    def __init__(
        self, num_embeddings : int, embedding_dim : int, commitment_cost : float, decay=0.99, epsilon=1e-5
    ):
        super(Quantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = torch.nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 4, 1)
        input_shape = inputs.shape

        flat_input = inputs.reshape(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = torch.nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = torch.nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach() # Trick to have identity backprop grads
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            loss,
            quantized.permute(0, 4, 1, 2, 3).contiguous(),
            perplexity,
            encodings,
            encoding_indices,
        )


class SubPixelConvolution3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=2, **conv_kwargs):
        super(SubPixelConvolution3D, self).__init__()
        assert upsample_factor == 2
        self.upsample_factor = upsample_factor

        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels * upsample_factor ** 3,
            **conv_kwargs
        )

        self.shuffle = PixelShuffle3D(
            upscale_factor=upsample_factor
        )

        self.initialize_weights()

    def forward(self, input):
        out = self.conv(input)
        out = self.shuffle(out)
        return out

    def initialize_weights(self):
        # Written by: Daniele Cattaneo (@catta202000)
        # Taken from: https://github.com/pytorch/pytorch/pull/5429/files

        new_shape = [
            int(self.conv.weight.shape[0] / (self.upsample_factor ** 2))
        ] + list(self.conv.weight.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = torch.nn.init.xavier_normal_(subkernel)
        subkernel = subkernel.transpose(0, 1)

        subkernel = subkernel.contiguous().view(
            subkernel.shape[0], subkernel.shape[1], -1
        )

        kernel = subkernel.repeat(1, 1, self.upsample_factor ** 2)

        transposed_shape = (
            [self.conv.weight.shape[1]]
            + [self.conv.weight.shape[0]]
            + list(self.conv.weight.shape[2:])
        )
        kernel = kernel.contiguous().view(transposed_shape)

        kernel = kernel.transpose(0, 1)

        self.conv.weight.data = kernel


class PixelShuffle3D(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
        self.upscale_factor_cubed = upscale_factor ** 3
        self._shuffle_out = None
        self._shuffle_in = None

    def forward(self, input):
        shuffle_out = input.new()

        batch_size, cubed_channels, in_depth, in_height, in_width = input.size()

        assert cubed_channels % self.upscale_factor_cubed == 0
        channels = cubed_channels // self.upscale_factor_cubed

        input_view = input.view(
            batch_size,
            channels,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )

        shuffle_out.resize_(
            input_view.size(0),
            input_view.size(1),
            input_view.size(5),
            input_view.size(2),
            input_view.size(6),
            input_view.size(3),
            input_view.size(7),
            input_view.size(4),
        )

        shuffle_out.copy_(input_view.permute(0, 1, 5, 2, 6, 3, 7, 4))

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        output = shuffle_out.reshape(
            batch_size, channels, out_depth, out_height, out_width
        )

        return output
