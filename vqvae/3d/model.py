"""
This is largely a refactor of https://github.com/danieltudosiu/nmpevqvae
"""

from itertools import zip_longest
from typing import Tuple
from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch import nn

from metrics.ssim import ssim3D
from metrics.baur import BaurLoss3D
from metrics.distribution import Logistic, mixture_nll_loss, generic_nll_loss


baur_loss_f = BaurLoss3D(lambda_reconstruction=1)

def _sub_metric_log_dict(sub_metric_name, sub_metric):
    return {
        f'{sub_metric_name}_{func_name}': func(sub_metric.detach())
        for func_name, func in (
            ('min', torch.min),
            ('max', torch.max),
            ('mean', torch.mean),
            ('std', torch.std)
        )
    }


class VQVAE(pl.LightningModule):

    # first in line is the default
    supported_metrics = ("mse", "normal_nll", "logistic_mixture_nll")

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
        base_lr, max_lr = 1e-5, 5e-3

        train_loader = self.train_dataloader()
        if self.use_ddp:
            multiplier = train_loader.batch_size * len(self.trainer.gpus) * self.trainer.world_size / 2
            base_lr *= multiplier
            max_lr *= multiplier

        optimizer = torch.optim.Adam(self.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=5, # *2 as advised by CLR github
            cycle_momentum=False
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='validation')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'validation')

        loss, log_dict = self.recon_loss_f(batch, batch_idx)

        if mode == 'train':
            result = pl.TrainResult(minimize=loss)
        else: # mode == 'validation'
            result = pl.EvalResult(checkpoint_on=loss)

        # logging
        result.log_dict({f'{mode}_{key}': val for key, val in log_dict.items()})

        return result

    def normal_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        loc, log_scale = torch.split(self(x), self.input_channels, dim=1)
        log_scale = torch.max(log_scale, torch.Tensor([-7]).to(device=self.device)) # lower bound log_scale
        unreduced_loss = generic_nll_loss(x, Normal, loc=loc, scale=log_scale.exp(), reduce_sum=False)

        log_dict = {
            **_sub_metric_log_dict('loss', unreduced_loss),
            **_sub_metric_log_dict('loc', loc),
            **_sub_metric_log_dict('log_scale', log_scale),
        }

        return unreduced_loss.mean(), log_dict

    def mse(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def logistic_mixture_nll(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
        x, _ = batch

        pi_k, loc, log_scale = torch.split(x, self.n_mix, dim=1)
        nll_loss = mixture_nll_loss(x, Logistic, self.n_mix, pi_k, loc=loc, scale=log_scale.exp(), reduce_sum=True)

        log_dict = {
            **_sub_metric_log_dict('pi_k', pi_k),
            **_sub_metric_log_dict('loc', loc),
            **_sub_metric_log_dict('log_scale', log_scale),
        }

        return nll_loss, log_dict


    def _parse_input_args(self, args: Namespace):
        assert args.metric in self.supported_metrics

        if args.metric == 'normal_nll':
            self.recon_loss_f = self.normal_nll
            out_multiplier = 2
        elif args.metric == 'mse':
            self.recon_loss_f = self.mse
            out_multiplier = 1
        elif args.metric == 'logistic_mixture_nll':
            self.recon_loss_f = self.logistic_mixture_nll
            self.n_mix = args.n_mix
            out_multiplier = args.n_mix

        self.output_channels = args.input_channels * out_multiplier
        self.input_channels = args.input_channels
        self.base_network_channels = args.base_network_channels
        self.n_bottleneck_blocks = args.n_bottleneck_blocks
        self.n_blocks_per_bottleneck = args.n_blocks_per_bottleneck

        self.num_layers = (
            (3 * args.n_bottleneck_blocks - 1)
            * args.n_blocks_per_bottleneck
            + (2 * args.n_bottleneck_blocks)
        )


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--input-channels', type=int, default=1)
        parser.add_argument('--base-network_channels', type=int, default=4)
        parser.add_argument('--n-bottleneck-blocks', type=int, default=3)
        parser.add_argument('--n-blocks-per-bottleneck', type=int, default=2)
        parser.add_argument('--metric', choices=cls.supported_metrics, default=cls.supported_metrics[0])

        # loss function specific arguments
        # FIXME: put this is a class or something, at least not here
        parser.add_argument('--n-mix', default=5)

        return parser



class DownBlock(nn.Module):
    def __init__(self, in_channels, n_down=2):
        super(DownBlock, self).__init__()

        self.layers = nn.Sequential(*(
            FixupResBlock(in_channels*2**i, in_channels*2**(i+1), mode='down') for i in range(n_down)
        ))

    def forward(self, data):
        return self.layers(data)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, aux_channels=0, n_up=2):
        super(UpBlock, self).__init__()

        # Some slight shenanigans required for the first layer because of possible concat beforehand
        self.layers = nn.Sequential(*(
            FixupResBlock(
                in_channels if i == n_up-1 else out_channels*(2**(i+1)),
                out_channels*(2**i),
                mode='up'
            )
            for i in range(n_up-1, -1, -1)
        ))

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
                Quantizer(num_embeddings=256, embedding_dim=embedding_dim, commitment_cost=7)
            )

            before_channels = after_channels


    def forward(self, data):
        preprocessed = self.parse_input(data)

        down = preprocessed
        downsampled = [(down := downblock(down)) for downblock in self.down]

        aux = None
        quantizations = []
        for down, pre_quantize, quantize in reversed(list(zip(downsampled, self.pre_quantize, self.quantize))):
            quantization = quantize(pre_quantize(down, aux))
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
                n_up=n_up_per_enc
            ))

            after_channels = before_channels

        self.out = FixupResBlock(base_network_channels, out_channels, mode='out')

    def forward(self, quantizations):
        for i, (quantization, up) in enumerate(reversed(list(zip(quantizations, self.up)))):
            out = quantization if i == 0 else torch.cat([quantization, out], dim=1)
            out = up(out)

        out = self.out(out)

        return out


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
            kernel_size, stride, padding = 3, 2, 1
        elif mode in ('same', 'out'):
            conv = nn.Conv3d
            kernel_size, stride, padding = 3, 1, 1
        elif mode == 'up':
            conv = nn.ConvTranspose3d
            kernel_size, stride, padding = 4, 2, 1

        self.branch_conv1, self.skip_conv = (
            conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, bias=False)
            for _ in range(2)
        )

        self.branch_conv2 = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
        )


    def forward(self, input):
        out = self.branch_conv1(input + self.bias1a)
        out = self.activation(out + self.bias1b)

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += self.skip_conv(input + self.bias1a)

        if self.mode != 'out':
            out = self.activation(out)

        return out

    def initialize_weights(self, num_layers):

        torch.nn.init.normal_(
            tensor=self.branch_conv1.weight,
            mean=0,
            std=np.sqrt(
                2 / (self.branch_conv1.weight.shape[0] * np.prod(self.branch_conv1.weight.shape[2:]))
            )
            * num_layers ** (-0.5),
        )
        torch.nn.init.constant_(tensor=self.branch_conv2.weight, val=0)
        torch.nn.init.normal_(
            tensor=self.skip_conv.weight,
            mean=0,
            std=np.sqrt(
                2
                / (
                    self.skip_conv.weight.shape[0]
                    * np.prod(self.skip_conv.weight.shape[2:])
                )
            ),
        )


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
    def __init__(self, in_channels, out_channels, upsample_factor=2):
        super(SubPixelConvolution3D, self).__init__()
        assert upsample_factor == 2
        self.upsample_factor = upsample_factor

        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels * upsample_factor ** 3,
            kernel_size=3,
            padding=1,
        )

        self.shuffle = PixelShuffle3D(
            upscale_factor=upsample_factor
        )

        self.initialize_weights()

    def forward(self, input):
        out = self.conv(input)
        out = self.shuffle(out)
        out = self.nca(out)
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
