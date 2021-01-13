
from itertools import chain
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


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
             ) if mode == 'decoder' else None,
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
    def __init__(self, in_channels, base_network_channels, num_embeddings: List[int], n_enc=3, n_down_per_enc=2):
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
                Quantizer(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim, commitment_cost=0.25)
            )

            before_channels = after_channels


    def forward(self, data):
        down = self.parse_input(data)
        downsampled = ((down := downblock(down)) for downblock in self.down)

        aux = None
        quantizations = []
        for down, pre_quantize, quantize in reversed(list(zip(downsampled, self.pre_quantize, self.quantize))):
            quantizations.append((quantization := quantize(pre_quantize(down, aux))))

            _, aux, *_ = quantization

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

            self.up.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=after_channels,
                    n_up=n_up_per_enc,
                    mode='decoder'
                )
            )

            after_channels = before_channels

        self.out = FixupResBlock(base_network_channels, out_channels, mode='out')

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

    def __init__(self, in_channels, out_channels, mode, activation=nn.ELU):
        super(FixupResBlock, self).__init__()

        assert mode in ("down", "same", "up", "out")
        self.mode = mode

        # branch_channels = max(in_channels, out_channels)
        branch_channels = out_channels

        self.activation = activation()

        self.bias1a, self.bias1b, self.bias2a, self.bias2b = (
            nn.Parameter(torch.zeros(1)) for _ in range(4)
        )
        self.scale = nn.Parameter(torch.ones(1))

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
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.skip_conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1 if mode != 'down' else 2),
            stride=(1 if mode != 'down' else 2),
            padding=(0 if mode != 'down' else 0),
            bias=True
        )

        self.branch_conv2 = torch.nn.Conv3d(
            in_channels=branch_channels,
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

        out = out + self.skip_conv(input)

        if self.mode != 'out':
            out = self.activation(out)

        return out

    def initialize_weights(self, num_layers):
        m = 2 # number of convs in a branch

        torch.nn.init.kaiming_normal_(self.branch_conv1.weight) * (num_layers ** (-1 / (2*m - m)))

        torch.nn.init.constant_(tensor=self.branch_conv2.weight, val=0)

        torch.nn.init.kaiming_normal_(self.skip_conv.weight)
        torch.nn.init.constant_(tensor=self.skip_conv.bias, val=0)


class Quantizer(torch.nn.Module):
    # Code taken from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I

    """
    EMA-updated Vector Quantizer
    """
    def __init__(
        self, num_embeddings : int, embedding_dim : int, commitment_cost : float, decay=0.99, laplace_alpha=1e-5
    ):
        super(Quantizer, self).__init__()

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed) # e_i
        self.register_buffer("embed_avg", embed.clone()) # m_i
        self.register_buffer("cluster_size", torch.zeros(num_embeddings)) # N_i

        # TODO: replace with bool, cuda_mul_bool issue fixed in
        # https://github.com/pytorch/pytorch/pull/48310
        #
        # Needs to be a buffer, otherwise doesn't get added to state dict
        self.register_buffer("first_pass", torch.as_tensor(1))

        self.commitment_cost = commitment_cost

        self.decay = decay
        self.laplace_alpha = laplace_alpha

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed)

    def _update_ema(self, flat_input, encoding_indices):
        # buffer updates need to be in-place because of distributed
        encodings_one_hot = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings
        ).type_as(flat_input)

        new_cluster_size = encodings_one_hot.sum(dim=0)
        dw = encodings_one_hot.T @ flat_input

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(new_cluster_size)
            torch.distributed.all_reduce(dw)

        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=(1-self.decay)
        )

        self.embed_avg.data.mul_(self.decay).add_(dw, alpha=(1-self.decay))

        # Laplacian smoothing
        n = self.cluster_size.sum()
        cluster_size = n * ( # times n because we don't want probabilities but counts
            (self.cluster_size + self.laplace_alpha)
            / (n + self.num_embeddings * self.laplace_alpha)
        )

        embed_normalized = self.embed_avg / cluster_size.unsqueeze(dim=-1)
        self.embed.data.copy_(embed_normalized)

    def _init_ema(self, flat_input):
        mean = flat_input.mean(dim=0)
        std = flat_input.std(dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean)
            torch.distributed.all_reduce(std)
            mean /= torch.distributed.get_world_size()
            std /= torch.distributed.get_world_size()

        self.embed.mul_(std)
        self.embed.add_(mean)
        self.embed_avg.copy_(self.embed)

        self.cluster_size.data.add_(flat_input.size(dim=0) / self.num_embeddings)
        self.first_pass.mul_(0)

    def forward(self, inputs):
        with torch.no_grad():
            channel_last = inputs.permute(0, 2, 3, 4, 1) # XXX: might not actually be necessary
            input_shape = channel_last.shape

            flat_input = channel_last.reshape(-1, self.embedding_dim)

            if self.training and self.first_pass:
                self._init_ema(flat_input)

            encoding_indices = torch.argmin(torch.cdist(flat_input, self.embed), dim=1)
            quantized = self.embed_code(encoding_indices).reshape(input_shape)

            if self.training:
                self._update_ema(flat_input, encoding_indices)

            # avg_probs = torch.mean(encodings_one_hot, dim=0)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            # Cast everything back to the same order and dimensions of the input
            quantized = quantized.permute(0, 4, 1, 2, 3)
            encoding_indices = encoding_indices.reshape(input_shape[:-1])

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Trick to have identity backprop grads
        quantized = inputs + (quantized - inputs).detach()

        return (
            loss,
            quantized,
            # perplexity,
            encoding_indices
        )
