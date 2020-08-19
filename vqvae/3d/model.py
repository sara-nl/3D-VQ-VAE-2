from itertools import zip_longest

import torch
from torch import nn
import pytorch_lightning as pl

"""
This is largely a refactor of https://github.com/danieltudosiu/nmpevqvae
"""

class VQVAE(pl.LightningModule):
    def __init__(
        self,
        input_channels=1,
        base_network_channels=4,
        n_bottleneck_blocks=3,
        n_blocks_per_bottleneck=2
    ):

        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_channels, base_network_channels, n_bottleneck_blocks, n_blocks_per_bottleneck)
        self.decoder = Decoder(input_channels, base_network_channels, n_bottleneck_blocks, n_blocks_per_bottleneck)


    def forward(self, data):
        _, quantizations, _, _, _ = zip(*self.encode(inputs)) # list transpose of encoder outputs
        decoded = self.decode(quantizations)

    def encode(self, data):
        return self.Encoder(data)

    def decode(self, quantizations):
        return self.Decoder(data)


class DownBlock(nn.Module):
    def __init__(self, in_channels, n_down=2):
        self.layers = nn.Sequential(*(
            FixupResBlock(in_channels*i, in_channels*(i+1), mode='down') for i in range(1, n_down+1)
        ))

    def forward(self, data):
        return self.layers(data)


class UpBlock(nn.Module):
    def __init__(self, out_channels, aux_channels=0, n_up=2):
        # Some slight schenanigans required for the first layer because of possible concat beforehand
        self.layers = nn.Sequential(*(
            FixupResBlock(out_channels/i + (aux_channels if i == n_up else 0),
                          out_channels/(i+1), mode='up')
            for i in range(1, n_up+1)
        ))

    def forward(self, data):
        return self.layers(data)


class PreQuantization(nn.Module):

    def __init__(self, in_channels, last=False, n_up=2):
        super(PreQuantization, self).__init__()
        self.last = last

        if not last:
            self.upsample = UpBlock(in_channels, n_up=n_up)
        self.pre_q = FixupResBlock(in_channels/(2**n_up), mode='same')

    def forward(self, data, auxilary=None):
        assert self.last is bool(auxilary), (
            "The last layer shouldn't be conditioned on anything!"
        )

        if not self.last:
            data = torch.cat([data, self.upsample(auxilary)], dim=1)

        return self.pre_q(data)



class Encoder(nn.Module):
    def __init__(self, in_channels, base_network_channels, n_enc=3, n_down_per_enc=2):
        super(Encoder, self).__init__()

        self.parse_input = FixupResBlock(in_channels, base_network_channels, mode='same')

        channels = base_network_channels
        self.down, self.pre_quantize, self.quantize = [], [], []
        for i in range(n_enc):
            self.down.append(DownBlock(channels, n_down_per_enc))
            self.pre_quantize.append(
                PreQuantization(channels, last=(i==n_enc-1), n_up=n_down_per_enc)
            )
            self.quantize.append(
                Quantizer(num_embeddings=channels/2, embedding_dim=256, commitment_cost=7)
            )

            channels *= 2 ** n_down_per_enc


    def forward(self, data):
        down = data
        downsampled = [(down := downblock(down)) for downblock in self.down]

        aux = None
        quantizations = []
        for down, pre_quantize, quantize in reversed(zip(downsampled, self.pre_quantize, self.quantize)):
            quantization = quantize(pre_quantize(down, aux))
            quantizations.append(quantization)

            _, aux, _, _, _ = quantization

        return quantizations


class Decoder(nn.Module):
    def __init__(self, out_channels, base_network_channels, n_enc=3, n_up_per_enc=2):
        super(Encoder, self).__init__()

        self.up = []
        channels = base_network_channels
        for i in range(n_enc):
            self.up.append(UpBlock(
                out_channels=channels, aux_channels=(0 if i==n_enc-1 else channels/2),
                n_up=n_up_per_enc
            ))
            channels *= 2 ** (n_enc * n_up_per_enc)

        self.subpixel = SubPixelConvolution3D(base_network_channels, out_channels)

    def forward(self, quantizations):
        for i, (quantization, up) in enumerate(reversed(zip(quantizations, self.up))):
            up_input = quantization if i != 0 else torch.cat([quantization, prev_up], dim=1)

            prev_up = up(up_input)

        out = self.subpixel(prev_up)

        return out


class FixupBlock(torch.nn.Module):
    # Adapted from:
    # https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py#L20

    """
    FIXME: check wether the biases should be a single scalar or a vector
    """

    def __init__(self, in_channels, out_channels, mode):
        super(FixupBlock, self).__init__()

        assert mode in ("down", "same", "up")

        self.bias1a, self.bias1b, self.bias2a, self.bias2b, self.scale = (
            nn.Parameter(torch.zeros(1) for _ in range(5))
        )

        self.activation = torch.nn.LeakyReLU(inplace=False)

        if mode == 'down':
            conv = nn.Conv3d
            kernel_size, stride = 3, 2
        elif mode == 'same':
            conv = nn.Conv3d
            kernel_size, stride = 3, 1
        else: # mode == 'up'
            conv = nn.ConvTranspose3d
            kernel_size, stride = 4, 2

        self.branch_conv1, self.skip_conv = (
            conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=stride, padding=1, bias=False)
            for _ in range(2)
        )


        self.branch_conv2 = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.nca = torch.nn.Sequential(
            torch.nn.ConstantPad3d(padding=(1, 0, 1, 0, 1, 0), value=0),
            torch.nn.AvgPool3d(kernel_size=2, stride=1),
        )


    def forward(self, input):
        out = self.branch_conv(input + self.bias1a)
        out = self.nca(self.activation(out + self.bias1b))

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += self.nca(self.skip_conv(input + self.bias1a))
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
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

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

        self.name = name

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
            upscale_factor=upsample_factor, name="PixelShuffle3D"
        )

        self.nca = torch.nn.Sequential(
            torch.nn.ConstantPad3d(padding=(1, 0, 1, 0, 1, 0), value=0),
            torch.nn.AvgPool3d(kernel_size=2, stride=1),
        )

        self.name = name

    def forward(self, input):
        with torch.autograd.profiler.record_function(self.name):
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
        self.name = name

    def forward(self, input):
        shuffle_out = input.new()

        batch_size, cubed_channels, in_depth, in_heigt, in_width = input.size()
        channels = cubed_channels / self.upscale_factor_cubed

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

        output = shuffle_out.view(
            batch_size, channels, out_depth, out_height, out_width
        )

        return output
