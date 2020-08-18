import pytorch_lightning as pl
from itertools import zip_longest

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

        self.input = FixupResBlock(in_channels, base_network_channels)

        self.encoder = Encoder(base_network_channels, n_bottleneck_blocks, n_blocks_per_bottleneck)
        self.decoder = Decoder(base_network_channels, n_bottleneck_blocks, n_blocks_per_bottleneck)


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
    def __init__(self, in_channels, n_enc=3, n_down_per_enc=2):
        super(Encoder, self).__init__()
        self.n_enc = n_enc

        channels = in_channels
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
        quantizations = [
            ((diff, aux, perplexity, one_hot_idx, idx) := quantize(pre_quantize(down, aux)))
            for down, pre_quantize, quantize
            in reversed(zip(downsampled, self.pre_quantize, self.quantize))
        ]

        return quantizations


class Decoder(nn.Module):
    def __init__(self, out_channels, n_enc=3, n_up_per_enc=2):
        super(Encoder, self).__init__()

        self.up = []
        channels = out_channels
        for i in range(n_enc):
            self.up.append(UpBlock(
                out_channels=channels, aux_channels=(0 if i==n_enc-1 else channels/2),
                n_up=n_up_per_enc
            ))
            channels *= 2 ** (n_enc * n_up_per_enc)

        self.n_enc = n_enc


    def forward(self, quantizations):
        for i, (quantization, up) in enumerate(reversed(zip(quantizations, self.up))):
            up_input = quantization
            if i != 0:
                up_input = torch.cat([up_input, prev_up], dim=1)
            prev_up = up(up_input)

        return prev_up

class FixupResBlock(nn.Module):
    pass

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
