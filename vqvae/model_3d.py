import pytorch_lightning as pl

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
        
        child = None
        in_channels = base_network_channels * 2 ** (n_bottleneck_blocks-1 * n_blocks_per_bottleneck)
        for i in range(n_bottleneck_blocks):
            out_channels = (
                in_channels
                if i == n_bottleneck_blocks-1 # if block is the upper-most bottleneck block
                else in_channels + in_channels / 8 # else compensate for combined up-conv
            )

            child = BottleNeckBlock(in_channels, out_channels, child)

        self.bottleneck_chain = child

        self.output = SubPixelConv(base_network_channels, input_channels)

    def forward(self, data):
        


class Quantize(nn.Module):
    pass

class DownSample(nn.Module):
    def __init__(self, n_down=1):
        super(DownSample, self).__init__()


class UpSample(nn.Module):
    def __init__(self, n_up=1):
        super(DownSample, self).__init__()

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, child=None):

        super(BottleNeckBlock, self).__init__():
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.child = child

        self.downsample = Downsample()
        self.pre_quantization = FixupResBlock(mode='regular')
        self.quantize = Quantize()
        self.upsample = Upsample()

    def forward(self, data):
        downsampled = self.downsample(data)

        if self.child is not None:
            child_output = self.child(downsampled)

            split_sizes = self.in_channels / 8, self.in_channels
            pre_q, post_q = torch.split(child_output, split_sizes, dim=1)

            downsampled = torch.cat([downsampled, pre_q])
        
        pre_quantization = self.pre_quantization(downsampled)
        quantized = self.quantize(pre_quantization)
        
        if self.child is not None:
            quantized = torch.cat([quantized, post_q], dim=1)

        upsampled = self.upsample(quantized)
        return upsampled

