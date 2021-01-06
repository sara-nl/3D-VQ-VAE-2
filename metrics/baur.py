from typing import Sequence

import torch
import torch.nn as nn

class BaurLoss3D(object):
    def __init__(self, lambda_reconstruction=1):
        super(BaurLoss3D).__init__()

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gdl = 0

        self.l1_loss = lambda x, y: nn.PairwiseDistance(p=1)(
            x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        ).sum()
        self.l2_loss = lambda x, y: nn.PairwiseDistance(p=2)(
            x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        ).sum()

    def __call__(self, recon: torch.Tensor, target: torch.Tensor, quantization_losses: Sequence[torch.Tensor]):
        originals = target
        reconstructions = recon

        l1_reconstruction = (
            self.l1_loss(originals, reconstructions) * self.lambda_reconstruction
        )
        l2_reconstruction = (
            self.l2_loss(originals, reconstructions) * self.lambda_reconstruction
        )

        originals_gradients = self.__image_gradients(originals)
        reconstructions_gradients = self.__image_gradients(reconstructions)

        l1_gdl = (
            self.l1_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l1_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l1_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        l2_gdl = (
            self.l2_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l2_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l2_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        quantization_loss = torch.Tensor([quantization_loss for quantization_loss in quantization_losses]).sum()

        loss_total = l1_reconstruction + l2_reconstruction + l1_gdl + l2_gdl + quantization_loss

        return loss_total


    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, features, depth, height, width = input_shape

        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        dzz = torch.zeros(
            (batch_size, features, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = torch.cat([dz, dzz], 2)
        dz = torch.reshape(dz, input_shape)

        dyz = torch.zeros(
            (batch_size, features, depth, 1, width), device=image.device, dtype=dy.dtype
        )
        dy = torch.cat([dy, dyz], 3)
        dy = torch.reshape(dy, input_shape)

        dxz = torch.zeros(
            (batch_size, features, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = torch.cat([dx, dxz], 4)
        dx = torch.reshape(dx, input_shape)

        return dx, dy, dz
