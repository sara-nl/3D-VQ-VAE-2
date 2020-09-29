from typing import Optional, Union, Tuple, Type
from functools import cached_property
from math import pi

import torch
import torch.distributions as dist
from torch.distributions.mixture_same_family import MixtureSameFamily


class Logistic(dist.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc, self.scale = dist.utils.broadcast_all(loc, scale)

        base_distribution = dist.Uniform(torch.Tensor([0]).to(loc.device), torch.Tensor([1]).to(loc.device)).expand(self.loc.shape)
        transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=loc, scale=scale)]

        super(Logistic, self).__init__(base_distribution, transforms)


def mixture_nll_loss(
    x: torch.Tensor,
    base_dist: Type[dist.Distribution],
    n_mix: int,
    mixture_comp_logits: torch.Tensor,
    reduce_mean: bool = True,
    **base_dist_kwargs
    ) -> torch.Tensor:
    '''
    x has minimum dim of B x W
    expects every base_dist_kwarg to be a Sequence of length n_mix
    '''

    # Checking correct shape of inputs and permuting to have mixture component dim last
    num_dims = len(mixture_comp_logits.shape)
    assert num_dims >= 2
    axes = (0, *range(2, num_dims), 1)

    _, channel, *_ = mixture_comp_logits.shape
    mixture_comp_logits = mixture_comp_logits.permute(axes)
    assert channel == n_mix
    for param, values in base_dist_kwargs.items():
        _, channel, *_ = values.shape
        assert channel == n_mix
        base_dist_kwargs[param] = values.permute(axes)

    pi_k = dist.Categorical(logits=mixture_comp_logits)
    dists = base_dist(**base_dist_kwargs)

    nll_loss = generic_nll_loss(
        x,
        base_dist=MixtureSameFamily,
        mixture_distribution=pi_k,
        component_distribution=dists,
        reduce_mean=reduce_mean
    )

    return nll_loss


def generic_nll_loss(
    x: torch.Tensor,
    base_dist: Type[dist.Distribution],
    reduce_mean: bool = True,
    **base_dist_kwargs
) -> torch.Tensor:

    nll_loss = -base_dist(**base_dist_kwargs).log_prob(x.squeeze())
    return (
        nll_loss
        if not reduce_mean
        else nll_loss.mean()
    )

if __name__ == '__main__':
    n_mix = 10
    batch_size = 5

    dim = (28, 28)

    test_in = torch.randn((batch_size, 1, *dim))

    mix, locs, scales = torch.randn((3, batch_size, n_mix, *dim))
    scaled = scales.exp()

    loss = mixture_nll_loss(test_in, Logistic, n_mix, mix, loc=locs, scale=scales)



