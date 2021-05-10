from random import randrange
from functools import partial

import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F


def bits_per_dim(mean_nll: torch.Tensor):
    '''Assumes the nll was calculated with the natural logarithm'''
    return mean_nll / np.log(2)

def idx_to_one_hot(data: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    return rearrange(F.one_hot(data, num_classes=num_classes),
                     'b d h w c -> b c d h w').to(torch.float)



def mixup_data(x, y, alpha=1.0, condition=None):

    def sattolo_cycle(batch_size) -> None:
        """
        Sattolo's algorithm.
        """

        # torch.arange doesn't return  
        out = np.arange(batch_size)
        out[0] = torch.as_tensor(0)
        i = batch_size
        while i > 1:
            i = i - 1
            j = randrange(i)  # 0 <= j <= i-1

            out[j], out[i] = out[i], out[j]
        return out

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]

    lam = torch.as_tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)

    # The randomness is controlled by the sampled lambda,
    # so there is no reason a sample should get mixed with itself.
    # Therefore we sample from the sattolo cycle
    index = sattolo_cycle(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]

    mixed_condition = (
        lam * condition + (1 - lam) * condition[index]
        if condition is not None else None
    )
    target = y, y[index]
    return mixed_x, mixed_condition, target, lam


def mixup_criterion(criterion, lam, **kwargs):
    def mixed_criterion(input, target, criterion, lam, **kwargs):
        y_a, y_b = target
        return lam * criterion(input, y_a, **kwargs) + (1 - lam) * criterion(input, y_b, **kwargs)

    return partial(mixed_criterion, criterion=criterion, lam=lam, **kwargs)
