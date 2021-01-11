"""
Parts copied from:
- https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
- https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/simple_metrics.py

Edited by: Robert Jan Schlimbach
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def nmse(orig: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(pred - orig) ** 2 / torch.norm(orig) ** 2


def psnr(orig: torch.Tensor, pred: torch.Tensor, data_range: float) -> torch.Tensor:
    return 10 * torch.log10((data_range ** 2) / F.mse_loss(pred, orig))
