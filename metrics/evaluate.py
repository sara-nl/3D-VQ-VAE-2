"""
Parts copied from:
- https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
- https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/__init__.py
- https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/simple_metrics.py

Edited by: Robert Jan Schlimbach
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = (
    "nmse",
    "psnr",
    "ssim3d"
)

def nmse(orig: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(pred, orig) ** 2 / torch.norm(orig) ** 2

def psnr(orig: torch.Tensor, pred: torch.Tensor, data_range: float) -> torch.Tensor:
    return 10 * torch.log10((data_range ** 2) / F.mse_loss(pred, orig))

def ssim3d(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    (_, channel, _, _, _) = img1.size()
    window = _create_window_3D(window_size, channel)
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.exp(-0.5 * ((torch.arange(window_size) - window_size // 2) / sigma) ** 2)
    return gauss/gauss.sum()

@lru_cache(maxsize=1)
def _create_window_3D(window_size: int, channel: int):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window
    
def _ssim_3D(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
