from argparse import ArgumentParser, Namespace
from pathlib import Path

import nrrd
import torch
from torch.utils.data import DataLoader
from monai import transforms

from model import VQVAE
from utils import CTScanDataset


def main(args: Namespace):
    key, min_val, max_val = 'img', -1000, 3000

    transform = transforms.Compose([
        transforms.AddChannel(),
        transforms.ThresholdIntensity(threshold=max_val, cval=max_val, above=False),
        transforms.ThresholdIntensity(threshold=min_val, cval=min_val, above=True),
        transforms.ScaleIntensity(minv=None, maxv=None, factor=(-1 - 1/min_val)),
        transforms.ShiftIntensity(offset=1),
        transforms.SpatialPad(spatial_size=(512, 512, 128), mode='constant'),
        transforms.RandSpatialCrop(roi_size=(512, 512, 128), random_size=False),
        # transforms.Resize(spatial_size=(256, 256, 64)),
        transforms.ToTensor()
    ])

    dataset = CTScanDataset(args.dataset_path, transform=transform, spacing=(0.976, 0.976, 3))
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    single_sample, _ = next(iter(train_loader))
    single_sample = single_sample.cuda()

    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()
    res = model(single_sample)

    res = torch.nn.functional.softplus(res)
    res =  res.squeeze().detach().cpu().numpy()

    # from metrics.distribution import Logistic, sample_mixture
    # from torch.distributions.normal import Normal
    # log_pi_k, locs, log_scales = torch.split(res, model.n_mix, dim=1)
    # loc, scale = torch.nn.functional.softplus(locs), log_scales.exp()

    # res = sample_mixture(Normal, model.n_mix, log_pi_k, greedy=True, loc=loc, scale=scale).squeeze().detach().cpu().numpy()
    # # breakpoint()
    # res[res < 0] = 0
    # res[res > 4] = 4
    breakpoint()
    nrrd.write(str(args.out_path), res)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    main(args)