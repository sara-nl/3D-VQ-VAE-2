from argparse import ArgumentParser, Namespace
from pathlib import Path

import nrrd
import torch
from torch.utils.data import DataLoader
from monai import transforms

from model import VQVAE
from _utils import CTScanDataset


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
        # transforms.Resize(spatial_size=(256, 256, 128)),
        transforms.ToTensor()
    ])

    dataset = CTScanDataset(args.dataset_path, transform=transform, spacing=(0.976, 0.976, 3))
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    single_sample, _ = next(iter(train_loader))
    single_sample = single_sample.cuda()

    model = VQVAE().cuda()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    out = model(single_sample).squeeze().detach().cpu().numpy()

    nrrd.write(str(args.out_path), out)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    main(args)