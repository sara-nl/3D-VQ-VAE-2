import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

def save_samples(model, imgs, epoch, sample_dir, sample_size=25):
    model.eval()

    if sample_size > (batch_size := imgs.shape[0]):
        sample_size = batch_size

    sample = imgs[:sample_size]
    with torch.no_grad():
        out, _ = model(sample)

    utils.save_image(
        torch.cat([sample, out], 0),
        f"{str(sample_dir)}/{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )

    model.train()


def train(epoch, loader, model, optimizer, scheduler, device, sample_dir):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):

        model.zero_grad()

        img = img.to(device)

        if dist.is_primary() and i == 0 and epoch % 10 == 0:
            save_samples(model, img, epoch, sample_dir)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )


def main(args):
    device = "cuda"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / current_time
    ckpt_dir = out_dir / 'checkpoint'
    sample_dir = out_dir / 'sample'
    if dist.is_primary():
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir()
        if not sample_dir.is_dir():
            sample_dir.mkdir()
        with open(out_dir / 'args.out', mode='w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=3*args.n_gpu, pin_memory=True
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, sample_dir)

        if dist.is_primary():
            torch.save(model.state_dict(), f"{str(ckpt_dir)}/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", default=Path('./out'), type=Path)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
