import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import datasets, utils

from tqdm import tqdm
import matplotlib.pyplot as plt

from pixelsnail import PixelSNAIL
from sample import load_model, sample_model
import os 
import argparse

parser = argparse.ArgumentParser(description='MNIST classification model')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument("--epochs", type=int, default=25, help='')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--local_rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
     

def train(epoch, loader, model, optimizer, device):
    model.train()
   
    loader = tqdm(loader)
    criterion = nn.CrossEntropyLoss()

    for i, (img, label) in enumerate(loader):
        # if i == 0:
        #     save_samples(model, img, epoch, './sample/')

        model.zero_grad()

        img = img.to(device)

        out, _ = model(img)

        loss = criterion(out, img)
        loss.backward()

        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == img).float()
        accuracy = correct.sum() / img.numel()

        loader.set_description(
                (f'epoch: {epoch + 1}; loss: {loss.item():.5f}; ' f'acc: {accuracy:.5f}')
        )
        if device == 0:
            
            loader.update()
        


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


def save_samples(model, imgs, epoch, sample_dir, sample_size=25):
    model.eval()

    if sample_size > (batch_size := imgs.shape[0]):
        sample_size = batch_size

    sample = imgs[:sample_size]
    with torch.no_grad():
        out, _ = model(sample)

    utils.save_image(
        torch.cat([sample, out.max(1)[1].cpu()], 0).to(torch.float),
        f"{str(sample_dir)}/mnist_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True,
        range=(0,255)
    )

    model.train()


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    sample = False
    print("Use GPU: {} for training".format(args.gpu))
        
    args.local_rank = args.local_rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.local_rank)
    
    pixelsnail_kwargs = {
        'shape': [28, 28],
        'n_class': 256,
        'channel': 128,
        'kernel_size': 5,
        'n_block': 2,
        'n_res_block': 4,
        'res_channel': 128
    }

    torch.cuda.set_device(args.gpu)
    
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)

    model = PixelSNAIL(**pixelsnail_kwargs)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.gpu == 0:
        print('The number of parameters of model is', num_params)

    if sample: # quick and dirty sampling
        ckpt_path = './checkpoint/mnist_025.pt'
        model.load_state_dict(torch.load(ckpt_path))
        sample = sample_model(model, device, batch=1, size=[28, 28], temperature=1)

        plt.imshow(sample.permute(1, 0, 2).reshape(28, -1).cpu().numpy())
        plt.show()
    else:
        dataset = datasets.MNIST('.', transform=PixelTransform(), download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for i in range(args.epochs):
            train(i, loader, model, optimizer, args.gpu)
            if args.gpu == 0:
                torch.save(model.state_dict(), f'checkpoint/mnist_{str(i + 1).zfill(3)}.pt')


def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.world_size, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()    
        