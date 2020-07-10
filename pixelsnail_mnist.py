import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from tqdm import tqdm

from pixelsnail import PixelSNAIL
from sample import load_model, sample_model


def train(epoch, loader, model, optimizer, device):
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



if __name__ == '__main__':
    sample = False
    device = 'cuda'

    pixelsnail_kwargs = {
        'shape': [28, 28],
        'n_class': 256,
        'channel': 128,
        'kernel_size': 5,
        'n_block': 2,
        'n_res_block': 4,
        'res_channel': 128
    }

    model = PixelSNAIL(**pixelsnail_kwargs)
    model = model.to(device)

    if sample: # quick and dirty sampling
        ckpt_path = './checkpoint/mnist_025.pt'
        model.load_state_dict(torch.load(ckpt_path))
        sample = sample_model(model, device, batch=1, size=[28, 28], temperature=1)
        
        import matplotlib.pyplot as plt

        plt.imshow(sample.permute(1, 0, 2).reshape(28, -1).cpu().numpy())
        plt.show()
    else:
        epoch = 1

        dataset = datasets.MNIST('.', transform=PixelTransform(), download=True)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for i in range(epoch):
            train(i, loader, model, optimizer, device)
            torch.save(model.state_dict(), f'checkpoint/mnist_{str(i + 1).zfill(3)}.pt')
        