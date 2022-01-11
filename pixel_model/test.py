import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from pixel_model.layers import CausalConv3dAdd

if __name__ == '__main__':
    inp = torch.zeros((1, 3, 20,20,20))

    inp[0, :, 9, 9, 9] = 1

    a_mask = CausalConv3dAdd(1, 1, 3, bias=False, mask='A')
    b_mask = CausalConv3dAdd(1, 1, 3, bias=False, mask='B')

    for layer in (a_mask, b_mask):
        layer.depth_conv.weight[:] = 1
        layer.height_conv.weight[:] = 1
        layer.width_conv.weight[:] = 1

    depth, height, width = inp[:, 0][None], inp[:, 1][None], inp[:, 2][None]

    layer_index = 10
    imgs = [torch.log(width[..., layer_index, :, :] + 1).detach()]

    depth, height, width = a_mask(depth=depth, height=height, width=height)
    for i in range(1, 10):
        imgs.append(torch.log(CausalConv3dAdd.stacks_to_output(depth, height, width)[..., layer_index, :, :] + 1).detach())
        depth, height, width = b_mask(depth, height, width)

    plot = make_grid(torch.cat(imgs), nrow=5, padding=0)
    plt.imshow(plot[0]) # I'm not sure why make_grid duplicates over the 0-th dim
    plt.show()
