from argparse import ArgumentParser, Namespace
from pathlib import Path
from uuid import UUID

import nrrd
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VQVAE
from utils import CTScanDataset, DepthPadAndCrop

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

@torch.no_grad()
def main(args: Namespace):

    min_val, max_val, scale_val = -1500, 3000, 1000

    print("- Loading model weights")
    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()

    db = torch.load(args.db_path)

    for i, embedding_0 in enumerate(db[1].values()):
        # embedding_1 = db[1][embedding_0['condition']]
        # embedding_2 = db[2][embedding_1['condition']]
        embedding_1 = embedding_0
        embedding_0 = {'data': torch.zeros((64,64,32)).long()}

        embeddings = [
            quantizer.embed_code(embedding['data'].cuda().unsqueeze(dim=0)).permute(0, 4, 1, 2, 3)
            for embedding, quantizer
            in zip((embedding_0, embedding_1), model.encoder.quantize)
        ]
        breakpoint()
        print("- Performing forward pass")
        with torch.cuda.amp.autocast():
            res = model.decode(embeddings)
            res = torch.nn.functional.elu(res)
        # res = inverse_softplus(res)

        res = res.squeeze().detach().cpu().numpy()
        res = res * scale_val - scale_val
        res = np.rint(res).astype(np.int)

        print("- Writing to nrrd")
        nrrd.write(str(args.out_path) + f'_{i}.nrrd', res, header={'spacings': (0.976, 0.976, 3)})

        print("- Done")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("db_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path, help='outpath without extension')
    args = parser.parse_args()

    main(args)