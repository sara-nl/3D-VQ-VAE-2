from argparse import ArgumentParser, Namespace
from pathlib import Path
from uuid import UUID

import nrrd
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vqvae.model import VQVAE
from utils import CTScanDataset, DepthPadAndCrop

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

@torch.no_grad()
def main(args: Namespace):

    min_val, max_val, scale_val = -1500, 3000, 1000

    print("- Loading model weights")
    model = VQVAE.load_from_checkpoint(str(args.ckpt_path)).cuda()

    db = torch.load(args.db_path)

    for embedding_0_key, embedding_0 in db[0].items():
        embedding_1_key = embedding_0['condition']
        embedding_1 = db[1][embedding_1_key]

        # issue where the pixelcnn samples 0's
        success = 'failure' if torch.all(embedding_0['data'][-1] == 0) else 'success'

        embeddings = [
            quantizer.embed_code(embedding['data'].cuda().unsqueeze(dim=0)).permute(0, 4, 1, 2, 3)
            for embedding, quantizer
            in zip((embedding_0, embedding_1), model.encoder.quantize)
        ]

        print("- Performing forward pass")
        with torch.cuda.amp.autocast():
            res = model.decode(embeddings)
            res = torch.nn.functional.elu(res)

        res = res.squeeze().detach().cpu().numpy()
        res = res * scale_val - scale_val
        res = np.rint(res).astype(np.int)

        print("- Writing to nrrd")
        nrrd.write(str(args.out_path) + f'_{success}_{str(embedding_1_key)}_{str(embedding_0_key)}.nrrd', res, header={'spacings': (0.976, 0.976, 3)})

        print("- Done")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("db_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path, help='outpath without extension')
    args = parser.parse_args()

    main(args)