# Install 
`conda env create -f environment.yml`  
`pip install .`  

# Checkpoints / Codes / Samples
All checkpoints, codes, and samples are available at [https://surfdrive.surf.nl/files/index.php/s/xY7bwjrgfnhPCAt]().  
Simply copy+pasting the folder structure over the root of this reposity is sufficient.  
For completeness, the relative path (from root) of each checkpoint/code/sample is listed in the following table:  
| Checkpoints | Full size (512×512×128) | Downscaled (256×256×128) |
|-|-|-|
| AE | [3-layer](slurm-jobs/lightning_logs/version_7446231/checkpoints/epoch=1214-step=128683.ckpt) | [2-layer](slurm-jobs/lightning_logs/version_7464547/checkpoints/epoch=279-step=29679.ckpt) |
| Codes | [3-layer-codes](vqvae/codes/version_7446231_epoch_1214.lmdb) | [2-layer-codes](vqvae/codes/version_7464547_epoch_279.lmdb) |
| Pixel Model | [Top](slurm-jobs/lightning_logs/version_7453175/checkpoints/last.ckpt)<br>[Mid](slurm-jobs/lightning_logs/version_7453174/checkpoints/epoch\=945-step\=100222.ckpt) <br>[Bottom](slurm-jobs/lightning_logs/version_7453173/checkpoints/epoch=252-step=26817.ckpt) | [Top](slurm-jobs/lightning_logs/version_7492735/checkpoints/last.ckpt)<br>[Bottom](slurm-jobs/lightning_logs/version_7490077/checkpoints/epoch=302-step=32064.ckpt)  |
| Samples | [Unconditional](pixel-model/codes/version_7446231_epoch_1214_lowtau.pt) | [Conditional](pixel-model/codes/version_7464547_epoch_279_zerozeroonetau_conditioned.pt)<br>[Unconditional](pixel-model/codes/version_7464547_epoch_279_zerozeroonetau_mixup_dropout.pt) |

# Due Credit
This implementation alters and re-uses ideas from a few places:
- VQ-VAE & Pixel-Model: https://github.com/rosinality/vq-vae-2-pytorch
- VQ-VAE: https://github.com/danieltudosiu/nmpevqvae
- Fixup: https://github.com/hongyi-zhang/Fixup
