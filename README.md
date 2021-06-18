# Install 
`conda env create -f environment.yml`  
`pip install .`  

# Checkpoints / Codes / Samples
## Full size (512×512×128):
### AE:
- slurm-jobs/lightning_logs/version_7446231/checkpoints/epoch=1214-step=128683.ckpt
### Codes
- vqvae/codes/version_7446231_epoch_1214.lmdb
### Pixel:
- slurm-jobs/lightning_logs/version_7453175/checkpoints/last.ckpt
- slurm-jobs/lightning_logs/version_7453174/checkpoints/epoch\=945-step\=100222.ckpt
- slurm-jobs/lightning_logs/version_7453173/checkpoints/epoch=252-step=26817.ckpt
### Samples
- pixel-model/codes/version_7446231_epoch_1214_lowtau.pt

## Downscaled (256×256×128):
### AE:
- slurm-jobs/lightning_logs/version_7464547/checkpoints/epoch=279-step=29679.ckpt
### Codes
- vqvae/codes/version_7464547_epoch_279.lmdb
### Pixel:
- slurm-jobs/lightning_logs/version_7492735/checkpoints/last.ckpt
- slurm-jobs/lightning_logs/version_7490077/checkpoints/epoch=302-step=32064.ckpt
### Samples:
- pixel-model/codes/version_7464547_epoch_279_zerozeroonetau_conditioned.pt
- pixel-model/codes/version_7464547_epoch_279_zerozeroonetau_mixup_dropout.pt

# Due Credit
This implementation alters and re-uses ideas from a few places:
- VQ-VAE & Pixel-Model: https://github.com/rosinality/vq-vae-2-pytorch
- VQ-VAE: https://github.com/danieltudosiu/nmpevqvae
- Fixup: https://github.com/hongyi-zhang/Fixup
