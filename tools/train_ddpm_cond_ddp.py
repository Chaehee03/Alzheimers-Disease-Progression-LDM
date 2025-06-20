import os
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from models.unet_cond import UNet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import utils
from utils.config_utils import *
from models import const
from utils.data import get_dataset_from_pd
from torch.utils.data.dataloader import default_collate
from monai import transforms
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, SpacingD,
    ResizeWithPadOrCropD, Lambda, ToTensorD
)
from monai.data.image_reader import NumpyReader
from monai.metrics import SSIMMetric, PSNRMetric
from tools.sample_ldm import sample

def setup():
    dist.init_process_group(backend='nccl')

def cleanup():
    dist.destroy_process_group()

def concat_covariates(_dict):
    keys_to_remove = [k for k in _dict if k.endswith("_transforms")]
    for k in keys_to_remove:
        _dict.pop(k)
    _dict['context'] = torch.tensor([_dict[c] for c in const.CONDITIONING_VARIABLES]).unsqueeze(0)
    return _dict

def train(args):
    setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    world_size = dist.get_world_size()

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    start_time = time.time()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    ldm_config = config['ldm_params']
    vqvae_config = config['vqvae_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)
    condition_types = condition_config['condition_types'] if condition_config else []

    dataset_df = pd.read_csv(train_config['A_csv'])
    dataset_df = dataset_df.dropna(subset=["image_path", "segm_path", "latent_path"])
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']

    transforms_fn = transforms.Compose([
        LoadImageD(keys=["latent_path"], reader=NumpyReader()),
        EnsureChannelFirstD(keys=["latent_path"], channel_dim=0),
        SpacingD(keys=["latent_path"], pixdim=const.RESOLUTION, mode="bilinear"),
        ResizeWithPadOrCropD(keys=["latent_path"], spatial_size=(32, 40, 32)),
        Lambda(func=concat_covariates),
        ToTensorD(keys=["latent_path", "context"], track_meta=False),
    ])

    trainset = get_dataset_from_pd(train_df, transforms_fn, None)
    validset = get_dataset_from_pd(valid_df, transforms_fn, None)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size,rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(validset, num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(trainset, sampler=train_sampler,
                              batch_size=train_config['ldm_batch_size'],
                              num_workers=train_config['num_workers'],
                              pin_memory=True, collate_fn=default_collate)

    valid_loader = DataLoader(validset, sampler=valid_sampler,
                              batch_size=train_config['ldm_batch_size'],
                              num_workers=train_config['num_workers'],
                              pin_memory=True, collate_fn=default_collate)

    unet = UNet(im_channels=vqvae_config['z_channels'], model_config=ldm_config).to(device)
    unet = DDP(unet, device_ids=[rank], find_unused_parameters=True)

    optimizer = Adam(unet.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    scaler = GradScaler()

    checkpoint_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        unet.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    vqvae = VQVAE(im_channels=vqvae_config['im_channels'], model_config=vqvae_config).to(device)
    vqvae.eval()
    vqvae_path = "/DataRead2/chsong/checkpoints/vqvae_best_ckpt_13.pth"
    if os.path.exists(vqvae_path):
        ckpt = torch.load(vqvae_path, map_location=device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        vqvae.load_state_dict(new_state_dict)
    else:
        raise Exception('VQVAE checkpoint not found')

    if rank == 0:
        writer = SummaryWriter(log_dir="/DataRead2/chsong/tensorboard/ldm_exp_w_ddp_7")

    scale_factor = 12.8
    # best_val = float("inf")
    # min_delta = 1e-5
    # patience = 10
    # wait = 0
    global_counter = {'train': 0, 'valid': 0}

    for epoch_idx in range(start_epoch, train_config['ldm_epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch_idx)
        valid_sampler.set_epoch(epoch_idx)

        for mode, loader in zip(['train', 'valid'], [train_loader, valid_loader]):
            unet.train() if mode == 'train' else unet.eval()
            epoch_loss = 0

            for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch_idx+1} - {mode}")):
                with autocast():
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    latents = batch['latent_path'].to(device) * scale_factor
                    cond_input = {'context': batch['context'].to(device)} if 'context' in condition_types else {}
                    noise = torch.randn_like(latents).to(device)
                    t = torch.randint(0, diffusion_config['num_timesteps'], (latents.size(0),), device=device).long()
                    noisy_im = scheduler.add_noise(latents, noise, t)
                    noise_pred = unet(noisy_im, t, cond_input=cond_input)
                    loss = criterion(noise_pred, noise)

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if rank == 0:
                    writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                global_counter[mode] += 1
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            if rank == 0:
                writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch_idx)
                print(f"[{mode}] Epoch {epoch_idx + 1}: Loss = {epoch_loss:.4f}")

            if mode == 'valid' and rank == 0:
                with torch.no_grad():
                    unet.eval()
                    vqvae.eval()
                    ssim_metric.reset()
                    psnr_metric.reset()

                    all_ssim = []
                    all_psnr = []

                    for i, val_batch in enumerate(valid_loader):
                        if i > 3: break
                        latents = val_batch['latent_path'].to(device) * scale_factor
                        context = val_batch['context'].to(device)

                        # Decode ground truth and prediction
                        gt_volume = vqvae.decode(latents).clamp(-1, 1).add(1).div(2)
                        pred_volume = sample(unet.module, vqvae, scheduler, context, diffusion_config, device=device)

                        print(f"pred_volume shape: {pred_volume.shape}, dtype: {pred_volume.dtype}")
                        print(f"gt_volume shape: {gt_volume.shape}, dtype: {gt_volume.dtype}")

                        pred_volume = pred_volume.to(device)
                        gt_volume = gt_volume.to(device)

                        ssim_val = ssim_metric(pred_volume, gt_volume).mean().item()
                        psnr_val = psnr_metric(pred_volume, gt_volume).mean().item()

                        all_ssim.append(ssim_val)
                        all_psnr.append(psnr_val)

                    avg_ssim = np.mean(all_ssim)
                    avg_psnr = np.mean(all_psnr)

                    writer.add_scalar("valid/psnr", avg_psnr, epoch_idx)
                    writer.add_scalar("valid/ssim", avg_ssim, epoch_idx)

                    print(f"[Epoch {epoch_idx + 1}] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

                    # if epoch_loss < best_val - min_delta:
                    #     best_val = epoch_loss
                    #     wait = 0
                    #     torch.save({
                    #         'epoch': epoch_idx,
                    #         'model_state_dict': unet.module.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #         'val_loss': best_val
                    #     }, os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name']))
                    # else:
                    #     wait += 1
                    #     if wait >= patience:
                    #         print(f"Early stopping at epoch {epoch_idx + 1}")
                    #         break

        if rank == 0:
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': unet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))
        if rank == 0:
            epoch_end = time.time()
            print(f"[Epoch {epoch_idx + 1}] Duration: {(epoch_end - epoch_start):.2f} seconds")

    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training complete. Total training time: {elapsed_time:.2f} seconds.")
        writer.close()

    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path', default='config/adni.yaml', type=str)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()


