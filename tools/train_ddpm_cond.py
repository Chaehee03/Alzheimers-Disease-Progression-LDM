import yaml
import torch
import os
import argparse
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet_cond import UNet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import utils
from utils.config_utils import *
from models import const
from monai import transforms
from utils.data import get_dataset_from_pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD,
    SpacingD, ResizeWithPadOrCropD,
    Lambda, ToTensorD
)
from monai.data.image_reader import NumpyReader
from torch.utils.data.dataloader import default_collate
import math
import itertools
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def montage(volume: torch.Tensor, fill: float = 0.0, normalize: bool = False) -> torch.Tensor:
    """
    Creates a 2D grid montage from a 3D tensor (D, H, W).
    :param volume: 3D tensor (D, H, W)
    :param fill: fill value between slices
    :param normalize: if True, normalize each slice to [0, 1]
    :return: 2D montage tensor (H_total, W_total)
    """
    if volume.ndim != 3:
        raise ValueError("Input must be a 3D tensor (D, H, W)")

    D, H, W = volume.shape
    grid_size = math.ceil(math.sqrt(D))
    grid_H = grid_size * H
    grid_W = grid_size * W

    canvas = torch.full((grid_H, grid_W), fill, dtype=volume.dtype)

    for idx in range(D):
        row = idx // grid_size
        col = idx % grid_size
        slice_img = volume[idx]
        if normalize:
            min_val = slice_img.min()
            max_val = slice_img.max()
            if (max_val - min_val) > 1e-5:
                slice_img = (slice_img - min_val) / (max_val - min_val)
        canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = slice_img

    return canvas

def axis_montage(vol, axis, **kw):
    # vol : (D, H, W)
    if axis == 0:  # axial(Z)
        v = vol
    elif axis == 1:  # coronal(Y)
        v = vol.permute(1, 0, 2)        # (H, D, W)
    else:            # sagittal(X)
        v = vol.permute(2, 0, 1)        # (W, D, H)
    return montage(v, **kw)

def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """

    # Drop all keys ending with '_transforms' to avoid DataLoader KeyError
    keys_to_remove = [k for k in _dict if k.endswith("_transforms")]
    for k in keys_to_remove:
        _dict.pop(k)

    _dict['context'] = torch.tensor([ _dict[c] for c in const.CONDITIONING_VARIABLES ]).unsqueeze(0)
    return _dict

# Visualization with TensorBoard
def images_to_tensorboard(
    writer,
    epoch,
    mode,
    autoencoder,
    z_channels,
    diffusion,
    scale_factor
):
    """
    Visualize the generation on tensorboard using MONAI montage.
    Now logs for each (size, view) pair.
    """
    sizes = ['small', 'medium', 'large']
    axes = {0: "axial", 1: "coronal", 2: "sagittal"}

    for tag_i, size in enumerate(sizes):
        context = torch.tensor([[
            (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,
            (torch.randint(1, 2, (1,)) - const.SEX_MIN) / const.SEX_DELTA,
            (torch.randint(1, 3, (1,)) - const.DIA_MIN) / const.DIA_DELTA,
            0.567, 0.539, 0.578, 0.558,
            0.30 * (tag_i + 1),
        ]], dtype=torch.float32, device=device)

        z = torch.randn((1, z_channels, 16, 16, 16), device=device) * scale_factor

        for t in reversed(range(1000)):
            z = diffusion(z, t=torch.tensor([t], device=device), cond_input={'context': context})

        with torch.no_grad():
            recon_image = autoencoder.decode(z)  # (1, 1, D, H, W)

        recon_image = (recon_image.clamp(-1, 1) + 1) / 2  # normalize to [0, 1]

        for axis, view_name in axes.items():
            grid = axis_montage(recon_image[0, 0], axis=axis, fill=0, normalize=True)
            tag = f"{mode}/{size}_{view_name}"  # ex) valid/small_axial
            writer.add_image(tag, grid.unsqueeze(0), epoch)


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file: # open config file
        try:
            config = yaml.safe_load(file) # YAML file -> python dictionary
        except yaml.YAMLError as exc: # fail to open config file -> exception
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    ldm_config = config['ldm_params']
    vqvae_config = config['vqvae_params']
    train_config = config['train_params']

    # Create the noise scheduler #
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Instantiate Condition related components
    condition_types = []
    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'context' in condition_types:
            validate_context_config(condition_config)

    ##########################

    dataset_df = pd.read_csv(train_config['A_csv'])


    transforms_fn = transforms.Compose([
        LoadImageD(keys=["latent_path"], reader=NumpyReader()),
        EnsureChannelFirstD(keys=["latent_path"], channel_dim=0),
        SpacingD(keys=["latent_path"], pixdim=const.RESOLUTION, mode="bilinear"),
        ResizeWithPadOrCropD(keys=["latent_path"], spatial_size=(32, 40, 32)),
        Lambda(func=concat_covariates),
        ToTensorD(keys=["latent_path", "context"], track_meta=False),
    ])

    dataset_df = dataset_df.dropna(subset=["image_path", "segm_path", "latent_path"])

    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']


    trainset = get_dataset_from_pd(train_df, transforms_fn, None)
    validset = get_dataset_from_pd(valid_df, transforms_fn, None)


    train_loader = DataLoader(dataset=trainset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    valid_loader = DataLoader(dataset=validset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=False,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    # Instantiate the model
    unet = UNet(im_channels=vqvae_config['z_channels'],
                model_config=ldm_config).to(device)  # load model to GPU/CPU
    unet.train()

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(unet.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    start_epoch = 0

    if os.path.exists(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])):
        print('Loaded Unet checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    scaler = GradScaler()

    log_path = "/DataRead2/chsong/tensorboard/ldm_exp3"
    writer = SummaryWriter(log_dir=log_path)

    print('Loading vqvae model.')
    vqvae = VQVAE(im_channels=vqvae_config['im_channels'],
                  model_config=vqvae_config).to(device)
    vqvae.eval()

    # Load trained vqvae if checkpoint exists
    path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(path):
        print('Loaded vqvae checkpoint')
        ckpt = torch.load(path, map_location=device)
        vqvae.load_state_dict(ckpt["model_state_dict"])
    else:
        raise Exception('VQVAE checkpoint not found')

    ## Compute Average Scale Factor
    # loader = DataLoader(trainset, batch_size=16, shuffle=True)
    # sigmas = []
    # with torch.no_grad():
    #     for batch in itertools.islice(loader, 200):
    #         sigmas.append(batch['latent_path'].std().item())
    # scale_factor = 1 / np.mean(sigmas)
    # print('Global scale factor =', scale_factor)
    # torch.save({'scale_factor': scale_factor}, 'scale_factor.pt')


    # scale_factor = torch.load('scale_factor.pt')['scale_factor']
    scale_factor = torch.load('scale_factor_eval.pt')['scale_factor_eval']
    print('Global scale factor loaded =', scale_factor)

    global_counter = {'train': 0, 'valid': 0}  # track batch index for TensorBoard

    best_val = float("inf")
    min_delta = 1e-5
    patience = 10
    wait = 0

    # Run training
    for epoch_idx in range(start_epoch, num_epochs): # repeat for epoch times
        for mode in ['train', 'valid']:
            unet.train() if mode == 'train' else unet.eval()
            loader = train_loader if mode == 'train' else valid_loader
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch_idx + 1}")

            for step, batch in progress_bar:

                # # training 루프 직전에 latent_std = latents.std().item() 찍어보기
                # print("latent batch std before scaling:", batch['latent_path'].std().item())
                # print("applied scale_factor:", scale_factor)

                with autocast(enabled=True):

                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    latents = batch['latent_path'].to(device) * scale_factor
                    n = latents.shape[0]
                    cond_input = {}

                    ########### Handling Conditional Input ###########
                    if 'context' in condition_types:
                        with torch.no_grad():
                            validate_context_config(condition_config)
                            cond_input['context'] = batch['context'].to(device)

                    with torch.set_grad_enabled(mode == 'train'):
                        # Sample random noise
                        noise = torch.randn_like(latents).to(device)

                        # Sample timestep
                        t = torch.randint(0, diffusion_config['num_timesteps'], (n,)).to(device).long()

                        # Add noise to images according to timestep
                        noisy_im = scheduler.add_noise(latents, noise, t)
                        noise_pred = unet(noisy_im, t, cond_input = cond_input) # U-Net predicts the noise

                        loss = criterion(noise_pred, noise) # calculate loss

                    if mode == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                    global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch_idx)
            print(f"[{mode}] Epoch {epoch_idx + 1}: Loss = {epoch_loss:.4f}")

            ########## Visualize result with TensorBoard #########

            images_to_tensorboard(
                writer=writer,
                epoch=epoch_idx,
                mode=mode,
                autoencoder=vqvae,
                z_channels=vqvae_config['z_channels'],
                diffusion=unet,
                scale_factor=scale_factor
            )
            ########################################################

            if mode == 'valid':
                if epoch_loss < best_val - min_delta:  # 미세한 수치 오차 허용
                    best_val = epoch_loss
                    wait = 0
                    torch.save(
                        {
                            'epoch': epoch_idx,
                            'model_state_dict': unet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val
                        },
                        os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name'])
                    )
                    print(f" Best checkpoint saved (epoch_loss = {best_val:.4g}, epoch = {epoch_idx + 1})")
                else:
                    wait += 1
                    if wait >= patience:
                        print(f" Early-stopping at epoch {epoch_idx + 1} (no val improvement for {patience} epochs)")
                        torch.save({
                            'epoch': epoch_idx,
                            'model_state_dict': unet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))  # save model
                        return

        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])) # save model

    print('Done Training ...')


if __name__ == '__main__': # runs only when this file script is run (don't run when imported)
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    train(args)
