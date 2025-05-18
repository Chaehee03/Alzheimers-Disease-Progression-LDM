import torch
import yaml
import argparse
import os
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from torch.optim.lr_scheduler import MultiStepLR
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from monai import transforms
from utils.data import get_dataset_from_pd
from monai.data.image_reader import NumpyReader
from torch.utils.data.dataloader import default_collate
from models.unet_cond import UNet
from models.controlnet import ControlNet
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import math
import nibabel as nib
from models import const
import pandas as pd
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

def sample_using_controlnet_and_z(autoencoder, diffusion_unet, controlnet,
                                  starting_z, starting_a, context,
                                  scale_factor, scheduler, device='cuda'):
    """
    Returns (C,D,H,W) tensor in [0,1] for TensorBoard.
    """
    controlnet.eval(); diffusion_unet.eval(); autoencoder.eval()

    with torch.no_grad(), autocast(True):
        z = torch.randn_like(starting_z)             # z_T  (T = scheduler.num_timesteps-1)
        n = starting_z.size(0)

        # ----- hint tensor (starting_z + age channel) -----
        starting_a = torch.as_tensor(starting_a, device=device)
        age_chan = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
        hint     = torch.cat([starting_z, age_chan], dim=1)

        # ----- reverse loop -----
        for t in reversed(range(scheduler.num_timesteps)):
            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)

            # Noise prediction
            noise_pred = controlnet(
                z,
                t_tensor,
                context,
                hint)

            # One DDPM step
            z, _ = scheduler.sample_prev_timestep(z, noise_pred, t)

        # decode & rescale to [0,1]
        with torch.no_grad():
            recon_image = autoencoder.decode(z)  # (1, 1, D, H, W)

        recon_image = (recon_image.clamp(-1, 1) + 1) / 2  # normalize to [0, 1]
        return recon_image.squeeze(0).squeeze(0)



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


# Visualization with TensorBoard
def images_to_tensorboard(writer, epoch, mode,
                          autoencoder, diffusion, controlnet,
                          dataset, scale_factor, scheduler):
    """
    visualise three random samples:
      baseline  |  predicted  |  (optional) ground-truth follow-up
    """

    def _to_3d(vol: torch.Tensor) -> torch.Tensor:
        """(1,D,H,W) 또는 (1,1,D,H,W) → (D,H,W) 로 변환"""
        while vol.ndim > 3:
            vol = vol.squeeze(0)
        return vol  # 최종적으로 3차원

    resample_fn = transforms.Spacing(pixdim=1.5)
    axes = {0: "axial", 1: "coronal", 2: "sagittal"}
    rand_ids = torch.randint(0, len(dataset), (3,))

    for tag_i, idx in enumerate(rand_ids):
        item = dataset[idx]
        starting_z = item['starting_latent'].unsqueeze(0).to(device) * scale_factor
        starting_a = torch.tensor(item['starting_age']).unsqueeze(0).to(device)
        context    = item['context'].to(device)

        # (1) sampling --------------------------------------------------------
        pred_vol = sample_using_controlnet_and_z(
            autoencoder, diffusion, controlnet,
            starting_z, starting_a, context,
            scale_factor, scheduler, device=device)     # (D,H,W) in [0,1]
        pred_vol = _to_3d(pred_vol)

        # (2) baseline image --------------------------------------------------
        if 'starting_image' in item:
            st_img = nib.load(item['starting_image']).get_fdata()
            st_img = resample_fn(torch.from_numpy(st_img).unsqueeze(0)).squeeze(0)
            st_img = (st_img - st_img.min()) / (st_img.max() - st_img.min() + 1e-6)
            st_img = _to_3d(st_img)
        else:
            st_img = autoencoder.decode(starting_z / scale_factor)
            st_img = _to_3d(st_img)

        # (3) optional GT follow-up ------------------------------------------
        gt_img = None
        if 'followup_image' in item:
            gt_img = nib.load(item['followup_image']).get_fdata()
            gt_img = resample_fn(torch.from_numpy(gt_img).unsqueeze(0)).squeeze(0)
            gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-6)
            gt_img = _to_3d(gt_img)

        for ax, ax_name in axes.items():
            def to_grid(vol):  # vol: (D,H,W)
                return axis_montage(vol, ax, fill=0, normalize=True)

            grid_list = [to_grid(st_img), to_grid(pred_vol)]
            if gt_img is not None:
                grid_list.append(to_grid(gt_img))

            grid = torch.stack(grid_list)  # (N, H, W)
            writer.add_images(f'{mode}/{ax_name}_cmp_{tag_i}',
                              grid.unsqueeze(1),  # add fake channel dim
                              epoch)

        # (4) 3-D → 2-D montage ----------------------------------------------
        def to_grid(vol):  # vol: (D,H,W)
            return axis_montage(vol, 0, fill=0, normalize=True)

        grids = [to_grid(st_img), to_grid(pred_vol)]
        if gt_img is not None:
            grids.append(to_grid(gt_img))
        grids = torch.stack(grids)                # (N, H, W)

        writer.add_images(f'{mode}/comparison_{tag_i}', grids.unsqueeze(1), epoch)


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    conditions = [
        _dict['followup_age'],
        _dict['sex'],
        _dict['followup_diagnosis'],
        _dict['followup_cerebral_cortex'],
        _dict['followup_hippocampus'],
        _dict['followup_amygdala'],
        _dict['followup_cerebral_white_matter'],
        _dict['followup_lateral_ventricle']
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    ldm_config = config['ldm_params']
    vqvae_config = config['vqvae_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     ldm_scheduler=True)

    def rename_keys(d):
        d['starting_latent'] = d.pop('starting_latent_path')
        d['followup_latent'] = d.pop('followup_latent_path')
        return d

    transforms_fn = transforms.Compose([
        transforms.LoadImaged(keys=['starting_latent_path', 'followup_latent_path'],
                   reader=NumpyReader()),
        transforms.Lambda(rename_keys),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.SpacingD(keys=['starting_latent', 'followup_latent'], pixdim=(2, 2, 2), mode="bilinear"),
        transforms.ResizeWithPadOrCropD(keys=['starting_latent', 'followup_latent'], spatial_size=(32, 40, 32)),
        transforms.Lambda(func=concat_covariates),
        transforms.ToTensorD(keys=['starting_latent', 'followup_latent', "context"], track_meta=False),
    ])

    dataset_df = pd.read_csv(train_config['B_csv'])
    dataset_df = dataset_df.dropna(subset=[
        "starting_image_path", "followup_image_path",
        "starting_segm_path", "followup_segm_path",
        "starting_latent_path", "followup_latent_path"
    ])

    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, train_config['cache_dir_B'])
    validset = get_dataset_from_pd(valid_df, transforms_fn, train_config['cache_dir_B'])


    train_loader = DataLoader(dataset=trainset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['controlnet_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    valid_loader = DataLoader(dataset=validset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['controlnet_batch_size'],
                              shuffle=False,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    model = ControlNet(im_channels=vqvae_config['z_channels'],
                       model_config=ldm_config,
                       model_locked=True,
                       model_ckpt=os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name']),
                       device=device).to(device)
    model.train()

    # Training parameters setting
    num_epochs = train_config['controlnet_epochs']
    optimizer = Adam(model.get_params(), lr=train_config['controlnet_lr'])
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['controlnet_lr_steps'], gamma=0.1)
    criterion = torch.nn.MSELoss()

    start_epoch = 0

    # Load ControlNet checkpoint to continue training.
    if os.path.exists(os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name'])):
        print('Continuing ControlNet training from checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # Load pretrained Unet
    unet = UNet(im_channels=vqvae_config['z_channels'],
                model_config=ldm_config).to(device)  # load model to GPU/CPU
    unet.train()

    if os.path.exists(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])):
        print('Loaded Unet checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))
        unet.load_state_dict(checkpoint['model_state_dict'])

    ########## Settings for TensorBoard Visualization ##########
    sample_dict = train_df.iloc[0].to_dict()
    out = transforms_fn(sample_dict, threading=False)
    z = out['followup_latent'].to(device)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

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
    ###############################################################

    scaler = GradScaler()

    best_val = float("inf")
    min_delta = 1e-5
    patience = 5
    wait = 0

    global_counter = {'train': 0, 'valid': 0}

    log_path = "/DataRead2/chsong/tensorboard/controlnet_exp1"
    writer = SummaryWriter(log_dir=log_path)

    visualize_period = 5

    # Run training
    for epoch_idx in range(start_epoch, num_epochs):  # repeat for epoch times
        for mode in ['train', 'valid']:
            model.train() if mode == 'train' else model.eval()
            loader = train_loader if mode == 'train' else valid_loader
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch_idx + 1}")

            for step, batch in progress_bar:

                with autocast(enabled=True):

                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(mode == 'train'):

                        starting_z = batch['starting_latent'].to(device) * scale_factor
                        followup_z = batch['followup_latent'].to(device) * scale_factor
                        starting_a = batch['starting_age'].to(device)
                        context = batch['context'].to(device)

                        n = starting_z.shape[0]

                        concatenating_age = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
                        controlnet_condition = torch.cat([starting_z, concatenating_age], dim=1)

                        # Sample random noise
                        noise = torch.randn_like(followup_z).to(device)

                        # Sample timestep
                        t = torch.randint(0, diffusion_config['num_timesteps'], (n,)).to(device).long()

                        # Add noise to images according to timestep
                        noisy_im = scheduler.add_noise(followup_z, noise, t)

                        noise_pred = model(
                            noisy_im.float(),
                            t,
                            context.float(),
                            controlnet_condition.float()
                        )

                        loss = criterion(noise_pred, noise)

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # -------------------------------
                # Iteration end
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch_idx)
            print(f"[{mode}] Epoch {epoch_idx + 1}: Loss = {epoch_loss:.4f}")

            ########## Visualize result with TensorBoard #########
            if epoch_idx % visualize_period == 0:
                images_to_tensorboard(writer=writer,
                                  epoch=epoch_idx + 1,
                                  mode=mode,
                                  autoencoder=vqvae,
                                  diffusion=unet,
                                  controlnet=model,
                                  dataset=trainset if mode == 'train' else validset,
                                  scale_factor=scale_factor,
                                  scheduler=scheduler)

            ########################################################

            if epoch_idx > 2 and mode == 'valid':
                if epoch_loss < best_val - min_delta:
                    best_val = epoch_loss
                    wait = 0
                    torch.save(
                        {
                            'epoch': epoch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val
                        },
                        os.path.join(train_config['task_name'], train_config['controlnet_best_ckpt_name'])
                    )
                    print(f"ControlNet Best checkpoint saved (epoch_loss = {best_val:.4g}, epoch = {epoch_idx + 1})")
                else:
                    wait += 1
                    if wait >= patience:
                        print(
                            f" Early-stopping at epoch {epoch_idx + 1} (no loss improvement for {patience} epochs)")
                        torch.save({
                            'epoch': epoch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name']))  # save model
                        return

        lr_scheduler.step()
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }, os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name']))  # save model

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq.yaml', type=str)
    args = parser.parse_args()
    train(args)
