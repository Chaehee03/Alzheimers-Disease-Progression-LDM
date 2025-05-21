import yaml
import argparse
import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from utils.data import get_dataset_from_pd
from models import const
import imageio
from torchvision.transforms import ToPILImage
from monai.metrics import SSIMMetric, PSNRMetric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler_g = GradScaler()
scaler_d = GradScaler()

ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
psnr_metric = PSNRMetric(max_val=1.0)

def get_commitment_beta(epoch, max_beta=0.5, warmup_epochs=10):
    """
    Warmup schedule for commitment beta.
    Linearly increases from 0 to max_beta over `warmup_epochs`.
    """
    return min(max_beta, (epoch + 1) / warmup_epochs * max_beta)


def compute_lpips_2d_slicewise(recon_volume, target_volume, lpips_model):
    """
    recon_volume, target_volume: (B, 1, D, H, W) in [0, 1]
    LPIPS는 (B, 3, H, W) 이미지를 요구하므로 각 슬라이스를 3채널로 변환 후 처리
    """
    B, C, D, H, W = recon_volume.shape
    assert C == 1, "Expected single channel input"

    lpips_scores = []
    for b in range(B):
        for d in range(D):
            pred = recon_volume[b, 0, d]  # (H, W)
            gt = target_volume[b, 0, d]   # (H, W)

            # Normalize to [0, 1]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

            # Convert to 3-channel image: (3, H, W)
            pred_rgb = pred.repeat(3, 1, 1).unsqueeze(0).to(device)
            gt_rgb = gt.repeat(3, 1, 1).unsqueeze(0).to(device)

            score = lpips_model(pred_rgb, gt_rgb, normalize=True)
            lpips_scores.append(score.item())

    return np.mean(lpips_scores)


def get_disc_weight(epoch, total_epochs):
    if epoch < 5:
        return 0.0
    elif epoch < 15:
        # warm-up: linear from e-4 to 0.1
        t = (epoch - 5) / 10
        return 1e-4 + (0.1 - 1e-4) * t
    elif epoch < 31:
        # grow to 0.25
        t = (epoch - 15) / 15
        return 0.1 + (0.25 - 0.1) * t
    elif epoch < 61:
        return 0.25
    elif epoch <= total_epochs:
        t = (epoch - 61) / (total_epochs - 61)
        return 0.25 + (0.35 - 0.25) * 0.5 * (1 - np.cos(np.pi * t))
    else:
        return 0.35

def save_gif(input_volume, recon_volume, epoch_idx, save_dir, tag='vqvae'):
    """
    Save a gif comparing input and reconstruction from a 3D volume.
    input_volume, recon_volume: (1, C, D, H, W) torch tensors scaled [0,1]
    """
    os.makedirs(save_dir, exist_ok=True)
    input_volume = input_volume[0, 0]  # (D, H, W)
    recon_volume = recon_volume[0, 0]
    frames = []
    to_pil = ToPILImage()

    def normalize(x):
        x = x - x.min()
        x = x / (x.max() + 1e-5)
        return x

    for d in range(input_volume.shape[0]):
        input_slice = normalize(input_volume[d].cpu().unsqueeze(0))
        recon_slice = normalize(recon_volume[d].cpu().unsqueeze(0))
        # concat horizontally
        stacked = torch.cat([input_slice, recon_slice], dim=-1)
        img = to_pil(torch.clamp(stacked, 0, 1))
        frames.append(img)

    gif_path = os.path.join(save_dir, f"{tag}_epoch{epoch_idx}.gif")
    imageio.mimsave(gif_path, frames, duration=0.12)


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['vqvae_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(train_config['dataset_csv'])

    dataset_df = dataset_df.dropna(subset=["image_path", "segm_path"])

    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, train_config['cache_dir'])
    validset = get_dataset_from_pd(valid_df, transforms_fn, train_config['cache_dir'])

    train_loader = DataLoader(dataset=trainset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['vqvae_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=validset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['vqvae_batch_size'],
                              shuffle=False,
                              persistent_workers=True,
                              pin_memory=True)

    val_fixed_sample = next(iter(valid_loader))['image'].float().to(device)[:1]

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['vqvae_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()

    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(in_channels=dataset_config['im_channels']).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['vae_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['vae_lr'], betas=(0.5, 0.999))

    pretrain_epochs = 5
    disc_step_start = pretrain_epochs * len(train_loader)
    # disc_step_start = train_config['disc_step_start']
    step_count = 0
    start_epoch = 0

    best_psnr = -1
    best_ssim = -1
    epochs_without_improvement = 0
    early_stopping_patience = train_config.get("early_stopping_patience", 4)

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vqvae autoencoder checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint['step_count']


    if os.path.exists(os.path.join(train_config['task_name'],
                                                train_config['vqvae_discriminator_ckpt_name'])):
        print('Loaded vqvae discriminator checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'],
                                                  train_config['vqvae_discriminator_ckpt_name']))
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint['step_count']


    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['vae_acc_steps']



    for epoch_idx in range(start_epoch, num_epochs):
        recon_losses = []
        codebook_losses = []
        # commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        disc_weight = get_disc_weight(epoch_idx, total_epochs=num_epochs)

        used_indices = set()

        for sample in tqdm(train_loader, desc="Training"):
            step_count += 1
            im = sample['image'].float().to(device)

            with autocast():
                # Fetch autoencoders output(reconstructions)
                model_output = model(im)
                output, z, quantize_losses = model_output

                with torch.no_grad():
                    latent, _ = model.encode(im)
                    _, _, encoding_indices = model.quantize(latent)
                    used_indices.update(torch.unique(encoding_indices).cpu().tolist())

                    flat = latent.permute(0, 2, 3, 4, 1).reshape(-1, latent.shape[1])
                    print(f"[DEBUG][Epoch {epoch_idx + 1}] latent std: {flat.std().item():.6f}")

                ######### Optimize Generator ##########
                # L2 Loss
                recon_loss = recon_criterion(output, im)
                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / acc_steps
                commitment_beta = get_commitment_beta(epoch_idx)
                g_loss = (recon_loss +
                          (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                          (commitment_beta * quantize_losses['commitment_loss'] / acc_steps))
                codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
                # Adversarial loss only if disc_step_start steps passed
                if step_count >= disc_step_start:
                    disc_fake_pred = discriminator(model_output[0])
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.ones(disc_fake_pred.shape,
                                                               device=disc_fake_pred.device))
                    gen_losses.append(disc_weight * disc_fake_loss.item())
                    g_loss += disc_weight * disc_fake_loss / acc_steps
            with torch.no_grad():
                model.eval()
                lpips_loss = compute_lpips_2d_slicewise(output, im, lpips_model)
                model.train()
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss)
            g_loss += (train_config['perceptual_weight'] * lpips_loss / acc_steps)

            scaler_g.scale(g_loss).backward()
            #####################################

            ######### Optimize Discriminator #######
            if step_count >= disc_step_start:
                with autocast():
                    fake = output
                    disc_fake_pred = discriminator(fake.detach())
                    disc_real_pred = discriminator(im)
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.zeros(disc_fake_pred.shape,
                                                                device=disc_fake_pred.device))
                    disc_real_loss = disc_criterion(disc_real_pred,
                                                    torch.ones(disc_real_pred.shape,
                                                               device=disc_real_pred.device))
                    disc_loss = disc_weight * (disc_fake_loss + disc_real_loss) / 2
                    disc_losses.append(disc_loss.item())
                    disc_loss = disc_loss / acc_steps
                scaler_d.scale(disc_loss).backward()
                if step_count % acc_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()
            #####################################

            if step_count % acc_steps == 0:
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

        if step_count % acc_steps != 0:
            if scaler_d._scale is not None:
                scaler_d.step(optimizer_d)
                scaler_d.update()
                optimizer_d.zero_grad()
            if scaler_g._scale is not None:
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

        # Test on Validation Set
        model.eval()
        with torch.no_grad():
            # Evaluation Metric (SSIM, PSNR)
            all_ssim = []
            all_psnr = []
            for sample in valid_loader:
                val_input = sample['image'].float().to(device)
                val_output, _, _ = model(val_input)

                val_input = torch.clamp(val_input, 0, 1)
                val_output = torch.clamp(val_output, 0, 1)

                ssim_score = ssim_metric(val_output, val_input).item()
                psnr_score = psnr_metric(val_output, val_input).item()
                all_ssim.append(ssim_score)
                all_psnr.append(psnr_score)

            # Sampled GIF Saving Logic
            val_recon, _, _ = model(val_fixed_sample)
            save_dir = os.path.join(train_config['task_name'], 'vqvae_gif_samples')
            save_gif(val_fixed_sample.cpu(), val_recon.cpu(), epoch_idx + 1, save_dir)
        model.train()
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)

        codebook_coverage = len(used_indices) / model.embedding.num_embeddings * 100

        if len(disc_losses) > 0 and len(gen_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | Codebook usage: {:.2f} | G Loss : {:.4f} | D Loss {:.4f} |'
                'SSIM: {:.4f} |  PSNR: {:.2f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       codebook_coverage,
                       np.mean(gen_losses),
                       np.mean(disc_losses),
                       np.mean(all_ssim),
                       np.mean(all_psnr)), flush=True)
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                  'Codebook : {:.4f} | Codebook usage: {:.2f} | '
                  'SSIM: {:.4f} |  PSNR: {:.2f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses),
                         codebook_coverage,
                         np.mean(all_ssim),
                         np.mean(all_psnr)), flush=True)

        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
        }, os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name']))

        torch.save({
            'epoch': epoch_idx,
            'step_count': step_count,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
        }, os.path.join(train_config['task_name'], train_config['vqvae_discriminator_ckpt_name']))

        # Save Best Checkpoint
        improved = False
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            improved = True
            print(f"[Best Checkpoint] New best PSNR: {avg_psnr:.2f} at epoch {epoch_idx + 1}")

            torch.save({
                'epoch': epoch_idx,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
            }, os.path.join(train_config['task_name'], 'vqvae_best_checkpoint_2.pth'))
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            improved = True

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"[EarlyStopping] No improvement. Count: {epochs_without_improvement}")

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"[EarlyStopping] Triggered at epoch {epoch_idx + 1}. Best PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}")
            break

    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    train(args)




