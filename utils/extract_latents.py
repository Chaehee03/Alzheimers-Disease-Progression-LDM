import yaml
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from models import const
from models.vqvae import VQVAE
from torch.optim import Adam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    parser.add_argument('--dataset_csv', type=str, required=True)
    args = parser.parse_args()

    # Read the config file #
    with open(args.config_path, 'r') as file:  # open config file
        try:
            config = yaml.safe_load(file)  # YAML file -> python dictionary
        except yaml.YAMLError as exc:  # fail to open config file -> exception
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    vqvae_config = config['vqvae_params']
    train_config = config['train_params']

    vae = VQVAE(im_channels=dataset_config['im_channels'],
              model_config=vqvae_config).to(DEVICE)
    vae.eval()  # don't need training, just create latents

    optimizer_g = Adam(vae.parameters(), lr=train_config['vae_lr'], betas=(0.5, 0.999))

    # Load trained vae if checkpoint exists
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vqvae checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'],
                                             train_config['vqvae_autoencoder_ckpt_name']))
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    df = pd.read_csv(train_config['dataset_csv'])
    df = df.dropna(subset=["image_path", "segm_path"])

    with torch.no_grad():
        for image_path in tqdm(df.image_path, total=len(df)):
            destpath = image_path.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz')
            # if os.path.exists(destpath): continue
            mri_tensor = transforms_fn({'image_path': image_path})['image'].to(DEVICE)
            mri_latent, _ = vae.encode(mri_tensor.unsqueeze(0))
            mri_latent = mri_latent.cpu().squeeze(0).numpy()
            np.savez_compressed(destpath, data=mri_latent)
            # Add latent_path field to dataset.csv & save
            df.loc[df.image_path == image_path, 'latent_path'] = destpath

    df.to_csv(train_config['dataset_csv'], index=False)
