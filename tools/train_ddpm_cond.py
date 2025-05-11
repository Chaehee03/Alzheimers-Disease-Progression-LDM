import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet_cond import UNet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils import utils
from utils.config_utils import *
from utils.diffusion_utils import *
from models import const
from monai import transforms
from monai.data.image_reader import NumpyReader
from utils.data import get_dataset_from_pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
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
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate([ 'small', 'medium', 'large' ]):

        context = torch.tensor([[
            (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age
            (torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567, # (mean) cerebral cortex
            0.539, # (mean) hippocampus
            0.578, # (mean) amygdala
            0.558, # (mean) cerebral white matter
            0.30 * (tag_i+1), # variable size lateral ventricles
        ]])

        # 샘플 latent 생성 (DDPM sampling을 유사하게 구현했다고 가정)
        z = torch.randn((1, z_channels, 16, 16, 16), device=device) * scale_factor
        for t in reversed(range(1000)):
            z = diffusion(z, t=torch.tensor([t], device=device), cond_input={'context': context})

        # 복원 이미지 생성
        with torch.no_grad():
            recon_image = autoencoder.decode(z)

        # [-1,1] → [0,1] 정규화 후 TensorBoard에 이미지 추가
        recon_image = (recon_image.clamp(-1, 1) + 1) / 2
        writer.add_images(tag=f"{mode}/{size}_ventricles", img_tensor=recon_image, global_step=epoch)

        # image = sample_using_diffusion(
        #     autoencoder=autoencoder,
        #     diffusion=diffusion,
        #     context=context,
        #     device=DEVICE,
        #     scale_factor=scale_factor
        # )
        #
        # utils.tb_display_generation(
        #     writer=writer,
        #     step=epoch,
        #     tag=f'{mode}/{size}_ventricles',
        #     image=image
        # )

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

    # Check missing latent paths
    dataset_df = pd.read_csv(args.dataset_csv)
    missing_latents = [
        row['latent_path']
        for _, row in dataset_df.iterrows()
        if not os.path.exists(row['latent_path'])
    ]
    if missing_latents:
        raise FileNotFoundError(
            f"[latent_path error] Following latent paths don't exist.\n"
            f"{missing_latents[:5]}...\nTotal {len(missing_latents)} latent paths are missing"
        )

    # Create the noise scheduler #
    scheduler = LinearNoiseScheduler(num_timesteps=train_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Instantiate Condition related components
    empty_context_embed = None
    condition_types = []
    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'context' in condition_types:
            validate_context_config(condition_config)

    ##########################
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)
    ])

    dataset_df = pd.read_csv(train_config['dataset_csv'])
    dataset_df = dataset_df.dropna(subset=["image_path", "segm_path"])
    
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, train_config['cache_dir'])
    validset = get_dataset_from_pd(valid_df, transforms_fn, train_config['cache_dir'])

    train_loader = DataLoader(dataset=trainset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    valid_loader = DataLoader(dataset=validset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=False,
                              persistent_workers=True,
                              pin_memory=True)
    ############################

    # Instantiate the model
    unet = UNet(im_channels=vqvae_config['z_channels'],
                 model_config=ldm_config).to(device) # load model to GPU/CPU
    unet.train()

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(unet.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter()


    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['latent']

    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    global_counter = {'train': 0, 'valid': 0}  # track batch index for TensorBoard

    # Run training
    for epoch_idx in range(num_epochs): # repeat for epoch times
        for mode in ['train', 'valid']:
            unet.train() if mode == 'train' else unet.eval()
            loader = train_loader if mode == 'train' else valid_loader
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch_idx}")

            for step, batch in progress_bar:

                with autocast(enabled=True):

                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    latents = batch['latent'].to(device) * scale_factor
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
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            print(f"[{mode}] Epoch {epoch_idx + 1}: Loss = {epoch_loss / len(loader):.4f}")

            ########## Visualize result with TensorBoard #########
            print('Loading vqvae model.')
            vqvae = VQVAE(im_channels=vqvae_config['im_channels'],
                      model_config=ldm_config).to(device)
            vqvae.eval()

            # Load trained vqvae if checkpoint exists
            if os.path.exists(os.path.join(train_config['task_name'],
                                           train_config['vqvae_autoencoder_ckpt_name'])):
                print('Loaded vqvae checkpoint')
                vqvae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                            train_config['vqvae_autoencoder_ckpt_name']),
                                               map_location=device))
            else:
                raise Exception('VQVAE checkpoint not found')

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

        torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name'])) # save model

    print('Done Training ...')


if __name__ == '__main__': # runs only when this file script is run (don't run when imported)
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    train(args)