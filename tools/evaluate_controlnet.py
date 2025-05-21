from models.controlnet import ControlNet
from models.vqvae import VQVAE
from models import const
import torch
import yaml
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from monai import transforms
from utils.data import get_dataset_from_pd
from monai.data.image_reader import NumpyReader
from torch.utils.data.dataloader import default_collate
from models.controlnet import ControlNet
import pandas as pd
from monai.metrics import SSIMMetric, PSNRMetric
import numpy as np
from tools.sample_controlnet import sample
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
psnr_metric = PSNRMetric(max_val=1.0)



def save_slice(gt_volume: torch.Tensor,
               pred_volume: torch.Tensor,
               step: int,
               save_dir: str):
    """
       gt_volume, pred_volume : (B=1, C=1, D, H, W) 또는 (D, H, W)
       두 볼륨에서 중앙 슬라이스를 뽑아
           [ pred | gt ]  형태로 붙여서 저장
       """

    os.makedirs(save_dir, exist_ok=True)
    to_pil = ToPILImage()

    # (1) 배치/채널 차원 제거 → (D,H,W)
    if gt_volume.dim() == 5:
        gt_volume   = gt_volume[0, 0]     # (D,H,W)
        pred_volume = pred_volume[0, 0]

    # (2) 정규화
    def norm(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        return x / (x.max() + 1e-5)

    # (3) 각 뷰의 중앙 슬라이스 추출
    def central_slice(vol: torch.Tensor, view: str) -> torch.Tensor:
        if view == "axial":      # z 축 기준 (D)
            idx = vol.shape[0] // 2
            return vol[idx]                      # (H,W)
        elif view == "coronal":  # y 축 기준 (H)
            idx = vol.shape[1] // 2
            return vol[:, idx, :]                # (D,W)
        elif view == "sagittal": # x 축 기준 (W)
            idx = vol.shape[2] // 2
            return vol[:, :, idx]                # (D,H)
        else:
            raise ValueError(f"Unknown view {view}")

    for view in ["axial", "coronal", "sagittal"]:
        gt_slice   = norm(central_slice(gt_volume, view)).unsqueeze(0)   # (1,H,W)
        pred_slice = norm(central_slice(pred_volume, view)).unsqueeze(0)

        vmin = torch.minimum(gt_volume.min(), pred_volume.min())
        vmax = torch.maximum(gt_volume.max(), pred_volume.max())

        def n(x): return (x - vmin) / (vmax - vmin + 1e-5)

        # (4) 좌측=pred, 우측=gt 로 grid화
        grid = make_grid(torch.stack([pred_slice, gt_slice]), nrow=2, padding=0)

        img = to_pil(grid)
        fn  = f"{view}_test{step}.png"
        img.save(os.path.join(save_dir, fn))
        img.close()


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


def to_unit_range(x):
    vmin = x.amin(dim=(-3, -2, -1), keepdim=True)
    vmax = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - vmin) / (vmax - vmin + 1e-8)


def evaluate(args):
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

    transforms_fn = transforms.Compose([
        transforms.LoadImaged(keys=['starting_latent_path', 'followup_image_path'],
                              reader=NumpyReader()),
        transforms.EnsureChannelFirstD(keys=["starting_latent_path"], channel_dim=0),
        transforms.EnsureChannelFirstD(keys=["followup_image_path"], channel_dim=None),
        transforms.SpacingD(keys=["starting_latent_path"],  pixdim=const.RESOLUTION, mode="bilinear"),
        transforms.SpacingD(keys=["followup_image_path"],  pixdim=const.RESOLUTION, mode="bilinear"),
        transforms.EnsureTypeD(keys=["starting_latent_path", "followup_image_path"],
                      data_type="tensor", track_meta=False),
        transforms.ResizeWithPadOrCropD(keys=['starting_latent_path'], spatial_size=(32, 40, 32)),
        transforms.ResizeWithPadOrCropD(keys=["followup_image_path"], spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.Lambda(func=concat_covariates),
        transforms.ToTensorD(keys=['starting_laten_path', 'followup_image_path', "context"], track_meta=False),
    ])

    dataset_df = pd.read_csv(train_config['B_csv'])
    dataset_df = dataset_df.dropna(subset=[
        "starting_image_path", "followup_image_path",
        "starting_segm_path", "followup_segm_path",
        "starting_latent_path", "followup_latent_path"
    ])

    test_df = dataset_df[dataset_df.split == 'test']
    # testset = get_dataset_from_pd(test_df, transforms_fn, train_config['cache_dir_sampling'])
    testset = get_dataset_from_pd(test_df, transforms_fn, None)

    test_loader = DataLoader(dataset=testset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['controlnet_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    # Instantiate the controlnet
    controlnet = ControlNet(im_channels=vqvae_config['z_channels'],
                       model_config=ldm_config,
                       model_locked=True,
                       model_ckpt=os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']),
                       device=device).to(device)
    controlnet.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['controlnet_best_ckpt_name'])), "Train ControlNet first"
    checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['controlnet_best_ckpt_name']))
    controlnet.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded controlnet checkpoint')

    vqvae = VQVAE(im_channels=vqvae_config['im_channels'],
              model_config=vqvae_config).to(device)
    vqvae.eval()

    # Load vqvae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])), \
        "VQVAE checkpoint not present. Train VQVAE first."
    checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name']))
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded vqvae checkpoint')

    scale_factor = sf = torch.load('scale_factor.pt')['scale_factor']
    print('Global scale factor loaded =', scale_factor)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    progress_bar.set_description(f"[Evaluating Test Set]")

    all_ssim = []
    all_psnr = []

    for step, batch in progress_bar:
        starting_z = batch['starting_latent_path'].to(device) * scale_factor
        starting_a = batch['starting_age'].to(device)
        followup_image = batch['followup_image_path'].to(device)
        context = batch['context'].to(device)

        n = starting_z.shape[0]

        concatenating_age = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
        controlnet_condition = torch.cat([starting_z, concatenating_age], dim=1)

        with torch.no_grad():

            recon, *_ = vqvae(followup_image.to(device))
            z, _ = vqvae.encode(followup_image.to(device))

            print("VQ-SSIM:", ssim_metric(to_unit_range(recon), to_unit_range(followup_image)).item())
            print("latent mean/std:", z.mean().item(), z.std().item())

            pred_follow_up = sample(controlnet, vqvae, scheduler, diffusion_config,
                   controlnet_condition, context, scale_factor).to(device).float()
            pred_follow_up = to_unit_range(pred_follow_up)

            gt_follow_up = to_unit_range(followup_image.float())

            assert pred_follow_up.shape == gt_follow_up.shape, \
                f"Shape mismatch: pred {pred_follow_up.shape}, gt {gt_follow_up.shape}"

            ssim_score = ssim_metric(pred_follow_up, gt_follow_up).item()
            psnr_score = psnr_metric(pred_follow_up, gt_follow_up).item()

        all_ssim.append(ssim_score)
        all_psnr.append(psnr_score)

        save_dir = os.path.join(train_config['task_name'], train_config['evaluate_controlnet_result_name'])
        if step < 10:
            print(f"[Step {step}] → running SSIM: {np.mean(all_ssim):.4f}, PSNR: {np.mean(all_psnr):.2f}")
            save_slice(gt_follow_up, pred_follow_up, step, save_dir)
        elif step % 20 == 0:
            print(f"[Step {step}] → running SSIM: {np.mean(all_ssim):.4f}, PSNR: {np.mean(all_psnr):.2f}")
            save_slice(gt_follow_up, pred_follow_up, step, save_dir)

    print(
        'Tested on Test Set - Average SSIM: {:.4f} |  Average PSNR: {:.2f}'.
        format(np.mean(all_ssim),
               np.mean(all_psnr)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for controlnet evaluation')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    evaluate(args)
