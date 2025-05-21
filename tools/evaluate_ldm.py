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
import pandas as pd
from monai.metrics import SSIMMetric, PSNRMetric
import numpy as np
from tools.sample_ldm import sample
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from models.unet_cond import UNet
import itertools


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

    # Drop all keys ending with '_transforms' to avoid DataLoader KeyError
    keys_to_remove = [k for k in _dict if k.endswith("_transforms")]
    for k in keys_to_remove:
        _dict.pop(k)

    _dict['context'] = torch.tensor([ _dict[c] for c in const.CONDITIONING_VARIABLES ]).unsqueeze(0)
    return _dict


# def to_unit_range(x):
#     vmin = x.amin(dim=(-3, -2, -1), keepdim=True)
#     vmax = x.amax(dim=(-3, -2, -1), keepdim=True)
#     return (x - vmin) / (vmax - vmin + 1e-8)




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

    dataset_df = pd.read_csv(train_config['A_csv'])

    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=["latent_path"], reader=NumpyReader()),
        transforms.EnsureChannelFirstD(keys=["latent_path"], channel_dim=0),
        # transforms.SpacingD(keys=["latent_path"], pixdim=const.RESOLUTION, mode="bilinear"),
        transforms.CenterSpatialCropD(keys=['latent_path'], roi_size=(32, 40, 32)),
        # transforms.ResizeWithPadOrCropD(keys=['latent_path'], spatial_size=(32, 40, 32)),
        transforms.Lambda(func=concat_covariates),
        transforms.ToTensorD(keys=["latent_path", "context"], track_meta=False),
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=(128, 160, 128), keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    # transforms_fn = transforms.Compose([
    #     transforms.LoadImageD(keys=["latent_path"], reader=NumpyReader()),
    #     transforms.EnsureChannelFirstD(keys=["latent_path"], channel_dim=0),
    #     # transforms.SpacingD(keys=["latent_path"], pixdim=const.RESOLUTION, mode="bilinear"),
    #     transforms.ResizeWithPadOrCropD(keys=["latent_path"], spatial_size=(32, 40, 32)),
    #     transforms.Lambda(func=concat_covariates),
    #     transforms.ToTensorD(keys=["latent_path", "context"], track_meta=False),
    #     transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    #     transforms.LoadImageD(image_only=True, keys=['image']),
    #     transforms.EnsureChannelFirstD(keys=['image']),
    #     transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    #     transforms.ResizeWithPadOrCropD(spatial_size=(128, 160, 128), keys=['image']),
    # ])

    dataset_df = dataset_df.dropna(subset=["image_path", "segm_path", "latent_path"])

    test_df = dataset_df[dataset_df.split == 'test']
    # testset = get_dataset_from_pd(test_df, transforms_fn, train_config['cache_dir_sampling'])
    testset = get_dataset_from_pd(test_df, transforms_fn, None)

    test_loader = DataLoader(dataset=testset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=default_collate)

    # Load pretrained Unet
    unet = UNet(im_channels=vqvae_config['z_channels'],
                model_config=ldm_config).to(device)  # load model to GPU/CPU
    unet.train()

    # if os.path.exists(os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name'])):
    #     print('Loaded Unet checkpoint')
    #     checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name']))
    #     unet.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     print('Loading Unet checkpoint failed')


    if os.path.exists('/DataRead2/chsong/checkpoints/ldm_best_checkpoint_eval_std_tmp.pth'):
        print('Loaded Unet checkpoint')
        checkpoint = torch.load('/DataRead2/chsong/checkpoints/ldm_best_checkpoint_eval_std_tmp.pth')
        unet.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Loading Unet checkpoint failed')


    vqvae = VQVAE(im_channels=vqvae_config['im_channels'], model_config=vqvae_config).to(device)
    vqvae.eval()

    # Load vqvae if found
    assert os.path.exists(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])), \
        "VQVAE checkpoint not present. Train VQVAE first."
    checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name']))
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded vqvae checkpoint')

    # scale_factor = torch.load('scale_factor.pt')['scale_factor']
    # print('Global scale factor loaded =', scale_factor)

    # Compute Average Scale Factor
    loader = DataLoader(testset, batch_size=16, shuffle=True)
    sigmas = []
    with torch.no_grad():
        for batch in itertools.islice(loader, 200):
            sigmas.append(batch['latent_path'].std().item())
    scale_factor = 1 / np.mean(sigmas)
    torch.save({'scale_factor_eval': scale_factor}, 'scale_factor_eval.pt')
    print('Global scale factor =', scale_factor)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    progress_bar.set_description(f"[Evaluating Test Set]")

    all_ssim = []
    all_psnr = []

    for step, batch in progress_bar:

        # # training 루프 직전에 latent_std = latents.std().item() 찍어보기
        # print("latent batch std before scaling:", batch['latent_path'].std().item())
        # print("applied scale_factor:", scale_factor)

        gt_image = batch['image'].to(device)
        context = batch['context'].to(device)

        with torch.no_grad():

            pred_image = sample(unet, vqvae, scheduler, context, diffusion_config, scale_factor).to(device).float()

            gt_image = gt_image.float()

            # print("pred_image std before scaling:", pred_image.std().item())
            # print("applied scale_factor:", scale_factor)

            gmin = torch.minimum(gt_image.min(), pred_image.min())
            gmax = torch.maximum(gt_image.max(), pred_image.max())

            def to_unit_range(x): return (x - gmin) / (gmax - gmin + 1e-5)

            pred_image = to_unit_range(pred_image)

            # gt_image = to_cp /path/to/train_dir/latest_ckpt.pth /tmp/tmp_ckpt_eval.pthunit_range(gt_image.float())

            assert pred_image.shape == gt_image.shape, \
                f"Shape mismatch: pred {pred_image.shape}, gt {gt_image.shape}"

            ssim_score = ssim_metric(pred_image, gt_image).item()
            psnr_score = psnr_metric(pred_image, gt_image).item()

        all_ssim.append(ssim_score)
        all_psnr.append(psnr_score)

        save_dir = os.path.join(train_config['task_name'], train_config['evaluate_ldm_result_name'])
        if step < 10:
            print(f"[Step {step}] → running SSIM: {np.mean(all_ssim):.4f}, PSNR: {np.mean(all_psnr):.2f}")
            save_slice(gt_image, pred_image, step, save_dir)
        elif step % 20 == 0:
            print(f"[Step {step}] → running SSIM: {np.mean(all_ssim):.4f}, PSNR: {np.mean(all_psnr):.2f}")
            save_slice(gt_image, pred_image, step, save_dir)

    print(
        'Tested on Test Set - Average SSIM: {:.4f} |  Average PSNR: {:.2f}'.
        format(np.mean(all_ssim),
               np.mean(all_psnr)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm evaluation')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    evaluate(args)
