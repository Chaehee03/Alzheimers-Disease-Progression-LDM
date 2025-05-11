import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
import pandas as pd
import tqdm

# from models.lpips import LPIPS3D
from models.vae import VAE
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from monai.data.image_reader import NumpyReader
from utils.data import get_dataset_from_pd
from models import const
# from monai.transforms import LoadImageD

# class DebugLoadImageD(LoadImageD):
#     def __call__(self, data):
#         try:
#             return super().__call__(data)
#         except Exception as e:
#             print("\n" + "="*60)
#             print(f"[❌ LoadImageD ERROR] Failed to load image from: {data.get('image_path', 'MISSING')}")
#             print(f"[❌ LoadImageD ERROR] Exception: {repr(e)}")
#             print("="*60 + "\n")
#             raise e  # 원래 에러를 다시 raise 해서 trace 유지



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("torch.cuda.is_available() = ", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
scaler_g = GradScaler()
scaler_d = GradScaler()


def kl_divergence(mean, logvar):
    return 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1)


def train(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    vae_config = config['vae_params']
    train_config = config['train_params']

    # Set the desired seed value
    seed = train_config['seed']
    torch.manual_seed(seed) # Pytorch
    np.random.seed(seed) # Numpy
    random.seed(seed) # Python
    if device == 'cuda': # Cuda
        torch.cuda.manual_seed_all(seed) # for cases using several GPUs

    # Create the model
    model = VAE(im_channels=dataset_config['im_channels'],
                model_config=vae_config).to(device)

    ###########################################
    # transforms_fn = transforms.Compose([
    #     transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    #     DebugLoadImageD(image_only=True, keys=['image']),  # ← 여기만 바꾸면 됨
    #     transforms.EnsureChannelFirstD(keys=['image']),
    #     transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    #     transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    #     transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    # ])

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

    # # 1. NaN 체크
    # nan_rows = dataset_df[dataset_df["image_path"].isna()]
    # if not nan_rows.empty:
    #     print(f"[🚨 NaN] {len(nan_rows)} rows have NaN as image_path:")
    #     print(nan_rows)
    #
    # # 2. 비문자열 (float 등) 확인
    # non_str_rows = dataset_df[~dataset_df["image_path"].apply(lambda x: isinstance(x, str))]
    # if not non_str_rows.empty:
    #     print(f"[🚨 Non-string] {len(non_str_rows)} rows have non-string image_path:")
    #     print(non_str_rows)
    #
    # # 3. 파일 존재 여부 확인 (문자열인 경우만)
    # valid_df = dataset_df[dataset_df["image_path"].apply(lambda x: isinstance(x, str))]
    # nonexistent_rows = valid_df[~valid_df["image_path"].apply(os.path.exists)]
    # if not nonexistent_rows.empty:
    #     print(f"[🚨 Not found] {len(nonexistent_rows)} paths do not exist:")
    #     print(nonexistent_rows["image_path"].tolist())

    train_df = dataset_df[dataset_df.split == 'train']
    trainset = get_dataset_from_pd(train_df, transforms_fn, train_config['cache_dir'])

    # train_loader = DataLoader(dataset=trainset,
    #                           num_workers=train_config['num_workers'],
    #                           batch_size=train_config['vae_batch_size'],
    #                           shuffle=True,
    #                           pin_memory=True)

    train_loader = DataLoader(dataset=trainset,
                              num_workers=train_config['num_workers'],
                              batch_size=train_config['vae_batch_size'],
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)
    ###########################################

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])

    num_epochs = train_config['vae_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Discriminator Loss (The Least Squares GAN)
    disc_criterion = torch.nn.MSELoss()

    # Instantiate LPIPS and Disc
    # freezing part is in lpips.py -> don't need to freeze lpips
    # lpips_model = LPIPS3D().eval().to(device)
    discriminator = Discriminator(in_channels=dataset_config['im_channels']).to(device)

    optimizer_g = Adam(model.parameters(), lr=train_config['vae_lr'], betas = (0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr = train_config['vae_lr'], betas = (0.5, 0.999))

    # step point when discriminator kicks in. (starts to train generating fake images)
    disc_step_start = len(train_loader)

    # Gradient Accumulation
    # Useful when dealing with high-resolution images
    # And when a large batch size is unavailable.
    acc_steps = train_config['vae_acc_steps']
    image_save_steps = train_config['vae_image_save_steps']
    image_save_count = 0

    start_epoch = 0
    step_count = 0

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vae_autoencoder_ckpt_name'])):
        print('Loaded vae autoencoder checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vae_autoencoder_ckpt_name']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint['step_count']

    if os.path.exists(os.path.join(train_config['task_name'],
                                                train_config['vae_discriminator_ckpt_name'])):
        print('Loaded vae discriminator checkpoint')
        checkpoint = torch.load(os.path.join(train_config['task_name'],
                                                  train_config['vae_discriminator_ckpt_name']))
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint['step_count']

    max_kl = train_config['kl_weight']  # 예: 1e-4
    initial_kl = train_config.get('kl_start', 1e-6)
    kl_anneal_epochs = train_config.get('kl_anneal_epochs', 10)

    pretrain_epochs = train_config.get('pretrain_epochs', 5)
    disc_schedule = [
        (pretrain_epochs, 0.1),  # epoch ≥ 5 → disc_weight = 0.1
        (pretrain_epochs + 5, 0.3),  # epoch ≥ 10 → disc_weight = 0.3
        (pretrain_epochs + 10, 0.5)  # epoch ≥ 15 → disc_weight = 0.5 (최대)
    ]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        recon_losses = []
        kl_losses = []
        # perceptual_losses = [] # LPIPS
        adversarial_losses = []
        disc_losses = []
        gen_losses = []

        # Initialize gradients to zeros
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        if epoch < kl_anneal_epochs:
            # 선형 증가: epoch=0→initial, epoch=kl_anneal_epochs→max_kl
            kl_weight = initial_kl + (max_kl - initial_kl) * (epoch / kl_anneal_epochs)
        else:
            kl_weight = max_kl

        # 1) Pretrain 기간 (=VAE만 학습): GAN weight=0
        if epoch < pretrain_epochs:
            disc_weight = 0.0
        else:
            # 2) 스케줄 테이블을 순회하면서, 해당 epoch 에 맞는 값 선택
            disc_weight = 0.0
            for e_start, w in disc_schedule:
                if epoch >= e_start:
                    disc_weight = w


        for sample in tqdm.tqdm(train_loader, desc="Training"):
            step_count += 1
            im = sample['image'].float().to(device)

            with autocast():
                # Fetch VAE output(reconstructed images)
                model_output = model(im)
                out, mean, logvar = model_output

                # Image Saving Logic #
                # print [Original Input - Model Output] pair
                if step_count % image_save_steps == 0 or step_count == 1: # save initially & per image_save_steps.
                    sample_size = min(8, im.shape[0]) # save max 8 samples
                    # model output image normalization for saving
                    # save_output = torch.clamp(out[:sample_size], -1, 1).detach().cpu()
                    # save_output = (save_output + 1) / 2 # [-1, 1] -> [0, 1]
                    # # original input image normalization for saving
                    # save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                    save_output = torch.clamp(out[:sample_size], 0, 1).detach().cpu()
                    save_input = torch.clamp(im[:sample_size], 0, 1).detach().cpu()

                    B, C, D, H, W = save_input.shape

                    # 1) 중앙 슬라이스 인덱스
                    mid = D // 2

                    # 2) 2D 슬라이스 선택 → shape: (B, C, H, W)
                    input_slice = save_input[:, :, mid, :, :]
                    output_slice = save_output[:, :, mid, :, :]

                    # 3) 배치로 합치고 그리기
                    # (B*2, C, H, W) 이 되고, nrow=sample_size 로 갤러리 생성
                    grid = make_grid(torch.cat([input_slice, output_slice], dim=0),
                                     nrow=sample_size)

                    # PIL 변환, 저장
                    img = torchvision.transforms.ToPILImage()(grid)

                    # create output directory
                    if not os.path.exists(os.path.join(train_config['task_name'], 'vae_samples')):
                        os.mkdir(os.path.join(train_config['task_name'], 'vae_samples'))
                    img.save(os.path.join(train_config['task_name'], 'vae_samples',
                                          'current_vae_sample_{}.png'.format(image_save_count)))
                    image_save_count += 1
                    img.close()

                ############## Optimize Generator ############
                # L2 Loss
                recon_loss = recon_criterion(out, im)
                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / acc_steps # divide loss by acc_steps before backprop -> effect of calculating gradients partially

                # KL divergence loss(VAE)
                kl_loss = kl_divergence(mean, logvar)/ acc_steps

                g_loss = recon_loss + kl_weight * kl_loss
                kl_losses.append(kl_weight * kl_loss.item())

                # Adversarial loss term is added after step count passed disc_step_start.
                if step_count >= disc_step_start:
                    disc_fake_pred = discriminator(model_output[0])
                    # Generators takes only a batch size of fake samples for training.
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.ones(disc_fake_pred.shape,
                                                               device = disc_fake_pred.device)) # ground truth(G): fake -> 1
                    adversarial_losses.append(disc_weight * disc_fake_loss.item())
                    g_loss += disc_weight * disc_fake_loss / acc_steps

                # LPIPS
                # for p in lpips_model.parameters():
                #     p.requires_grad = False
                #
                # with torch.no_grad():
                #     lpips_loss = torch.mean(lpips_model(out, im)) # batch-wise mean of (B, 1, 1, 1) tensors.
                # perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
                # g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps

                gen_losses.append(g_loss.item()) # total generator's loss (recon + kl + adversarial + lpips)

            scaler_g.scale(g_loss).backward() # calculate gradient
            ##############################################

            ############ Optimize Discriminator ###########
            # Discriminator starts training after step_count passed disc_step_start.
            if step_count > disc_step_start:
                with autocast():
                    fake = out.detach()
                    disc_fake_pred = discriminator(fake.detach()) # detach fake(generator's gradient) from backpropagation
                    disc_real_pred = discriminator(im)
                    # Discriminator takes both fake and real samples and train classifying them.
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.zeros(disc_fake_pred.shape, # ground truth(D): fake -> 0
                                                                device = disc_fake_pred.device))
                    disc_real_loss = disc_criterion(disc_real_pred,
                                                    torch.ones(disc_real_pred.shape, # ground truth(D): real -> 1
                                                               device = disc_real_pred.device))
                    # Total loss = Adding BCE of y=0 (ground truth: fake) & y=1 (ground truth: real)
                    disc_loss = disc_weight * (disc_fake_loss + disc_real_loss) / 2
                    disc_losses.append(disc_loss.item())
                    disc_loss = disc_loss / acc_steps

                scaler_d.scale(disc_loss).backward() # backprop

                # After accumulating for acc_steps, update discriminator's params once.
                if step_count % acc_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad() # Initialize gradient to zero before next batch iteration.
            ################################################

            # After accumulating for acc_steps, update generator's params once.
            if step_count % acc_steps == 0:
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad()

        # After entire dataset training
        optimizer_d.step() # Update D's params
        optimizer_d.zero_grad() # Initialize before next epoch
        optimizer_g.step() # Update G's params
        optimizer_g.zero_grad() # Initialize before next epoch

        # print results per 1 epoch
        # disc_losses can be empty if D kicks in after training G for entire dataset.
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | '
                'KL Divergence Loss : {:.4f} | Adversarial Loss: {:.4f} '
                '| G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch + 1,
                       np.mean(recon_losses),
                       np.mean(kl_losses),
                       np.mean(adversarial_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} |  '
                  'KL Divergence Loss : {:.4f} | G Loss : {:.4f}'.
                  format(epoch + 1,
                         np.mean(recon_losses),
                         np.mean(kl_losses),
                         np.mean(gen_losses)))

        torch.save({
                'epoch': epoch,
                'step_count': step_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
        }, os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name']))

        torch.save({
                'epoch': epoch,
                'step_count': step_count,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_d.state_dict(),
        }, os.path.join(train_config['task_name'], train_config['vae_discriminator_ckpt_name']))

    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()
    train(args)