import torch
from tqdm import tqdm
from models import const
from monai import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

def resize_batch_tensor(x, spatial_size):
    B, C = x.shape[:2]
    resized_slices = []
    resize = transforms.Resize(spatial_size)  # 3D 크기 지정 (D,H,W)
    for b in range(B):
        for c in range(C):
            resized = resize(x[b, c].unsqueeze(0))  # (1, D, H, W)
            resized_slices.append(resized.squeeze(0))
    return torch.stack(resized_slices).view(B, C, *spatial_size)


# Sample stepwise by going backward one timestep at a time.
# Save the x0 predictions
@torch.no_grad()
def sample(unet, vqvae, scheduler, context, diffusion_config, scale_factor, device='cuda') -> torch.Tensor:
    unet.eval()
    vqvae.eval()

    # drawing a random z_T ~ N(0,I)
    xt = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)

    for t in tqdm(reversed(range(diffusion_config['num_timesteps']))):

        n = xt.size(0)
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)

        # Get LDM prediction of noise
        noise_pred = unet(
            xt,
            t_tensor,
            {'context': context})

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if t == 0:
            # Decode ONLY the final image to save time
            ims = vqvae.to(device).decode(xt / scale_factor)
            print("ims.shape:", ims.shape)
            print("decode raw min/max:", ims.min().item(), ims.max().item())
            # ims = transforms.Spacing(pixdim=const.RESOLUTION)(ims)
            # ims= resize_batch_tensor(ims, const.INPUT_SHAPE_AE)
            # # ims = transforms.ResizeWithPadOrCrop(spatial_size=(1, 120, 144, 120))(ims)
            # print("decode raw min/max after processing:", ims.min().item(), ims.max().item())
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

    return ims



