import torch
from tqdm import tqdm
from models.controlnet import ControlNet
from models.vqvae import VQVAE
from models import const

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

# Sample stepwise by going backward one timestep at a time.
# Save the x0 predictions
@torch.no_grad()
def sample(controlnet, vqvae, scheduler, diffusion_config,
           controlnet_condition, context, device='cuda') -> torch.Tensor:
    controlnet.eval();
    vqvae.eval()

    # drawing a random z_T ~ N(0,I)
    xt = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)

    for t in tqdm(reversed(range(diffusion_config['num_timesteps']))):

        n = xt.size(0)
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)

        # Get Controlnet prediction of noise
        noise_pred = controlnet(
            xt,
            t_tensor,
            context,
            controlnet_condition,)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if t == 0:
            # Decode ONLY the final image to save time
            ims = vqvae.to(device).decode(xt)
            print("decode raw min/max:", ims.min().item(), ims.max().item())
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2

    return ims



