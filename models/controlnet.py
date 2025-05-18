import torch
import torch.nn as nn
from models.unet_cond import UNet
from models.blocks import get_time_embedding


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):

    def __init__(self, im_channels,
                 model_config,
                 model_locked=True,
                 model_ckpt=None,
                 device=None):
        super().__init__()
        # Trained DDPM
        self.model_locked = model_locked
        self.trained_unet = UNet(im_channels, model_config)

        # Load weights for the trained model
        if model_ckpt is not None and device is not None:
            print('Loading Trained Diffusion Model')
            ckpt = torch.load(model_ckpt, map_location=device)
            state_dict = ckpt['model_state_dict']
            self.trained_unet.load_state_dict(state_dict, strict=True)

        # ControlNet Copy of Trained DDPM
        # use_up = False removes the upblocks(decoder layers) from DDPM Unet
        self.control_unet = UNet(im_channels, model_config, use_up=False)
        # Load same weights as the trained model
        if model_ckpt is not None and device is not None:
            print('Loading ControlNet Diffusion Model')
            ckpt = torch.load(model_ckpt, map_location=device)
            state_dict = ckpt['model_state_dict']
            self.control_unet.load_state_dict(state_dict, strict=False)

        ######### Hint Block for ControlNet ##########
        hint = model_config['hint_channels']
        target = self.trained_unet.down_channels[0]
        self.control_unet_hint_block = nn.Sequential(
            make_zero_module(nn.Conv3d(hint, target, kernel_size=1, padding=0))
        )
        #########################################################

        # Zero Convolution Module for Downblocks(encoder Layers)
        self.control_unet_down_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv3d(self.trained_unet.down_channels[i],
                                       self.trained_unet.down_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(len(self.trained_unet.down_channels)-1)
        ])

        # Zero Convolution Module for MidBlocks
        self.control_unet_mid_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv3d(self.trained_unet.mid_channels[i],
                                       self.trained_unet.mid_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(1, len(self.trained_unet.mid_channels))
        ])

        for p in self.trained_unet.parameters():
            p.requires_grad_(False)

    def get_params(self):
        # Add all ControlNet parameters
        # First is our copy of unet
        params = list(self.control_unet.parameters())

        # Add parameters of hint Blocks & Zero convolutions for down/mid blocks
        params += list(self.control_unet_hint_block.parameters())
        params += list(self.control_unet_down_zero_convs.parameters())
        params += list(self.control_unet_mid_zero_convs.parameters())

        # If we desire to not have the decoder layers locked, then add
        # them as well
        if not self.model_locked:
            params += list(self.trained_unet.ups.parameters())
            params += list(self.trained_unet.norm_out.parameters())
            params += list(self.trained_unet.conv_out.parameters())
        return params

    def forward(self, x, t, context, hint):
        # Time embedding and timestep projection layers of trained unet
        trained_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.trained_unet.t_emb_dim)
        trained_unet_t_emb = self.trained_unet.t_proj(trained_unet_t_emb)

        # Get all downblocks output of trained unet
        trained_unet_down_outs = []
        if context.dim() == 2:
            context = context.unsqueeze(1)
        with torch.no_grad():
            train_unet_out = self.trained_unet.conv_in(x)
            for idx, down in enumerate(self.trained_unet.downs):
                trained_unet_down_outs.append(train_unet_out)
                train_unet_out = down(train_unet_out, trained_unet_t_emb, context)

        ############# ControlNet Layers #############
        # Time embedding and timestep projection layers of controlnet copy of unet
        control_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.control_unet.t_emb_dim)
        control_unet_t_emb = self.control_unet.t_proj(control_unet_t_emb)

        if hint.dtype != x.dtype:
            hint = hint.to(dtype=x.dtype)

        # Hint block of controlnet copy of unet
        control_unet_hint_out = self.control_unet_hint_block(hint)

        # Call conv_in layer for controlnet copy of unet
        # and add hint blocks output to it
        control_unet_out = self.control_unet.conv_in(x)
        control_unet_out += control_unet_hint_out

        # Get all downblocks output for controlnet copy
        control_unet_down_outs = []
        for idx, down in enumerate(self.control_unet.downs):
            # Controlnet copy output -> pass zero conv layers -> save it in list
            control_unet_down_outs.append(self.control_unet_down_zero_convs[idx](control_unet_out))
            control_unet_out = down(control_unet_out, control_unet_t_emb, context)

        for idx in range(len(self.control_unet.mids)):
            # Get midblock output of controlnet copy
            control_unet_out = self.control_unet.mids[idx](control_unet_out, control_unet_t_emb, context)

            # Get midblock output of trained unet
            train_unet_out = self.trained_unet.mids[idx](train_unet_out, trained_unet_t_emb, context)

            # Controlnet copy midblock output -> pass zero conv layers
            # -> added to trained unet midblock output
            train_unet_out += self.control_unet_mid_zero_convs[idx](control_unet_out)

        # Upblocks of trained unet
        for up in self.trained_unet.ups:
            # Get downblock outputs from trained unet and controlnet copy
            trained_unet_down_out = trained_unet_down_outs.pop()
            control_unet_down_out = control_unet_down_outs.pop()

            # Controlnet output + Trained Unet (encoder) output
            # -> skip connection (concat to trained unet decoder output)
            train_unet_out = up(train_unet_out,
                                control_unet_down_out + trained_unet_down_out,
                                trained_unet_t_emb)

        # Output layers of trained unet
        train_unet_out = self.trained_unet.norm_out(train_unet_out)
        train_unet_out = nn.SiLU()(train_unet_out)
        train_unet_out = self.trained_unet.conv_out(train_unet_out)
        # out B x C x H x W
        return train_unet_out






