import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):

    assert t_emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=t_emb_dim//2, dtype=torch.float32, device = time_steps.device) / (t_emb_dim//2)
                        ))

    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    # 1D tensor(B,) -> column vector(B, 1) -> (B, t_emb_dim // 2) -> (B, t_emb_dim)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv3d(in_channels if i==0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]) # (B, t_emb_dim) -> (B, out_channels) (out_channels == feature_map_channels)
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
            )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv3d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv3d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x

        for i in range(self.num_layers):

            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i](out) # (B, out_channels, H, W)
            # [:, :, None, None]: keep two dimension + add two dimension (default value: 1)
            # (B, t_emb_dim) -> (B, out_channels) -> (B, out_channels, 1, 1)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None] # time embedding (broad casting)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input) # skip connection

            if self.attn:
                # Attention block
                batch_size, channels, d, h, w = out.shape
                # self-attention -> tokenize each pixel => flattened spatial dimension (B, C, seq_len)
                in_attn = out.reshape(batch_size, channels, d*h*w)
                in_attn = self.attention_norms[i](in_attn) # normalize before softmax of attention
                # self.attention(q, k, v) input shape: (B, seq_len, emb_dim) -> change 2nd, 3rd dimension
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn) # query, key, value (self-attention -> all in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w) # reconstruct to original shape
                out = out + out_attn # skip connection

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, attn, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv3d(in_channels if i==0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers+1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers+1)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv3d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x
        # first resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):

            if self.attn:
                # attention block
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d*h*w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

            # second resnet block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv3d(in_channels if i==0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4,
                                                 stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)

        # skip connection (concat symmetric encoder result)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i](out)  # (B, out_channels, H, W)
            # [:, :, None, None]: keep two dimension + add two dimension (default value: 1)
            # (B, t_emb_dim) -> (B, out_channels) -> (B, out_channels, 1, 1)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]  # time embedding (broad casting)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)  # skip connection

            # Attention block
            if self.attn:
                batch_size, channels, d, h, w = out.shape
                # self-attention -> tokenize each pixel => flattened spatial dimension (B, C, seq_len)
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.attention_norms[i](in_attn)
                # self.attention(q, k, v) input shape: (B, seq_len, emb_dim) -> change 2nd, 3rd dimension
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)  # query, key, value (self-attention -> all in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)  # reconstruct to original shape
                out = out + out_attn  # skip connection

        return out


class DownBlockUnet(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv3d(in_channels if i==0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]) # (B, t_emb_dim) -> (B, out_channels) (out_channels == feature_map_channels)
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
            )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv3d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv3d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x

        for i in range(self.num_layers):

            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i](out) # (B, out_channels, H, W)
            # [:, :, None, None]: keep two dimension + add two dimension (default value: 1)
            # (B, t_emb_dim) -> (B, out_channels) -> (B, out_channels, 1, 1)
            if self.t_emb_dim is not None:
                emb = self.t_emb_layers[i](t_emb)  # (B, C)
                emb = emb[:, :, None, None, None]  # (B, C, 1, 1, 1)
                out = out + emb
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input) # skip connection

            if self.attn:
                # Attention block
                batch_size, channels, d, h, w = out.shape
                # self-attention -> tokenize each pixel => flattened spatial dimension (B, C, seq_len)
                in_attn = out.reshape(batch_size, channels, d*h*w)
                in_attn = self.attention_norms[i](in_attn) # normalize before softmax of attention
                # self.attention(q, k, v) input shape: (B, seq_len, emb_dim) -> change 2nd, 3rd dimension
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn) # query, key, value (self-attention -> all in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w) # reconstruct to original shape
                out = out + out_attn # skip connection

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                if context_proj.dim() == 2:
                    context_proj = context_proj.unsqueeze(1)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out

class MidBlockUnet(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, attn, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv3d(in_channels if i==0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers+1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers+1)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv3d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x
        # first resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            emb = self.t_emb_layers[0](t_emb)  # (B, C)
            emb = emb[:, :, None, None, None]  # (B, C, 1, 1, 1)
            out = out + emb
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            if self.attn:
                # attention block
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d*h*w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                if context_proj.dim() == 2:
                    context_proj = context_proj.unsqueeze(1)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)
                out = out + out_attn

            # second resnet block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            if self.t_emb_dim is not None:
                emb = self.t_emb_layers[0](t_emb)  # (B, C)
                emb = emb[:, :, None, None, None]  # (B, C, 1, 1, 1)
                out = out + emb
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)

        return out


class UpBlockUnet(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.up = nn.ConvTranspose3d(in_channels, in_channels,
                                     kernel_size=4, stride=2, padding=1)

        merged_channels = in_channels + skip_channels

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, merged_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv3d(merged_channels if i == 0 else out_channels,
                          out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers)
        ])

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList([
            nn.Conv3d(merged_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size = 1)
            for i in range(num_layers)]
            )
        self.up_sample_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4,
                                                 stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)

        # skip connection (concat symmetric encoder result)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i](out)  # (B, out_channels, H, W)
            # [:, :, None, None]: keep two dimension + add two dimension (default value: 1)
            # (B, t_emb_dim) -> (B, out_channels) -> (B, out_channels, 1, 1)
            if self.t_emb_dim is not None:
                emb = self.t_emb_layers[i](t_emb)  # (B, C)
                emb = emb[:, :, None, None, None]  # (B, C, 1, 1, 1)
                out = out + emb
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)  # skip connection

            # Attention block
            if self.attn:
                batch_size, channels, d, h, w = out.shape
                # self-attention -> tokenize each pixel => flattened spatial dimension (B, C, seq_len)
                in_attn = out.reshape(batch_size, channels, d * h * w)
                in_attn = self.attention_norms[i](in_attn)
                # self.attention(q, k, v) input shape: (B, seq_len, emb_dim) -> change 2nd, 3rd dimension
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)  # query, key, value (self-attention -> all in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, d, h, w)  # reconstruct to original shape
                out = out + out_attn  # skip connection

        return out