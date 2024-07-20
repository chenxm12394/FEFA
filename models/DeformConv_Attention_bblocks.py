import math
from models.basicblock import BasicLayer, PatchExpandSkip, PatchMerging, SpectralBlock
from models.humus_block import (
    ConvBlock,
    DownsampConvBlock,
    TransposeConvBlock,
)
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import fastmri
import sys

from models.utils.align_blocks_MultiScale2 import alignment, multiModalFusion


def calc_freq(feature_map):
    def fourier(x):  # 2D Fourier transform
        f = torch.fft.fft2(x)
        f = f.abs() + 1e-6
        f = f.log()
        return f

    def shift(x):  # shift Fourier transformed feature map
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    latent = feature_map.cpu()

    if len(latent.shape) == 3:  # for ViT
        b, n, c = latent.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
    elif len(latent.shape) == 4:  # for CNN
        b, c, h, w = latent.shape
    else:
        raise Exception("shape: %s" % str(latent.shape))
    latent = fourier(latent)
    latent = shift(latent).mean(dim=(0, 1))
    # only use the half-diagonal components
    latent = latent.diag()[int(h / 2) :]
    latent = latent  # visualize 'relative' log amplitudes
    # (i.e., low-freq amp - high freq amp
    return latent


class tokenRSTBEncoder(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
    ):
        super(tokenRSTBEncoder, self).__init__()
        self.dim = dim
        conv_dim = dim // (patch_size**2)
        divide_out_ch = 1
        self.input_resolution = input_resolution
        self.num_windows = int(input_resolution[0] // window_size) ** 2
        self.num_heads = num_heads

        # basic layers
        self.residual_group_x = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.residual_group_c = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        # self.refine = SpectralBlock(dim, self.input_resolution, mlp_ratio=mlp_ratio)

        self.conv_x = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)
        self.conv_c = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)
        # self.propmt_fuse_c = nn.Conv2d(dim * 5 // 4, dim * 5 // 4, 1)
        # self.propmt_fuse = nn.Linear(dim * 5 // 4, dim)
        # self.last = nn.Linear(2*dim, dim)

        self.reshape_x = PatchMerging(input_resolution, dim)
        self.reshape_c = PatchMerging(input_resolution, dim)
        # self.align = alignment(dim)

    def forward(self, x, size, c):
        c = self.conv_c(self.residual_group_c(c, size)) + c
        x = self.conv_x(self.residual_group_x(x, size)) + x
        # x = self.refine(x)
        # x = self.last(torch.cat([x, c], dim=-1))
        # cross_x = self.cross(x, c, size)
        # skip_x = self.align(self.patch_unembed(x, size), self.patch_unembed(c, size))
        # skip_x = x
        x = (
            self.reshape_x(x),
            x,
        )  # return skip connection
        c = (
            self.reshape_c(c),
            c,
        )  # return skip connection

        # return x, c, attns
        return x, c


class tokenRSTBBottleNeck(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
    ):
        super(tokenRSTBBottleNeck, self).__init__()
        self.dim = dim
        conv_dim = dim // (patch_size**2)
        divide_out_ch = 1
        self.input_resolution = input_resolution
        self.num_windows = int(input_resolution[0] // window_size) ** 2
        self.num_heads = num_heads

        # basic layers
        # basic layers
        self.residual_group_x = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.residual_group_c = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        # self.refine = SpectralBlock(dim, self.input_resolution, mlp_ratio=mlp_ratio)

        self.conv_x = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)
        self.conv_c = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)

        self.align = multiModalFusion(dim)
        # self.align = ref_back_projection_concat(dim, 1)

        # self.reshape_x = PatchMerging(input_resolution, dim)
        # self.reshape_c = PatchMerging(input_resolution, dim)

    def forward(self, x, size, c):
        c = self.conv_c(self.residual_group_c(c, size)) + c
        # token_emb = self.cross_att_c(token, self.patch_embed(out_c)) + token
        x = self.conv_x(self.residual_group_x(x, size)) + x
        x = self.align(x, c)
        # x = self.refine(x)
        # offset_feat = self.offset_up(offset_feat)
        # x = self.last(torch.cat([self.patch_unembed(x, size), self.patch_unembed(c, size)], dim=1))

        return x, c


class RSTBDecoder(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        prev=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
    ):
        super(RSTBDecoder, self).__init__()
        self.dim = dim
        conv_dim = dim // (patch_size**2)
        divide_out_ch = 1
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.refine = SpectralBlock(dim, self.input_resolution, mlp_ratio=mlp_ratio)
        # self.refine = SpectralGatingNetwork2(hidden_size=dim, num_blocks=dim)
        self.conv = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)

        self.align = alignment(
            dim=dim, prev=prev, 
        )

        self.reshape = PatchExpandSkip([res // 2 for res in input_resolution], dim * 2)
        # self.offset_up =nn.Sequential(
        #     nn.Conv2d(dim*2, dim, 1),
        #     nn.ConvTranspose2d(dim, dim, 3, stride=2, padding=1, output_padding=1),
        # )
        # self.reshape = PatchExpand([res // 2 for res in input_resolution],
        #                            dim * 2)
        # self.cross_attention = CrossAttention(dim, num_heads)

    def forward(
        self, x, x_size, skip_x, c, prev_offset_feat=None, prev_aligned_feat=None
    ):
        # offset_feat = self.offset_up(offset_feat) * 2
        skip_x, offset_feat, aligned_feat = self.align(
            skip_x, c, prev_offset_feat, prev_aligned_feat
        )

        x = self.reshape(x, skip_x)
        x = self.conv(self.residual_group(x, x_size)) + x
        x = self.refine(x)

        return x, offset_feat, aligned_feat


class HUMUSBlock(nn.Module):
    r"""HUMUS-Block
    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of each Swin Transformer layer in encoder and decoder paths.
        num_heads (tuple(int)): Number of attention heads in different layers of encoder and decoder.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        conv_downsample_first: use convolutional downsampling before MUST to reduce compute load on Transformers
    """

    def __init__(
        self,
        img_size,
        in_chans,
        patch_size=1,
        embed_dim=48,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        img_range=1.0,
        bottleneck_depth=2,
        bottleneck_heads=24,
        out_chans=None,
        no_residual_learning=False,
        flow_skip=False,
        **kwargs
    ):
        super(HUMUSBlock, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans if out_chans is None else out_chans

        self.center_slice_out = out_chans == 1

        self.img_range = img_range
        self.window_size = window_size
        self.no_residual_learning = no_residual_learning
        self.flow_skip = flow_skip

        #####################################################################################################
        ################################### 1, input block ###################################
        input_conv_dim = embed_dim
        self.conv_first_x = nn.Conv2d(2, input_conv_dim // 2, 3, 1, 1)
        self.conv_first_c = nn.Conv2d(1, input_conv_dim // 2, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**self.num_layers)
        self.mlp_ratio = mlp_ratio
        self.prev = [True, True, False]
        self.prev_last = True


        # Downsample for low-res feature extraction



        img_size = [im // 2 for im in img_size]
        self.conv_down_block_x = nn.Sequential(
            ConvBlock(input_conv_dim // 2, input_conv_dim, 0.0),
            DownsampConvBlock(input_conv_dim, input_conv_dim),
            # DownsampConvBlock(input_conv_dim, input_conv_dim),
        )
        # DownsampConvBlock(input_conv_dim, input_conv_dim))
        self.conv_down_block_c = nn.Sequential(
            ConvBlock(input_conv_dim // 2, input_conv_dim, 0.0),
            DownsampConvBlock(input_conv_dim, input_conv_dim),
            # DownsampConvBlock(input_conv_dim, input_conv_dim),
        )
        # DownsampConvBlock(input_conv_dim, input_conv_dim))

        # split image into non-overlapping patches
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # Build MUST
        # encoder
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim_scaler = 2**i_layer
            layer = tokenRSTBEncoder(
                dim=int(embed_dim * dim_scaler),
                input_resolution=(
                    img_size[0] // dim_scaler,
                    img_size[1] // dim_scaler,
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
                img_size=[im // dim_scaler for im in img_size],
                patch_size=1,
            )
            self.layers_down.append(layer)
        # self.norm_down_x = norm_layer()
        # self.norm_down_c = norm_layer()
        # bottleneck
        dim_scaler = 2**self.num_layers

        self.layer_bottleneck = tokenRSTBBottleNeck(
            dim=int(embed_dim * dim_scaler),
            input_resolution=(
                img_size[0] // dim_scaler,
                img_size[1] // dim_scaler,
            ),
            depth=bottleneck_depth,
            num_heads=bottleneck_heads,
            window_size=8,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=[im // dim_scaler for im in img_size],
            patch_size=1,
        )

        # decoder
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim_scaler = 2 ** (self.num_layers - i_layer - 1)
            layer = RSTBDecoder(
                dim=int(embed_dim * dim_scaler),
                input_resolution=(
                    img_size[0] // dim_scaler,
                    img_size[1] // dim_scaler,
                ),
                depth=depths[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                prev=self.prev[(self.num_layers - 1 - i_layer)],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[: (self.num_layers - 1 - i_layer)]) : sum(
                        depths[: (self.num_layers - 1 - i_layer) + 1]
                    )
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
                img_size=[im // dim_scaler for im in img_size],
                patch_size=1,
            )
            self.layers_up.append(layer)

        self.conv_after_body = nn.Conv2d(input_conv_dim, input_conv_dim, 3, 1, 1)
        self.align = alignment(
            dim=input_conv_dim // 2,
            prev=self.prev_last,
            groups=6,

        )

        self.conv_up = nn.Sequential(
            TransposeConvBlock(input_conv_dim, input_conv_dim // 2),
        )

        self.conv_up_block = ConvBlock(input_conv_dim, input_conv_dim // 2, 0.0)

        #####################################################################################################
        ################################ 3, output block ################################
        self.conv_last_complex = nn.Conv2d(input_conv_dim // 2, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()

        # divisible by window size
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        # divisible by total downsampling ratio
        # this could be done more efficiently by combining the two
        total_downsamp = int(2 ** (self.num_layers - 1))
        pad_h = (total_downsamp - h % total_downsamp) % total_downsamp
        pad_w = (total_downsamp - w % total_downsamp) % total_downsamp
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward_features(self, x, c):
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(x)
        c = self.pos_drop(c)

        # encode
        skip_cons_x = []
        skip_cons_c = []
        # attns = {}
        for i, layer in enumerate(self.layers_down):
            (x, skip_x), (c, skip_c) = layer(x, layer.input_resolution, c)
            skip_cons_x.append(skip_x)
            skip_cons_c.append(skip_c)
            # latents[i] = self.calc_freq(x)
            # attns[i] = attn

        x, c = self.layer_bottleneck(x, self.layer_bottleneck.input_resolution, c)
        # ref_skips.append(c)
        # skip_cons_c.append(tokens)
        # latents[len(self.layers_down)] = self.calc_freq(x)
        # attns[len(self.layers_down)] = attn_c
        # attns[len(self.layers_down)+1] = attn_x
        # freqs = {}
        # offset_feats = []
        # prev_flows = []
        offset_feat = None
        prev_aligned_feat=c
        # decode
        for i, layer in enumerate(self.layers_up):
            x, offset_feat, prev_aligned_feat = layer(
                x,
                x_size=layer.input_resolution,
                skip_x=skip_cons_x[-i - 1],
                c=skip_cons_c[-i - 1],
                prev_offset_feat=offset_feat,
                prev_aligned_feat=prev_aligned_feat
            )
            # prev_flows.append(flow)
            # ref_skips.append(aligned_ref)

        return x, offset_feat, prev_aligned_feat

    def forward(self, x, c):
        C, H, W = x.shape[1:]
            

        # x_first = self.conv_first_x(torch.cat([x, c], dim=1))
        x_first = self.conv_first_x(x)
        c_first = self.conv_first_c(c)

        x_down = self.conv_down_block_x(x_first)
        c_down = self.conv_down_block_c(c_first)

        res, offset_feat, prev_aligned_feat = self.forward_features(
            x_down, c_down
        )
        after_freq = res
        res = self.conv_after_body(res)
        res = self.conv_up(res)
        res = torch.cat([res, x_first], dim=1)
        res = self.conv_up_block(res)
        # res = self.refine(res, (H, W))
        # res = self.mrf(res, c_first)
        res, offset_feat, aligned_feat = self.align(res, c_first, offset_feat, prev_aligned_feat)
        res = self.conv_last_complex(res)
        # ref_skips.append(aligned_ref)
        # for offset_feat in offset_skips_new:
        #     print(offset_feat.shape)

        if self.no_residual_learning:
            x = res
        else:
            x = x + res
        return x


if __name__ == "__main__":
    model = HUMUSBlock((320, 320), 2).to("cuda:7")
    img = torch.randn((2, 2, 320, 320)).to("cuda:7")
    total = sum(
        [param.nelement() for param in model.parameters() if param.requires_grad]
    )
    print("Number of parameter: %.2fM" % (total / 1e6))
    c = torch.randn((2, 1, 320, 320)).to("cuda:7")
    x, latents = model(img, c)
    print(x.shape)
    print(latents)
