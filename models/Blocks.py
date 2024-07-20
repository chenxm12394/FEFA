import numbers
from einops import rearrange
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class DownsampConvBlock(nn.Module):
    """
    A Downsampling Convolutional Block that consists of one strided convolution
    layer followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=2,
                      stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H/2, W/2)`.
        """
        return self.layers(image)


class MRF(nn.Module):
    def __init__(self, nf):
        super(MRF, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_a = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_b = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out


class SobelConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        requires_grad=True,
    ):
        assert kernel_size % 2 == 1, "SobelConv2d's kernel_size must be odd."
        assert (
            out_channels % 4 == 0
        ), "SobelConv2d's out_channels must be a multiple of 4."
        assert (
            out_channels % groups == 0
        ), "SobelConv2d's out_channels must be a multiple of groups."

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(
                torch.zeros(size=(out_channels,), dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(
            torch.zeros(
                size=(out_channels, int(in_channels / groups),
                      kernel_size, kernel_size)
            ),
            requires_grad=False,
        )
        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size -
                                      1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(
                torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.sobel_factor = nn.Parameter(
                torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                requires_grad=False,
            )

    def forward(self, x):
        # if torch.cuda.is_available():
        #     self.sobel_factor = self.sobel_factor.cuda()
        #     if isinstance(self.bias, nn.Parameter):
        #         self.bias = self.bias.cuda()
        self.sobel_factor.to(x.device)
        if isinstance(self.bias, nn.Parameter):
            self.bias.to(x.device)

        sobel_weight = self.sobel_weight * self.sobel_factor
        sobel_weight.to(x.device)

        out = F.conv2d(
            x,
            sobel_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        return out


class CAM_Module(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_1 = nn.Conv2d(self.chanel_in, self.chanel_in, 3, padding=1)
        self.conv_2 = nn.Conv2d(self.chanel_in, self.chanel_in, 3, padding=1)
        self.conv_3 = nn.Conv2d(self.chanel_in, self.chanel_in, 3, padding=1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        proj_query = x1.view(m_batchsize, C, -1)

        proj_key = x2.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        proj_value = x3.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x3
        return out


class SkipMix(nn.Module):
    def __init__(self, n_feats) -> None:
        super().__init__()
        self.CAM_x = CAM_Module(n_feats)
        self.CAM_c = CAM_Module(n_feats)
        self.mix = nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1, bias=False)
        # self.lambda_ = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU(True)

    def forward(self, x, c):
        x = self.CAM_x(x)
        c = self.CAM_c(c)
        skip = self.act(self.mix(torch.cat([x, c], dim=1)))

        return skip


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 num_tokens,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
        #                 num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :,
        #                                  None] - coords_flatten[:,
        #                                                         None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(
        #     1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :,
        #                 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # relative_position_index = relative_position_index.unsqueeze(-1)
        # self.register_buffer("relative_position_index",
        #                      relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index.view(-1)].view(
        #         self.window_size[0] * self.window_size[1],
        #         self.window_size[0] * self.window_size[1],
        #         -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(
        #     2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # # relative_position_bias = relative_position_bias.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 1, self.num_prompts)
        # # relative_position_bias =
        # attn = attn + relative_position_bias.repeat(1, self.num_tokens, self.num_tokens).unsqueeze(0) # 1, nH, Wh*Ww, Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, size):
        super().__init__()

        self.h = size[0]  # H
        self.w = int((size[1] / 2) + 1)  # (W/2)+1, this is due to rfft2
        self.complex_weight = nn.Parameter(
            torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # B, H, W, C=x.shape
        x = x.to(torch.float32)
        # print(x.dtype)
        # Add above for this error, RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
        # print(x.shape, H, W)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # print(x.shape, H, W)
        # print('wno',x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight',weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        # print('wno',x.shape)
        x = x.reshape(B, N, C)  # permute is not same as reshape or view
        return x

class SpectralGatingNetwork2(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, hidden_size, num_blocks=1, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = 0.01
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.act = self.build_act_layer()
        self.act2 = self.build_act_layer()

    @staticmethod
    def build_act_layer() -> nn.Module:
        act_layer = nn.ReLU()
        return act_layer

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, x_size):
        dtype = x.dtype
        x = x.float()
        H, W = x_size
        B, N, C = x.shape
        x = rearrange(x, 'B (H W) C -> B C H W ', H=H, W=W)
        # x = self.fu(x)

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])


        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        # x = x.type(dtype) + bias
        x =rearrange(x, 'B C H W -> B (H W) C')

        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, size):
        super().__init__()

        self.h = size[0]  # H
        self.w = int((size[1] / 2) + 1)  # (W/2)+1, this is due to rfft2
        self.complex_weight = nn.Parameter(
            torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x, x_size=None):
        if len(x.shape) == 3:
            H, W = x_size
            B, N, C = x.shape
            x = x.view(B, H, W, C)
        elif len(x.shape) == 4:
            N = 0
            x = x.permute(0, 2, 3, 1)
        B, H, W, C=x.shape
        x = x.to(torch.float32)
        # print(x.dtype)
        # Add above for this error, RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
        # print(x.shape, H, W)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # print(x.shape, H, W)
        # print('wno',x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight',weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        # print('wno',x.shape)
        if N:
            x = x.reshape(B, N, C)  # permute is not same as reshape or view
        else:
            x = x.permute(0, 3, 1, 2)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SpectralBlock(nn.Module):

    def __init__(self, dim, size, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size=None):
        input_shape = x.shape
        if len(input_shape) == 4:
            B, C, H, W = input_shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x), size))))
        if len(input_shape) == 4:
            x = rearrange(x, 'b (h w) c -> b c h w',h=H,w=W)
        
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous(
        ).view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PromptWindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        num_prompts,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.num_prompts = num_prompts

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1)
                        * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        _C, _H, _W = relative_position_bias.shape

        relative_position_bias = torch.cat(
            (
                torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                relative_position_bias,
            ),
            dim=1,
        )
        relative_position_bias = torch.cat(
            (
                torch.zeros(
                    _C, _H + self.num_prompts, self.num_prompts, device=attn.device
                ),
                relative_position_bias,
            ),
            dim=-1,
        )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            # expand relative_position_bias
            mask = torch.cat(
                (torch.zeros(nW, self.num_prompts, _W, device=attn.device), mask), dim=1
            )
            mask = torch.cat(
                (
                    torch.zeros(
                        nW, _H + self.num_prompts, self.num_prompts, device=attn.device
                    ),
                    mask,
                ),
                dim=-1,
            )
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class Prompt_adjust(nn.Module):
    def __init__(
        self,
        dim,
        num_prompts,
        input_resolution,
        prompt_window_size,
        window_size,
        num_heads,
    ):
        super(Prompt_adjust, self).__init__()
        self.in_chans = dim
        # prompts_per_window
        self.num_prompts = num_prompts
        # image resolution
        self.prompt_window_size = prompt_window_size
        self.prompt_resolution = [
            size // window_size for size in input_resolution]
        self.att = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.prompt_window_size),
            num_heads=num_heads
        )
        self.mlp = Mlp(dim, dim*2)

    def forward(self, prompt):
        """
        prompt: (B, N, C)
        """
        """ Forward pass with input x. """
        B, N, C = prompt.shape
        H, W = self.prompt_resolution[0], self.prompt_resolution[1]
        # print(N, self.prompt_window_size , self.num_prompts)
        assert (
            N == H * W * self.num_prompts
        ), "wrong input size!!"
        prompt = rearrange(prompt, 'B (h ws1 w ws2 np) c -> (B h w) (ws1 ws2 np) c', h=H//self.prompt_window_size,
                           ws1=self.prompt_window_size, ws2=self.prompt_window_size, w=W//self.prompt_window_size, np=self.num_prompts)
        prompt = self.att(prompt) + prompt
        prompt = rearrange(prompt, '(B h w) (ws1 ws2 np) c -> B (h ws1 w ws2 np) c ', h=H//self.prompt_window_size,
                           ws1=self.prompt_window_size, ws2=self.prompt_window_size, w=W//self.prompt_window_size, np=self.num_prompts)
        prompt = self.mlp(prompt) + prompt
        return prompt


class PromptSwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_prompts,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_prompts = num_prompts
        self.total_prompts = self.num_prompts * int(
            (self.input_resolution[0] / self.window_size)
            * (self.input_resolution[1] / self.window_size)
        )
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = PromptWindowAttention(
            num_prompts=num_prompts,
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.mlp = FeedForward(in_features=dim,
        #                        hidden_features=mlp_hidden_dim,
        #                        act_layer=act_layer,
        #                        drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): B N + num_prompts*num_windows C
            prompt (_type_):

        Returns:
            _type_: _description_
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)

        prompt_emb = x[:, : self.total_prompts, :]
        x = x[:, self.total_prompts:, :]
        L = L - self.total_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(
            H * W, L
        )

        # change input size
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C

        # expand prompts_embs
        # B,nw,num_prompts, C --> B*nw, num_prompts, C
        # prompt_emb = prompt_emb.unsqueeze(0)
        # B * num_windows, num_prompts, C
        prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
        x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # seperate prompt embs --> nW*B, num_prompts, C

        # change input size
        prompt_emb = attn_windows[:, : self.num_prompts, :]
        attn_windows = attn_windows[:, self.num_prompts:, :]
        # change prompt_embs's shape:
        # nW*B, num_prompts, C - B, num_prompts, C

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(
                    self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        prompt_emb = prompt_emb.reshape(B, -1, C)
        # add the prompt back:
        # FFN
        x = torch.cat((prompt_emb, x), dim=1)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PromptBasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        num_prompts,
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
        use_checkpoint=False,
        # add two more parameters for prompt
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.total_prompts = num_prompts * \
            int(input_resolution[0] // window_size) ** 2
        # self.tap = nn.ModuleList([TAP(in_channels=dim, K=5, input_resolution=input_resolution) for i in range(depth)])
        # build blocks
        self.blocks = nn.ModuleList(
            [
                PromptSwinTransformerBlock(
                    dim=dim,
                    num_prompts=num_prompts,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,  # noqa
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, deep_prompt_embd=None):
        # add the prompt embed before each blk call
        B = x.shape[0]  # batchsize
        num_blocks = len(self.blocks)
        prompts = []
        # if deep_prompt_embd.shape[0] != num_blocks:
        # first layer
        for i in range(num_blocks):
            _, N, C = deep_prompt_embd[i].shape
            prompt_emb = deep_prompt_embd[i].to(x.device).expand(B, N, C)

            if i == 0:
                # x = torch.cat((prompt_emb, self.tap[i](x)), dim=1)
                x = torch.cat((prompt_emb, x), dim=1)
            else:
                # x = torch.cat((prompt_emb, self.tap[i](x[:, self.total_prompts :, :])), dim=1)
                x = torch.cat(
                    (prompt_emb, x[:, self.total_prompts:, :]), dim=1)
            x = self.blocks[i](x)
            prompts.append(x[:, : self.total_prompts, :])

        return x[:, self.total_prompts:, :], prompts


class PromptBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_prompts,
                 input_resolution,
                 prompt_window_size,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 new_prompts=False,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.attn = PromptCrossAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_heads = num_heads
        self.new_prompts = new_prompts
        self.num_prompts = num_prompts
        self.window_size = window_size
        self.input_resolution = input_resolution
        self.norm_x = norm_layer(dim)
        self.norm_p = norm_layer(dim)
        if self.new_prompts:
            self.prompt_proj = Prompt_adjust(dim=dim, num_prompts=num_prompts, input_resolution=input_resolution,
                                             prompt_window_size=prompt_window_size, window_size=window_size, num_heads=num_heads)

    def forward(self, x, prompt_emb):

        H, W = self.input_resolution
        B, N, C = x.shape
        x = self.norm_x(x)
        prompt_emb = self.norm_p(prompt_emb)

        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.new_prompts:
            prompt_emb, x_windows = self.attn(prompt_emb, x_windows)
        else:
            x_windows, prompt_emb = self.attn(x_windows, prompt_emb)

        # change input size
        # change prompt_embs's shape:
        # nW*B, num_prompts, C - B, num_prompts, C

        # merge windows
        # print(x_windows.shape)
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)  # B H W C

        x = x.view(B, H * W, C)
        prompt_emb = prompt_emb.reshape(B, -1, C)
        # add the prompt back:
        # FFN
        if self.new_prompts:
            prompt_emb = self.prompt_proj(prompt_emb)
        return x, prompt_emb


class PromptCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads,  qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        # self.num_prompt=num_prompt
        # self.permuted_window_size = (window_size[0] // 2,window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # compresses the channel dimension of KV
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv):
        """
        Args:
            q: input features with shape of (num_windows*b, n2, c)
            kv: input features with shape of (num_windows*b, n1, c)

        """
        shortcut_kv = kv
        B_, N1, C = kv.shape
        B_, N2, C = q.shape
        # compress the channel dimension of KV :(num_windows*b, num_heads, n1, c//num_heads)
        kv = self.kv(kv).reshape(B_, N1, 2, self.num_heads,
                                 C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # keep the channel dimension of Q: (num_windows*b, num_heads, n2, c//num_heads)
        q = self.q(q).reshape(B_, N2, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # (num_windows*b, num_heads, n2, n1)

        # relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1], -1)  # (n, n//4)
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n//4)
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        q = (attn @ v).transpose(1, 2).reshape(B_, N2, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q, shortcut_kv


class tokenWindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        num_prompts,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.num_prompts = num_prompts

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1)
                        * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, is_encoder=True):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        prompt_embed = x[:, :self.num_prompts, :]
        if is_encoder:
            q = x
            kv = x[:, self.num_prompts:, :]
        else:
            q = x[:, self.num_prompts:, :]
            kv = x

        B_, q_N, C = q.shape
        B_, kv_N, C = kv.shape
        q = (
            self.q(q)
            .reshape(B_, q_N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        kv = (
            self.kv(kv)
            .reshape(B_, kv_N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            q[0],
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        _C, _H, _W = relative_position_bias.shape
        if is_encoder:
            relative_position_bias = torch.cat(
                (
                    torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                    relative_position_bias,
                ),
                dim=1,
            )
        else:
            relative_position_bias = torch.cat(
                (
                    torch.zeros(
                        _C, _H, self.num_prompts, device=attn.device
                    ),
                    relative_position_bias,
                ),
                dim=-1,
            )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            # expand relative_position_bias
            if is_encoder:
                mask = torch.cat(
                    (torch.zeros(nW, self.num_prompts, _W, device=attn.device), mask), dim=1
                )
            else:
                mask = torch.cat(
                    (
                        torch.zeros(
                            nW, _H, self.num_prompts, device=attn.device
                        ),
                        mask,
                    ),
                    dim=-1,
                )
            attn = attn.view(B_ // nW, nW, self.num_heads, q_N, kv_N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, q_N, kv_N)
            # B*num_windows, num_heads, qN, kv_N
            # B*num_windows, qN, kv_N
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(is_encoder, x.shape)
        if not is_encoder:
            x = torch.cat([prompt_embed, x], dim=1)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='Bias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
