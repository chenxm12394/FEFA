import math
from typing import List, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.DeformConv_Attention_bblocks import HUMUSBlock
# from models.abalation.abalation_resnet import HUMUSBlock
# from models.PromptAttention_cross_attention import HUMUSBlock
# from models.DeformConv_Attention import HUMUSBlock
# from models.Restomer import HUMUSBlock
# from models.ORI_skip import HUMUSBlock
    
class NormHumus(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        embed_dim: int = 24,
        use_ref: bool = False,
        flow_skip=False,
        img_size: int = 320,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
        """
        super().__init__()
        self.use_ref = use_ref
        if self.use_ref:
            self.unet = HUMUSBlock(
                img_size=[img_size, img_size],
                in_chans=2,
                out_chans=2,
                embed_dim=embed_dim,
                flow_skip=flow_skip,
            )
            self.ref_norm = torch.nn.InstanceNorm2d(in_chans)
        else:
            self.unet = HUMUSBlock(
                img_size=[img_size, img_size],
                in_chans=2,
                out_chans=2,
                embed_dim=embed_dim,
                flow_skip=flow_skip,
            )
        self.in_chans = in_chans
        self.out_chans = out_chans

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        return torch.cat([x.real, x.imag], dim=1)

    def chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(x)
        _, c, _, _ = x.shape
        assert c % 2 == 0
        c = c // 2
        return torch.complex(x[:,:c], x[:,c:])

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        assert c%2 == 0
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std + 1e-6), mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(
        self, 
        x: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        assert len(x.shape) == 4
        assert torch.is_complex(x)
        assert x.shape[1] == self.in_chans

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        if self.use_ref:
            assert not torch.is_complex(ref)
            ref = self.ref_norm(ref)
            # ref, _ = self.pad(ref)
        #     # x = torch.cat([x, ref], dim=1)
        # else:
        #     assert ref is None
        # x, attns = self.unet(x, ref)
        # x, flow_skips, flow = self.unet(x, ref, flow_skips)
        x = self.unet(x, ref)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_dim_to_complex(x)

        assert x.shape[1] == self.out_chans

        # return x, flow_skips, flow
        return x