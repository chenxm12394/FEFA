from einops import rearrange
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
import sys


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1

class multiModalFusion(nn.Module):
    def __init__(self, in_channels, ks=3, stride=1, padding=1):

        super(multiModalFusion, self).__init__()
                
        self.conv_1 = nn.Conv2d(in_channels*2, in_channels, ks, stride, padding)       
        self.conv_2 = nn.Conv2d(in_channels, in_channels*2, ks, stride, padding)
        self.conv_3 = nn.Conv2d(in_channels*2, in_channels, ks, stride, padding)
        self.gelu = nn.GELU() 

    def forward(self, tar, aux):

        feat = torch.cat([aux, tar], dim=1)
        
        fused_feat = self.gelu(self.conv_1(feat))
        
        exp_feat = self.gelu(self.conv_2(fused_feat))
        
        fused_feat = fused_feat + self.gelu(self.conv_3(exp_feat - feat))

        # fused_feat = self.fuse(torch.cat([fused_feat, x], dim=1))
        # print(fused_feat.shape)

        return fused_feat
    
class alignment(nn.Module):
    # def __init__(self, window_size, input_resolution, num_heads, dim=48, memory=False,  stride=1, type='group_conv', groups=8):
    def __init__(self, dim=48, prev=False, groups=8):
        
        super(alignment, self).__init__()
        
        act = nn.GELU()
        bias = False

        self.offset_conv = nn.Conv2d(dim, groups*27, 3, stride=1, padding=1, bias=bias)
        # self.offset_conv = nn.Sequential(
        #     ResASPP(dim),
        #     nn.Conv2d(dim, out_channels, 1, bias=bias)
        # )
        
        # 为groups * 3 * (3 * 3)卷积核数目 
        self.deform = DeformConv2d(dim, dim, 3, padding = 2, groups = groups, dilation=2)            
        self.MMF = multiModalFusion(dim)

        self.aligned_up = nn.Sequential(nn.Conv2d(dim*2, dim, 1), act)
        self.aligned_feat_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1), act)
        self.prev=prev
        # self.nff = NFF(in_chan=dim,
        # nums=2
        self.bottleneck = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        if self.prev:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            self.up = nn.Sequential(nn.Conv2d(2*dim, dim, 1), act)

    def offset_gen(self, x):
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        
        return offset, mask
        
    def forward(self, x, ref, prev_offset_feat=None, prev_aligned=None):
        # ref = self.style_transfer(x, ref)
        offset_feat = self.bottleneck(torch.cat([ref, x], dim=1))


        if self.prev:
            # print('memory', prev_offset_feat.shape)
            prev_offset_feat = F.interpolate(prev_offset_feat, scale_factor=2, mode='bilinear', align_corners=False)
            prev_offset_feat = self.up(prev_offset_feat)
            offset_feat = self.bottleneck_o(torch.cat([prev_offset_feat*2, offset_feat], dim=1))


        offset, mask = self.offset_gen(self.offset_conv(offset_feat))

        aligned_feat = self.deform(ref, offset, mask)

        if prev_aligned is not None:
            prev_aligned = F.interpolate(prev_aligned, scale_factor=2, mode='bilinear', align_corners=False)
            prev_aligned = self.aligned_up(prev_aligned)
            aligned_feat = self.aligned_feat_conv(torch.cat([aligned_feat, prev_aligned], dim=1))
        # aligned_feat_copy = aligned_feat
        mixed_feat = self.MMF(x, aligned_feat)
        # aligned_feat = self.back_projection(torch.cat([aligned_feat, x], dim=1))
        # aligned_feat = self.back_projection(ref, x)
        # aligned_feat = self.fuse(x, aligned_feat)
        # return aligned_feat, offset_feat
        return mixed_feat, offset_feat, aligned_feat


if __name__ == '__main__':
    model = alignment(24, memory=True)
    x = torch.rand((3, 24, 40, 40))
    ref = torch.rand((3, 24, 40, 40))
    offset = torch.rand((3, 48, 20, 20))
    aligned, offset, _ = model(x, ref, offset)
    print(aligned.shape, offset.shape)