import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from NAFcommon import LayerNorm2d, NAFBlock
from NAFcommon import MySequential
from NAFcommon import Local_Base
import math


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])

        return feats

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        #print('scale:', scale)
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        elif scale%5 == 0:
            m.append(conv(n_feats, 100 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(10))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=5, drop_path_rate=0., drop_out_rate=0., dual=False, conv = default_conv):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlockSR(
                    width, 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        self.upsample = Upsampler(conv, up_scale, width)
        self.skip_conv = conv(img_channel, width, kernel_size=3)
        self.final = conv(width, img_channel, kernel_size=3)

    def forward(self, inp, lms):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bicubic')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        #print('1___out shape:', out.shape)
        #print('1___inp_hr shape:', inp_hr.shape)

        out = torch.cat([self.upsample(x) for x in feats], dim=1)

        inp_hr = self.skip_conv(inp_hr)
        #print('2___out shape:', out.shape)
        #print('inp_hr shape:', inp_hr.shape)
        #print('inp_hr shape:', inp_hr.shape)

        out = out + inp_hr
        #out = out + lms

        out = self.final(out)
        return out
