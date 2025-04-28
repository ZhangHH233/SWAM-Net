import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from models.basics.unet import encoder, decoder
from models.blocks.dwt_modules.DWT_IDWT_layer import *
from models.blocks.att_modules import HW_DWT_module


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

class SWAM_Net(nn.Module):
    pass

        

if __name__ == '__main__':
    x = torch.randn([1,3,512,512]).cuda()
    model = SWAM_Net(in_channels=3, num_classes=4,
                Down_type=4,
                Fusion_type=0).cuda()
    print(model(x).shape)
