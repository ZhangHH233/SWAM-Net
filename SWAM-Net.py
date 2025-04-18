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
    def __init__(self,in_channels,num_classes,Down_type=3,Fusion_type=1,WF_type='WF5'):
        super(SWAM_Net,self).__init__()   
        
        Down_type,Fusion_type = str(Down_type),str(Fusion_type)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
          
        self.start_conv = x2conv(in_channels, 64)   
        
        self.DWT1 = HW_DWT_module(64,Down_type,Fusion_type,WF_type)
        self.conv1 = x2conv(64, 128) 
         
        self.DWT2 = HW_DWT_module(128,Down_type,Fusion_type,WF_type)
        self.conv2 = x2conv(128, 256)  
        
        self.DWT3 = HW_DWT_module(256,Down_type,Fusion_type,WF_type)
        self.conv3 = x2conv(256, 512)  
        
        self.DWT4 = HW_DWT_module(512,Down_type,Fusion_type,WF_type)
        self.conv4 = x2conv(512, 1024)  
        
        self.middle_conv = x2conv(1024, 1024)  
        
        self.uppool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)       
        self.dec_conv4 = x2conv(1024, 512)       
        
        self.uppool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)        
        self.dec_conv3 = x2conv(512, 256) 
        
        self.uppool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)        
        self.dec_conv2 = x2conv(256, 128) 
        
        self.uppool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        
        self.dec_conv1 = x2conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self,x):
        x1 = self.start_conv(x)  
        
        att1,copy1 = self.DWT1.Down(x1)
        x2 = torch.mul(self.Maxpool(x1), att1)          
        x2 = self.conv1(x2)
        
        att2,copy2 = self.DWT2.Down(x2)
        x3 = torch.mul(self.Maxpool(x2), att2)           
        x3 = self.conv2(x3)
        
        att3,copy3 = self.DWT3.Down(x3)
        x4 = torch.mul(self.Maxpool(x3), att3)           
        x4 = self.conv3(x4)
        
        att4,_ = self.DWT4.Down(x4)
        x5 = torch.mul(self.Maxpool(x4), att4)   
        x5 = self.conv4(x5)
                
        x5 = self.middle_conv(x5)
                
        d4 = self.uppool4(x5)
        x4 = self.DWT3.Fusion(x4,copy3)
        d4 = torch.cat((x4,d4), dim=1)
        d4 = self.dec_conv4(d4)
        
        d3 = self.uppool3(d4)
        x3 = self.DWT2.Fusion(x3,copy2)
        d3 = torch.cat((x3,d3), dim=1)
        d3 = self.dec_conv3(d3)
        
        d2 = self.uppool2(d3)
        x2 = self.DWT1.Fusion(x2,copy1)
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.dec_conv2(d2)
        
        d1 = self.uppool1(d2)
        d1 = torch.cat((x1,d1),dim=1)
        d1 = self.dec_conv1(d1)
               
        d = self.final_conv(d1)      

        return d

if __name__ == '__main__':
    x = torch.randn([1,3,512,512]).cuda()
    model = SWAM_Net(in_channels=3, num_classes=4,
                Down_type=4,
                Fusion_type=0).cuda()
    print(model(x).shape)