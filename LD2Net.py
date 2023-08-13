import math
import os

import cv2

from torchvision import transforms


import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import math







class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.r1 = nn.ReLU()
        self.r2 = nn.ReLU()
        self.r3 = nn.ReLU()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')

        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())

        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        feats0 = self.r1(feats0)

        featss = self.net1_convs(feats0)

        outs = self.net1_recon(featss)


        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])

        return R, L

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2),

                                  )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()


        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))


    def forward(self, x):
        return self.body(x)

class HDCB(nn.Module):
    def __init__(self, dim, ksize, drate1, drate2, drate3):
        super(HDCB, self).__init__()

        self.c1 = nn.Conv2d(dim, dim, kernel_size=ksize, dilation=drate1, padding='same')
        self.lr1 = nn.LeakyReLU(0.1)
        self.c2 = nn.Conv2d(dim, dim, kernel_size=ksize, dilation=drate2, padding='same')
        self.lr2 = nn.LeakyReLU(0.1)
        self.c3 = nn.Conv2d(dim, dim, kernel_size=ksize, dilation=drate3, padding='same')
        self.lr3 = nn.LeakyReLU(0.1)





    def forward(self, x):
        x1_in = x
        x1_= self.c1(x1_in)
        x1_out = self.lr1(x1_)


        x2_in = x1_out+x1_in
        x2_ = self.c2(x2_in)
        x2_out = self.lr2(x2_)



        x3_in = x2_out+x2_in
        x3_ = self.c3(x3_in)
        x3_out = self.lr3(x3_)

        out = x3_out+x3_in



        return  out



class fuse(nn.Module):
    def __init__(self,dim):
        super(fuse, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conva = nn.Conv2d(dim,dim,kernel_size=3,padding=1)

        self.sigmoid = nn.Sigmoid()
        self.ADM = nn.Sequential(
            nn.Linear(dim ,dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim//4, 2, bias=False),
        )

    def forward(self, x,l1,l2):

        a, b, c, d = x.shape

        px = self.conva(x)


        y = self.avg_pool(px).view(a, b)
        y = self.ADM(y)
        ax = F.softmax(y, dim=1)

        out =   l1* ax[:,0].view(a,1,1,1) + l2 * ax[:,1].view(a,1,1,1)
        return out


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.layer_0 = nn.Conv2d(4,  24, 3, padding='same')
        self.layer_1 = nn.Conv2d(24, 3, 3, padding='same')

        self.down1 = Downsample(24)  #48
        self.down2 = Downsample(48)  #96


        self.up2 = Upsample(96)   #48
        self.up1 = Upsample(48)   #24

        self.HDCB_0 = HDCB(24,7, 7,3,1)
        self.HDCB_1 = HDCB(48,5, 5,3,1)
        self.HDCB_2 = HDCB(96,3, 3,2,1)
        self.fuse1= fuse(48)
        self.fuse0 = fuse(24)








    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        x_l0 = self.layer_0(input_img)
        x_l0_p =self.HDCB_0(x_l0)
        print(x_l0_p.shape)




        x_l1 = self.down1(x_l0)
        x_l1_p = self.HDCB_1(x_l1)


        print(x_l1_p.shape)
        x_l2 = self.down2(x_l1)
        x_l2_p = self.HDCB_2(x_l2)
        print(x_l2_p.shape)

        l1_new = self.up2(x_l2_p)
        l1 = self.fuse1(x_l1,x_l1_p,l1_new)
        l1 = self.HDCB_1(l1)


        l0_new = self.up1(l1)
        l0 = self.fuse0(x_l0,x_l0_p,l0_new)
        l0 = self.HDCB_0(l0)



        out = self.layer_1(l0)




        return out


class APMG(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.main =nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )

        self.mag = nn.Sequential(
        nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

      )
        self.magconv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.phaconv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)



        mag_ = self.mag(mag)
        mag_res = mag_ - mag


        pha_ = self.pha(pha)
        pha_res = pha_ - pha





        pooling1 = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling1 = torch.nn.functional.softmax(pooling1, dim=1)
        pha1 = pha_ * pooling1

        pooling2 = torch.nn.functional.adaptive_avg_pool2d(pha_res, (1, 1))
        pooling2 = torch.nn.functional.softmax(pooling2, dim=1)
        mag1 = mag_ * pooling2



        mag_out = mag1+mag
        mag_out = self.magconv(mag_out)

        pha_out = pha1 + pha
        pha_out = self.phaconv(pha_out)




        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return self.main(y) + x




class DNUDC(nn.Module):
    def __init__(self):
        super(DNUDC, self).__init__()

        self.apmg= APMG(3)
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()


    def forward(self, input):
        b, c, h, w = input.shape



        input = F.interpolate(input, (512, 1024),
                           mode='bilinear')

        R_low, I_low = self.DecomNet(input)
        I_delta= self.RelightNet(I_low, R_low)



        R_low_fft = self.apmg(R_low)

 
        output = R_low_fft * I_delta
        output = output+input

        output = F.interpolate(output, (1024, 2048),
                           mode='bilinear')
    

        return output




