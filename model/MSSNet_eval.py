"""
## Multi-Scale-Stage Network
## Code is based on Multi-Stage Progressive Image Restoration(MPRNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Residual Block
class ResBlock(nn.Module):
    def __init__(self, wf, kernel_size, reduction, bias, act):
        super(ResBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(wf, wf, kernel_size, bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(wf, scale)
        self.down23  = DownSample(wf+scale, vscale)

        if csff:
            self.csff_enc1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(wf,              wf,              kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(wf+scale,        wf+scale,        kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(wf+scale+vscale, wf+scale+vscale, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, wf, scale, vscale, kernel_size, reduction, act, bias):
        super(Decoder, self).__init__()

        self.decoder_level1 = [ResBlock(wf,              kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [ResBlock(wf+scale,        kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [ResBlock(wf+scale+vscale, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = ResBlock(wf,       kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = ResBlock(wf+scale, kernel_size, reduction, bias=bias, act=act)

        self.up32  = SkipUpSample(wf+scale, vscale)
        self.up21  = SkipUpSample(wf, scale)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class EDUp(nn.Module):
    def __init__(self):
        super(EDUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, elist, dlist):

        up_elist = []
        for feat in elist:
            up_elist.append(self.up(feat))

        up_dlist = []
        for feat in dlist:
            up_dlist.append(self.up(feat))

        return up_elist, up_dlist

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class ScaleUpSample(nn.Module):
    def __init__(self, in_channels):
        super(ScaleUpSample,self).__init__()
        self.scaleUp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                     nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self,feat):
        return self.scaleUp(feat)

#https://github.com/fangwei123456/PixelUnshuffle-pytorch
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * (k*w) * (k*h)
        kdownscale_factor: k
        batchSize * c * (k*w) * (k*h) -> batchSize * (k*k*c) * w * h
        '''
        c = input.shape[1]

        kernel = torch.zeros(size=[self.downscale_factor * self.downscale_factor * c,
                                   1, self.downscale_factor, self.downscale_factor],
                             device=input.device)
        for y in range(self.downscale_factor):
            for x in range(self.downscale_factor):
                kernel[x + y * self.downscale_factor::self.downscale_factor*self.downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=self.downscale_factor, groups=c)

class Tail_shuffle(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,bias):
        super(Tail_shuffle,self).__init__()
        self.tail = conv(in_channels, out_channels, kernel_size, bias=bias)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, feat):
        return self.pixelshuffle(self.tail(feat))

##########################################################################

class DeblurNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, wf=54, scale=42, vscale=42, kernel_size=3, reduction=4, bias=False):
        super(DeblurNet, self).__init__()

        act=nn.PReLU()
        self.pixel_unshuffle = PixelUnshuffle(2)
        self.ED_up = EDUp()
        #scale1
        self.shallow_feat1 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.E1_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=False)
        self.D1_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)
        self.tail1_1 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        ##########################################################################
        #scale2
        self.shallow_feat2 = nn.Sequential(conv(12, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.up_scale1_feat = ScaleUpSample(wf)
        self.fusion12 = conv(wf*2, wf, kernel_size, bias=bias)

        self.E2_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D2_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E2_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D2_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.tail2_1 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        self.tail2_2 = Tail_shuffle(wf, 12, kernel_size, bias=bias)
        ################################################################################
        #scale3
        self.shallow_feat3 = nn.Sequential(conv(3, wf, kernel_size, bias=bias), ResBlock(wf,kernel_size, reduction, bias=bias, act=act))
        self.up_scale2_feat = ScaleUpSample(wf)
        self.fusion23 = conv(wf*2, wf, kernel_size, bias=bias)

        self.E3_1 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_1 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E3_2 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_2 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.E3_3 = Encoder(wf, scale, vscale, kernel_size, reduction, act, bias, csff=True)
        self.D3_3 = Decoder(wf, scale, vscale, kernel_size, reduction, act, bias)

        self.tail3_1 = conv(wf, 3, kernel_size, bias=bias)
        self.tail3_2 = conv(wf, 3, kernel_size, bias=bias)
        self.tail3_3 = conv(wf, 3, kernel_size, bias=bias)

    def forward(self, s3_blur):

        interpolation = nn.Upsample(scale_factor=0.5, mode = 'bilinear',align_corners=True)
        s2_blur = interpolation(s3_blur)

        ##-------------------------------------------
        ##-------------- Scale 1---------------------
        ##-------------------------------------------
        s1_blur_ps = self.pixel_unshuffle(s2_blur)
        shfeat1 = self.shallow_feat1(s1_blur_ps)

        e1_1f = self.E1_1(shfeat1)
        d1_1f = self.D1_1(e1_1f)

        ##-------------------------------------------
        ##-------------- Scale 2---------------------
        ##-------------------------------------------
        s2_blur_ps = self.pixel_unshuffle(s3_blur)

        shfeat2 = self.shallow_feat2(s2_blur_ps)
        s1_sol_feat = self.up_scale1_feat(d1_1f[0])
        fusion12 = self.fusion12(torch.cat([shfeat2,s1_sol_feat],1))

        e_list,d_list = self.ED_up(e1_1f,d1_1f)
        e2_1f = self.E2_1(fusion12,e_list,d_list)
        d2_1f = self.D2_1(e2_1f)

        e2_2f = self.E2_2(d2_1f[0],e2_1f,d2_1f)
        d2_2f = self.D2_2(e2_2f)

        ##-------------------------------------------
        ##-------------- Scale 3---------------------
        ##-------------------------------------------
        shfeat3 = self.shallow_feat3(s3_blur)
        s2_sol_feat = self.up_scale2_feat(d2_2f[0])
        fusion23 = self.fusion23(torch.cat([shfeat3,s2_sol_feat],1))

        e_list,d_list = self.ED_up(e2_2f,d2_2f)
        e3_1f = self.E3_1(fusion23,e_list,d_list)
        d3_1f = self.D3_1(e3_1f)

        e3_2f = self.E3_2(d3_1f[0],e3_1f,d3_1f)
        d3_2f = self.D3_2(e3_2f)

        e3_3f = self.E3_3(d3_2f[0],e3_2f,d3_2f)
        d3_3f = self.D3_3(e3_3f)

        res3_3 = self.tail3_3(d3_3f[0]) + s3_blur

        return res3_3
