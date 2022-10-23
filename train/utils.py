import torch
import numpy as np

def gim2uint8(x):
    x = x.detach().cpu().numpy()
    x= (x)*255
    out = np.clip(x,0,255)+0.5
    return out.astype(np.uint8)

def im2uint8(x):
    x = x.cpu().numpy()
    x= (x)*255
    out = np.clip(x,0,255)+0.5
    return out.astype(np.uint8)

def get_pyramid(img):
    img_pyramid=[]
    for i in range(2):
        scale_f = 0.5 ** (3 - i -1)
        down = torch.nn.Upsample(scale_factor=scale_f, mode = 'bilinear',align_corners=True)
        img_pyramid.append(down(img))

    img_pyramid.append(img)
    return img_pyramid
