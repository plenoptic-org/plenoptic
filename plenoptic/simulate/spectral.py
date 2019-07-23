import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from .steerable_pyramid_freq import Steerable_Pyramid_Freq


class Spectral(nn.Module):

    def __init__(self,image_size, Nsc=4, Nor=4):
        super().__init__()
        self.complex_steerable_pyramid =  Steerable_Pyramid_Freq(image_size,height=Nsc,is_complex=True,order = Nor-1)
        self.Nsc = Nsc
        self.Nor = Nor

        # weighted local statistics

    def forward(self, im0):

        # pixel statistics
        mn0 = torch.min(im0)
        mx0 = torch.max(im0)
        mean0 = torch.mean(im0)
        var0 = torch.var(im0)
        skew0 = Spectral.skew(im0, mean0, var0)
        kurt0 = Spectral.kurtosis(im0, mean0,var0)

        statg0 = torch.stack((mean0, var0,skew0,kurt0,mn0,mx0)).view(6,1)
        # im0 = (im0-mean0)/var0
        # build steerable pyramid
        self.complex_steerable_pyramid.forward(im0)
        pyr0 = self.complex_steerable_pyramid.coeffout

        # stats = torch.empty((len(pyr0),1))
        stats = torch.empty((len(pyr0),1))  
        cnt=0
        for mat in pyr0:
            tmp = torch.unbind(mat,-1)
            stats[cnt]=torch.abs(((tmp[0]**2+tmp[1]**2)**.5).squeeze()).mean()
            cnt+=1            



        stats = torch.cat((stats,statg0))

        return stats

    def kurtosis(mtx, mn, v):
        # implementation is only for real components
        return torch.mean(torch.abs(mtx-mn).pow(4))/(v.pow(2))

    def skew(mtx, mn, v):
        return torch.mean((mtx-mn).pow(3))/(v.pow(1.5))
