import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ..canonical_computations.non_linearities import rectangular_to_polar_dict
from ...tools.stats import skew, kurtosis
from ...tools.signal import min, max


class Spectral(nn.Module):
    """
    """
    def __init__(self,image_size, n_ori=6, n_scale=4):
        super().__init__()

        self.complex_steerable_pyramid =  Steerable_Pyramid_Freq(image_size, height=n_scale, is_complex=True, order=n_ori-1, downsample=True)
        self.non_linearity = rectangular_to_polar_dict

    def forward(self, x):

        dims = (1, 2, 3)
        # pixel statistics
        x_min = min(x, dim=dims)
        x_max = max(x, dim=dims)
        x_mean = torch.mean(x, dim=dims)
        x_var = torch.var(x, dim=dims)
        x_skew = skew(x, dim=dims, keepdim=True).view(x.shape[0])
        x_kurt = kurtosis(x, dim=dims, keepdim=True).view(x.shape[0])
        x_stats = torch.stack((x_mean, x_var, x_skew, x_kurt, x_min, x_max)).view(x.shape[0], 6)

        # build steerable pyramid
        x = (x-x_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1))/x_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        y = self.complex_steerable_pyramid(x)

        stats = torch.empty((x.shape[0], len(y)))
        cnt=0
        for channel in y.values():
            if channel.shape[-1] == 2:
                real, imag = torch.unbind(channel, -1)
                stats[:, cnt]=torch.abs(((real**2 + imag**2)**.5)).mean(dim=dims)
            else:
                stats[:, cnt]=torch.mean(torch.abs(channel))
            cnt+=1

        # TODO
        # energy, phase = self.non_linearity(y, residuals=True)
        # y_stats = torch.cat([torch.abs(e).mean(dim=dims) for e in energy.values()]).view(x.shape[0], len(energy.values()))
        # print(stats)
        # print(y_stats)
        # print(stats - y_stats)

        stats = torch.cat((x_stats, stats), 1)
        return stats
