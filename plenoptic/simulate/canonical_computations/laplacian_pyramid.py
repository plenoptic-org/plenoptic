import torch
import torch.nn as nn
from torchvision import transforms
from ...tools.conv import blur_downsample, upsample_blur

class Laplacian_Pyramid(nn.Module):
    """
    Adelson Burt 1981

    """

    def __init__(self, n_scales=5, downsample_filter=None, upsample_filter=None, edge_type='reflect1'):
        super(Laplacian_Pyramid, self).__init__()

        self.n_scales = n_scales

    def analysis(self, x):

        y = []
        for scale in range(self.n_scales - 1):
            x_down = blur_downsample(x)
            x_up = upsample_blur(x_down)
            y.append(x - x_up)
            x = x_down
        y.append(x)

        return y

    def synthesis(self, y):

        x = y[self.n_scales - 1]
        for scale in range(self.n_scales - 1, 0, -1):
            y_up = upsample_blur(x)
            x = y[scale - 1] + y_up

        return x


class Laplacian_Pyramid_Learnable(nn.Module):
    """
    TODO learning filters

    """

    def __init__(self, n_scales=5, downsample_filter=None, upsample_filter=None, edge_type='reflect1'):
        super(Laplacian_Pyramid_Learnable, self).__init__()

        self.n_scales = n_scales

        if edge_type == 'zero':
            self.correlate_downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=downsample_filter.shape[-1], stride=2, padding=2, bias=False)
            self.correlate_downsample.weight = nn.Parameter(torch.tensor(downsample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

            self.upsample_convolve = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=upsample_filter.shape[-1], stride=2, padding=2, output_padding=1, bias=False)
            self.upsample_convolve.weight = nn.Parameter(torch.tensor(upsample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        elif edge_type == 'reflect1':
            correlate_downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=downsample_filter.shape[-1], stride=2, padding=0, bias=False)
            correlate_downsample.weight = nn.Parameter(torch.tensor(downsample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

            pad_down = nn.ReflectionPad2d(downsample_filter.shape[-1]//2)
            self.correlate_downsample = transforms.Compose([pad_down, correlate_downsample])

            # TODO - generalize to other image / filter sizes!
            # this solution is specific to power of two images and filter [5 x 5]
            # need start and stop arguments, two tuples of boolean values, even / odd
            upsample_convolve = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=upsample_filter.shape[-1], stride=2, padding=4, output_padding=1, bias=False)
            upsample_convolve.weight = nn.Parameter(torch.tensor(upsample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

            pad_up = nn.ReflectionPad2d(1)
            self.upsample_convolve = transforms.Compose([pad_up, upsample_convolve])

    def analysis(self, x):

        y = []
        for scale in range(self.n_scales-1):
            x_down = self.correlate_downsample(x)
            x_up = self.upsample_convolve(x_down) #, output_size=x.shape)
            y.append(x - x_up)
            x = x_down
        y.append(x)

        return y

    def synthesis(self, y):

        x = y[self.n_scales-1]
        for scale in range(self.n_scales-1, 0, -1):
            y_up = self.upsample_convolve(x) #, output_size=y[scale-1].shape)
            x = y[scale-1] + y_up

        return x
