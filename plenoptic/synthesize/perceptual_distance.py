import numpy as np
import torch
import torch.nn.functional as F

from ..simulate.laplacian_pyramid import Laplacian_Pyramid
from ..simulate.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ..simulate.non_linearities import local_gain_control

import os
dirname = os.path.dirname(__file__)


def gaussian(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def nlp(im, N_levels=6):

    (_, channel, height, width) = im.size()

    DN_filts = np.load(dirname + '/DN_filts.npy')
    sigmas = np.load(dirname + '/DN_sigmas.npy')

    L = Laplacian_Pyramid(n_scales=N_levels)
    Lap_dom = L.analysis(im)

    padd = 2
    DN_dom = []
    for N_b in range(0, N_levels):
        A2 = F.conv2d(torch.abs(Lap_dom[N_b]), torch.tensor(DN_filts[N_b], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=padd, groups=channel)

        # Divisive Normalization
        DN_dom.append(Lap_dom[N_b]/(sigmas[N_b]+A2))

    return DN_dom


def nlpd(IM_1, IM_2):
    """Normalized Laplacian Pyramid Distance

    compute root mean squared error for each scale, and then average over these,
    effectively giving larger weight to the lower frequency coefficients
    (which are fewer in number, due to subsampling)


    Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual image quality assessment using a normalized Laplacian pyramid. Electronic Imaging, 2016(16), pp.1-6.
    """

    y1 = nlp(IM_1)
    y2 = nlp(IM_2)

    dist = []
    for i in range(6):
        dist.append(torch.sqrt(torch.mean((y1[i] - y2[i]) ** 2)))

    return 1 / torch.stack(dist).mean()


def nspd(IM_1, IM_2, O=1, S=5):
    """normalized steerable pyramid distance

    ongoing work
    """

    L1 = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S)
    G1 = local_gain_control

    pyr_1 = L1(IM_1)
    pyr_2 = L1(IM_2)

    norm_1, state_1 = G1(pyr_1)
    norm_2, state_2 = G1(pyr_2)

    dist = []
    for key in state_1.keys():
        dist.append(torch.sqrt(torch.mean((norm_1[key] - norm_2[key]) ** 2)))
        dist.append(torch.sqrt(torch.mean((state_1[key] - state_2[key]) ** 2)))

    return 1 / torch.stack(dist).mean()
