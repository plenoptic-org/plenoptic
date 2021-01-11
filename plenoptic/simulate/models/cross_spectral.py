import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq \
    import Steerable_Pyramid_Freq
from ...tools.conv import blur_downsample


class CrossSpectral(nn.Module):
    """Under development
    """

    def __init__(self, image_size, n_ori=6, n_scale=4):
        super().__init__()

        # self.complex_pyr = Steerable_Pyramid_Freq(image_size,
        #                                           height=n_scale,
        #                                           is_complex=True,
        #                                           order=n_ori-1,
        #                                           downsample=False)
        self.pyr = Steerable_Pyramid_Freq(image_size,
                                          height=n_scale,
                                          order=n_ori-1,
                                          downsample=False)

    def forward(self, x, vectorize=False):
        assert x.ndim == 4

        # y = self.complex_pyr(x)
        # energy, phase = rectangular_to_polar(y[:, 1:-1:2], y[:, 2:-1:2])

        coeffs = self.pyr(x)
        lp = blur_downsample(torch.sqrt(coeffs[:, 0:1]**2))
        hp = blur_downsample(torch.sqrt(coeffs[:, -1:]**2))
        y = coeffs[:, 1:-1]
        # local spatial averaging
        S = self.pyr.num_scales
        O = self.pyr.num_orientations

        stats = torch.cat([torch.sqrt(blur_downsample(y[:, c].unsqueeze(1)**2))
                           for c in range(S * O)],
                          dim=1)
        # torch.sqrt
        cross_ori = []
        for s in range(S):
            for o in range(O):
                cross_ori.append((
                            blur_downsample(y[:, o+s*O].unsqueeze(1) *
                                            y[:, ((o+1) % O+s*O)].unsqueeze(1)
                                            )))
                print(o + s*O, ((o+1) % O+s*O))
        cross_ori = torch.cat(cross_ori, dim=1)

        # torch.sqrt
        cross_scale = []
        for o in range(O):
            for s in range(S-1):
                cross_scale.append((
                            blur_downsample(y[:, o + s*O].unsqueeze(1) *
                                            y[:, o + (s+1)*O].unsqueeze(1)
                                            )))
        cross_scale = torch.cat(cross_scale, dim=1)

        stats = torch.cat([stats, cross_ori, cross_scale, lp, hp], dim=1)
        # if vectorize:
        #     stats  = stats.view(x.shape[0], -1)

        # global spatial averaging
        # stats = torch.cat([torch.sqrt(y[:, 0:1]**2).mean(dim=(2, 3)),
        #                    energy.mean(dim=(2, 3)),
        #                    torch.sqrt(y[:, -1:]**2).mean(dim=(2, 3))],
        #                   dim=1).view(x.shape[0], -1)

        return stats
