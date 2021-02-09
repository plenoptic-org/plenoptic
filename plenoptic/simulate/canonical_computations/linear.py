import torch
import torch.nn as nn


class Linear(nn.Module):
    """Simplistic convolutional model:
    It splits the input greyscale image into low and high frequencies.
    """
    def __init__(self):
        super().__init__()

        kernel_size = (3, 3)

        self.conv = nn.Conv2d(1, 2, kernel_size,
                              padding=(1, 1),
                              bias=False)

        # Gaussian (low pass) filter
        variance = 3
        xs = torch.linspace(-2, 2, kernel_size[0])

        g1 = torch.exp(-xs**2 / (2 * variance))
        f1 = torch.outer(g1, g1)
        f1 = f1 / f1.sum()

        # difference of Gaussian (high pass) filter
        g2 = torch.exp(-xs**2 / (variance / 3))
        f2 = torch.outer(g2, g2)
        f2 = f2 / f2.sum() - f1
        f2 = f2 - f2.sum()

        f1 = torch.tensor(f1, dtype=torch.float32)
        f2 = torch.tensor(f2, dtype=torch.float32)

        self.conv.weight.data[0, 0] = f1
        self.conv.weight.data[1, 0] = f2

    def forward(self, x):

        h = self.conv(x)

        return h
