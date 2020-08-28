import torch
import torch.nn as nn
from pyrtools.tools.synthetic_images import gaussian


class Linear(nn.Module):

    def __init__(self, kernel_size=(7, 7)):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=kernel_size, padding=(0, 0), bias=False)
        f1 = gaussian(kernel_size, covariance=2)
        f2 = gaussian(kernel_size, covariance=1) - f1
        self.conv1.weight.data[0,0] = nn.Parameter(torch.tensor(f1, dtype=torch.float32))
        self.conv1.weight.data[1,0] = nn.Parameter(torch.tensor(f2, dtype=torch.float32))

    def forward(self, x):

        h = self.conv1(x)

        return h
