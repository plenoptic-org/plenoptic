import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, kernel_size=(7, 7)):
        super(Linear, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=kernel_size, padding=(0, 0), bias=False)

    def forward(self, x):

        h = self.conv1(x)

        return h
