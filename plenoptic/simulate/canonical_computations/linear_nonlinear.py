import torch.nn as nn


class Linear_Nonlinear(nn.Module):

    def __init__(self):
        super(Linear_Nonlinear, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(1, 1), bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        return h
