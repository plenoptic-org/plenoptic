import torch
import torch.nn as nn


class Linear_Nonlinear(nn.Module):
    """Canonical functional model of early visual processing.

    The idea of "Threshold Logic Unit" goes at least as far back as: 
    McCulloch, W.S. and Pitts, W., 1943. A logical calculus of the ideas
    immanent in nervous activity. The bulletin of mathematical biophysics

    Notes
    -----
    (cascades of) linear filters and simple nonlinearities are
    surprisingly effective at performing visual tasks and at capturing
    cell responses in the early visual system:
    - the linear filters capture basic cell properties (tuning/invariance)
    - the ReLU captures the point (a.k.a. memoryless or static) nonlinearity
        of cells typicall input-output curve. To capture modulatory phenomenon,
        use other nonlinearities like Gain control / Divisive normalization.

    Parameters
    ----------
    n_channels: int, optional
        number of convolutional channels
    kernel_size : tuple of two ints
        size of the receptive fields (odd size supported), [3x3] by default
    default_filters : bool, optional
        Initialize the first three filters to: low pass, high pass on / off.
    """
    def __init__(self, n_channels=3, kernel_size=(3, 3), default_filters=True):
        super().__init__()

        self.conv = nn.Conv2d(1, n_channels, kernel_size,
                              padding=[int((k-1)/2) for k in kernel_size],
                              bias=False)

        if default_filters and n_channels >= 3:
            variance = 3
            xs = torch.linspace(-2, 2, kernel_size[0])

            g1 = torch.exp(-xs**2 / (2 * variance))
            f1 = torch.outer(g1, g1)
            f1 = f1 / f1.sum()

            g2 = torch.exp(-xs**2 / (variance / 3))
            f2 = torch.outer(g2, g2)
            f2 = f2 / f2.sum() - f1
            f2 = f2 - f2.sum()

            f1 = torch.tensor(f1, dtype=torch.float32)
            f2 = torch.tensor(f2, dtype=torch.float32)

            self.conv.weight.data[0, 0] = nn.Parameter(f1)
            self.conv.weight.data[1, 0] = nn.Parameter(f2)
            self.conv.weight.data[2, 0] = nn.Parameter(-f2)

        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv(x))

        return h
