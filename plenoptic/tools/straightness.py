import torch
import torch.nn as nn
import numpy as np


def make_straight_line(start, stop, n_step):
    """

    Parameters
    ----------
    start:
        [1, C, H, W]
    stop:
        [1, C, H, W]
    n_step:

    Returns
    -------

    """
    assert start.shape == stop.shape
    tt = torch.linspace(0, 1, n_step).view(n_step, 1, 1, 1)

    return (1 - tt) * start + tt * stop
    # x = torch.Tensor(self.n_steps-2, *self.image_size[1:])
    # for i in range(self.n_steps-2):
    #     t = (i+1)/(self.n_steps-1)
    #     x[i].copy_(self.xA[0] * (1 - t)+(t * self.xB[0]))


def sample_brownian_bridge(start, stop, n_step):
    """
    start:
        [1, C, H, W]
    stop:
        [1, C, H, W]
    n_step:

    example
    ```
    d1 = 10
    d2 = 20
    B = make_brownian_bridge(np.random.randn(d1,d2),
                             np.random.randn(d1,d2), 10)
    plt.plot(B.reshape(10,-1))
    plt.axhline(0, ls='--', c='k')
    ```
    """
    assert start.shape == stop.shape
    d = start.numel()
    dt = torch.tensor(1/(n_step - 1))
    tt = torch.linspace(0, 1, n_step)[:, None]

    dW = torch.sqrt(dt) * torch.randn(n_step, d)
    dW[0] = start.flatten()
    W = torch.cumsum(dW, axis=0)

    B = W - tt * (W[-1] - stop.flatten())[None, :]

    # print(torch.norm(((W - tt * W[-1][None, :])[-1])) < 1e-6)
    # print(torch.norm((W - W[-1:])[-1]) < 1e-6)
    # print(torch.norm(B[-1] - stop.flatten())) # why less exact?
    B = torch.reshape(B, (n_step, *start.shape[1:]))

    assert torch.norm(B[0] - start) < 1e-6
    assert torch.norm(B[-1] - stop) < 1e-5  # why less exact?
    return B


def get_angles_dist_accel(x):
    """
    under dev


    in:  B  T X Y
    out: theta B T-2
         dist  B T-1
         accel B T-2 X Y

    """
    B, T, X, Y = x.shape

    x = x.view((B, T, -1))
    v = x[:, 0:T-1] - x[:, 1:T]
    d = torch.norm(v, dim=2, keepdim=True)
    v_hat = torch.div(v, d)

    theta = torch.empty((B, T-2))
    accel = torch.empty((B, T-2, X * Y))

    for t in range(T-2):
        theta[:, t] = torch.acos(
            torch.bmm(v_hat[:, t].view((B, 1, X * Y)),
                      v_hat[:, t+1].view((B, X * Y, 1))).squeeze()
        ) / torch.tensor(np.pi)
        accel[:, t] = v_hat[:, t] - torch.bmm(v_hat[:, t].view((B, 1, -1)),
                                              v_hat[:, t+1].view((B, -1, 1))
                                              )[..., 0] * v_hat[:, t]

    accel_hat = torch.div(accel, torch.norm(accel, dim=2, keepdim=True))
    accel_hat = accel_hat.view((B, T-2, X, Y))

    return theta, d, accel_hat


def Haar_1d(x, n_scales=None):
    """
    tool for multiscale geodesic

    in: B 1 T X Y
    Haar decomposition along T axis

    todo:
    work with [T,C,H,W]
    use functionals, to avoid parameters
    """

    if n_scales is None:
        n_scales = int(np.log2(x.shape[-3]))

    diff = nn.Conv3d(1, 1, (2, 1, 1), bias=False)
    blur = nn.Conv3d(1, 1, (2, 1, 1), bias=False, stride=(2, 1, 1))

    diff.weight = nn.Parameter(torch.ones_like(diff.weight))
    diff.weight.select(2, 0).mul_(-1)  # padding = 1 and pop ?
    blur.weight = nn.Parameter(torch.ones_like(diff.weight))

    y = []
    for s in range(n_scales):
        #         print(s, x.shape)
        y.append(diff(x))
        x = blur(x)

    return y
