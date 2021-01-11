import torch


def make_straight_line(start, stop, n_steps):
    """
    Parameters
    ----------
    start (resp. stop ): torch.Tensor
        signal of shape [1, C, H, W]
        the pylons between which to draw the line
    n_steps : int
        number of steps

    Returns
    -------
    straight : torch.Tensor
        sequence of shape [n_steps+1, C, H, W]
    """
    assert start.shape == stop.shape
    tt = torch.linspace(0, 1, steps=n_steps+1).view(n_steps+1, 1, 1, 1)
    straight = (1 - tt) * start + tt * stop

    return straight


def sample_brownian_bridge(start, stop, n_steps, max_norm=1):
    """
    Parameters
    ----------
    start (resp. stop): torch.Tensor
        signal of shape [1, C, H, W]
        the pylons on which to build the bridge
    n_steps: int
        number of steps on the bridge
    max_norm: float
        controls variability of the bridge by setting how far (in l2 norm)
        it veers from the straight line interpolation at the midpoint
        between pylons.
        each component of the bridge will reach a maximal variability with
        std = max_norm / sqrt(d), where d is the dimension of the signal.
        (ie. d = C*H*W).

    Returns
    -------
    B: torch.Tensor
        sequence of shape [n_steps+1, C, H, W]
        a brownian bridge accros the two pylons
    """
    assert start.shape == stop.shape
    assert max_norm >= 0

    d = start.numel()
    dt = torch.tensor(1/n_steps)
    tt = torch.linspace(0, 1, steps=n_steps+1)[:, None]

    sigma = torch.sqrt(dt / d) * 2. * max_norm
    dW = sigma * torch.randn(n_steps+1, d)
    dW[0] = start.flatten()
    W = torch.cumsum(dW, axis=0)

    B = W - tt * (W[-1] - stop.flatten())[None, :]
    B = torch.reshape(B, (n_steps+1, *start.shape[1:]))
    return B


def distance_from_line(y, y0=None, y1=None):
    """Compute the l2 distance of each y to its projection onto the
    straight line going accross the pylons y0 to y1.

    Parameters
    ----------
    y: torch.FloatTensor
        sequence of signals of shape [T, C, H, W]
    y0 (resp. y1): torch.Tensor, optional
        signal of shape [1, C, H, W] set by default to the endpoints of y

    Returns
    -------
    dist: torch.FloatTensor
        sequence of T euclidian distances to the line
    """
    if y0 is None:
        y0 = y[0:1]
    if y1 is None:
        y1 = y[-1:]
    assert y0.shape == y1.shape
    assert y0.shape == y[0:1].shape

    line = make_straight_line(y0, y1, len(y)-1)
    dist = (y - line).pow(2).sum(dim=(1, 2, 3)).pow(.5)

    return dist


def translation_sequence(image, n_steps=10):
    """horizontal translation

    Parameters
    ---------
    image: torch.Tensor
        Base image of shape, [C, H, W]
    n_steps: int, optional
        Number of steps in the sequence. Defaults to 10.
        The length of the sequence is n_steps + 1

    Returns
    -------
    sequence: torch.Tensor
        [T, C, H, W], with T = n_steps + 1

    """
    sequence = torch.empty(n_steps+1, *image.shape)

    for shift in range(n_steps+1):
        sequence[shift] = torch.roll(image, shift, [2])

    return sequence
