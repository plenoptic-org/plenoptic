import torch


def make_straight_line(start, stop, n_steps):
    """make a straight line between `start` and `stop`
    made up of `n_steps`

    Parameters
    ----------
    start, stop: torch.Tensor
        signal of shape [1, D], the anchor points between which
        a line will be made
    n_steps : int
        number of steps

    Returns
    -------
    straight : torch.Tensor
        sequence of shape [n_steps+1, D]
    """
    assert start.shape == stop.shape
    assert start.shape[0] == 1

    device = start.device
    tt = torch.linspace(0, 1, steps=n_steps+1, device=device
                        ).view(n_steps+1, 1)
    straight = (1 - tt) * start + tt * stop

    return straight


def sample_brownian_bridge(start, stop, n_steps, max_norm=1):
    """Sample a brownian bridge between `start` and `stop` made up of `n_steps`

    Parameters
    ----------
    start, stop: torch.Tensor
        signal of shape [1, D], the anchor points between which
        a random path will be sampled (like pylons on which
        the bridge will rest)
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
        sequence of shape [n_steps+1, D]
        a brownian bridge accros the two pylons
    """
    assert start.shape == stop.shape
    assert start.shape[0] == 1
    assert max_norm >= 0

    device = start.device
    D = start.shape[1]
    dt = torch.tensor(1/n_steps)
    tt = torch.linspace(0, 1, steps=n_steps+1, device=device)[:, None]

    sigma = torch.sqrt(dt / D) * 2. * max_norm
    dW = sigma * torch.randn(n_steps+1, D, device=device)
    dW[0] = start.flatten()
    W = torch.cumsum(dW, dim=0)

    B = W - tt * (W[-1:] - stop)

    return B


def deviation_from_line(y, y0=None, y1=None, normalize=True):
    """Compute the deviation of `y` to the straight line between `y0` and `y1`
    for visualization purposes.

    Project each point of the path `y` onto the line defined by
    the anchor points, and measure the two sides of a right triangle:
    - from the projected point to the first anchor point
      (aka. distance along line)
    - from the projected point to the corresponding point on the path `y`
      (aka. distance from line).

    Parameters
    ----------
    y: torch.FloatTensor
        sequence of signals of shape [T, D]
    y0, y1: torch.Tensor or None, optional
        signal of shape [1, D], the points that define the line. If None,
        the endpoints of y.
    normalize: bool, optional
        use the distance between the anchor points as a unit of measurement

    Returns
    -------
    dist_along_line: torch.FloatTensor
        sequence of T euclidian distances along the line
    dist_from_line: torch.FloatTensor
        sequence of T euclidian distances to the line

    """
    T, D = y.shape
    if y0 is None:
        y0 = y[0].view(1, D)
    if y1 is None:
        y1 = y[-1].view(1, D)
    assert y0.shape == y1.shape

    line = (y1 - y0)
    line_length = torch.norm(line)
    line = line / line_length
    y_centered = y - y0
    dist_along_line = y_centered @ line[0]
    projection = dist_along_line.view(T, 1) * line
    dist_from_line = torch.norm(y_centered - projection, dim=1)

    if normalize:
        dist_along_line /= line_length
        dist_from_line /= line_length

    return dist_along_line, dist_from_line


def translation_sequence(image, n_steps=10):
    """make a horizontal translation sequence on `image`

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
