import torch
from torch import Tensor
from typing import Tuple
from .validate import validate_input


def make_straight_line(start: Tensor, stop: Tensor, n_steps: int) -> Tensor:
    """make a straight line between `start` and `stop` with `n_steps` transitions.

    Parameters
    ----------
    start, stop
        Images of shape (1, channel, height, width), the anchor points between
        which a line
        will be made.
    n_steps
        Number of steps (i.e., transitions) to create between the two anchor
        points. Must be positive.

    Returns
    -------
    straight
        Tensor of shape (n_steps+1, channel, height, width)

    """
    validate_input(start, no_batch=True)
    validate_input(stop, no_batch=True)
    if start.shape != stop.shape:
        raise ValueError(f"start and stop must be same shape, but got {start.shape} and {stop.shape}!")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, but got {n_steps}")
    shape = start.shape[1:]

    device = start.device
    start = start.reshape(1, -1)
    stop = stop.reshape(1, -1)
    tt = torch.linspace(0, 1, steps=n_steps+1, device=device
                        ).view(n_steps+1, 1)
    straight = (1 - tt) * start + tt * stop

    return straight.reshape((n_steps+1, *shape))


def sample_brownian_bridge(start: Tensor, stop: Tensor,
                           n_steps: int, max_norm: float = 1) -> Tensor:
    """Sample a brownian bridge between `start` and `stop` made up of `n_steps`

    Parameters
    ----------
    start, stop
        signal of shape (1, channel, height, width), the anchor points between
        which a random path will be sampled (like pylons on which the bridge
        will rest)
    n_steps
        number of steps on the bridge
    max_norm
        controls variability of the bridge by setting how far (in l2 norm)
        it veers from the straight line interpolation at the midpoint between
        pylons. each component of the bridge will reach a maximal variability
        with std = max_norm / sqrt(d), where d is the dimension of the signal.
        (ie. d = C*H*W). Must be non-negative.

    Returns
    -------
    bridge
        sequence of shape (n_steps+1, channel, height, width) a brownian bridge
        across the two pylons

    """
    validate_input(start, no_batch=True)
    validate_input(stop, no_batch=True)
    if start.shape != stop.shape:
        raise ValueError(f"start and stop must be same shape, but got {start.shape} and {stop.shape}!")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, but got {n_steps}")
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative but got {max_norm}!")
    shape = start.shape[1:]

    device = start.device
    start = start.reshape(1, -1)
    stop = stop.reshape(1, -1)
    D = start.shape[1]
    dt = torch.tensor(1/n_steps)
    tt = torch.linspace(0, 1, steps=n_steps+1, device=device)[:, None]

    sigma = torch.sqrt(dt / D) * 2. * max_norm
    dW = sigma * torch.randn(n_steps+1, D, device=device)
    dW[0] = start.flatten()
    W = torch.cumsum(dW, dim=0)

    bridge = W - tt * (W[-1:] - stop)

    return bridge.reshape((n_steps+1, *shape))


def deviation_from_line(sequence: Tensor,
                        normalize: bool = True) -> Tuple[Tensor, Tensor]:
    """Compute the deviation of `sequence` to the straight line between its endpoints.

    Project each point of the path `sequence` onto the line defined by
    the anchor points, and measure the two sides of a right triangle:
    - from the projected point to the first anchor point
      (aka. distance along line)
    - from the projected point to the corresponding point on the path `sequence`
      (aka. distance from line).

    Parameters
    ----------
    sequence
        sequence of signals of shape (T, channel, height, width)
    normalize
        use the distance between the anchor points as a unit of measurement

    Returns
    -------
    dist_along_line
        sequence of T euclidian distances along the line
    dist_from_line
        sequence of T euclidian distances to the line

    """
    validate_input(sequence)
    y = sequence.reshape(sequence.shape[0], -1)
    T, D = y.shape
    y0 = y[0].view(1, D)
    y1 = y[-1].view(1, D)

    line = (y1 - y0)
    line_length = torch.linalg.vector_norm(line, ord=2)
    line = line / line_length
    y_centered = y - y0
    dist_along_line = y_centered @ line[0]
    projection = dist_along_line.view(T, 1) * line
    dist_from_line = torch.linalg.vector_norm(y_centered - projection, dim=1,
                                              ord=2)

    if normalize:
        dist_along_line /= line_length
        dist_from_line /= line_length

    return dist_along_line, dist_from_line


def translation_sequence(image: Tensor, n_steps: int = 10) -> Tensor:
    """make a horizontal translation sequence on `image`

    Parameters
    ----------
    image
        Base image of shape, (1, channel, height, width)
    n_steps
        Number of steps in the sequence. The length of the sequence is n_steps
        + 1. Must be positive.

    Returns
    -------
    sequence
        translation sequence of shape (n_steps+1, channel, height, width)

    """
    validate_input(image, no_batch=True)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, but got {n_steps}")
    sequence = torch.empty(n_steps+1, *image.shape[1:]).to(image.device)

    for shift in range(n_steps+1):
        sequence[shift] = torch.roll(image, shift, [-1])

    return sequence
