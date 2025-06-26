"""
Some useful non-linearities for visual models.

The functions operate on dictionaries or tensors.
"""

import torch

from ...tools import signal
from ...tools.conv import blur_downsample, upsample_blur


def rectangular_to_polar_dict(
    coeff_dict: dict, residuals: bool = False
) -> tuple[dict, dict]:
    """
    Return the complex modulus and the phase of each complex tensor in a dictionary.

    Keys are preserved, with the option of dropping ``"residual_lowpass"`` and
    ``"residual_highpass"`` by setting ``residuals=False``.

    Parameters
    ----------
    coeff_dict
        A dictionary containing complex tensors.
    residuals
        An option to include residuals in the returned ``energy`` dict.

    Returns
    -------
    energy
        The dictionary of :class:`torch.Tensor` containing the local complex
        modulus of ``coeff_dict``.
    state
        The dictionary of :class:`torch.Tensor` containing the local phase of
        ``coeff_dict``.

    See Also
    --------
    :func:`~plenoptic.tools.signal.rectangular_to_polar`
        Same operation on tensors.
    polar_to_rectangular_dict
        The inverse operation.
    local_gain_control_dict
        The analogous function for complex-valued signals.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], is_complex=True)
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.rectangular_to_polar_dict(coeffs)
        >>> po.pyrshow(energy)
        <PyrFigure size ...>
        >>> po.pyrshow(state)
        <PyrFigure size ...>
    """
    energy = {}
    state = {}
    for key in coeff_dict:
        # ignore residuals
        if isinstance(key, tuple) or not key.startswith("residual"):
            energy[key], state[key] = signal.rectangular_to_polar(coeff_dict[key])

    if residuals:
        energy["residual_lowpass"] = coeff_dict["residual_lowpass"]
        energy["residual_highpass"] = coeff_dict["residual_highpass"]

    return energy, state


def polar_to_rectangular_dict(
    energy: dict,
    state: dict,
) -> dict:
    """
    Return the real and imaginary parts of tensor in a dictionary.

    Keys in the output are identical to those in the input. Will grab residuals from
    ``energy``, if present, with keys ``"residual_highpass"`` and
    ``"residual_lowpass"``.

    Parameters
    ----------
    energy
        The dictionary of :class:`torch.Tensor` containing the local complex
        modulus.
    state
        The dictionary of :class:`torch.Tensor` containing the local phase.

    Returns
    -------
    coeff_dict
       A dictionary containing complex tensors of coefficients.

    See Also
    --------
    :func:`~plenoptic.tools.signal.polar_to_rectangular`
        Same operation on tensors.
    rectangular_to_polar_dict
        The inverse operation.
    local_gain_release_dict
        The analogous function for real-valued signals.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], is_complex=True)
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.rectangular_to_polar_dict(coeffs)
        >>> coeffs = po.simul.non_linearities.polar_to_rectangular_dict(energy, state)
        >>> po.pyrshow(coeffs)
        <PyrFigure size ...>
    """
    coeff_dict = {}
    for key in energy:
        # ignore residuals here
        if isinstance(key, tuple) or not key.startswith("residual"):
            coeff_dict[key] = signal.polar_to_rectangular(energy[key], state[key])

    if "residual_lowpass" in energy:
        coeff_dict["residual_lowpass"] = energy["residual_lowpass"]
        coeff_dict["residual_highpass"] = energy["residual_highpass"]

    return coeff_dict


def local_gain_control(
    x: torch.Tensor, epsilon: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spatially local gain control.

    Compute the local energy and phase of a real-valued tensor.

    Parameters
    ----------
    x
        Tensor of shape (batch, channel, height, width).
    epsilon
        Small constant to avoid division by zero.

    Returns
    -------
    norm
        The local energy of ``x``, shape (batch, channel, height/2,
        width/2).
    direction
        The local phase of ``x`` (a.k.a. local unit vector, or local
        state), shape (batch, channel, height, width).

    See Also
    --------
    local_gain_control_dict
        Same operation on dictionaries.
    local_gain_release
        The inverse operation.
    :func:`~plenoptic.tools.signal.rectangular_to_polar`
        The analogous function for complex-valued signals.

    Notes
    -----
    Norm and direction (analogous to complex modulus and phase) are defined using
    blurring operator and division. Indeed blurring the responses removes high
    frequencies introduced by the squaring operation. In the complex case adding the
    quadrature pair response has the same effect (note that this is most clearly seen in
    the frequency domain). Here computing the direction (phase) reduces to dividing out
    the norm (modulus), indeed the signal only has one real component. This is a
    normalization operation (local unit vector), hence the connection to local gain
    control.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> norm, direction = po.simul.non_linearities.local_gain_control(img)
        >>> po.imshow([img, norm, direction], title=["image", "norm", "direction"])
        <PyrFigure size ...>
    """
    # these could be parameters, but no use case so far
    p = 2.0

    norm = blur_downsample(torch.abs(x**p)).pow(1 / p)
    odd = torch.as_tensor(x.shape)[2:4] % 2
    direction = x / (upsample_blur(norm, odd) + epsilon)

    return norm, direction


def local_gain_release(
    norm: torch.Tensor, direction: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Spatially local gain release.

    Convert the local energy and phase to a single real-valued tensor.

    Parameters
    ----------
    norm
        The local energy of a tensor, with shape (batch, channel, height/2,
        width/2).
    direction
        The local phase of a tensor (a.k.a. local unit vector, or local state),
        with shape (batch, channel, height, width).
    epsilon
        Small constant to avoid division by zero.

    Returns
    -------
    x
        Tensor of shape (batch, channel, height, width).

    See Also
    --------
    local_gain_release_dict
        Same operation on dictionaries.
    local_gain_control
        The inverse operation.
    :func:`~plenoptic.tools.signal.polar_to_rectangular`
        The analogous function for complex-valued signals.

    Notes
    -----
    Norm and direction (analogous to complex modulus and phase) are defined using
    blurring operator and division. Indeed blurring the responses removes high
    frequencies introduced by the squaring operation. In the complex case adding the
    quadrature pair response has the same effect (note that this is most clearly seen in
    the frequency domain). Here computing the direction (phase) reduces to dividing out
    the norm (modulus), indeed the signal only has one real component. This is a
    normalization operation (local unit vector), hence the connection to local gain
    control.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> norm, direction = po.simul.non_linearities.local_gain_control(img)
        >>> x = po.simul.non_linearities.local_gain_release(norm, direction)
        >>> po.imshow(
        ...     [img, norm, direction, x], title=["image", "norm", "direction", "x"]
        ... )
        <PyrFigure size ...>
    """
    odd = torch.as_tensor(direction.shape)[2:4] % 2
    x = direction * (upsample_blur(norm, odd) + epsilon)
    return x


def local_gain_control_dict(
    coeff_dict: dict, residuals: bool = True
) -> tuple[dict, dict]:
    """
    Spatially local gain control, for each element in a dictionary.

    For more details, see :func:`local_gain_control`.

    Parameters
    ----------
    coeff_dict
        A dictionary containing tensors of shape (batch, channel, height, width).
    residuals
        An option to carry around residuals in the energy dict.
        Note that the transformation is not applied to the residuals,
        that is dictionary elements with a key starting in "residual".

    Returns
    -------
    energy
        The dictionary of :class:`torch.Tensor` containing the local energy of
        ``x``.
    state
        The dictionary of :class:`torch.Tensor` containing the local phase of
        ``x``.

    See Also
    --------
    local_gain_control
        Same operation on tensors.
    local_gain_release_dict
        The inverse operation.
    rectangular_to_polar_dict
        The analogous function for complex-valued signals.

    Notes
    -----
    Note that energy and state are not computed on the residuals.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:])
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.local_gain_control_dict(coeffs)
        >>> po.pyrshow(energy)
        <PyrFigure size ...>
        >>> po.pyrshow(state)
        <PyrFigure size ...>
    """
    energy = {}
    state = {}

    for key in coeff_dict:
        if isinstance(key, tuple) or not key.startswith("residual"):
            energy[key], state[key] = local_gain_control(coeff_dict[key])

    if residuals:
        energy["residual_lowpass"] = coeff_dict["residual_lowpass"]
        energy["residual_highpass"] = coeff_dict["residual_highpass"]

    return energy, state


def local_gain_release_dict(energy: dict, state: dict, residuals: bool = True) -> dict:
    """
    Spatially local gain release, for each element in a dictionary.

    For more details, see :func:`local_gain_release`.

    Parameters
    ----------
    energy
        The dictionary of :class:`torch.Tensor` containing the local energy of
        ``x``.
    state
        The dictionary of :class:`torch.Tensor` containing the local phase of
        ``x``.
    residuals
        An option to carry around residuals in the energy dict.
        Note that the transformation is not applied to the residuals,
        that is dictionary elements with a key starting in "residual".

    Returns
    -------
    coeff_dict
        A dictionary containing tensors of shape (batch, channel, height, width).

    See Also
    --------
    local_gain_release
        Same operation on tensors.
    local_gain_control_dict
        The inverse operation.
    polar_to_rectangular_dict
        The analogous function for complex-valued signals.

    Examples
    --------
    .. plot::

        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:])
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.local_gain_control_dict(coeffs)
        >>> coeffs_dict = po.simul.non_linearities.local_gain_release_dict(
        ...     energy, state
        ... )
        >>> po.pyrshow(coeffs_dict)
        <PyrFigure size ...>
    """
    coeff_dict = {}

    for key in energy:
        if isinstance(key, tuple) or not key.startswith("residual"):
            coeff_dict[key] = local_gain_release(energy[key], state[key])

    if residuals:
        coeff_dict["residual_lowpass"] = energy["residual_lowpass"]
        coeff_dict["residual_highpass"] = energy["residual_highpass"]

    return coeff_dict
