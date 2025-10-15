"""
Some useful non-linearities for visual models.

The functions operate on dictionaries or tensors.
"""

import torch

from ...tools import signal
from ...tools.conv import blur_downsample, upsample_blur


def rectangular_to_polar_dict(
    coeff_dict: dict,
    residuals: bool = False,
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
        >>> spyr = po.simul.SteerablePyramidFreq(
        ...     img.shape[-2:], is_complex=True, height=3
        ... )
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
        if not isinstance(key, str) or not key.startswith("residual"):
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
        >>> import numpy as np
        >>> import torch
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(
        ...     img.shape[-2:], is_complex=True, height=3
        ... )
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.rectangular_to_polar_dict(
        ...     coeffs, residuals=True
        ... )
        >>> coeffs_back = po.simul.non_linearities.polar_to_rectangular_dict(
        ...     energy, state
        ... )
        >>> all(torch.allclose(coeffs[key], coeffs_back[key]) for key in coeffs)
        True
        >>> po.pyrshow(coeffs_back)
        <PyrFigure size ...>
    """
    coeff_dict = {}

    for key in energy:
        # ignore residuals here
        if not isinstance(key, str) or not key.startswith("residual"):
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
        Tensor of shape (batch, channel, height, width) or
        (batch, channel, angle, height, width).
    epsilon
        Small constant to avoid division by zero.

    Returns
    -------
    norm
        The local energy of ``x``, shape (batch, channel, height/2,
        width/2) or (batch, channel, angle, height/2, width/2), depending on
        dimensionality of ``x``.
    direction
        The local phase of ``x`` (a.k.a. local unit vector, or local
        state), shape (batch, channel, height, width) or (batch, channel,
        angle, height, width), depending on dimensionality of ``x``.

    Raises
    ------
    ValueError
        If ``x`` does not have 4 or 5 dimensions.

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

    def _local_gain_control(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gain control in helper function we can vmap."""  # noqa: DOC201
        # numpydoc ignore=ES01,PR01,RT01
        norm = blur_downsample(torch.abs(x**p)).pow(1 / p)
        odd = torch.as_tensor(x.shape)[-2:] % 2
        direction = x / (upsample_blur(norm, odd) + epsilon)
        return norm, direction

    if x.ndim == 5:
        func = torch.vmap(_local_gain_control, in_dims=2, out_dims=2)
    elif x.ndim == 4:
        func = _local_gain_control
    else:
        raise ValueError("Tensor must have 4 or 5 dimensions!")

    return func(x)


def local_gain_release(
    norm: torch.Tensor,
    direction: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Spatially local gain release.

    Convert the local energy and phase to a single real-valued tensor.

    Parameters
    ----------
    norm
        The local energy of a tensor, with shape (batch, channel, height/2,
        width/2) or (batch, channel, angle, height/2, width/2).
    direction
        The local phase of a tensor (a.k.a. local unit vector, or local state),
        with shape (batch, channel, height, width) or (batch, channel,
        angle, height, width).
    epsilon
        Small constant to avoid division by zero.

    Returns
    -------
    x
        Tensor of shape (batch, channel, height, width) or (batch, channel,
        angle, height, width), depending on input tensor dimensionality.

    Raises
    ------
    ValueError
        If input tensors do not have 4 or 5 dimensions.

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
        ...     [img, x, img - x],
        ...     title=["Original image", "Gain release output", "Difference"],
        ... )
        <PyrFigure size ...>
    """

    def _local_gain_release(
        direction: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute gain release in helper function we can vmap."""  # noqa: DOC201
        # numpydoc ignore=ES01,PR01,RT01
        odd = torch.as_tensor(direction.shape)[-2:] % 2
        return direction * (upsample_blur(norm, odd) + epsilon)

    if direction.ndim == 5:
        func = torch.vmap(_local_gain_release, in_dims=2, out_dims=2)
    elif direction.ndim == 4:
        func = _local_gain_release
    else:
        raise ValueError("Tensor must have 4 or 5 dimensions!")

    return func(direction, norm)


def local_gain_control_dict(
    coeff_dict: dict,
    residuals: bool = True,
) -> tuple[dict, dict]:
    """
    Spatially local gain control, for each element in a dictionary.

    For more details, see :func:`local_gain_control`.

    Parameters
    ----------
    coeff_dict
        A dictionary containing tensors of shape (batch, channel, height, width)
        or (batch, channel, angle, height, width).
    residuals
        An option to carry around residuals in the energy dict.
        Note that the transformation is not applied to the residuals,
        that is dictionary elements with a key starting in "residual".

    Returns
    -------
    energy
        The dictionary of :class:`torch.Tensor` containing the local energy of
        ``x``. Tensor shapes match those found in ``coeff_dict``.
    state
        The dictionary of :class:`torch.Tensor` containing the local phase of
        ``x``. Tensor shapes match those found in ``coeff_dict``.

    Raises
    ------
    ValueError
        If the tensors contained within ``coeff_dict`` do not have 4 or 5 dimensions.

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
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], height=3)
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
        if not isinstance(key, str) or not key.startswith("residual"):
            energy[key], state[key] = local_gain_control(coeff_dict[key])

    if residuals:
        energy["residual_lowpass"] = coeff_dict["residual_lowpass"]
        energy["residual_highpass"] = coeff_dict["residual_highpass"]

    return energy, state


def local_gain_release_dict(
    energy: dict,
    state: dict,
    residuals: bool = True,
) -> dict:
    """
    Spatially local gain release, for each element in a dictionary.

    For more details, see :func:`local_gain_release`.

    Parameters
    ----------
    energy
        The dictionary of :class:`torch.Tensor` containing the local energy of
        ``x``, with shape (batch, channel, height, width) or (batch, channel,
        angle, height, width).
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
        A dictionary containing the "gain released" tensors, with shapes matching
        those found in ``energy``.

    Raises
    ------
    ValueError
        If the tensors contained within ``energy`` and ``state`` do not have 4 or 5
        dimensions.

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
        >>> import torch
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], height=3)
        >>> coeffs = spyr(img)
        >>> energy, state = po.simul.non_linearities.local_gain_control_dict(coeffs)
        >>> coeffs_dict = po.simul.non_linearities.local_gain_release_dict(
        ...     energy, state
        ... )
        >>> all([torch.allclose(coeffs[k], coeffs_dict[k]) for k in coeffs.keys()])
        True
        >>> po.pyrshow(coeffs_dict)
        <PyrFigure size ...>
    """
    coeff_dict = {}

    for key in energy:
        if not isinstance(key, str) or not key.startswith("residual"):
            coeff_dict[key] = local_gain_release(energy[key], state[key])

    if residuals:
        coeff_dict["residual_lowpass"] = energy["residual_lowpass"]
        coeff_dict["residual_highpass"] = energy["residual_highpass"]

    return coeff_dict
