"""
Steerable frequency pyramid.

Construct a steerable pyramid on matrix two dimensional signals, in the
Fourier domain.
"""  # numpydoc ignore=EX01

import warnings
from collections import OrderedDict
from typing import Literal

import einops
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from numpy.typing import NDArray
from scipy.special import factorial
from torch import Tensor

from ...tools.signal import _interpolate1d, _raised_cosine, _steer

complex_types = [torch.cdouble, torch.cfloat]

SCALES_TYPE = int | Literal["residual_lowpass", "residual_highpass"]


class SteerablePyramidFreq(nn.Module):
    r"""
    Steerable frequency pyramid in Torch.

    Construct a steerable pyramid on matrix two dimensional signals, in the Fourier
    domain. Boundary-handling is circular. Reconstruction is exact (within floating
    point errors). However, if the image has an odd-shape, the reconstruction will not
    be exact due to boundary-handling issues that have not been resolved. Similarly,
    a complex pyramid of order=0 has non-exact reconstruction and cannot be tight-frame.

    The squared radial functions tile the Fourier plane with a raised-cosine
    falloff. Angular functions are

    .. math::

       \cos\left(\frac{\theta-k*\pi}{o+1}\right)^o

    where :math:`o` is the order parameter set at initialization and :math:`k` runs from
    0 to :math:`o` for a total of :math:`o+1` orientations.

    Parameters
    ----------
    image_shape
        Shape of input image.
    height
        The height of the pyramid. If ``'auto'``, will automatically determine based on
        the size of ``image``. If an ``int``, must be non-negative and less than
        ``log2(min(image_shape[1], image_shape[1]))-2``. If ``height=0``, this only
        returns the residuals.
    order
        The Gaussian derivative order used for the steerable filters, in ``[0, 15]``.
        Note that to achieve steerability the minimum number of orientation is
        ``order + 1``, which is used here. To get more orientations at the same order,
        use the method :meth:`steer_coeffs`.
    twidth
        The width of the transition region of the radial lowpass function, in
        octaves.
    is_complex
        Whether the pyramid coefficients should be complex or not. If ``True``, the real
        and imaginary parts correspond to a pair of odd and even symmetric filters. If
        ``False``, the coefficients only include the real part. Regardless of the value
        of ``is_complex``, the symmetry of the real part is determined by the ``order``
        parameter: if ``order`` is even, then the real coefficients are even symmetric;
        if ``order`` is odd, then the real coefficients are odd symmetric. (If
        ``is_complex=True``, then the imaginary coefficients will have the opposite
        symmetry of the real ones).
    downsample
        Whether to downsample each scale in the pyramid or keep the output
        pyramid coefficients in fixed bands of size ``image_shape``. When
        downsample is ``False``, the forward method returns a tensor.
    tight_frame
        Whether the pyramid obeys the generalized parseval theorem or not (i.e.
        is a tight frame). If ``True``, the energy of the pyr_coeffs equals the energy
        of the image. In order to match the `matlabPyrTools
        <http://github.com/labForComputationalVision/matlabpyrtools>`_ or `pyrtools
        <https://github.com/labForComputationalVision/pyrtools>`_ implementations, this
        must be set to ``False``.

    Attributes
    ----------
    image_shape : tuple
        Shape of input image.
    pyr_size : OrderedDict
        Dictionary containing the height and width of the pyramid coefficients. Keys are
        the same as those in ``pyr_coeffs`` returned by :meth:`forward`, in order:
        ``"residual_highpass"``, the integers from ``0`` to (the initialization
        argument) ``order``, and ``"residual_lowpass"``. The values are 2-tuples of
        ints. While the dictionary is initialized with the object, the values are not
        set until the first time :meth:`forward` is called.
    fft_norm : str
        The way the ffts are normalized, see :func:`torch.fft.fft2` for more details.
    is_complex : bool
        Whether the coefficients are complex- or real-valued.
    scales : list
        All the scales of the representation (including residuals) in coarse-to-fine
        order. A subset of this list can be passed to the :meth:`forward` method to
        restrict the output.

    Raises
    ------
    ValueError
        If ``image_shape`` contains non-integers.
    ValueError
        If ``len(image_shape) != 2`` .
    ValueError
        If ``height`` is not a non-negative integer or is larger than the biggest
        possible value (determined by ``image_shape``).
    ValueError
        If ``order`` not an integer in ``[0, 15]``.
    ValueError
        If ``order == 0`` and ``is_complex is False``. See
        https://github.com/plenoptic-org/plenoptic/issues/326 for an explanation
    ValueError
        If ``twidth`` not positive.

    Warns
    -----
    UserWarning
        If ``image_shape`` has an odd value, because then reconstruction will be
        imperfect.

    Notes
    -----
    Transform described in Simoncelli and Freeman, 1995 [1]_, filter kernel design
    described in Karasaridis and Smoncelli, 1996 [2]_. For further information see
    online [3]_.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible
       Architecture for Multi-Scale Derivative Computation," Second Int'l Conf
       on Image Processing, Washington, DC, Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for
       Steerable Pyramid Image Transforms", ICASSP, Atlanta, GA, May 1996. ..
    .. [3] `<https://www.cns.nyu.edu/~eero/steerpyr/>`_

    Examples
    --------
    >>> import plenoptic as po
    >>> spyr = po.simul.SteerablePyramidFreq((256, 256))
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        height: Literal["auto"] | int = "auto",
        order: int = 3,
        twidth: int = 1,
        is_complex: bool = False,
        downsample: bool = True,
        tight_frame: bool = False,
    ):
        super().__init__()

        self.order = order
        # complex_const comes from the Fourier transform of a gaussian derivative.
        self._complex_const_forward = np.power(complex(0, -1), self.order)
        self._complex_const_recon = np.power(complex(0, 1), self.order)
        try:
            self.image_shape = tuple([int(i) for i in image_shape])
        except ValueError:
            raise ValueError(
                f"image_shape must be castable to ints, but got {image_shape}!"
            )
        if self.image_shape != tuple(image_shape):
            raise ValueError(
                f"image_shape must be castable to ints, but got {image_shape}!"
            )
        if len(self.image_shape) != 2:
            raise ValueError(
                f"image_shape must be a tuple of length 2, but got {self.image_shape}!"
            )

        if (self.image_shape[0] % 2 != 0) or (self.image_shape[1] % 2 != 0):
            warnings.warn("Reconstruction will not be perfect with odd-sized images")

        self.is_complex = is_complex
        self.downsample = downsample
        self.tight_frame = tight_frame
        if self.tight_frame:
            self.fft_norm = "ortho"
        else:
            self.fft_norm = "backward"
        # cache constants
        self.lutsize = 1024
        self.Xcosn = (
            np.pi
            * np.array(range(-(2 * self.lutsize + 1), (self.lutsize + 2)))
            / self.lutsize
        )
        self.alpha = (self.Xcosn + np.pi) % (2 * np.pi) - np.pi

        max_ht = np.floor(np.log2(min(self.image_shape[0], self.image_shape[1]))) - 2
        if height == "auto":
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise ValueError(f"Cannot build pyramid higher than {max_ht:.0f} levels.")
        elif height < 0:
            raise ValueError("Height must be a non-negative integer.")
        else:
            self.num_scales = int(height)

        if self.order > 15 or self.order < 0:
            raise ValueError("order must be an integer in the range [0, 15].")
        if self.order == 0 and self.is_complex:
            raise ValueError(
                "Complex pyramid cannot have order=0! See "
                "https://github.com/plenoptic-org/plenoptic/issues/326 "
                "for an explanation."
            )
        self.num_orientations = int(self.order + 1)

        if twidth <= 0:
            raise ValueError("twidth must be positive.")
        twidth = int(twidth)

        dims = np.array(self.image_shape)

        # make a grid for the raised cosine interpolation
        ctr = np.ceil((np.array(dims) + 0.5) / 2).astype(int)

        (xramp, yramp) = np.meshgrid(
            np.linspace(-1, 1, dims[1] + 1)[:-1],
            np.linspace(-1, 1, dims[0] + 1)[:-1],
            indexing="xy",
        )

        self.angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
        self.log_rad = np.log2(log_rad)

        # radial transition function (a raised cosine in log-frequency):
        self.Xrcos, Yrcos = _raised_cosine(twidth, (-twidth / 2.0), np.array([0, 1]))
        self.Yrcos = np.sqrt(Yrcos)

        self.YIrcos = np.sqrt(1.0 - self.Yrcos**2)

        # create low and high masks
        lo0mask = _interpolate1d(self.log_rad, self.YIrcos, self.Xrcos)
        hi0mask = _interpolate1d(self.log_rad, self.Yrcos, self.Xrcos)
        self.register_buffer(
            "lo0mask", einops.rearrange(torch.as_tensor(lo0mask), "h w -> 1 1 1 h w")
        )
        self.register_buffer("hi0mask", torch.as_tensor(hi0mask).unsqueeze(0))

        # need a mock image to down-sample so that we correctly
        # construct the differently-sized masks
        mock_image = np.random.rand(*self.image_shape)
        imdft = np.fft.fftshift(np.fft.fft2(mock_image))
        lodft = imdft * lo0mask

        # this list, used by coarse-to-fine optimization, gives all the
        # scales (including residuals) from coarse to fine
        self.scales = (
            ["residual_lowpass"]
            + list(range(self.num_scales))[::-1]
            + ["residual_highpass"]
        )
        self.pyr_size = OrderedDict({k: () for k in self.scales[::-1]})

        # we create these copies because they will be modified in the
        # following loops
        Xrcos = self.Xrcos.copy()
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        # pre-generate the angle, hi and lo masks, as well as the
        # indices used for down-sampling.
        self._loindices = []
        for i in range(self.num_scales):
            Xrcos -= np.log2(2)
            const = (
                (2 ** (2 * self.order))
                * (factorial(self.order, exact=True) ** 2)
                / float(self.num_orientations * factorial(2 * self.order, exact=True))
            )

            if self.is_complex:
                Ycosn_forward = (
                    2.0
                    * np.sqrt(const)
                    * (np.cos(self.Xcosn) ** self.order)
                    * (np.abs(self.alpha) < np.pi / 2.0).astype(int)
                )
                Ycosn_recon = np.sqrt(const) * (np.cos(self.Xcosn)) ** self.order

            else:
                Ycosn_forward = np.sqrt(const) * (np.cos(self.Xcosn)) ** self.order
                Ycosn_recon = Ycosn_forward

            himask = _interpolate1d(log_rad, self.Yrcos, Xrcos)
            self.register_buffer(
                f"_himasks_scale_{i}", torch.as_tensor(himask).unsqueeze(0)
            )

            anglemasks = []
            anglemasks_recon = []
            for b in range(self.num_orientations):
                anglemask = _interpolate1d(
                    angle,
                    Ycosn_forward,
                    self.Xcosn + np.pi * b / self.num_orientations,
                )
                anglemask_recon = _interpolate1d(
                    angle,
                    Ycosn_recon,
                    self.Xcosn + np.pi * b / self.num_orientations,
                )
                anglemasks.append(torch.as_tensor(anglemask).unsqueeze(0))
                anglemasks_recon.append(torch.as_tensor(anglemask_recon).unsqueeze(0))

            self.register_buffer(
                f"_anglemasks_scale_{i}",
                einops.rearrange(anglemasks, "o 1 h w -> 1 1 o h w"),
            )
            self.register_buffer(
                f"_anglemasks_recon_scale_{i}", torch.cat(anglemasks_recon)
            )
            if not self.downsample:
                lomask = _interpolate1d(log_rad, self.YIrcos, Xrcos)
                self.register_buffer(
                    f"_lomasks_scale_{i}", torch.as_tensor(lomask).unsqueeze(0)
                )
                self._loindices.append([np.array([0, 0]), dims])
                lodft = lodft * lomask

            else:
                # subsample lowpass
                dims = np.array([lodft.shape[0], lodft.shape[1]])
                ctr = np.ceil((dims + 0.5) / 2).astype(int)
                lodims = np.ceil((dims - 0.5) / 2).astype(int)
                loctr = np.ceil((lodims + 0.5) / 2).astype(int)
                lostart = ctr - loctr
                loend = lostart + lodims
                self._loindices.append([lostart, loend])

                # subsample indices
                log_rad = log_rad[lostart[0] : loend[0], lostart[1] : loend[1]]
                angle = angle[lostart[0] : loend[0], lostart[1] : loend[1]]

                lomask = _interpolate1d(log_rad, self.YIrcos, Xrcos)
                self.register_buffer(
                    f"_lomasks_scale_{i}", torch.as_tensor(lomask).unsqueeze(0)
                )
                # subsampling
                lodft = lodft[lostart[0] : loend[0], lostart[1] : loend[1]]
                # convolution in spatial domain
                lodft = lodft * lomask

        # reasonable default dtype
        self.to(torch.float32)
        # This model has no trainable parameters, so it's always in eval mode
        self.eval()

    def forward(
        self,
        image: Tensor,
        scales: list[SCALES_TYPE] | None = None,
    ) -> OrderedDict:
        r"""
        Generate the steerable pyramid coefficients for an image.

        The steerable pyramid coefficients run from fine to coarse and split the image
        into subbands corresponding to different orientations and scales (i.e., spatial
        frequencies).

        .. versionchanged:: 1.4
           The returned ``pyr_coeffs`` dictionary's keys are now either strings
           specifying the residual or integers specifying the scale. The non-residual
           coefficients are now 5d tensors of shape (batch, channel, num_orientations,
           height, width).

        Parameters
        ----------
        image
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width).
        scales
            Which scales to include in the returned representation. If None, we
            include all scales. Otherwise, can contain subset of values present
            in this model's ``scales`` attribute (ints from 0 up to
            ``self.num_scales-1`` and the strs 'residual_highpass' and
            'residual_lowpass'. Can contain a single value or multiple values.
            If it's an int, we include all orientations from that scale. Order
            within the list does not matter.

        Returns
        -------
        pyr_coeffs
            Pyramid coefficients. These will be stored in an ordered dictionary with
            keys that are, in order: ``"residual_highpass"``, the integers from ``0`` to
            (the initialization argument) ``order``, and ``"residual_lowpass"``.
            Coefficients have shape ``(*image.shape[:2], self.num_orientations,
            image.shape[2] / 2**scale, image.shape[3] / 2**scale)``, with the
            ``"residual_highpass"`` height and width matching that of ``image``, and
            ``"residual_lowpass"`` having height and width ``(image.shape[2] /
            2**self.num_scales, image.shape[3] / 2**self.num_scales)``. They are
            ordered from fine to coarse: ``"residual_highpass", 0, 1, ...,
            num_scales-1, "residual_lowpass"``.

        Raises
        ------
        ValueError
            If ``image`` is the wrong shape, i.e. ``image.shape[-2:] !=
            self.image_shape``.

        Examples
        --------
        .. plot::
          :context: reset

          >>> import plenoptic as po
          >>> img = po.data.einstein()
          >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:])
          >>> po.pyrshow(spyr(img))
          <PyrFigure ...>
        """
        if self.image_shape != image.shape[-2:]:
            raise ValueError(
                f"Input tensor height/width {tuple(image.shape[-2:])} does not match "
                f"image_shape set at initialization {tuple(self.image_shape)}. "
                "Either resize the input or re-initialize this model."
            )
        pyr_coeffs = OrderedDict()
        if scales is None:
            scales = self.scales
        scale_ints = [s for s in scales if isinstance(s, int)]
        if len(scale_ints) != 0:
            assert (max(scale_ints) < self.num_scales) and (min(scale_ints) >= 0), (
                "Scales must be within 0 and num_scales-1"
            )

        # image is a torch tensor batch of images of size (batch, channel, height,
        # width)
        assert len(image.shape) == 4, "Input must be batch of images of shape BxCxHxW"

        imdft = fft.fft2(image, dim=(-2, -1), norm=self.fft_norm)
        imdft = fft.fftshift(imdft, dim=(-2, -1))

        if "residual_highpass" in scales:
            # high-pass
            hi0dft = imdft * self.hi0mask
            hi0 = fft.ifftshift(hi0dft, dim=(-2, -1))
            hi0 = fft.ifft2(hi0, dim=(-2, -1), norm=self.fft_norm)
            pyr_coeffs["residual_highpass"] = hi0.real
            self.pyr_size["residual_highpass"] = tuple(hi0.real.shape[-2:])

        # input to the next scale is the low-pass filtered component. after this
        # multiplication, lodft will be shape (batch, channel, orientations, height,
        # width)
        lodft = einops.einsum(imdft, self.lo0mask, "b c h w, b c o h w -> b c o h w")

        for i in range(self.num_scales):
            if i in scales:
                # high-pass mask is selected based on the current scale
                himask = getattr(self, f"_himasks_scale_{i}")
                mask = getattr(self, f"_anglemasks_scale_{i}") * himask
                # compute filter output at each orientation

                # band pass filtering is done in the fourier space as multiplying
                #  by the fft of a gaussian derivative.
                # The oriented dft is computed as a product of the fft of the
                # low-passed component, the precomputed anglemask (specifies
                # orientation), and the precomputed hipass mask (creating a bandpass
                # filter) the complex_const variable comes from the Fourier
                # transform of a gaussian derivative.
                # Based on the order of the gaussian, this constant changes.

                banddft = self._complex_const_forward * lodft * mask
                # fft output is then shifted to center frequencies
                band = fft.ifftshift(banddft, dim=(-2, -1))
                # ifft is applied to recover the filtered representation in spatial
                # domain
                band = fft.ifft2(band, dim=(-2, -1), norm=self.fft_norm)

                # for real pyramid, take the real component of the complex band
                if not self.is_complex:
                    pyr_coeffs[i] = band.real
                else:
                    # Because the input signal is real, to maintain a tight frame
                    # if the complex pyramid is used, magnitudes need to be divided
                    # by sqrt(2) because energy is doubled.

                    if self.tight_frame:
                        band = band / np.sqrt(2)
                    pyr_coeffs[i] = band
                self.pyr_size[i] = tuple(band.shape[-2:])

            if not self.downsample:
                # no subsampling of angle and rad
                # just use lo0mask
                lomask = getattr(self, f"_lomasks_scale_{i}")
                lodft = lodft * lomask

                # Since we don't subsample here, if we are not using
                # orthonormalization that we need to manually account for the
                # subsampling, so that energy in each band remains the same
                # the energy is cut by factor of 4 so we need to scale magnitudes
                # by factor of 2.
                if self.fft_norm != "ortho":
                    lodft = 2 * lodft
            else:
                # subsample indices
                lostart, loend = self._loindices[i]

                # subsampling of the dft for next scale
                lodft = lodft[..., lostart[0] : loend[0], lostart[1] : loend[1]]
                # low-pass filter mask is selected
                lomask = getattr(self, f"_lomasks_scale_{i}")
                # again multiply dft by subsampled mask (convolution in spatial domain)
                lodft = lodft * lomask

        if "residual_lowpass" in scales:
            # compute residual lowpass when height <=1
            lo0 = fft.ifftshift(lodft, dim=(-2, -1))
            lo0 = fft.ifft2(lo0, dim=(-2, -1), norm=self.fft_norm)
            pyr_coeffs["residual_lowpass"] = lo0.real.squeeze(2)
            self.pyr_size["residual_lowpass"] = tuple(lo0.real.shape[-2:])

        return pyr_coeffs

    @staticmethod
    def convert_pyr_to_tensor(
        pyr_coeffs: OrderedDict, split_complex: bool = False
    ) -> tuple[
        Tensor, tuple[int, list[SCALES_TYPE], list[torch.Size], list[torch.Size] | bool]
    ]:
        r"""
        Convert coefficient dictionary to a tensor.

        The output tensor has shape (batch, channel, height, width) and is
        intended to be used in an :class:`torch.nn.Module` downstream. In the
        multichannel case, all bands for each channel will be stacked together
        (i.e. if there are 2 channels and 18 bands per channel,
        ``pyr_tensor[:,0:18,...]`` will contain the pyr responses for channel 1 and
        ``pyr_tensor[:, 18:36, ...]`` will contain the responses for channel 2). In
        the case of a complex, multichannel pyramid with ``split_complex=True``,
        the real/imaginary bands will be intereleaved so that they appear as
        pairs with neighboring indices in the channel dimension of the tensor.
        (Note: the residual bands are always real so they will only ever have a
        single band even when ``split_complex=True``.)

        This only works if ``pyr_coeffs`` was created with a pyramid with
        ``downsample=False``

        Parameters
        ----------
        pyr_coeffs
            The pyramid coefficients.
        split_complex
            Indicates whether the output should split complex bands into
            real/imag channels or keep them as a single channel. This should be
            ``True`` if you intend to use a convolutional layer on top of the
            output.

        Returns
        -------
        pyr_tensor
            Tensor with shape (batch, channel, height, width). pyramid coefficients
            reshaped into tensor. The first channel will be the residual
            highpass and the last will be the residual lowpass. Each band is
            then a separate channel, going from fine to coarse (i.e., starting with all
            of scale 0's orientations, then scale 1's, etc.).
        pyr_info
            Information required to recreate the dictionary, containing the
            number of channels, the list of pyramid keys for the dictionary,
            info on how to unpack the coefficients, and info on how ``split_complex``
            was used.

        Raises
        ------
        RuntimeError
            If ``self.downsample is True``. In this case, we can't concatenate across
            scales, because each scale is a different size.

        See Also
        --------
        convert_tensor_to_pyr
            Convert tensor representation to pyramid dictionary.

        Examples
        --------
        .. plot::
          :context: reset

          >>> import plenoptic as po
          >>> img = po.data.einstein()
          >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], downsample=False)
          >>> coeffs = spyr(img)
          >>> coeffs_tensor, _ = spyr.convert_pyr_to_tensor(coeffs)
          >>> coeffs_tensor.shape
          torch.Size([1, 26, 256, 256])
          >>> # rearrange so that the residuals are at the end
          >>> coeffs_tensor = [
          ...     coeffs_tensor[:, 1:-1],
          ...     coeffs_tensor[:, :1],
          ...     coeffs_tensor[:, -1:],
          ... ]
          >>> po.imshow(coeffs_tensor, col_wrap=spyr.num_orientations)
          <PyrFigure ...>
        """
        pyr_keys = list(pyr_coeffs.keys())
        num_channels = pyr_coeffs[pyr_keys[0]].size(1)
        try:
            packed, pack_info = einops.pack(list(pyr_coeffs.values()), "b c * h w")
        except RuntimeError:
            raise RuntimeError(
                "feature maps could not be concatenated into tensor. Check that you"
                "are using coefficients that are not downsampled across scales."
                "This is done with the 'downsample=False' argument for the pyramid"
            )

        # if the second half of this is False, then pyr_coeffs only contains residuals
        if split_complex and not all([isinstance(k, str) for k in pyr_keys]):
            start_idx = 0
            end_idx = None
            if "residual_highpass" in pyr_keys:
                start_idx = 1
            if "residual_lowpass" in pyr_keys:
                end_idx = -1
            complex_coeffs = packed[:, :, start_idx:end_idx]
            try:
                separated = einops.rearrange(
                    [complex_coeffs.real, complex_coeffs.imag],
                    "complex b c o h w -> b c (o complex) h w",
                )
            except RuntimeError:
                raise RuntimeError(
                    "split_complex=True but coefficient tensors are real-valued! "
                    "Either set split_complex=False or regenerate the coefficients "
                    "with a complex pyramid."
                )
            to_pack = []
            if "residual_highpass" in pyr_keys:
                to_pack.append(packed[:, :, 0].real)
            to_pack.append(separated)
            if "residual_lowpass" in pyr_keys:
                to_pack.append(packed[:, :, -1].real)
            packed, split_complex = einops.pack(to_pack, "b c * h w")

        pyr_info = (num_channels, pyr_keys, pack_info, split_complex)
        return einops.rearrange(packed, "b c o h w -> b (c o) h w"), pyr_info

    @staticmethod
    def convert_tensor_to_pyr(
        pyr_tensor: Tensor,
        num_channels: int,
        pyr_keys: list[SCALES_TYPE],
        pack_info: list[torch.Size],
        split_complex_pack_info: list[torch.Size] | bool,
    ) -> OrderedDict:
        r"""
        Convert pyramid coefficient tensor to dictionary format.

        The arguments other than ``pyr_tensor`` are elements of the
        ``pyr_info`` tuple returned by :meth:`convert_pyr_to_tensor`. You
        should always unpack the arguments for this function from that
        ``pyr_info`` tuple. See Examples section below.

        Parameters
        ----------
        pyr_tensor
            Shape (batch, channel, height, width). The pyramid coefficients.
        num_channels
            Number of channels in the original input tensor the pyramid was
            created for (i.e. if the input was an RGB image, this would be 3).
        pyr_keys
            Keys from the original pyramid dictionary.
        pack_info
            List of sizes of the fifth dimension for each coefficient (i.e., the number
            of orientations) used to pack/unpack the tensors.
        split_complex_pack_info
            If :meth:`convert_pyr_to_tensor` was called with ``split_complex=True``,
            another list of sizes used to pack/unpack the tensors. Else, ``False``.

        Returns
        -------
        pyr_coeffs
            Pyramid coefficients in dictionary format as returned by :meth:`forward`.

        See Also
        --------
        convert_pyr_to_tensor
            Convert pyramid dictionary representation to tensor.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> spyr = po.simul.SteerablePyramidFreq(
        ...     img.shape[-2:], downsample=False, is_complex=True
        ... )
        >>> coeffs = spyr(img)
        >>> coeffs_tensor, pyr_info = spyr.convert_pyr_to_tensor(coeffs)
        >>> coeffs_tensor.shape
        torch.Size([1, 26, 256, 256])
        >>> coeffs_tensor.dtype
        torch.complex64
        >>> new_coeffs = spyr.convert_tensor_to_pyr(coeffs_tensor, *pyr_info)
        >>> all([torch.equal(v, new_coeffs[k]) for k, v in coeffs.items()])
        True
        >>> coeffs_tensor, pyr_info = spyr.convert_pyr_to_tensor(
        ...     coeffs, split_complex=True
        ... )
        >>> coeffs_tensor.shape
        torch.Size([1, 50, 256, 256])
        >>> coeffs_tensor.dtype
        torch.float32
        >>> new_coeffs = spyr.convert_tensor_to_pyr(coeffs_tensor, *pyr_info)
        >>> all([torch.equal(v, new_coeffs[k]) for k, v in coeffs.items()])
        True
        """
        # this function just undoes the einops calls in convert_pyr_to_tensor
        unpacked = einops.rearrange(
            pyr_tensor, "b (c o) h w -> b c o h w", c=num_channels
        )
        if not isinstance(split_complex_pack_info, bool):
            unpacked = einops.unpack(unpacked, split_complex_pack_info, "b c * h w")
            if "residual_highpass" in pyr_keys:
                complex_coeffs = unpacked[1]
                complex_pack_info = pack_info[1:]
            else:
                complex_coeffs = unpacked[0]
                complex_pack_info = pack_info
            if "residual_lowpass" in pyr_keys:
                complex_pack_info = complex_pack_info[:-1]
            bands = einops.rearrange(
                complex_coeffs, "b c (o complex) h w -> b c o h w complex", complex=2
            ).contiguous()
            bands = torch.view_as_complex(bands)
            bands = einops.unpack(bands, complex_pack_info, "b c * h w")
            coeffs = []
            if "residual_highpass" in pyr_keys:
                coeffs.append(unpacked[0])
            coeffs.extend(bands)
            if "residual_lowpass" in pyr_keys:
                coeffs.append(unpacked[-1])
        else:
            coeffs = einops.unpack(unpacked, pack_info, "b c * h w")
        pyr_coeffs = OrderedDict({k: v for k, v in zip(pyr_keys, coeffs)})
        # make sure these are real-valued
        for k in ["residual_lowpass", "residual_highpass"]:
            if k in pyr_coeffs:
                pyr_coeffs[k] = pyr_coeffs[k].real
        return pyr_coeffs

    def _recon_levels_check(
        self, levels: Literal["all"] | list[SCALES_TYPE]
    ) -> list[SCALES_TYPE]:
        r"""
        Check whether levels arg is valid for reconstruction and return valid version.

        When reconstructing the input image (i.e., when calling :meth:`recon_pyr()`),
        the user specifies which levels to include. This makes sure those
        levels are valid and gets them in the form we expect for the rest of
        the reconstruction. If the user passes ``"all"``, this constructs the
        appropriate list (based on the values of ``pyr_coeffs``).

        Parameters
        ----------
        levels
            If ``list`` should contain some subset of integers from ``0`` to
            ``self.num_scales-1`` (inclusive) and ``"residual_highpass"`` and
            ``"residual_lowpass"`` (if appropriate for the pyramid). If ``"all"``,
            returned value will contain all valid levels.

        Returns
        -------
        levels
            List containing the valid levels for reconstruction.

        Raises
        ------
        TypeError
            If ``levels`` is not one of the allowed values.
        """  # numpydoc ignore=EX01
        if isinstance(levels, str):
            if levels != "all":
                raise TypeError(
                    "levels must be a list of levels or the string 'all' but"
                    f" got {levels}"
                )
            levels = (
                ["residual_highpass"]
                + list(range(self.num_scales))
                + ["residual_lowpass"]
            )
        else:
            if not hasattr(levels, "__iter__"):
                raise TypeError(
                    "levels must be a list of levels or the string 'all' but"
                    f" got {levels}"
                )
            levs_nums = np.array([int(i) for i in levels if isinstance(i, int)])
            assert (levs_nums >= 0).all(), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(), (
                f"Level numbers must be in the range [0, {self.num_scales - 1:d}]"
            )
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if "residual_highpass" in levels:
                levs_tmp = ["residual_highpass"] + levs_tmp
            if "residual_lowpass" in levels:
                levs_tmp = levs_tmp + ["residual_lowpass"]
            levels = levs_tmp
        # not all pyramids have residual highpass / lowpass, but it's easier
        # to construct the list including them, then remove them if necessary.
        if "residual_lowpass" not in self.pyr_size and "residual_lowpass" in levels:
            levels.pop(-1)
        if "residual_highpass" not in self.pyr_size and "residual_highpass" in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands: Literal["all"] | list[int]) -> list[int]:
        """
        Check whether bands arg is valid for reconstruction and return valid version.

        When reconstructing the input image (i.e., when calling :meth:`recon_pyr()`),
        the user specifies which orientations to include. This makes sure those
        orientations are valid and gets them in the form we expect for the rest
        of the reconstruction. If the user passes ``'all'``, this
        constructs the appropriate list (based on the values of ``pyr_coeffs``).

        Parameters
        ----------
        bands
            If list, should contain some subset of integers from ``0`` to
            ``self.num_orientations-1``. If ``'all'``, returned value will
            contain all valid orientations.

        Returns
        -------
        bands
            List containing the valid orientations for reconstruction.

        Raises
        ------
        TypeError
            If ``bands`` is not an int or ``"all"``.
        ValueError
            If ``bands`` is an integer outside of the range ``[0,
            self.num_orientations-1]``.
        """  # numpydoc ignore=EX01
        if isinstance(bands, str):
            if bands != "all":
                raise TypeError(
                    f"bands must be a list of ints or the string 'all' but got {bands}"
                )
        else:
            if not hasattr(bands, "__iter__"):
                raise TypeError(
                    f"bands must be a list of ints or the string 'all' but got {bands}"
                )
            bands: NDArray = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            if any(bands > self.num_orientations):
                raise ValueError(
                    "Error: band numbers must be in the range "
                    f"[0, {self.num_orientations - 1:d}]"
                )
        return bands

    def recon_pyr(
        self,
        pyr_coeffs: OrderedDict,
        levels: Literal["all"] | list[SCALES_TYPE] = "all",
        bands: Literal["all"] | list[int] = "all",
    ) -> Tensor:
        """
        Reconstruct image from coefficients, optionally using a subset.

        Parameters
        ----------
        pyr_coeffs
            Pyramid coefficients to reconstruct from.
        levels
            If ``list`` should contain some subset of integers from ``0`` to
            ``self.num_scales-1`` (inclusive), ``"residual_lowpass"``, and
            ``"residual_highpass"``. If ``"all"``, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands
            If list, should contain some subset of integers from ``0`` to
            ``self.num_orientations-1``. If ``"all"``, returned value will contain
            all valid orientations. Otherwise, must be one of the valid
            orientations.

        Returns
        -------
        recon
            The reconstructed image, of shape (batch, channel, height, width).

        Raises
        ------
        ValueError
            If ``self.forward()`` was called with ``scales`` argument not ``None``.
        TypeError
            If ``levels`` is not one of the allowed values.
        TypeError
            If ``bands`` is not an integer or ``"all"`` .
        ValueError
            If ``bands`` is an integer outside of the range ``[0,
            self.num_orientations-1]``.

        Examples
        --------
        .. plot::
          :context: reset

          >>> import plenoptic as po
          >>> import torch
          >>> img = po.data.einstein()
          >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:])
          >>> coeffs = spyr(img)
          >>> recon = spyr.recon_pyr(coeffs)
          >>> torch.allclose(recon, img, rtol=1e-8, atol=1e-5)
          True
          >>> titles = ["Original", "Reconstructed", "Difference"]
          >>> po.imshow([img, recon, img - recon], title=titles)
          <PyrFigure ...>
        """  # numpydoc ignore=ES01
        # For reconstruction to work, last time we called forward needed
        # to include all levels
        for s in self.scales:
            if s not in pyr_coeffs:
                raise ValueError(
                    f"scale {s} not in pyr_coeffs! pyr_coeffs must include"
                    " all scales, so make sure forward() was called with"
                    " arg scales=None"
                )

        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        scale = 0

        # Recursively generate the reconstruction - function starts with
        # fine scales going down to coarse and then the reconstruction
        # is built recursively from the coarse scale up

        recondft = self._recon_levels(pyr_coeffs, levels, bands, scale)

        outdft = recondft * self.lo0mask.squeeze()
        # generate highpass residual Reconstruction
        if "residual_highpass" in levels:
            hidft = fft.fft2(
                pyr_coeffs["residual_highpass"],
                dim=(-2, -1),
                norm=self.fft_norm,
            )
            hidft = fft.fftshift(hidft, dim=(-2, -1))

            # output dft is the sum of the recondft from the recursive
            # function times the lomask (low pass component) with the
            # highpass dft * the highpass mask
            outdft = outdft + hidft * self.hi0mask

        # get output reconstruction by inverting the fft
        reconstruction = fft.ifftshift(outdft, dim=(-2, -1))
        reconstruction = fft.ifft2(reconstruction, dim=(-2, -1), norm=self.fft_norm)

        # get real part of reconstruction (if complex)
        return reconstruction.real

    def _recon_levels(
        self,
        pyr_coeffs: OrderedDict,
        recon_levels: list[SCALES_TYPE],
        recon_bands: list[int] | Literal["all"],
        scale: int,
    ) -> Tensor:
        """
        Recursive function used to build the reconstruction. Called by recon_pyr.

        Each time this function is called, it reconstructs a single scale.

        Parameters
        ----------
        pyr_coeffs
            Dictionary containing the coefficients of the pyramid. Keys are
            ``(level, band)`` tuples and the strings ``"residual_lowpass"`` and
            ``"residual_highpass"`` and values are Tensors of shape (batch,
            channel, height, width).
        recon_levels
            List of scales to include in the reconstruction.
        recon_bands
            Either ``"all"`` (in which case we include all bands)
            or list of bands to include in the reconstruction.
        scale
            Current scale that is being used to build the reconstruction
            scale is incremented by 1 on each call of the function.

        Returns
        -------
        recondft
            Current reconstruction based on the orientation band dft from the
            current scale summed with the output of recursive call with the
            next scale incremented.
        """  # numpydoc ignore=EX01
        # base case, return the low-pass residual
        if scale == self.num_scales:
            if "residual_lowpass" in recon_levels:
                lodft = fft.fft2(
                    pyr_coeffs["residual_lowpass"],
                    dim=(-2, -1),
                    norm=self.fft_norm,
                )
                lodft = fft.fftshift(lodft)
            else:
                lodft = fft.fft2(
                    torch.zeros_like(pyr_coeffs["residual_lowpass"]),
                    dim=(-2, -1),
                    norm=self.fft_norm,
                )

            return lodft

        # Reconstruct from orientation bands
        # update himask
        if scale in recon_levels:
            himask = getattr(self, f"_himasks_scale_{scale}")
            mask = getattr(self, f"_anglemasks_recon_scale_{scale}") * himask
            coeffs = pyr_coeffs[scale]
            # then recon_bands is not "all" and we're subselecting them
            if not isinstance(recon_bands, str):
                coeffs = coeffs[:, :, recon_bands]
                mask = mask[recon_bands]
            if self.tight_frame and self.is_complex:
                coeffs = coeffs * np.sqrt(2)
            orientdft = fft.fft2(coeffs, dim=(-2, -1), norm=self.fft_norm)
            orientdft = fft.fftshift(orientdft, dim=(-2, -1))
            orientdft = self._complex_const_recon * orientdft * mask
            orientdft = orientdft.sum(2)
        else:
            orientdft = torch.zeros_like(pyr_coeffs[scale][:, :, 0])

        # get the bounding box indices for the low-pass component
        lostart, loend = self._loindices[scale]

        # create lowpass mask
        lomask = getattr(self, f"_lomasks_scale_{scale}")

        # Recursively reconstruct by going to the next scale
        reslevdft = self._recon_levels(pyr_coeffs, recon_levels, recon_bands, scale + 1)
        # in not downsampled case, rescale the magnitudes of the reconstructed
        # dft at each level by factor of 2 to account for the scaling in the forward
        if (not self.tight_frame) and (not self.downsample):
            reslevdft = reslevdft / 2
        # create output for reconstruction result
        resdft = torch.zeros_like(pyr_coeffs[scale][:, :, 0], dtype=torch.complex64)

        # place upsample and convolve lowpass component
        resdft[..., lostart[0] : loend[0], lostart[1] : loend[1]] = reslevdft * lomask
        recondft = resdft + orientdft
        # add orientation interpolated and added images to the lowpass image
        return recondft

    def steer_coeffs(
        self,
        pyr_coeffs: OrderedDict,
        angles: list[float],
        even_phase: bool = True,
    ) -> tuple[dict, dict]:
        """
        Steer pyramid coefficients to the specified angles.

        This allows you to have filters that have the Gaussian derivative order
        specified in construction, but arbitrary angles or number of orientations.

        .. versionchanged:: 1.4
           The returned ``resteered_coeffs`` dictionary now only contains the new
           angles, as opposed to concatenating the new angles onto those found in
           the input ``pyr_coeffs``. Like the input ``pyr_coeffs``, the dictionary
           keys are now integers specifying the scale and the coefficients are
           5d tensors of shape (batch, channel, angles, height, width).

        Parameters
        ----------
        pyr_coeffs
            The pyramid coefficients to steer, as returned by :meth:`forward`.
        angles
            List of angles (in radians) to steer the pyramid coefficients to.
        even_phase
            Specifies whether the harmonics are cosine or sine phase aligned
            about those positions.

        Returns
        -------
        resteered_coeffs
            Dictionary of re-steered pyramid coefficients. will have the same
            number of scales as the original pyramid (though it will not
            contain the residual highpass or lowpass). Like the input ``pyr_coeffs``,
            keys are ints indexing the scale and values are tensors of shape (batch,
            channel, orientations, height, width), but now orientations index ``angles``
            instead of ``self.num_orientations``.
        resteering_weights :
            Dictionary of weights used to re-steer the pyramid coefficients.
            will have the same keys as ``resteered_coeffs``.

        Examples
        --------

        .. plot::

            >>> import plenoptic as po
            >>> import torch
            >>> img = po.data.einstein()
            >>> spyr = po.simul.SteerablePyramidFreq(img.shape[-2:], height=3)
            >>> coeffs = spyr(img)
            >>> resteered_coeffs, resteering_weights = spyr.steer_coeffs(
            ...     coeffs, torch.linspace(0, 2 * torch.pi, 64)
            ... )
            >>> ani = po.animshow(resteered_coeffs[2], repeat=True, framerate=6, zoom=4)
            >>> # Save the video (here we're saving it as a .gif)
            >>> ani.save("resteered_coeffs.gif")

        .. image:: resteered_coeffs.gif
        """
        assert pyr_coeffs[0].dtype not in complex_types, (
            "steering only implemented for real coefficients"
        )
        resteered_coeffs = {}
        resteering_weights = {}
        num_scales = self.num_scales
        for i in range(num_scales):
            # put orientation on the last dimension
            basis = einops.rearrange(pyr_coeffs[i], "b c o h w -> b c h w o")

            res, steervect = [], []
            for j, a in enumerate(angles):
                r, s = _steer(basis, a, even_phase=even_phase)
                res.append(r)
                steervect.append(s)
            # when called like above, the output of steer always has a singleton
            # dimension at the end corresponding to the single angle it was steered to
            resteered_coeffs[i] = einops.rearrange(
                res, "o b c h w dummy -> b c (o dummy) h w"
            )
            resteering_weights[i] = torch.stack(steervect, dim=-1)

        return resteered_coeffs, resteering_weights
