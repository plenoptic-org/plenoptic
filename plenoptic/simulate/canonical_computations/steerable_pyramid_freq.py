import warnings
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
from ...tools.signal import rcosFn, batch_fftshift, batch_ifftshift, pointOp, steer
import torch
import torch.nn as nn


class Steerable_Pyramid_Freq(nn.Module):
    r"""Steerable frequency pyramid in Torch

    Construct a steerable pyramid on matrix IM, in the Fourier domain.
    Reconstruction is exact (within floating point errors). However, if the image has odd-shape, the reconstruction
    will not be exact due to boundary-handling issues that have not been resolved.
    Boundary-handling is circular.

    The squared radial functions tile the Fourier plane with a
    raised-cosine falloff. Angular functions are cos(theta-
    k*pi/order+1)^(order).

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.
    For further information see the project webpage_

    Parameters
    ----------
    image_shape : `list or tuple`
        shape of input image
    height : 'auto' or `int`
        The height of the pyramid. If 'auto', will automatically determine based on the size of
        `image`.
    order : `int`.
        The Gaussian derivative order used for the steerable filters. Default value is 3.
        Note that to achieve steerability the minimum number of orientation is `order` + 1,
        and is used here. To get more orientations at the same order, use the method `steer_coeffs`
    twidth : `int`
        The width of the transition region of the radial lowpass function, in octaves
    is_complex : `bool`
        Whether the pyramid coefficients should be complex or not. If True, the real and imaginary
        parts correspond to a pair of even and odd symmetric filters. If False, the coefficients
        only include the real part / even symmetric filter.
    downsample: `bool`
        Whether to downsample each scale in the pyramid or keep the output pyramid coefficients
        in fixed bands of size imshapeximshape. When downsample is False, the forward method returns a tensor.
    tight_frame: `bool` default: True
        Whether the pyramid obeys the generalized parseval theorem or not (i.e. is a tight frame).
        If True, the energy of the pyr_coeffs = energy of the image. If not this is not true.
        In order to match the matlabPyrTools or pyrtools pyramids, this must be set to False

    Attributes
    ----------
    image_shape : `list or tuple`
        shape of input image
    pyr_type : `str` or `None`
        Human-readable string specifying the type of pyramid. For base class, is None.
    pyr_coeffs : `dict`
        Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
        values are 1d or 2d numpy arrays (same number of dimensions as the input image)
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
    fft_normalize : `bool`
        Whether the fft's are normalized or not. It is automatically set to True when tight_frame is true
        else it is set to False
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.
    .. _webpage: https://www.cns.nyu.edu/~eero/steerpyr/

    """

    def __init__(self, image_shape, height='auto', order=3, twidth=1, is_complex=False,
                  downsample=True, tight_frame=True):

        super().__init__()

        self.order = order
        self.image_shape = image_shape

        if (self.image_shape[0] % 2 != 0) or (self.image_shape[1] % 2 != 0):
            warnings.warn("Reconstruction will not be perfect with odd-sized images")

        self.is_complex = is_complex
        self.downsample = downsample
        self.tight_frame = tight_frame
        if self.tight_frame:
            self.fft_normalize = True
        else:
            self.fft_normalize = False
        # cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize + 1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi

        self.pyr_size = {}

        max_ht = np.floor(np.log2(min(self.image_shape[0], self.image_shape[1])))-2
        if height == 'auto':
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        if self.order > 15 or self.order <= 0:
            warnings.warn("order must be an integer in the range [1,15]. Truncating.")
            self.order = min(max(self.order, 1), 15)
        self.num_orientations = int(self.order + 1)

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1
        twidth = int(twidth)

        dims = np.array(self.image_shape)

        # make a grid for the raised cosine interpolation
        ctr = np.ceil((np.array(dims)+0.5)/2).astype(int)

        (xramp, yramp) = np.meshgrid(np.linspace(-1, 1, dims[1]+1)[:-1],
                                     np.linspace(-1, 1, dims[0]+1)[:-1])

        self.angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        self.log_rad = np.log2(log_rad)

        # radial transition function (a raised cosine in log-frequency):
        self.Xrcos, Yrcos = rcosFn(twidth, (-twidth/2.0), np.array([0, 1]))
        self.Yrcos = np.sqrt(Yrcos)

        self.YIrcos = np.sqrt(1.0 - self.Yrcos**2)

        # create low and high masks
        lo0mask = pointOp(self.log_rad, self.YIrcos, self.Xrcos)
        hi0mask = pointOp(self.log_rad, self.Yrcos, self.Xrcos)
        self.lo0mask = torch.tensor(lo0mask).unsqueeze(0).unsqueeze(-1)
        self.hi0mask = torch.tensor(hi0mask).unsqueeze(0).unsqueeze(-1)

        # pre-generate the angle, hi and lo masks, as well as the
        # indices used for down-sampling
        self._anglemasks = []
        self._anglemasks_recon = []
        self._himasks = []
        self._lomasks = []
        self._loindices = []

        # need a mock image to down-sample so that we correctly
        # construct the differently-sized masks
        mock_image = np.random.rand(*self.image_shape)
        imdft = np.fft.fftshift(np.fft.fft2(mock_image))
        lodft = imdft * lo0mask

        # this list, used by coarse-to-fine optimization, gives all the
        # scales (including residuals) from coarse to fine
        self.scales = (['residual_lowpass'] + list(range(self.num_scales))[::-1] +
                       ['residual_highpass'])

        # we create these copies because they will be modified in the
        # following loops
        Xrcos = self.Xrcos.copy()
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        for i in range(self.num_scales):
            Xrcos -= np.log2(2)
            const = ((2 ** (2*self.order)) * (factorial(self.order, exact=True)**2) /
                     float(self.num_orientations * factorial(2*self.order, exact=True)))

            if self.is_complex:
                Ycosn_forward = (2.0 * np.sqrt(const) * (np.cos(self.Xcosn) ** self.order) *
                                 (np.abs(self.alpha) < np.pi/2.0).astype(int))
                Ycosn_recon = np.sqrt(const) * (np.cos(self.Xcosn))**self.order

            else:
                Ycosn_forward = np.sqrt(const) * (np.cos(self.Xcosn))**self.order
                Ycosn_recon = Ycosn_forward

            himask = pointOp(log_rad, self.Yrcos, Xrcos)
            self._himasks.append(torch.tensor(himask).unsqueeze(0).unsqueeze(-1))

            anglemasks = []
            anglemasks_recon = []
            for b in range(self.num_orientations):
                anglemask = pointOp(angle, Ycosn_forward, self.Xcosn + np.pi*b/self.num_orientations)
                anglemask_recon = pointOp(angle, Ycosn_recon, self.Xcosn + np.pi*b/self.num_orientations)
                anglemasks.append(torch.tensor(anglemask).unsqueeze(0).unsqueeze(-1))
                anglemasks_recon.append(torch.tensor(anglemask_recon).unsqueeze(0).unsqueeze(-1))

            self._anglemasks.append(anglemasks)
            self._anglemasks_recon.append(anglemasks_recon)
            if not self.downsample:
                lomask = pointOp(log_rad, self.YIrcos, Xrcos)
                self._lomasks.append(torch.tensor(lomask).unsqueeze(0).unsqueeze(-1))
                self._loindices.append([np.array([0,0]), dims])
                lodft = lodft * lomask

            else:
                # subsample lowpass
                dims = np.array([lodft.shape[0], lodft.shape[1]])
                ctr = np.ceil((dims+0.5)/2).astype(int)
                lodims = np.ceil((dims-0.5)/2).astype(int)
                loctr = np.ceil((lodims+0.5)/2).astype(int)
                lostart = ctr - loctr
                loend = lostart + lodims
                self._loindices.append([lostart, loend])

                # subsample indices
                log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
                angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

                lomask = pointOp(log_rad, self.YIrcos, Xrcos)
                self._lomasks.append(torch.tensor(lomask).unsqueeze(0).unsqueeze(-1))
                # subsampling
                lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
                # convolution in spatial domain
                lodft = lodft * lomask

        # reasonable default dtype
        self = self.to(torch.float32)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self
        """
        self.lo0mask = self.lo0mask.to(*args, **kwargs)
        self.hi0mask = self.hi0mask.to(*args, **kwargs)
        self._himasks = [m.to(*args, **kwargs) for m in self._himasks]
        self._lomasks = [m.to(*args, **kwargs) for m in self._lomasks]
        angles = []
        angles_recon = []
        for a, ar in zip(self._anglemasks, self._anglemasks_recon):
            angles.append([m.to(*args, **kwargs) for m in a])
            angles_recon.append([m.to(*args, **kwargs) for m in ar])
        self._anglemasks = angles
        self._anglemasks_recon = angles_recon
        return self

    def forward(self, x, scales=[]):
        r"""Generate the steerable pyramid coefficients for an image

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width).
        scales : list, optional
            Which scales to include in the returned representation. If
            an empty list (the default), we include all
            scales. Otherwise, can contain subset of values present in
            this model's ``scales`` attribute (ints from 0 up to
            self.num_scales-1 and the strs 'residual_highpass' and
            'residual_lowpass'. Can contain a single value or multiple
            values. If it's an int, we include all orientations from
            that scale. Order within the list does not matter

        Returns
        -------
        representation: torch.Tensor or OrderedDict
            if the not downsampled version is used, representation is returned
            as a torch tensor with each band as a channel in BxCxHxW. The order
            of the channels is the same order as the keys in the pyr_coeffs dictonary.
            If the pyramid is complex, the channels are ordered such that for each band,
            the real channel comes first, followed by the imaginary channel.

            If downsample is true, representation is an OrderedDict of the coefficients.

        """
        self.pyr_coeffs = OrderedDict()
        if not isinstance(scales, list):
            raise Exception("scales must be a list!")
        if not scales:
            scales = self.scales
        scale_ints = [s for s in scales if isinstance(s,int)]
        if len(scale_ints) != 0:
            assert (max(scale_ints) < self.num_scales) and (min(scale_ints) >= 0), "Scales must be within 0 and num_scales-1"
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        lo0mask = self.lo0mask.clone()
        hi0mask = self.hi0mask.clone()

        # x is a torch tensor batch of images of size [N,C,W,H]
        assert len(x.shape) == 4, "Input must be batch of images of shape BxCxHxW"
        # x = x.squeeze(1) #flatten channel dimension first
        imdft = torch.rfft(x, signal_ndim=2, onesided=False, normalized=self.fft_normalize)
        imdft = batch_fftshift(imdft)

        if 'residual_highpass' in scales:
            # high-pass
            hi0dft = imdft * hi0mask
            hi0 = batch_ifftshift(hi0dft)
            hi0 = torch.ifft(hi0, signal_ndim=2, normalized=self.fft_normalize)
            hi0_real = torch.unbind(hi0, -1)[0]
            self.pyr_coeffs['residual_highpass'] = hi0_real
            self.pyr_size['residual_highpass'] = tuple(hi0_real.shape[-2:])

        lodft = imdft * lo0mask


        for i in range(self.num_scales):

            if i in scales:

                himask = self._himasks[i]
                for b in range(self.num_orientations):
                    anglemask = self._anglemasks[i][b]

                    # bandpass filtering
                    banddft = lodft * anglemask * himask
                    banddft = torch.unbind(banddft, -1)
                    # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                    complex_const = np.power(np.complex(0, -1), self.order)
                    banddft_real = complex_const.real * banddft[0] - complex_const.imag * banddft[1]
                    banddft_imag = complex_const.real * banddft[1] + complex_const.imag * banddft[0]
                    # preallocation and then filling in is much more
                    # efficient than using stack
                    banddft = torch.empty((*banddft_real.shape, 2), device=banddft_real.device)
                    banddft[..., 0] = banddft_real
                    banddft[..., 1] = banddft_imag

                    band = batch_ifftshift(banddft)
                    band = torch.ifft(band, signal_ndim=2, normalized=self.fft_normalize)
                    if not self.is_complex:
                        band = torch.unbind(band, -1)[0]
                        self.pyr_coeffs[(i, b)] = band
                        self.pyr_size[(i, b)] = tuple(band.shape[-2:])
                    else:
                        if self.tight_frame:
                            band = band / np.sqrt(2)
                        self.pyr_coeffs[(i, b)] = band
                        self.pyr_size[(i, b)] = tuple(band.shape[2:4])

            if not self.downsample:
                # no subsampling of angle and rad
                # just use lo0mask
                lomask = self._lomasks[i]
                lodft = lodft * lomask
            else:
                # subsample indices
                lostart, loend = self._loindices[i]

                log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
                angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

                # subsampling
                lodft = lodft[:, :, lostart[0]:loend[0], lostart[1]:loend[1], :]
                # filtering
                lomask = self._lomasks[i]
                # convolution in spatial domain
                lodft = lodft * lomask

        if 'residual_lowpass' in scales:
            # compute residual lowpass when height <=1
            lo0 = batch_ifftshift(lodft)
            lo0 = torch.ifft(lo0, signal_ndim=2, normalized=self.fft_normalize)
            lo0_real = torch.unbind(lo0, -1)[0]
            self.pyr_coeffs['residual_lowpass'] = lo0_real
            self.pyr_size['residual_lowpass'] = tuple(lo0_real.shape[-2:])

        if not self.downsample:
            return self.convert_pyr_to_tensor(self.pyr_coeffs)
        else:
            return self.pyr_coeffs


    def convert_pyr_to_tensor(self, pyr_coeffs):
        r"""
        Function that takes a torch pyramid (without downsampling) dictonary and converts the output into a single tensor
        of BxCxHxW for use in an nn module downstream.

        Parameters
        ----------
        pyr_coeffs: `OrderedDict`
            the pyramid coefficients

        Returns
        -----------
        coeff_out: `torch.Tensor` (BxCxHxW)
            pyramid coefficients reshaped into tensor
        """

        assert not self.downsample, "conversion to tensor only works for pyramids without downsampling of feature maps"
        coeff_list = []
        coeff_list_resid = []
        for k in pyr_coeffs.keys():
            if 'residual' in k:
                coeff_list_resid.append(pyr_coeffs[k])
            else:
                coeff_list.append(pyr_coeffs[k])
        if len(coeff_list) > 0:
            coeff_bands = torch.cat(coeff_list, dim=1)
            batch_size = coeff_bands.shape[0]
            imshape = [coeff_bands.shape[2], coeff_bands.shape[3]]
            if self.is_complex:
                coeff_bands = coeff_bands.permute(0,1,4,2,3).contiguous().view(batch_size,-1,imshape[0],imshape[1])
            if len(coeff_list_resid) == 1:
                coeff_resid = torch.cat(coeff_list_resid, dim=1)
                coeff_out = torch.cat([coeff_resid, coeff_bands], dim=1)
            elif len(coeff_list_resid) == 2:
                coeff_out = torch.cat([coeff_list_resid[0], coeff_bands, coeff_list_resid[1]], dim=1)
            else:
                coeff_out = coeff_bands
        else:
            coeff_out = torch.cat(coeff_list_resid, dim=1)

        return coeff_out

    def convert_tensor_to_pyr(self, pyr_tensor):
        r"""
        Function that takes a torch pyramid coefficient tensor and converts the output into
        the dictionary format where

        Parameters
        ----------
        pyr_tensor: `torch.Tensor` (BxCxHxW)
            the pyramid coefficients

        Returns
        ----------
        pyr_coeffs: `OrderedDict`
            pyramid coefficients in dictionary format
        """

        pyr_coeffs = OrderedDict()
        key_list = list(self.pyr_coeffs.keys())
        i = 0
        for k in key_list:
            if ('residual' in k) or (not self.is_complex):
                pyr_coeffs[k] = pyr_tensor[:,i,...].unsqueeze(1)
                i += 1
            else:
                pyr_coeffs[k] = pyr_tensor[:,i:i+2,:,:].permute(0,2,3,1).unsqueeze(1)
                i += 2

        return pyr_coeffs

    def _recon_levels_check(self, levels):
        r"""Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which levels to include. This makes sure those levels are valid and gets them in the form
        we expect for the rest of the reconstruction. If the user passes `'all'`, this constructs
        the appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, or `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.

        Returns
        -------
        levels : `list`
            List containing the valid levels for reconstruction.

        """
        if isinstance(levels, str) and levels == 'all':
            levels = ['residual_highpass'] + list(range(self.num_scales)) + ['residual_lowpass']
        else:
            if not hasattr(levels, '__iter__') or isinstance(levels, str):
                # then it's a single int or string
                levels = [levels]
            levs_nums = np.array([int(i) for i in levels if isinstance(i, int) or i.isdigit()])
            assert (levs_nums >= 0).all(), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(), "Level numbers must be in the range [0, %d]" % (self.num_scales-1)
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if 'residual_highpass' in levels:
                levs_tmp = ['residual_highpass'] + levs_tmp
            if 'residual_lowpass' in levels:
                levs_tmp = levs_tmp + ['residual_lowpass']
            levels = levs_tmp
        # not all pyramids have residual highpass / lowpass, but it's easier to construct the list
        # including them, then remove them if necessary.
        if 'residual_lowpass' not in self.pyr_coeffs.keys() and 'residual_lowpass' in levels:
            levels.pop(-1)
        if 'residual_highpass' not in self.pyr_coeffs.keys() and 'residual_highpass' in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands):
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which orientations to include. This makes sure those orientations are valid and gets them
        in the form we expect for the rest of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `self.pyr_coeffs`).

        Parameters
        ----------
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.

        Returns
        -------
        bands: `list`
            List containing the valid orientations for reconstruction.
        """
        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands

    def _recon_keys(self, levels, bands, max_orientations=None):
        """Make a list of all the relevant keys from `pyr_coeffs` to use in pyramid reconstruction

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        some subset of the pyramid coefficients to include in the reconstruction. This function
        takes in those specifications, checks that they're valid, and returns a list of tuples
        that are keys into the `pyr_coeffs` dictionary.

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.
        max_orientations: `None` or `int`.
            The maximum number of orientations we allow in the reconstruction. when we determine
            which ints are allowed for bands, we ignore all those greater than max_orientations.

        Returns
        -------
        recon_keys : `list`
            List of `tuples`, all of which are keys in `pyr_coeffs`. These are the coefficients to
            include in the reconstruction of the image.

        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        if max_orientations is not None:
            for i in bands:
                if i >= max_orientations:
                    warnings.warn(("You wanted band %d in the reconstruction but max_orientation"
                                   " is %d, so we're ignoring that band" % (i, max_orientations)))
            bands = [i for i in bands if i < max_orientations]
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band) for band in bands])
        return recon_keys

    def recon_pyr(self, levels='all', bands='all', twidth=1):
        """Reconstruct the image or batch of images, optionally using subset of pyramid coefficients.

        NOTE: in order to call this function, you need to have
        previously called `self.forward(x)`, where `x` is the tensor you
        wish to reconstruct. This will fail if you called `forward()`
        with a subset of scales.

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.
        twidth : `int`
            The width of the transition region of the radial lowpass function, in octaves

        Returns
        -------
        recon : `torch.Tensor`
            The reconstructed image or batch of images.
            Output is of size BxCxHxW

        """
        # For reconstruction to work, last time we called forward needed
        # to include all levels
        for s in self.scales:
            if isinstance(s, str):
                if s not in self.pyr_coeffs.keys():
                    raise Exception(f"scale {s} not in self.pyr_coeffs! pyr_coeffs must include"
                                    " all scales, so make sure forward() was called with arg "
                                    "scales=[]")
            else:
                for b in range(self.num_orientations):
                    if (s, b) not in self.pyr_coeffs.keys():
                        raise Exception(f"scale {s} not in self.pyr_coeffs! pyr_coeffs must "
                                        "include all scales, so make sure forward() was called "
                                        "with arg scales=[]")


        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1

        recon_keys = self._recon_keys(levels, bands)
        scale = 0


        # load masks from model
        lo0mask = self.lo0mask
        hi0mask = self.hi0mask

        # Recursively generate the reconstruction - function starts with
        # fine scales going down to coarse and then the reconstruction
        # is built recursively from the coarse scale up

        recondft = self._recon_levels(self.pyr_coeffs, recon_keys, scale)

        # generate highpass residual Reconstruction
        if 'residual_highpass' in recon_keys:
            hidft = torch.rfft(self.pyr_coeffs['residual_highpass'], signal_ndim=2, onesided=False, normalized=self.fft_normalize)
            hidft = batch_fftshift(hidft)

            # output dft is the sum of the recondft from the recursive
            # function times the lomask (low pass component) with the
            # highpass dft * the highpass mask
            outdft = recondft * lo0mask + hidft * hi0mask
        else:
            outdft = recondft * lo0mask

        # get output reconstruction by inverting the fft
        reconstruction = batch_ifftshift(outdft)
        reconstruction = torch.ifft(reconstruction, signal_ndim=2, normalized=self.fft_normalize)

        # get real part of reconstruction (if complex)
        reconstruction = torch.unbind(reconstruction, -1)[0]

        return reconstruction

    def _recon_levels(self, pyr_coeffs, recon_keys, scale):
        """Recursive function used to build the reconstruction. Called by recon_pyr

        Parameters
        ----------
        pyr_coeffs : `dict`
            Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
            values are 1d or 2d numpy arrays (same number of dimensions as the input image)
        recon_keys : `list of tuples and/or strings`
            list of the keys that index into the pyr_coeffs Dictionary
        scale : `int`
            current scale that is being used to build the reconstruction
            scale is incremented by 1 on each call of the function

        Returns
        -------
        recondft : `torch.Tensor`
            Current reconstruction based on the orientation band dft from the current scale
            summed with the output of recursive call with the next scale incremented

        """
        # base case, return the low-pass residual
        if scale == self.num_scales:
            if 'residual_lowpass' in recon_keys:
                lodft = torch.rfft(pyr_coeffs['residual_lowpass'], signal_ndim=2, onesided=False, normalized=self.fft_normalize)
                lodft = batch_fftshift(lodft)
            else:
                lodft = torch.rfft(torch.zeros_like(pyr_coeffs['residual_lowpass']), signal_ndim=2,
                                   onesided=False, normalized=self.fft_normalize)

            return lodft

        # Reconstruct from orientation bands
        # update himask
        himask = self._himasks[scale]
        orientdft = torch.zeros_like(pyr_coeffs[(scale, 0)])
        if not self.is_complex:
            # if the pyramid is not complex, the values in pyr_coeffs
            # will have shape (batch, channel, height, width), but
            # orientdft is going to take the outputs of a Fourier
            # transform, which is always complex-valued so it also needs
            # an extra dimension at the end for real and imaginary. If
            # the pyramid is complex, the values in pyr_coeffs will have
            # already have this shape.
            orientdft = torch.zeros((*orientdft.shape, 2), device=orientdft.device)

        for b in range(self.num_orientations):
            if (scale, b) in recon_keys:
                anglemask = self._anglemasks_recon[scale][b]
                if self.is_complex:
                    if self.tight_frame:
                        coeffs = pyr_coeffs[(scale,b)]*np.sqrt(2)
                    else:
                        coeffs = pyr_coeffs[(scale,b)]
                    banddft = torch.fft(coeffs, signal_ndim=2, normalized=self.fft_normalize)
                else:
                    banddft = torch.rfft(pyr_coeffs[(scale, b)], signal_ndim=2, onesided=False, normalized=self.fft_normalize)
                banddft = batch_fftshift(banddft)

                banddft = banddft * anglemask * himask
                banddft = torch.unbind(banddft, -1)
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                complex_const = np.power(np.complex(0, 1), self.order)
                banddft_real = complex_const.real * banddft[0] - complex_const.imag * banddft[1]
                banddft_imag = complex_const.real * banddft[1] + complex_const.imag * banddft[0]
                banddft = torch.empty((*banddft_real.shape, 2), device=banddft_real.device)
                banddft[..., 0] = banddft_real
                banddft[..., 1] = banddft_imag
                orientdft = orientdft + banddft

        # get the bounding box indices for the low-pass component
        lostart, loend = self._loindices[scale]

        # create lowpass mask

        lomask = self._lomasks[scale]
        # Recursively reconstruct by going to the next scale
        reslevdft = self._recon_levels(pyr_coeffs, recon_keys, scale+1)
        # create output for reconstruction result
        resdft = torch.zeros_like(pyr_coeffs[(scale, 0)])
        if not self.is_complex:
            # Similar to orientdft above, if the pyramid is not complex,
            # the values in pyr_coeffs will have shape (batch, channel,
            # height, width), but resdft is going to take the outputs
            # of a Fourier transform, which is always complex-valued so
            # it also needs an extra dimension at the end for real and
            # imaginary. If the pyramid is complex, the values in
            # pyr_coeffs will have already have this shape.
            resdft = torch.zeros((*resdft.shape, 2), device=resdft.device)

        # place upsample and convolve lowpass component
        resdft[:, :, lostart[0]:loend[0], lostart[1]:loend[1]] = reslevdft*lomask
        recondft = resdft + orientdft
        # add orientation interpolated and added images to the lowpass image
        return recondft



    def steer_coeffs(self, angles, even_phase=True):
        """Steer pyramid coefficients to the specified angles

        This allows you to have filters that have the Gaussian derivative order specified in
        construction, but arbitrary angles or number of orientations.

        Parameters
        ----------
        angles : `list`
            list of angles (in radians) to steer the pyramid coefficients to
        even_phase : `bool`
            specifies whether the harmonics are cosine or sine phase aligned about those positions.

        Returns
        -------
        resteered_coeffs : `dict`
            dictionary of re-steered pyramid coefficients. will have the same number of scales as
            the original pyramid (though it will not contain the residual highpass or lowpass).
            like `self.pyr_coeffs`, keys are 2-tuples of ints indexing the scale and orientation,
            but now we're indexing `angles` instead of `self.num_orientations`.
        resteering_weights : `dict`
            dictionary of weights used to re-steer the pyramid coefficients. will have the same
            keys as `resteered_coeffs`.

        """

        resteered_coeffs = {}
        resteering_weights = {}
        for i in range(self.num_scales):
            basis = torch.cat([self.pyr_coeffs[(i, j)].squeeze().unsqueeze(-1) for j in
                               range(self.num_orientations)], dim=-1)

            for j, a in enumerate(angles):
                res, steervect = steer(basis, a, return_weights=True, even_phase=even_phase)
                resteering_weights[(i, j)] = steervect
                resteered_coeffs[(i, self.num_orientations + j)] = res.reshape(self.pyr_coeffs[(i, 0)].shape)


        return resteered_coeffs, resteering_weights
