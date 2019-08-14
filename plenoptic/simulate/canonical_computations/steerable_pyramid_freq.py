import warnings
import numpy as np
from scipy.special import factorial
from ...tools.signal import rcosFn, batch_fftshift2d, batch_ifftshift2d, pointOp
import torch
import torch.nn as nn
# from ..config import *
dtype = torch.float32


class Steerable_Pyramid_Freq(nn.Module):
    """Steerable frequency pyramid in Torch
    # TODO: adapt documentation to pytorch (batch, dtype, shapes, args)

    Construct a steerable pyramid on matrix IM, in the Fourier domain.
    This is similar to Spyr, except that:

        + Reconstruction is exact (within floating point errors)
        + It can produce any number of orientation bands.
        - Typically slower, especially for non-power-of-two sizes.
        - Boundary-handling is circular.

    The squared radial functions tile the Fourier plane with a raised-cosine falloff. Angular functions are cos(theta- k*pi/order+1)^(order).

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.
    For further information see the project webpage_

    Parameters
    ----------
    image : `array_like`
        2d image upon which to construct to the pyramid.
    height : 'auto' or `int`.
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

    Attributes
    ----------
    image : `array_like`
        The input image used to construct the pyramid.
    image_size : `tuple`
        The size of the input image.
    pyr_type : `str` or `None`
        Human-readable string specifying the type of pyramid. For base class, is None.
    pyr_coeffs : `dict`
        Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
        values are 1d or 2d numpy arrays (same number of dimensions as the input image)
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
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
                 store_unoriented_bands=False, return_list=False):

        super(Steerable_Pyramid_Freq, self).__init__()

        self.order = order
        self.image_shape = image_shape
        self.is_complex = is_complex
        self.store_unoriented_bands = store_unoriented_bands
        self.return_list = return_list

        # cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize + 1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi

        self.pyr_size = {}

        max_ht = np.floor(np.log2(min(self.image_shape[0], self.image_shape[1]))) - 2
        if height == 'auto':
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        if self.order > 15 or self.order < 0:
            warnings.warn("order must be an integer in the range [0,15]. Truncating.")
            self.order = min(max(self.order, 0), 15)
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

        self.lo0mask = torch.tensor(lo0mask, dtype=dtype)[None,:,:,None]
        self.hi0mask = torch.tensor(hi0mask, dtype=dtype)[None,:,:,None]

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
        self.hi0mask = self.lo0mask.to(*args, **kwargs)
        return self

    def forward(self, x):
        pyr_coeffs = {}

        # create local variables from class variables
        Xrcos = self.Xrcos.copy()
        Yrcos = self.Yrcos.copy()
        YIrcos = self.YIrcos.copy()
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        lo0mask = self.lo0mask.clone()
        hi0mask = self.hi0mask.clone()

        # x is a torch tensor batch of images of size [N,C,W,H]
        imdft = torch.rfft(x, signal_ndim=2, onesided=False)
        imdft = batch_fftshift2d(imdft)

        # high-pass
        hi0dft = imdft * hi0mask
        hi0 = batch_ifftshift2d(hi0dft)
        hi0 = torch.ifft(hi0, signal_ndim=2)
        hi0_real = torch.unbind(hi0, -1)[0]
        # self.coeffout = [hi0_real]
        pyr_coeffs['residual_highpass'] = hi0_real


        lodft = imdft * lo0mask

        self._anglemasks = []
        self._himasks = []
        self._lomasks = []
        if self.store_unoriented_bands:
            self.unoriented_bands = []


        for i in range(self.num_scales):

            if self.store_unoriented_bands:
                lo0 = batch_ifftshift2d(lodft)
                lo0 = torch.ifft(lo0, signal_ndim=2)
                lo0_real = torch.unbind(lo0, -1)[0]
                self.unoriented_bands.append(lo0_real)

            Xrcos -= np.log2(2)
            const = (2 ** (2*self.order)) * (factorial(self.order, exact=True)**2) / float(self.num_orientations * factorial(2*self.order, exact=True))

            if self.is_complex:
                Ycosn = (2.0 * np.sqrt(const) * (np.cos(self.Xcosn) ** self.order) *
                     (np.abs(self.alpha) < np.pi/2.0).astype(int))

            else:
                Ycosn = np.sqrt(const) * (np.cos(self.Xcosn))**self.order

            himask = pointOp(log_rad, Yrcos, Xrcos)
            self._himasks.append(himask)
            himask = torch.tensor(himask, dtype=dtype)[None, :, :, None].to(x.device)

            anglemasks = []
            for b in range(self.num_orientations):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.num_orientations)
                anglemasks.append(anglemask)
                anglemask = torch.tensor(anglemask, dtype=dtype)[None, :, :, None].to(x.device)

                # bandpass filtering
                banddft = lodft * anglemask * himask
                banddft = torch.unbind(banddft, -1)
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                complex_const = np.power(np.complex(0, -1), self.order)
                banddft_real = complex_const.real * banddft[0] - complex_const.imag * banddft[1]
                banddft_imag = complex_const.real * banddft[1] + complex_const.imag * banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)

                band = batch_ifftshift2d(banddft)
                band = torch.ifft(band, signal_ndim=2)
                if not self.is_complex:
                    band = torch.unbind(band, -1)[0]
                    # self.coeffout.append(band)
                    pyr_coeffs[(i, b)] = band

                else:
                    # self.coeffout.append(band)
                    pyr_coeffs[(i, b)] = band

            self._anglemasks.append(anglemasks)

            # subsample lowpass
            dims = np.array([lodft.shape[2], lodft.shape[3]])
            ctr = np.ceil((dims+0.5)/2).astype(int)
            lodims = np.ceil((dims-0.5)/2).astype(int)
            loctr = np.ceil((lodims+0.5)/2).astype(int)
            lostart = ctr - loctr
            loend = lostart + lodims

            # subsample indices
            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

            # subsampling
            lodft = lodft[:, :, lostart[0]:loend[0], lostart[1]:loend[1], :]
            # filtering
            YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            self._lomasks.append(lomask)
            lomask = torch.tensor(lomask, dtype=dtype)[None, :, :, None].to(x.device)
            # convolution in spatial domain
            lodft = lodft * lomask

        # compute residual lowpass when height <=1
        lo0 = batch_ifftshift2d(lodft)
        lo0 = torch.ifft(lo0, signal_ndim=2)
        lo0_real = torch.unbind(lo0, -1)[0]
        # self.coeffout.append(lo0_real)

        pyr_coeffs['residual_lowpass'] = lo0_real

        if self.return_list:
            return [k for k in pyr_coeffs.values()]
        else:
            return pyr_coeffs

    # TODO
    # def steer_coeffs(self, angles, even_phase=True):

    # TODO
    def recon_pyr(self, coeff, twidth=1):

        if self.num_orientations != len(coeff[1]):
            raise Exception("Number of orientations in pyramid don't match coefficients")

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1

        dims = (coeff[0].shape[2], coeff[0].shape[1])

        ctr = np.ceil((dims+0.5)/2.0).astype(int)

        (xramp, yramp) = np.meshgrid((np.arange(1, dims[1]+1)-ctr[1]) / (dims[1]/2.),
                                     (np.arange(1, dims[0]+1)-ctr[0]) / (dims[0]/2.))
        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = rcosFn(twidth, (-twidth/2.0), np.array([0, 1]))
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1.0 - Yrcos**2))

        # from reconSFpyrLevs
        lutsize = 1024

        Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize

        order = self.num_orientations - 1
        const = (2**(2*order))*(factorial(order, exact=True)**2) / float(self.num_orientations*factorial(2*order, exact=True))
        Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order

        # lowest band
        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            nresdft = np.fft.fftshift(np.fft.fft2(pyr_coeffs['residual_lowpass']))
        else:
            nresdft = np.zeros_like(pyr_coeffs['residual_lowpass'])
        resdft = np.zeros(dim_list[1]) + 0j

        bounds = (0, 0, 0, 0)
        for idx in range(len(bound_list)-2, 0, -1):
            diff = (bound_list[idx][2]-bound_list[idx][0],
                    bound_list[idx][3]-bound_list[idx][1])
            bounds = (bounds[0]+bound_list[idx][0], bounds[1]+bound_list[idx][1],
                      bounds[0]+bound_list[idx][0] + diff[0],
                      bounds[1]+bound_list[idx][1] + diff[1])
            Xrcos -= np.log2(2.0)
        nlog_rad = log_rad[bounds[0]:bounds[2], bounds[1]:bounds[3]]

        nlog_rad_tmp = np.reshape(nlog_rad, (1, nlog_rad.shape[0]*nlog_rad.shape[1]))
        lomask = pointOp(nlog_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
        lomask = lomask.reshape(nresdft.shape[0], nresdft.shape[1])
        lomask = lomask + 0j
        resdft[bound_list[1][0]:bound_list[1][2],
               bound_list[1][1]:bound_list[1][3]] = nresdft * lomask

        # middle bands
        for idx in range(1, len(bound_list)-1):
            bounds1 = (0, 0, 0, 0)
            bounds2 = (0, 0, 0, 0)
            for boundIdx in range(len(bound_list) - 1, idx - 1, -1):
                diff = (bound_list[boundIdx][2]-bound_list[boundIdx][0],
                        bound_list[boundIdx][3]-bound_list[boundIdx][1])
                bound2tmp = bounds2
                bounds2 = (bounds2[0]+bound_list[boundIdx][0],
                           bounds2[1]+bound_list[boundIdx][1],
                           bounds2[0]+bound_list[boundIdx][0] + diff[0],
                           bounds2[1]+bound_list[boundIdx][1] + diff[1])
                bounds1 = bound2tmp
            nlog_rad1 = log_rad[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            nlog_rad2 = log_rad[bounds2[0]:bounds2[2], bounds2[1]:bounds2[3]]
            dims = dim_list[idx]
            nangle = angle[bounds1[0]:bounds1[2], bounds1[1]:bounds1[3]]
            YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
            if idx > 1:
                Xrcos += np.log2(2.0)
                nlog_rad2_tmp = np.reshape(nlog_rad2, (1, nlog_rad2.shape[0]*nlog_rad2.shape[1]))
                lomask = pointOp(nlog_rad2_tmp, YIrcos, Xrcos[0],
                                 Xrcos[1]-Xrcos[0])
                lomask = lomask.reshape(bounds2[2]-bounds2[0],
                                        bounds2[3]-bounds2[1])
                lomask = lomask + 0j
                nresdft = np.zeros(dim_list[idx]) + 0j
                nresdft[bound_list[idx][0]:bound_list[idx][2],
                        bound_list[idx][1]:bound_list[idx][3]] = resdft * lomask
                resdft = nresdft.copy()

            # reconSFpyrLevs
            if idx != 0 and idx != len(bound_list)-1:
                for b in range(self.num_orientations):
                    nlog_rad1_tmp = np.reshape(nlog_rad1,
                                               (1, nlog_rad1.shape[0]*nlog_rad1.shape[1]))
                    himask = pointOp(nlog_rad1_tmp, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

                    himask = himask.reshape(nlog_rad1.shape)
                    nangle_tmp = np.reshape(nangle, (1, nangle.shape[0]*nangle.shape[1]))
                    anglemask = pointOp(nangle_tmp, Ycosn,
                                        Xcosn[0]+np.pi*b/self.num_orientations,
                                        Xcosn[1]-Xcosn[0])

                    anglemask = anglemask.reshape(nangle.shape)
                    # either the coefficients will already be real-valued (if
                    # self.is_complex=False) or complex (if self.is_complex=True). in the
                    # former case, this np.real() does nothing. in the latter, we want to only
                    # reconstruct with the real portion
                    curLev = self.num_scales - 1 - (idx-1)
                    band = np.real(pyr_coeffs[(curLev, b)])
                    if (curLev, b) in recon_keys:
                        banddft = np.fft.fftshift(np.fft.fft2(band))
                    else:
                        banddft = np.zeros(band.shape)
                    resdft += ((np.power(-1+0j, 0.5))**(self.num_orientations-1) *
                               banddft * anglemask * himask)

        # apply lo0mask
        Xrcos += np.log2(2.0)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos, Xrcos[1]-Xrcos[0])

        lo0mask = lo0mask.reshape(dims[0], dims[1])
        resdft = resdft * lo0mask

        # residual highpass subband
        hi0mask = pointOp(log_rad, Yrcos, Xrcos, Xrcos[1]-Xrcos[0])

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 'residual_highpass' in recon_keys:
            hidft = np.fft.fftshift(np.fft.fft2(pyr_coeffs['residual_highpass']))
        else:
            hidft = np.zeros_like(pyr_coeffs['residual_highpass'])
        resdft += hidft * hi0mask

        outresdft = np.real(np.fft.ifft2(np.fft.ifftshift(resdft)))

        return outresdft
