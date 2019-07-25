import warnings
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
from ...tools.signal import rcosFn, batch_fftshift2d, batch_ifftshift2d, pointOp
import torch
import torch.nn as nn
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, image_shape, height='auto', order=3, twidth=1, is_complex=False, store_unoriented_bands=False, return_list=False):
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
        self.complex_fact_reconstruct = np.power(np.complex(0, 1), self.order)
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

        self.lo0mask = torch.tensor(lo0mask, dtype=dtype)[None,None,:,:,None].to(device)
        self.hi0mask = torch.tensor(hi0mask, dtype=dtype)[None,None,:,:,None].to(device)


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

        #x = x.squeeze(1) #flatten channel dimension first
        imdft = torch.rfft(x, signal_ndim=2, onesided=False)
        imdft = batch_fftshift2d(imdft)

        # high-pass
        hi0dft = imdft * hi0mask
        hi0 = batch_ifftshift2d(hi0dft)
        hi0 = torch.ifft(hi0, signal_ndim=2)
        hi0_real = torch.unbind(hi0, -1)[0]
        pyr_coeffs['residual_highpass'] = hi0_real
        self.pyr_size['residual_highpass'] = tuple(hi0_real.shape[-2:])


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
            himask = torch.tensor(himask, dtype=dtype)[None, None, :, :, None].to(device)

            anglemasks = []
            for b in range(self.num_orientations):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.num_orientations)
                anglemasks.append(anglemask)
                anglemask = torch.tensor(anglemask, dtype=dtype)[None, None, :, :, None].to(device)

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
                    pyr_coeffs[(i, b)] = band
                    self.pyr_size[(i,b)] = tuple(band.shape[-2:])
                else:
                    #TODO: decide what to do with imaginary/real axis
                    band = torch.einsum('bchwi->bcihw', band)
                    pyr_coeffs[(i, b)] = band
                    self.pyr_size[(i,b)] = tuple(band.shape[-2:])

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
            lomask = torch.tensor(lomask, dtype=dtype)[None,None, :, :, None].to(device)
            # convolution in spatial domain
            lodft = lodft * lomask

        # compute residual lowpass when height <=1
        lo0 = batch_ifftshift2d(lodft)
        lo0 = torch.ifft(lo0, signal_ndim=2)
        lo0_real = torch.unbind(lo0, -1)[0]

        pyr_coeffs['residual_lowpass'] = lo0_real
        self.pyr_size['residual_lowpass'] = tuple(lo0_real.shape[-2:])


        if self.return_list:
            return [k for k in pyr_coeffs.values()]
        else:
            return pyr_coeffs

    def _reorder_complex(self, pyr_coeffs):
        """helper function for reconstruction from complex pyramid filters.

        Torch ffts assume imaginary/real dimensions are the last dimension but in the
        pyramid we output filters such that the real/imaginary are treated as channels (2nd dim)
        This function reorders the channel back to the last dimension for processing of the ffts
        in the Reconstruction

        Arguments
        ----------
        pyr_coeffs: input coeff dict where each
        filter (except residuals) are tensors of shape (NCHW) (where C is of size 2 for the two imaginary/real dimensions)

        Returns
        -------
        coeff_out: output dict with the reordered filters

        """

        coeff_out = OrderedDict()
        for k in pyr_coeffs.keys():
            if pyr_coeffs[k].shape[1] == 2:
                coeff_out[k] = torch.einsum('nchw->nhwc', pyr_coeffs[k]).unsqueeze(1)
            else:
                coeff_out[k] = pyr_coeffs[k]

        return coeff_out

    def _recon_levels_check(self, levels):
        """Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which levels to include. This makes sure those levels are valid and gets them in the form
        we expect for the rest of the reconstruction. If the user passes `'all'`, this constructs
        the appropriate list (based on the values of `self.pyr_coeffs`).

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

    def recon_pyr(self, pyr_coeffs, levels = 'all', bands = 'all', twidth = 1):

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1

        recon_keys = self._recon_keys(levels, bands)
        pyr_coeffs = self._reorder_complex(self.pyr_coeffs)


        dims = np.array(self.pyr_size['residual_lowpass'])
        ctr = np.ceil((dims+0.5)/2.0).astype(int)

        (xramp, yramp) = np.meshgrid((np.arange(1, dims[1]+1)-ctr[1]) / (dims[1]/2.),
                                     (np.arange(1, dims[0]+1)-ctr[0]) / (dims[0]/2.))
        angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = np.log2(log_rad)

        # Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(twidth, (-twidth/2.0), np.array([0, 1]))
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1.0 - Yrcos**2)

        #create masks
        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        #hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None, None, :,:,None].to(device)
        #hi0mask = torch.from_numpy(hi0mask).float()[None, None, :,:,None].to(device)

        if 'residual_lowpass' in recon_keys:
            lodft = torch.rfft(pyr_coeffs['residual_lowpass'], signal_ndim=2, onesided=False)
            lodft = batch_fftshift2d(lodft)
        else:
            lodft = torch.zeros_like(self.pyr_coeffs['residual_lowpass'])

        lodft = lodft*lo0mask
        dims = np.array(pyr_coeffs[(self.num_scales-1,self.num_scales-1)].shape[2:4])
        lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)
        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)

        resdft[:,:, lostart[0]:loend[0], lostart[1]:loend[1],:] = lodft

        for i in range(self.num_scales):

            Xrcos += np.log(2)

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None, None, :, :,None]).float().to(device)

            Xcosn = np.pi * np.arange(-(2*self.lutsize+1), (self.lutsize+2)) / self.lutsize
            const = (2**(2*self.order))*(factorial(self.order, exact=True)**2) / float(self.num_orientations*factorial(2*self.order, exact=True))
            Ycosn = np.sqrt(const) * (np.cos(Xcosn))**self.order

            curlev = self.num_scales-1 - i
            orientdft = torch.zeros_like(pyr_coeffs[(curlev,0)]).to(device)

            anglemasks = []
            for b in range(self.num_orientations):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.num_orientations)
                anglemasks.append(anglemask)
                anglemask = torch.tensor(anglemask, dtype=dtype)[None, None, :, :, None].to(device)

                if (curlev,b) in recon_keys:
                    banddft = torch.fft(pyr_coeffs[(curlev,b)], signal_ndim=2)
                    banddft = batch_fftshift2d(banddft)
                    banddft = banddft * anglemask * himask
                    print(banddft.shape, anglemask.shape, himask.shape)
                    banddft = torch.unbind(banddft, -1)

                    banddft_real = self.complex_fact_reconstruct.real*banddft[0] - self.complex_fact_reconstruct.imag*banddft[1]
                    if self.is_complex:
                        banddft_imag = self.complex_fact_reconstruct.real*banddft[1] + self.complex_fact_reconstruct.imag*banddft[0]
                        banddft = torch.stack((banddft_real, banddft_imag), -1)
                    else:
                        banddft = banddft_real
                else:
                    banddft = torch.zeros_like(self.pyr_coeffs[(curlev,b)])

                orientdft = orientdft + banddft

            #now upsample and convolve the lowpass component

            dims = np.array(pyr_coeffs[(curlev,0)].shape[2:4])
            lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
            loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)
            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
            lomask = pointOp(nlog_rad, YIrcos, Xrcos)

            # Filtering
            lomask = pointOp(nlog_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None,None,:,:,None])
            lomask = lomask.float().to(device)

            resdft[:,:, lostart[0]:loend[0], lostart[1]:loend[1],:] = orientdft * lomask

        #add highpass residual
        hidft = torch.rfft(pyr_coeffs['highpass_residual'], signal_ndim=2, onesided=False)
        hidft = math.utils.batch_fftshift2d(hidft)

        outdft = resdft*lo0mask + hidft*hi0mask

        recons = math_utils.batch_ifftshift2d(outdft)
        recons = torch.ifft(recons, signal_ndim=2)
        recons = torch.unbind(recons,-1)[0]

        return recons
