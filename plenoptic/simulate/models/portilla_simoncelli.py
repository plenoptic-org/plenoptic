import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ...tools.signal import batch_fftshift
from ...tools.conv import blur_downsample
import numpy as np
from collections import OrderedDict 
import matplotlib.pyplot as plt
import matplotlib as mpl
from ...tools.display import clean_stem_plot


class Portilla_Simoncelli(nn.Module):
    r""" Model for measuring statistics originally proposed in [1] for synthesis.
    
    Currently we do not: support batch measurement of images.
    
    Parameters
    ----------
    n_scales: uint
        The number of pyramid scales used to measure the statistics (default=4)
    
    n_orientations: uint
        The number of orientations used to measure the statistics (default=4)
    
    spatial_corr_width: uint
        The width of the spatial cross- and auto-correlation statistics in the representation

    use_true_correlations: bool
        In the original Portilla-Simoncelli model the statistics in the representation
        that are labelled correlations were actually covariance matrices (i.e. not properly
        scaled).  In order to match the original statistics use_true_correlations must be 
        set to false. But in order to synthesize metamers from this model use_true_correlations
        must be set to true (default).

    Attributes
    ----------
    pyr: Steerable_Pyramid_Freq
        The complex steerable pyramid object used to calculate the portilla-simoncelli representation

    pyr_coeffs: OrderedDict
        The coefficients of the complex steerable pyramid.

    mag_pyr_coeffs: OrderedDict
        The magnitude of the pyramid coefficients.

    real_pyr_coeffs: OrderedDict
        The real parts of the pyramid coefficients.

    scales: list
        The names of the unique scales of coefficients in the pyramid.
    
    representation_scales: list
        The scale for each coefficient in its vector form

    representation: dictionary
        A dictionary containing the Portilla-Simoncelli statistics
    
    References
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on Joint Statistics of
    Complex Wavelet Coefficients. Int'l Journal of Computer Vision. 40(1):49-71, October, 2000.
    http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
    http://www.cns.nyu.edu/~lcv/texture/
    
    """
    def __init__(self, im_shape, n_scales=4, n_orientations=4, spatial_corr_width=9, use_true_correlations=True):
        super().__init__()

        self.image_shape = im_shape
        self.spatial_corr_width = spatial_corr_width
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = Steerable_Pyramid_Freq(self.image_shape, height=self.n_scales,
            order=self.n_orientations-1, is_complex=True,tight_frame=False)

        self.use_true_correlations = use_true_correlations
        self.scales = ['residual_lowpass']+[ii for ii in range(n_scales-1,-1,-1)]+['residual_highpass','pixel_statistics']
        self.representation_scales = self._get_representation_scales()


    def _get_representation_scales(self):
        r""" returns a vector that indicates the scale of each value in the representation (Portilla-Simoncelli statistics)
            
            """
        pixel_statistics = ['pixel_statistics']*6

        # magnitude_means
        magnitude_means = ['residual_highpass'] + [s for s in range(0,self.n_scales) for i in range(0,self.n_orientations)] \
                            + ['residual_lowpass']
        
        sc = [s for s in range(0,self.n_scales)]
        sc_lowpass =  sc + ['residual_lowpass']

        # skew_reconstructed
        skew_reconstructed = sc_lowpass

        # kurtosis_reconstructed
        kurtosis_reconstructed = sc_lowpass
        
        auto_correlation = (self.spatial_corr_width*self.spatial_corr_width) * sc_lowpass
        auto_correlation_magnitude = (self.spatial_corr_width*self.spatial_corr_width)*[s for s in sc for i in range(0,self.n_orientations)]
        
        cross_orientation_correlation_magnitude = (self.n_orientations*self.n_orientations) * sc_lowpass
        cross_orientation_correlation_real = (4*self.n_orientations*self.n_orientations) * sc_lowpass
        
        cross_scale_correlation_magnitude = (self.n_orientations*self.n_orientations) * sc
        cross_scale_correlation_real = (2*self.n_orientations*max(2*self.n_orientations,5)) * sc
        var_highpass_residual = [self.scales[-1]]

        scales = pixel_statistics + magnitude_means + auto_correlation_magnitude + skew_reconstructed + \
        kurtosis_reconstructed + auto_correlation + cross_orientation_correlation_magnitude + \
        cross_scale_correlation_magnitude + cross_orientation_correlation_real + \
        cross_scale_correlation_real + var_highpass_residual
        
        return scales


    def forward(self, image, scales=None):
        r"""Generate Texture Statistics representation of an image (see reference [1])
            
        Parameters
        ----------
        image : torch.tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d
        
        Returns
        =======
        representation_vector: torch.tensor
            A flattened tensor (1d) containing the measured representation statistics.
            
        """
        
        while image.ndimension() < 4:
            image = image.unsqueeze(0)

        self.pyr_coeffs = self.pyr.forward(image)
        
        self.representation = OrderedDict() 

        ### SECTION 1 (STATISTIC: pixel_statistics) ##################
        #  Calculate pixel statistics (mean, variance, skew, kurtosis, min, max).
        self.representation['pixel_statistics'] = OrderedDict()
        self.representation['pixel_statistics']['mean'] = torch.mean(image)
        self.representation['pixel_statistics']['var'] = torch.var(image)
        self.representation['pixel_statistics']['skew'] = Portilla_Simoncelli.skew(image)
        self.representation['pixel_statistics']['kurtosis'] = Portilla_Simoncelli.kurtosis(image)
        self.representation['pixel_statistics']['min'] = torch.min(image)
        self.representation['pixel_statistics']['max'] = torch.max(image)
        
        

        ### SECTION 2 (STATISTIC: mean_magnitude) ####################
        # Calculate the mean of the magnitude of each band of pyramid
        # coefficients.  Additionally, this section creates two
        # other dictionaries of coefficients: magnitude_pyr_coeffs
        # and real_pyr_coeffs, which contain the magnitude of the
        # pyramid coefficients and the real part of the pyramid
        # coefficients respectively.
        self.representation['magnitude_means'] = self.calculate_magnitude_means()

       
        ### SECTION 3 (STATISTICS: auto_correlation_magnitude, 
        #                          skew_reconstructed, 
        #                          kurtosis_reconstructed,
        #                          auto_correlation_reconstructed) #####
        #
        # Calculates the central auto-correlation of the magnitude of each
        # orientation/scale band.
        #
        # Calculates the skew and the kurtosis of the reconstructed
        # low-pass residuals (skew_reconstructed, kurtosis_reconstructed).
        #
        # Calculates the central auto-correlation of the low-pass residuals
        # for each scale of the pyrmid (auto_correlation_reconstructed),
        # where the residual at each scale is reconstructed from the
        # previous scale.  (Note: the lowpass residual of the pyramid
        # is low-pass filtered before this reconstruction process begins,
        # see below).

        # Initialize statistics
        self.representation['auto_correlation_magnitude'] = torch.zeros([self.spatial_corr_width, self.spatial_corr_width, self.n_scales, self.n_orientations])
        self.representation['skew_reconstructed'] = torch.empty((self.n_scales+1,1))
        self.representation['kurtosis_reconstructed'] = torch.empty((self.n_scales+1,1))
        self.representation['auto_correlation_reconstructed'] = torch.zeros([self.spatial_corr_width, self.spatial_corr_width, self.n_scales+1])
        
        self.calculate_autocorrelation_skew_kurtosis()


        ### SECTION 4 (STATISTICS: cross_orientation_correlation_magnitude, 
        #                          cross_scale_correlation_magnitude, 
        #                          cross_orientation_correlation_real,
        #                          cross_scale_correlation_real) ###########
        # Calculates cross-orientation and cross-scale correlations for the
        # real parts and the magnitude of the pyramid coefficients.
        #
        
        # Initialize statistics
        self.representation['cross_orientation_correlation_magnitude'] = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales+1)
        self.representation['cross_scale_correlation_magnitude'] = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales)
        self.representation['cross_orientation_correlation_real'] = torch.zeros(max(2*self.n_orientations,5), max(2*self.n_orientations,5), self.n_scales+1)
        self.representation['cross_scale_correlation_real'] = torch.zeros(2*self.n_orientations, max(2*self.n_orientations,5), self.n_scales)

        self.calculate_crosscorrelations()

        
        # STATISTIC: var_highpass_residual or the variance of the high-pass residual
        self.representation['var_highpass_residual'] = self.pyr_coeffs['residual_highpass'].pow(2).mean().unsqueeze(0)

        representation_vector = self.convert_to_vector() 

        # This is likely NOT efficient and should be replaced [did some timing tests... and it actually doesn't seem to matter]
        if scales is not None:
            ind = torch.LongTensor([i for i,s in enumerate(self.representation_scales) if s in self.scales])
            return representation_vector.index_select(0,ind)

        return representation_vector

    def convert_to_vector(self):
        r"""  Converts dictionary of statistics to a vector (for synthesis).

        Returns
        =======
         -- : torch.Tensor
            Flattened 1d vector of statistics.

        """

        list_of_stats = [torch.cat([vv.flatten() for vv in val.values()]) \
                        if isinstance(val,OrderedDict) \
                        else val.flatten() \
                        for (key,val) in self.representation.items()]
        return torch.cat(list_of_stats)

    # def convert_to_dict(self,vec):
    #     rep = OrderedDict()
    #     rep['pixel_statistics'] = OrderedDict()
    #     rep['pixel_statistics']['mean'] = vec[0]
    #     rep['pixel_statistics']['var'] = vec[1]
    #     rep['pixel_statistics']['skew'] = vec[2]
    #     rep['pixel_statistics']['kurtosis'] = vec[3]
    #     rep['pixel_statistics']['min'] = vec[4]
    #     rep['pixel_statistics']['max'] = vec[5]
        

    #     tmp = ['residual_lowpass'] + [(s,i) for s in range(0,self.n_scales) for i in range(0,self.n_orientations)] \
    #                         + ['residual_highpass']
    #     print(tmp)
    #     # print(self.representation['magnitude_means'])



    def calculate_magnitude_means(self):
        r""" Calculates the mean of the pyramid coefficient magnitudes.  Also
        stores two dictionaries, one containing the magnitudes of the pyramid
        coefficient and the other containing the real parts.
        
        Returns
        =======
        magnitude_means: OrderedDict
            The mean of the pyramid coefficient magnitudes.
        
        """

        # subtract mean from lowest scale band
        self.pyr_coeffs['residual_lowpass'] = self.pyr_coeffs['residual_lowpass'] - torch.mean(self.pyr_coeffs['residual_lowpass'])
        

        # calculate two new sets of coefficients: 1) magnitude of the pyramid coefficients, 2) real part of the pyramid coefficients
        self.magnitude_pyr_coeffs = OrderedDict();
        self.real_pyr_coeffs = OrderedDict();
        for key,val in self.pyr_coeffs.items():
            if key in ['residual_lowpass','residual_highpass']: # not complex
                self.magnitude_pyr_coeffs[key] = torch.abs(val).squeeze()
                self.real_pyr_coeffs[key] = val.squeeze()
            else: # complex
                tmp = torch.unbind(val,-1)
                self.magnitude_pyr_coeffs[key] = ((tmp[0]**2+tmp[1]**2)**.5).squeeze()
                self.real_pyr_coeffs[key] = tmp[0].squeeze()

        # STATISTIC: magnitude_means or the mean magnitude of each pyramid band
        magnitude_means = OrderedDict()
        for (key, val) in self.magnitude_pyr_coeffs.items():
            magnitude_means[key] = torch.mean(val)
            self.magnitude_pyr_coeffs[key] = self.magnitude_pyr_coeffs[key] - magnitude_means[key]   # subtract mean of magnitude

        return magnitude_means


    def expand(im,mult):
        r""" Resize an image (im) by a multiplier (mult).

        Parameters
        ----------
        im: torch.Tensor
            An image for expansion.

        mult: uint
            Multiplier by which to resize image.
        
        Returns
        =======
        

        """

        im = im.squeeze()
        ndim = len(im.shape)

        mx = im.shape[0]
        my = im.shape[1]
        my = mult*my
        mx = mult*mx

        fourier_large = torch.zeros(my,mx,2)

        if ndim==2:
            im=torch.stack((im,torch.zeros_like(im)),-1)

        fourier = mult**2*batch_fftshift(torch.fft(im,2).unsqueeze(0)).squeeze()

        y1=int(my/2 + 1 - my/(2*mult))
        y2=int(my/2 + my/(2*mult))
        x1=int(mx/2 + 1 - mx/(2*mult))
        x2=int(mx/2 + mx/(2*mult))


        fourier_large[y1:y2,x1:x2,:]=fourier[1:int(my/mult),1:int(mx/mult),:]
        fourier_large[y1-1,x1:x2,:]=fourier[0,1:int(mx/mult),:]/2;
        fourier_large[y2,x1:x2,:]=fourier[0,1:int(mx/mult),:].flip(1)/2;
        fourier_large[y1:y2,x1-1,:]=fourier[1:int(my/mult),0,:]/2;
        fourier_large[y1:y2,x2,:]=fourier[1:int(my/mult),0,:].flip(0)/2;
        esq=fourier[0,0,:]/4;
        fourier_large[y1-1,x1-1,:]=esq;
        fourier_large[y1-1,x2,:]=esq;
        fourier_large[y2,x1-1,:]=esq;
        fourier_large[y2,x2,:]=esq;

        fourier_large = batch_fftshift(fourier_large.unsqueeze(0)).squeeze()

        # finish this
        im_large = torch.ifft(fourier_large,2)

        if ndim==2:
            return(im_large[:,:,0])
        return im_large


    def calculate_autocorrelation_skew_kurtosis(self):
        r""" Calculate the autocorrelation for the real parts and magnitudes of the 
        coefficients. Calculate the skew and kurtosis at each scale.
        
        """

        # low-pass filter the low-pass residual.  We're still not sure why the original matlab code does this...
        lowpass = self.pyr_coeffs['residual_lowpass']
        filterPyr = Steerable_Pyramid_Freq(lowpass.shape[-2:], height=0, order=1, tight_frame=False)
        _ = filterPyr.forward(lowpass)
        reconstructed_image = filterPyr.pyr_coeffs['residual_lowpass'].squeeze()
        
        # Find the auto-correlation of the low-pass residual
        channel_size = torch.min(torch.tensor(lowpass.shape[-2:])).to(float)
        center = int(np.floor([(self.spatial_corr_width-1)/2]))
        le = int(np.min((channel_size/2-1,center)))
        self.representation['auto_correlation_reconstructed'][center-le:center+le+1, center-le:center+le+1, self.n_scales], vari = self.compute_autocorrelation(reconstructed_image)
        self.representation['skew_reconstructed'][self.n_scales],self.representation['kurtosis_reconstructed'][self.n_scales] = self.compute_skew_kurtosis(reconstructed_image,vari)

        
        for this_scale in range(self.n_scales-1, -1, -1):
            for nor in range(0, self.n_orientations):
                ch = self.magnitude_pyr_coeffs[(this_scale,nor)]
                channel_size = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((channel_size/2.0-1, center)))
                # Find the auto-correlation of the magnitude band
                self.representation['auto_correlation_magnitude'][center-le:center+le+1, center-le:center+le+1, this_scale, nor], vari = self.compute_autocorrelation(ch)
            
            reconstructed_image = Portilla_Simoncelli.expand(reconstructed_image,2)/4.0
            reconstructed_image = reconstructed_image.unsqueeze(0).unsqueeze(0)
            
            # reconstruct the unoriented band for this scale
            unoriented_band_pyr = Steerable_Pyramid_Freq(reconstructed_image.shape[-2:],height=1, order=self.n_orientations-1, is_complex=False,tight_frame=False);
            _ = unoriented_band_pyr.forward(reconstructed_image)
            for ii in range(0,self.n_orientations):
                unoriented_band_pyr.pyr_coeffs[(0,ii)] = self.real_pyr_coeffs[(this_scale,ii)].unsqueeze(0).unsqueeze(0)
            unoriented_band = unoriented_band_pyr.recon_pyr(levels=[0])
            
            # Add the unoriented band to the image reconstruction
            reconstructed_image = reconstructed_image + unoriented_band
            
            # Find auto-correlation of the reconstructed image
            self.representation['auto_correlation_reconstructed'][center-le:center+le+1, center-le:center+le+1, this_scale], vari = self.compute_autocorrelation(reconstructed_image)
            
            # Find skew and kurtosis of the reconstructed image
            self.representation['skew_reconstructed'][this_scale],self.representation['kurtosis_reconstructed'][this_scale] = self.compute_skew_kurtosis(reconstructed_image,vari)

    
    def calculate_crosscorrelations(self):
        r""" Calculate the cross-orientation and cross-scale correlations for the real parts
        and the magnitudes of the pyramid coefficients.

        """

        for this_scale in range(0, self.n_scales):
            band_num_el = self.real_pyr_coeffs[(this_scale,0)].numel()
            if this_scale < self.n_scales-1:
                next_scale_mag = torch.empty((band_num_el, self.n_orientations))
                next_scale_real = torch.empty((band_num_el, self.n_orientations*2))
                
                for nor in range(0, self.n_orientations):
                    upsampled = Portilla_Simoncelli.expand(self.pyr_coeffs[(this_scale+1,nor)].squeeze(),2)/4.0

                    # double the phase of the upsampled band -- why? so there is something to correlate (better explanation here)
                    X = (upsampled[:,:,0]**2 + upsampled[:,:,1]**2)**.5 * torch.cos(2*torch.atan2(upsampled[:,:,0], upsampled[:,:,1]))
                    Y = (upsampled[:,:,0]**2 + upsampled[:,:,1]**2)**.5 * torch.sin(2*torch.atan2(upsampled[:,:,0], upsampled[:,:,1]))
                    
                    # Save the components -- why both?
                    next_scale_real[:, nor] = X.t().flatten()
                    next_scale_real[:, nor+self.n_orientations] = Y.t().flatten()

                    # Save the magnitude
                    mag = (X**2+Y**2)**.5
                    next_scale_mag[:,nor] = (mag - mag.mean()).t().flatten()

            else:
                upsampled = Portilla_Simoncelli.expand(self.real_pyr_coeffs['residual_lowpass'].squeeze(),2)/4.0
                upsampled = upsampled.t()
                next_scale_real= torch.stack((upsampled.flatten(), 
                    upsampled.roll(1,0).flatten(),
                    upsampled.roll(-1,0).flatten(),
                    upsampled.roll(1,1).flatten(),
                    upsampled.roll(-1,1).flatten()),1)
                next_scale_mag=torch.empty((0))

            orientation_bands_mag = torch.stack(tuple([aa.t() for aa in [self.magnitude_pyr_coeffs[(this_scale,ii)] for ii in range(0,self.n_orientations)]])).view((self.n_orientations, band_num_el)).t()
            
            if next_scale_mag.shape[0] > 0:
                np0 = next_scale_mag.shape[1]
            else:
                np0 = 0
            self.representation['cross_orientation_correlation_magnitude'][0:self.n_orientations, 0:self.n_orientations, this_scale] = self.compute_crosscorrelation(orientation_bands_mag.t(),orientation_bands_mag,band_num_el)
            if np0 > 0:
                self.representation['cross_scale_correlation_magnitude'][0:self.n_orientations, 0:np0, this_scale] = self.compute_crosscorrelation(orientation_bands_mag.t(),next_scale_mag,band_num_el)
                if this_scale==self.n_scales-1: # correlations on the low-pass residuals
                    self.representation['cross_orientation_correlation_magnitude'][0:np0, 0:np0, this_scale+1] = self.compute_crosscorrelation(next_scale_mag.t(),next_scale_mag,band_num_el/4.0)
            

            orientation_bands_real = torch.stack(tuple([aa.t() for aa in [self.real_pyr_coeffs[(this_scale,ii)].squeeze() for ii in range(0,self.n_orientations)]])).view((self.n_orientations,band_num_el)).t()
            
            if next_scale_real.shape[0]>0:
                nrp = next_scale_real.shape[1]
            else:
                nrp=0
            self.representation['cross_orientation_correlation_real'][0:self.n_orientations,0:self.n_orientations,this_scale]=self.compute_crosscorrelation(orientation_bands_real.t(),orientation_bands_real,band_num_el)
            if nrp>0:
                self.representation['cross_scale_correlation_real'][0:self.n_orientations,0:nrp,this_scale] = self.compute_crosscorrelation(orientation_bands_real.t(),next_scale_real,band_num_el)
                if this_scale==self.n_scales-1: # correlations on the low-pass residuals
                    self.representation['cross_orientation_correlation_real'][0:nrp,0:nrp,this_scale+1]=self.compute_crosscorrelation(next_scale_real.t(),next_scale_real,(band_num_el/4.0))



    def compute_crosscorrelation(self,ch1,ch2,band_num_el):
        r""" Computes either the covariance of the two matrices or the cross-correlation
        depending on the value self.use_true_correlations.

        Parameters
        ----------
        ch1: torch.Tensor
            First matrix for cross correlation.
        ch2: torch.Tensor
            Second matrix for cross correlation.
        band_num_el: uint
            Number of elements for bands in the scale
        
        Returns
        =======

        --: torch.Tensor
            cross-correlation.
        
        """

        if self.use_true_correlations:
            return ch1@ch2/(band_num_el*ch1.std()*ch2.std())
        else:
            return ch1@ch2/(band_num_el)


    def compute_autocorrelation(self,ch):
        r""" Computes the autocorrelation and variance of a given matrix (ch)

        Parameters
        ----------
        ch: torch.Tensor

        
        Returns
        =======
        ac: torch.Tensor
            Autocorrelation of matrix (ch).

        vari: torch.Tensor
            Variance of matrix (ch).
        
        """

        channel_size = torch.min(torch.tensor(ch.shape[-2:])).to(float)
        
        # Calculate the edges of the central auto-correlation
        center = int(np.floor([(self.spatial_corr_width-1)/2]))
        le = int(np.min((channel_size/2.0-1,center)))

        # Find the center of the channel
        cy = int(ch.shape[-1]/2)
        cx = int(ch.shape[-2]/2)

        # Calculate the auto-correlation
        ac = torch.rfft(ch.squeeze(), 2, onesided=False)
        ac = torch.stack((ac[:, :, 0].pow(2) + ac[:, :, 1].pow(2), torch.zeros_like(ac[:, :, 0])), -1)
        ac = torch.ifft(ac, 2)
        ac = batch_fftshift(ac.unsqueeze(0)).squeeze()/torch.numel(ch)
        
        # Return only the central auto-correlation
        ac = ac[cx-le:cx+le+1, cy-le:cy+le+1,0]
        vari = ac[le,le]

        
        if self.use_true_correlations:
            ac = ac/vari

        return ac,vari

    def compute_skew_kurtosis(self,ch,vari):
        r""" Computes the skew and kurtosis of ch given the ratio of its
        variance (vari) and the pixel variance of the original image are
        above a certain threshold.  If the ratio does not meet that threshold
        it returns the default values (0,3).

        Parameters
        ----------
        ch: torch.Tensor


        vari: torch.Tensor
            variance of ch
        
        Returns
        =======

        skew: torch.Tensor
            skew of ch or default value (0)

        kurtosis: torch.Tensor
            kurtosis of ch or default value (3)
        
        """
        
        # Find the skew and the kurtosis of the low-pass residual
        if vari/self.representation['pixel_statistics']['var']> 1e-6:
            skew = Portilla_Simoncelli.skew(ch,mu=0,var=vari)
            kurtosis =Portilla_Simoncelli.kurtosis(ch,mu=0,var=vari)

        else:
            skew = 0
            kurtosis = 3

        return skew, kurtosis


    def skew(X, mu=None,var=None):
        r""" Computes the skew of a matrix X.

        Parameters
        ----------
        X: torch.Tensor

        mu: torch.Tensor
            pre-computed mean.

        var: torch.Tensor
            pre-computed variance.
        
        Returns
        =======

        skew: torch.Tensor
            skew of the matrix X
        
        """

        if mu is None:
            mu = X.mean()
        if var is None:
            var=X.var()
        return torch.mean((X-mu).pow(3))/(var.pow(1.5))

    def kurtosis(X,mu=None,var=None):
        r""" Computes the kurtosis of a matrix X.

        Parameters
        ----------
        X: torch.Tensor

        mu: torch.Tensor
            pre-computed mean.

        var: torch.Tensor
            pre-computed variance.
        
        Returns
        =======

        kurtosis: torch.Tensor
            kurtosis of the matrix X
        
        """

        # implementation is only for real components
        if mu is None:
            mu = X.mean()
        if var is None:
            var=X.var()
        return torch.mean(torch.abs(X-mu).pow(4))/(var.pow(2))
    
    # def plot_representation(self,figsize=(15, 15), ylim=None, ax=None, title=None, batch_idx=0, data=None):
    #     n_rows = 3
    #     n_cols = 3

    #     if data is None:
    #         rep = self.representation
    #         data = OrderedDict()
    #         data['pixels+var_highpass'] = rep['pixel_statistics']
    #         data['pixels+var_highpass']['var_highpass_residual'] = rep['var_highpass_residual']
    #         data['skew+kurtosis'] = torch.stack((rep['skew_reconstructed'],rep['kurtosis_reconstructed']))

    #         for (k,v) in rep.items():
    #             if k not in ['pixel_statistics','var_highpass_residual','kurtosis_reconstructed','skew_reconstructed']:
    #                 data[k] = v
    #         # data = self.representation

    #     # Set up grid spec
    #     if ax is None:
    #         # we add 2 to order because we're adding one to get the
    #         # number of orientations and then another one to add an
    #         # extra column for the mean luminance plot
    #         fig = plt.figure(figsize=figsize)
    #         gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
    #     else:
    #         warnings.warn("ax is not None, so we're ignoring figsize...")
    #         # want to make sure the axis we're taking over is basically invisible.
    #         ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
    #         gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
    #         fig = ax.figure
        
    #     # if isinstance(title, str):
    #     #     # then this is a single str, so we'll make it the same on
    #     #     # every subplot
    #     #     title = (n_rows * n_cols) * [title]

    #     # plot data
    #     axes = []
    #     for i, (k, v) in enumerate(data.items()):

    #         if isinstance(v,OrderedDict):
    #             ax = fig.add_subplot(gs[i//3,i%3])
    #             ax = clean_stem_plot(list(v.values()),ax,k,False)
    #             if k == 'pixels+var_highpass':
    #                 print('here')
    #         else:
    #             ax = fig.add_subplot(gs[i//3,i%3])
    #             ax = clean_stem_plot(v.flatten().detach().numpy(),ax,k,False)
            
    #         axes.append(ax)


    #     return fig, axes


