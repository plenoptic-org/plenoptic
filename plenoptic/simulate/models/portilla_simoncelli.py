import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ...tools.signal import batch_fftshift
from ...tools.conv import blur_downsample
import numpy as np
from collections import OrderedDict 


class Portilla_Simoncelli(nn.Module):
    ''' Model for measuring statistics originally proposed in [1] for synthesis.
    
    Currently we do not: support batch measurement of images.
    
    Parameters
    ----------
    n_scales:
    
    n_orientations:
    
    Na:

    normalize:

    normalizeFactor:

    
    Attributes
    ----------
    pyr:

    representation:


    
    
    References
    -----
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on Joint Statistics of
    Complex Wavelet Coefficients. Int'l Journal of Computer Vision. 40(1):49-71, October, 2000.
    http://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
    http://www.cns.nyu.edu/~lcv/texture/
    
    TODO
    ----
    = [ ] Operate on Steerable Pyramid coefficients in dictionaries not lists.
    
    '''
    def __init__(self, im_shape, n_scales=4, n_orientations=4, Na=9,normalize=False,normalization_mu=None,normalization_std=None):
        super(Portilla_Simoncelli, self).__init__()

        self.image_shape = im_shape
        self.Na = Na
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = Steerable_Pyramid_Freq(self.image_shape, height=self.n_scales,
            order=self.n_orientations-1, is_complex=True,tight_frame=False)

        self.normalize = normalize
        self.normalization_mu = normalization_mu
        self.normalization_std = normalization_std
        self.scales = ['residual_lowpass']+[ii for ii in range(n_scales-1,-1,-1)]+['residual_highpass','pixel_statistics']
        self.statistic_scales = self._get_statistic_scales()

    def forward(self, image, scales=None):
        """Generate Texture Statistics representation of an image (see reference [1])
            
        Parameters
        ----------
        image : torch.tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d
        
        Returns
        =======
        representation: torch.tensor
            A flattened tensor (1d) containing the measured statistics.
            
        """
        
        while image.ndimension() < 4:
            image = image.unsqueeze(0)

        self.pyr_coeffs = self.pyr.forward(image)
        
        stats = OrderedDict() 

        ### SECTION 1 (STATISTIC: pixel_statistics) ##################
        #  Calculate pixel statistics.
        stats['pixel_statistics'] = OrderedDict()
        stats['pixel_statistics']['mean'] = torch.mean(image)
        stats['pixel_statistics']['var'] = torch.var(image)
        stats['pixel_statistics']['skew'] = Portilla_Simoncelli.skew(image)
        stats['pixel_statistics']['kurtosis'] = Portilla_Simoncelli.kurtosis(image)
        stats['pixel_statistics']['min'] = torch.min(image)
        stats['pixel_statistics']['max'] = torch.max(image)
        
        

        ### SECTION 2 (STATISTIC: mean_magnitude) ####################
        # Calculates the mean of the magnitude of each band of pyramid
        # coefficients.  Additionally, this section creates two
        # additonal dictionaries of coefficients: magnitude_pyr_coeffs
        # and real_pyr_coeffs, which contain the magnitude of the
        # pyramid coefficients and the real part of the pyramid
        # coefficients respectively.

        # STATISTIC: magnitude_means or the mean magnitude of each pyramid band
        stats['magnitude_means'] = self.calculate_magnitude_means()

       
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

        # STATISTIC: auto_correlation_magnitude or the auto-correlation of each magnitude band
        stats['auto_correlation_magnitude'] = torch.zeros([self.Na, self.Na, self.n_scales, self.n_orientations])
        # STATISTIC: skew_reconstructed or the skew of the 
        stats['skew_reconstructed'] = torch.empty((self.n_scales+1,1))
        # STATISTIC: kurtosis_reconstructed or the kurtosis of the 
        stats['kurtosis_reconstructed'] = torch.empty((self.n_scales+1,1))
        # STATISTIC: auto_correlation or the central auto-correlation
        stats['auto_correlation_reconstructed'] = torch.zeros([self.Na, self.Na, self.n_scales+1])
        
        # low-pass filter the low-pass residual.  We're still not sure why the original matlab code does this...
        lowpass = self.pyr_coeffs['residual_lowpass']
        filterPyr = Steerable_Pyramid_Freq(lowpass.shape[-2:], height=0, order=1, tight_frame=False)
        _ = filterPyr.forward(lowpass)
        reconstructed_image = filterPyr.pyr_coeffs['residual_lowpass'].squeeze()
        
        # Find the auto-correlation of the low-pass residual
        Sch = torch.min(torch.tensor(lowpass.shape[-2:])).to(float)
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2-1,la)))
        stats['auto_correlation_reconstructed'][la-le:la+le+1, la-le:la+le+1, self.n_scales], vari = self.compute_autocorr(reconstructed_image)
        
        # Find the skew and the kurtosis of the low-pass residual
        if vari/stats['pixel_statistics']['var']> 1e-6:
            stats['skew_reconstructed'][self.n_scales] = Portilla_Simoncelli.skew(reconstructed_image,mu=0,var=vari)
            stats['kurtosis_reconstructed'][self.n_scales] =Portilla_Simoncelli.kurtosis(reconstructed_image,mu=0,var=vari)

        else:
            stats['skew_reconstructed'][self.n_scales] = 0
            stats['kurtosis_reconstructed'][self.n_scales] = 3
        
        for this_scale in range(self.n_scales-1, -1, -1):
            for nor in range(0, self.n_orientations):
                ch = self.magnitude_pyr_coeffs[(this_scale,nor)]
                Sch = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((Sch/2.0-1, la)))
                # Find the auto-correlation of the magnitude band
                stats['auto_correlation_magnitude'][la-le:la+le+1, la-le:la+le+1, this_scale, nor], vari = self.compute_autocorr(ch)
            
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
            stats['auto_correlation_reconstructed'][la-le:la+le+1, la-le:la+le+1, this_scale], vari = self.compute_autocorr(reconstructed_image)
            
            # Find skew and kurtosis of the reconstructed image
            if vari/stats['pixel_statistics']['var']> 1e-6:
                stats['skew_reconstructed'][this_scale] = Portilla_Simoncelli.skew(reconstructed_image,mu=0,var=vari)
                stats['kurtosis_reconstructed'][this_scale] =Portilla_Simoncelli.kurtosis(reconstructed_image,mu=0,var=vari)

            else:
                stats['skew_reconstructed'][this_scale] = 0
                stats['kurtosis_reconstructed'][this_scale] = 3



        ### SECTION 4 (STATISTICS: cross_orientation_correlation_magnitude, 
        #                          cross_scale_correlation_magnitude, 
        #                          cross_orientation_correlation_real,
        #                          cross_scale_correlation_real) ###########
        # Calculates cross-orientation and cross-scale correlations for the
        # real parts and the magnitude of the pyramid coefficients.
        #
        #



        # compute the cross-correlation matrices of the coefficient magnitudes pyramid at the different levels and orientations
        stats['cross_orientation_correlation_magnitude'] = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales+1)
        stats['cross_scale_correlation_magnitude'] = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales)
 
        stats['cross_orientation_correlation_real'] = torch.zeros(max(2*self.n_orientations,5), max(2*self.n_orientations,5), self.n_scales+1)
        stats['cross_scale_correlation_real'] = torch.zeros(2*self.n_orientations, max(2*self.n_orientations,5), self.n_scales)

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
            stats['cross_orientation_correlation_magnitude'][0:self.n_orientations, 0:self.n_orientations, this_scale] = (orientation_bands_mag.t()@orientation_bands_mag)/band_num_el
            if np0 > 0:
                stats['cross_scale_correlation_magnitude'][0:self.n_orientations, 0:np0, this_scale] = (orientation_bands_mag.t()@next_scale_mag)/band_num_el
                if this_scale==self.n_scales-1: # correlations on the low-pass residuals
                    stats['cross_orientation_correlation_magnitude'][0:np0, 0:np0, this_scale+1] = (next_scale_mag.t()@next_scale_mag)/(band_num_el/4.0)
            

            orientation_bands_real = torch.stack(tuple([aa.t() for aa in [self.real_pyr_coeffs[(this_scale,ii)].squeeze() for ii in range(0,self.n_orientations)]])).view((self.n_orientations,band_num_el)).t()
            
            if next_scale_real.shape[0]>0:
                nrp = next_scale_real.shape[1]
            else:
                nrp=0
            stats['cross_orientation_correlation_real'][0:self.n_orientations,0:self.n_orientations,this_scale]=(orientation_bands_real.t()@orientation_bands_real)/band_num_el
            if nrp>0:
                stats['cross_scale_correlation_real'][0:self.n_orientations,0:nrp,this_scale] = (orientation_bands_real.t()@next_scale_real)/band_num_el
                if this_scale==self.n_scales-1: # correlations on the low-pass residuals
                    stats['cross_orientation_correlation_real'][0:nrp,0:nrp,this_scale+1]=(next_scale_real.t()@next_scale_real)/(band_num_el/4.0)

        # STATISTIC: var_highpass_residual or the variance of the high-pass residual
        stats['var_highpass_residual'] = self.pyr_coeffs['residual_highpass'].pow(2).mean().unsqueeze(0)

        representation = Portilla_Simoncelli._convert_to_vector(stats) 

        if self.normalize:
            representation = (representation-self.normalization_mu) / self.normalization_std

        # This is likely NOT efficient and should be replaced [did some timing tests... and it actually doesn't seem to matter]
        if scales is not None:
            ind = torch.LongTensor([i for i,s in enumerate(self.statistic_scales) if s in self.scales])
            return representation.index_select(0,ind)

        return representation

    def _convert_to_vector(stats):

        list_of_stats = [torch.cat([vv.flatten() for vv in val.values()]) \
                        if isinstance(val,OrderedDict) \
                        else val.flatten() \
                        for (key,val) in stats.items()]
        return torch.cat(list_of_stats)


    def _get_statistic_scales(self):
        """
            
            """
        pixel_statistics = ['pixel_statistics']*6

        # magnitude_means
        magnitude_means = ['residual_lowpass'] + [s for s in self.scales[2:-1] for i in range(0,self.n_orientations)] \
                            + ['residual_highpass']
        
        # skew_reconstructed
        skew_reconstructed = self.scales[1:-1]

        # kurtosis_reconstructed
        kurtosis_reconstructed = self.scales[1:-1]
        
        auto_correlation = (self.Na*self.Na)*self.scales[1:-1]
        cross_orientation_correlation_magnitude = (self.n_orientations*self.n_orientations) * self.scales[1:-1]
        cross_orientation_correlation_real = (4*self.n_orientations*self.n_orientations) * self.scales[1:-1]
        
        auto_correlation_magnitude = (self.Na*self.Na)*[s for s in self.scales[2:-1] for i in range(0,self.n_orientations)]
        cross_scale_correlation_magnitude = (self.n_orientations*self.n_orientations) * self.scales[2:-1]
        cross_scale_correlation_real = (2*self.n_orientations*max(2*self.n_orientations,5)) * self.scales[2:-1]
        var_highpass_residual = [self.scales[-1]]

        scales = pixel_statistics + magnitude_means + auto_correlation_magnitude + skew_reconstructed + \
        kurtosis_reconstructed + auto_correlation + cross_orientation_correlation_magnitude + \
        cross_scale_correlation_magnitude + cross_orientation_correlation_real + \
        cross_scale_correlation_real + var_highpass_residual
        
        return scales

    def calculate_magnitude_means(self):

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


    def expand(t,f):

        t = t.squeeze()
        ndim = len(t.shape)

        mx = t.shape[0]
        my = t.shape[1]
        my = f*my
        mx = f*mx

        Te = torch.zeros(my,mx,2)

        if ndim==2:
            t=torch.stack((t,torch.zeros_like(t)),-1)

        T =  f**2*batch_fftshift(torch.fft(t,2).unsqueeze(0)).squeeze()

        y1=int(my/2 + 1 - my/(2*f))
        y2=int(my/2 + my/(2*f))
        x1=int(mx/2 + 1 - mx/(2*f))
        x2=int(mx/2 + mx/(2*f))


        Te[y1:y2,x1:x2,:]=T[1:int(my/f),1:int(mx/f),:]
        Te[y1-1,x1:x2,:]=T[0,1:int(mx/f),:]/2;
        Te[y2,x1:x2,:]=T[0,1:int(mx/f),:].flip(1)/2;
        Te[y1:y2,x1-1,:]=T[1:int(my/f),0,:]/2;
        Te[y1:y2,x2,:]=T[1:int(my/f),0,:].flip(0)/2;
        esq=T[0,0,:]/4;
        Te[y1-1,x1-1,:]=esq;
        Te[y1-1,x2,:]=esq;
        Te[y2,x1-1,:]=esq;
        Te[y2,x2,:]=esq;

        Te = batch_fftshift(Te.unsqueeze(0)).squeeze()

        # finish this
        te = torch.ifft(Te,2)

        if ndim==2:
            return(te[:,:,0])
        return te

    def compute_autocorr(self,ch):

        Sch = torch.min(torch.tensor(ch.shape[-2:])).to(float)
        
        # Calculate the edges of the central auto-correlation
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2.0-1,la)))

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

        return ac,vari


    def skew(X, mu=None,var=None):
        if mu is None:
            mu = X.mean()
        if var is None:
            var=X.var()
        return torch.mean((X-mu).pow(3))/(var.pow(1.5))

    def kurtosis(X,mu=None,var=None):
        # implementation is only for real components
        if mu is None:
            mu = X.mean()
        if var is None:
            var=X.var()
        return torch.mean(torch.abs(X-mu).pow(4))/(var.pow(2))

