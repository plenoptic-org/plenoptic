import torch
import torch.nn as nn
from ..canonical_computations.steerable_pyramid_freq import Steerable_Pyramid_Freq
from ...tools.signal import batch_fftshift
import numpy as np


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
        self.scales = ['pixel_statistics','residual_lowpass']+[ii for ii in range(n_scales-1,-1,-1)]+['residual_highpass']
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
        pyr_coeffs = self.pyr.forward(image)
        
        # STATISTIC: calculate pixel statistics

        pixel_statistics = torch.ones((6,1))
        pixel_statistics[0] = torch.mean(image)
        pixel_statistics[1] = torch.var(image)
        pixel_statistics[2] = Portilla_Simoncelli.skew(image)
        pixel_statistics[3] = Portilla_Simoncelli.kurtosis(image)
        pixel_statistics[4] = torch.min(image)
        pixel_statistics[5] = torch.max(image)        
        
        # subtract mean from lowest scale band
        pyr_coeffs['residual_lowpass'] = pyr_coeffs['residual_lowpass'] - torch.mean(pyr_coeffs['residual_lowpass'])
        

        # create two sets of coefficients: 1) magnitude of the pyramid coefficients, 2) real part of the pyramid coefficients
        magnitude_pyr_coeffs = dict();
        real_pyr_coeffs = dict();
        for key,val in pyr_coeffs.items():
            if self.pyr.is_complex==True and val.shape[-1] == 2:
                tmp = torch.unbind(val,-1)
                magnitude_pyr_coeffs[key] = ((tmp[0]**2+tmp[1]**2)**.5).squeeze()
                real_pyr_coeffs[key] = tmp[0].squeeze()
            else:
                magnitude_pyr_coeffs[key] = torch.abs(val).squeeze()
                real_pyr_coeffs[key] = val.squeeze()

        # STATISTIC: magnitude_means or the mean magnitude of each pyramid band
        magnitude_means = torch.empty([len(pyr_coeffs.keys()), 1])
        for bb, (key, val) in enumerate(magnitude_pyr_coeffs.items()):
            magnitude_means[bb] = torch.mean(val)
            magnitude_pyr_coeffs[key] = magnitude_pyr_coeffs[key] - magnitude_means[bb]   # subtract mean of magnitude

        # STATISTIC: auto_correlation or the central auto-correlation
        auto_correlation = torch.zeros([self.Na, self.Na, self.n_scales+1])
        # STATISTIC: skew_reconstructed or the skew of the unoriented bands
        skew_reconstructed = torch.empty((self.n_scales+1,1))
        # STATISTIC: kurtosis_reconstructed or the kurtosis of the unoriented bands
        kurtosis_reconstructed = torch.empty((self.n_scales+1,1))

        # low-pass filter the low-pass residual.  We're still not sure why the original matlab code does this...
        lowpass = pyr_coeffs['residual_lowpass']
        filterPyr = Steerable_Pyramid_Freq(lowpass.shape[-2:], height=0, order=1, tight_frame=False)
        _ = filterPyr.forward(lowpass.squeeze().unsqueeze(0).unsqueeze(0))
        reconstructed_image = filterPyr.pyr_coeffs['residual_lowpass'].squeeze()
        
        # Find the auto-correlation of the low-pass residual
        Sch = torch.min(torch.tensor(lowpass.shape[-2:])).to(float)
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2-1,la)))
        auto_correlation[la-le:la+le+1, la-le:la+le+1, self.n_scales], vari = self.compute_autocorr(reconstructed_image)
        skew_reconstructed[self.n_scales], kurtosis_reconstructed[self.n_scales] =  self.compute_skew_kurt(reconstructed_image,vari,pixel_statistics[1])

        # STATISTIC: auto_correlation_magnitude or the auto-correlation of each magnitude band
        auto_correlation_magnitude = torch.zeros([self.Na, self.Na, self.n_scales, self.n_orientations])
        for n_scales in range(self.n_scales-1, -1, -1):
            for nor in range(0, self.n_orientations):
                ch = magnitude_pyr_coeffs[(n_scales,nor)]
                Sch = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((Sch/2.0-1, la)))
                # Find the auto-correlation of the magnitude band
                auto_correlation_magnitude[la-le:la+le+1, la-le:la+le+1, n_scales, nor], vari = self.compute_autocorr(ch)
            
            reconstructed_image = Portilla_Simoncelli.expand(reconstructed_image,2)/4.0
            reconstructed_image = reconstructed_image[:,:,0].unsqueeze(0).unsqueeze(0)
            
            # reconstruct the unoriented band for this scale
            unoriented_band_pyr = Steerable_Pyramid_Freq(reconstructed_image.shape[-2:],height=1, order=self.n_orientations-1, is_complex=False,tight_frame=False);
            _ = unoriented_band_pyr.forward(reconstructed_image)
            for ii in range(0,self.n_orientations):
                unoriented_band_pyr.pyr_coeffs[(0,ii)] = real_pyr_coeffs[(n_scales,ii)].unsqueeze(0).unsqueeze(0)
            unoriented_band = unoriented_band_pyr.recon_pyr(levels=[0])
            
            # Add the unoriented band to the image reconstruction
            reconstructed_image = reconstructed_image + unoriented_band
            
            # Find auto-correlation of the reconstructed image
            auto_correlation[la-le:la+le+1, la-le:la+le+1, n_scales], vari = self.compute_autocorr(reconstructed_image)
            skew_reconstructed[n_scales], kurtosis_reconstructed[n_scales] =  self.compute_skew_kurt(reconstructed_image,vari,pixel_statistics[1])

        # compute the cross-correlation matrices of the coefficient magnitudes pyramid at the different levels and orientations
        cross_orientation_correlation_magnitude = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales+1)
        cross_scale_correlation_magnitude = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales)
 
        cross_orientation_correlation_real = torch.zeros(max(2*self.n_orientations,5), max(2*self.n_orientations,5), self.n_scales+1)
        cross_scale_correlation_real = torch.zeros(2*self.n_orientations, max(2*self.n_orientations,5), self.n_scales)

        for n_scales in range(0, self.n_scales):
            cousinSz = real_pyr_coeffs[(n_scales,0)].shape[0]*real_pyr_coeffs[(n_scales,0)].shape[1]
            if n_scales < self.n_scales-1:
                parents = torch.empty((cousinSz, self.n_orientations))
                rparents = torch.empty((cousinSz, self.n_orientations*2))
                
                for nor in range(0, self.n_orientations):
                    tmp = Portilla_Simoncelli.expand(pyr_coeffs[(n_scales+1,nor)].squeeze(),2)/4.0

                    rtmp = tmp[:,:,0]
                    itmp = tmp[:,:,1]

                    rtmp2 = (rtmp**2 + itmp**2)**.5 * torch.cos(2*torch.atan2(rtmp, itmp))
                    itmp2 = (rtmp**2 + itmp**2)**.5 * torch.sin(2*torch.atan2(rtmp, itmp))
                    
                    rparents[:, nor] = rtmp2.t().flatten()
                    rparents[:, nor+self.n_orientations] = itmp2.t().flatten()

                    tmp2 = (rtmp2**2+itmp2**2)**.5
                    parents[:,nor] = (tmp2 - tmp2.mean()).t().flatten()

            else:
                tmp = Portilla_Simoncelli.expand(real_pyr_coeffs['residual_lowpass'].squeeze(),2)/4.0
                tmp = tmp[:,:,0].t()
                rparents= torch.stack((tmp.flatten(), 
                    tmp.roll(1,0).flatten(),
                    tmp.roll(-1,0).flatten(),
                    tmp.roll(1,1).flatten(),
                    tmp.roll(-1,1).flatten()),1)

                parents=torch.empty((0))

            cousins = torch.stack(tuple([aa.t() for aa in [magnitude_pyr_coeffs[(n_scales,ii)] for ii in range(0,self.n_orientations)]])).view((self.n_orientations, cousinSz)).t()

            nc = cousins.shape[1]
            if parents.shape[0] > 0:
                np0 = parents.shape[1]
            else:
                np0 = 0
            cross_orientation_correlation_magnitude[0:nc, 0:nc, n_scales] = (cousins.t()@cousins)/cousinSz
            if np0 > 0:
                cross_scale_correlation_magnitude[0:nc, 0:np0, n_scales] = (cousins.t()@parents)/cousinSz
                if n_scales==self.n_scales-1:
                    cross_orientation_correlation_magnitude[0:np0, 0:np0, n_scales+1] = (parents.t()@parents)/(cousinSz/4.0)

            cousins = torch.stack(tuple([aa[:,:,0].t() for aa in [pyr_coeffs[(n_scales,ii)].squeeze() for ii in range(0,self.n_orientations)]])).view((self.n_orientations,cousinSz)).t()
            nrc = cousins.shape[1]
            nrp = 0
            if rparents.shape[0]>0:
                nrp = rparents.shape[1]

            cross_orientation_correlation_real[0:nrc,0:nrc,n_scales]=(cousins.t()@cousins)/cousinSz
            if nrp>0:
                cross_scale_correlation_real[0:nrc,0:nrp,n_scales] = (cousins.t()@rparents)/cousinSz
                
                if n_scales==self.n_scales-1:
                    
                    cross_orientation_correlation_real[0:nrp,0:nrp,n_scales+1]=(rparents.t()@rparents)/(cousinSz/4.0)

        # STATISTC: var_highpass_residual or the variance of the high-pass residual
        var_highpass_residual = pyr_coeffs['residual_highpass'].pow(2).mean()

        representation = torch.cat((pixel_statistics.flatten(),magnitude_means.flatten(),
            auto_correlation_magnitude.flatten(), skew_reconstructed.flatten(),
            kurtosis_reconstructed.flatten(),auto_correlation.flatten(), cross_orientation_correlation_magnitude.flatten(), 
            cross_scale_correlation_magnitude.flatten(), cross_orientation_correlation_real.flatten(), 
            cross_scale_correlation_real.flatten(), var_highpass_residual.unsqueeze(0))) 

        if self.normalize:
            representation = (representation-self.normalization_mu) / self.normalization_std

        # This is likely NOT efficient and should be replaced [did some timing tests... and it actually doesn't seem to matter]
        if scales is not None:
            ind = torch.LongTensor([i for i,s in enumerate(self.statistic_scales) if s in self.scales])
            return representation.index_select(0,ind)

        return representation


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


    def expand(t,f):

        t = t.squeeze()

        mx = t.shape[0]
        my = t.shape[1]
        my = f*my
        mx = f*mx

        Te = torch.zeros(my,mx,2)

        if len(t.shape)==2:
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
        return te

    def compute_autocorr(self,ch):

        Sch = torch.min(torch.tensor(ch.shape[-2:])).to(float)
        
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2.0-1,la)))

        # find the center of the channel
        cy = int(ch.shape[-1]/2)
        cx = int(ch.shape[-2]/2)

        # calculate the auto-correlation
        ac = torch.rfft(ch.squeeze(), 2, onesided=False)
        ac = torch.stack((ac[:, :, 0].pow(2) + ac[:, :, 1].pow(2), torch.zeros_like(ac[:, :, 0])), -1)
        ac = torch.ifft(ac, 2)
        ac = batch_fftshift(ac.unsqueeze(0)).squeeze()/torch.numel(ch)
        ac = ac[cx-le:cx+le+1, cy-le:cy+le+1,0]
        vari = ac[le,le]

        return ac,vari


    def compute_skew_kurt(self,ch,vari,var0):

        if vari/var0 > 1e-6:
            skew_reconstructed = torch.mean(ch.pow(3))/(vari.pow(1.5))
            kurtosis_reconstructed = torch.mean(ch.pow(4))/(vari.pow(2))

        else:
            skew_reconstructed = 0
            kurtosis_reconstructed = 3
        return skew_reconstructed,kurtosis_reconstructed

    

    def skew(mtx):
        return torch.mean((mtx-mtx.mean()).pow(3))/(mtx.var().pow(1.5))

    def kurtosis(mtx):
        # implementation is only for real components
        return torch.mean(torch.abs(mtx-mtx.mean()).pow(4))/(mtx.var().pow(2))


