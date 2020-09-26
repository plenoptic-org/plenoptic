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
    def __init__(self, im_shape, n_scales=4, n_orientations=4, Na=9,normalize=False,normalizationFactor=None):
        super(Portilla_Simoncelli, self).__init__()

        self.image_shape = im_shape
        self.Na = Na
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = Steerable_Pyramid_Freq(self.image_shape, height=self.n_scales,
            order=self.n_orientations-1, is_complex=True,tight_frame=False)

        self.normalize = normalize
        self.normalizationFactor = normalizationFactor


    def forward(self, image):
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
        pyr_coeffs_dict = self.pyr.forward(image)
        pyr_coeffs = [pyr_coeffs_dict[key] for key in pyr_coeffs_dict.keys()]

        # STATISTIC: pixel statistics
        pixel_statistics = torch.zeros((6,1))
        pixel_statistics[0] = torch.mean(image)
        pixel_statistics[1] = torch.var(image)
        pixel_statistics[2] = Portilla_Simoncelli.skew(image)
        pixel_statistics[3] = Portilla_Simoncelli.kurtosis(image)
        pixel_statistics[4] = torch.min(image)
        pixel_statistics[5] = torch.max(image)
        
        
        # subtract mean from lowest scale band
        nbands = len(pyr_coeffs)
        pyr_coeffs[-1] = (pyr_coeffs[-1])-torch.mean((pyr_coeffs[-1]))

        
        apyr = []
        rpyr = []
        for bb in range(0, nbands):

            if self.pyr.is_complex==True and pyr_coeffs[bb].shape[-1] == 2:  # if the pyramid complex
            # if self.complex_steerable_pyramid.complex and len(pyr_coeffs[bb].shape)==3:
                tmp = torch.unbind(pyr_coeffs[bb],-1)
                apyr.append(((tmp[0]**2+tmp[1]**2)**.5).squeeze())
                rpyr.append(tmp[0].squeeze())
            else:
                rpyr.append(pyr_coeffs[bb].squeeze())
                apyr.append(torch.abs(pyr_coeffs[bb]).squeeze())

            pyr_coeffs[bb] = pyr_coeffs[bb].squeeze()

        # STATISTIC: magMeans0 or the mean magnitude of each pyramid band
        magMeans0 = torch.empty([len(pyr_coeffs), 1])
        for bb in range(0, nbands):
            magMeans0[bb] = torch.mean(apyr[bb])
            apyr[bb] = apyr[bb] - magMeans0[bb]   # subtract mean of magnitude
        print(magMeans0)
        # STATISTIC: acr or the central auto-correlation
        acr = torch.zeros([self.Na, self.Na, self.n_scales+1])
        # STATISTIC: skew0p or the skew of the unoriented bands
        skew0p = torch.empty((self.n_scales+1,1))
        # STATISTIC: kurt0p or the kurtosis of the unoriented bands
        kurt0p = torch.empty((self.n_scales+1,1))

        # high-pass filter the low-pass residual.  We're still not sure why the original paper does this...
        ch = pyr_coeffs[-1]
        mpyrM = Steerable_Pyramid_Freq(ch.shape[-2:], height=0, order=0,tight_frame=False)
        mpyr_dict = mpyrM.forward(ch.squeeze().unsqueeze(0).unsqueeze(0))
        mpyr = [mpyr_dict[key] for key in mpyr_dict.keys()]
        im = mpyr[1].squeeze()
        
        # Find the auto-correlation of the low-pass residual
        Sch = torch.min(torch.tensor(ch.shape[-2:])).to(float)
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2-1,la)))
        self.acr_im = im
        acr[la-le:la+le+1, la-le:la+le+1, self.n_scales], vari = self.compute_autocorr(im)
        skew0p[self.n_scales], kurt0p[self.n_scales] =  self.compute_skew_kurt(im,vari,pixel_statistics[1])

        # STATISTIC: ace or the auto-correlation of each magnitude band
        ace = torch.zeros([self.Na, self.Na, self.n_scales, self.n_orientations])
        for n_scales in range(self.n_scales-1, -1, -1):
            for nor in range(0, self.n_orientations):
                nband = n_scales*self.n_orientations + nor + 1
                ch = apyr[nband]
                Sch = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((Sch/2.0-1, la)))
                # Find the auto-correlation of the magnitude band
                ace[la-le:la+le+1, la-le:la+le+1, n_scales, nor], vari = self.compute_autocorr(ch)
            
            im = Portilla_Simoncelli.expand(im,2)/4.0
            im = im[:,:,0].unsqueeze(0).unsqueeze(0)
            
            # reconstruct unoriented band
            tmp_pyr = Steerable_Pyramid_Freq(im.shape[-2:],height=1, order=self.n_orientations-1, is_complex=False,tight_frame=False);
            _ = tmp_pyr.forward(im)
            for ii in range(0,self.n_orientations):
                tmp_pyr.pyr_coeffs[(0,ii)] = rpyr[n_scales*self.n_orientations+1+ii].unsqueeze(0).unsqueeze(0)
            ch = tmp_pyr.recon_pyr(levels=[0])
            
            im = im + ch
            
            # Find auto-correlation of the unoriented band
            acr[la-le:la+le+1, la-le:la+le+1, n_scales], vari = self.compute_autocorr(im)
            skew0p[n_scales], kurt0p[n_scales] =  self.compute_skew_kurt(im,vari,pixel_statistics[1])

        # compute the cross-correlation matrices of the coefficient magnitudes pyramid at the different levels and orientations
        C0 = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales+1)
        Cx0 = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales)
 
        Cr0 = torch.zeros(max(2*self.n_orientations,5), max(2*self.n_orientations,5), self.n_scales+1)
        Crx0 = torch.zeros(2*self.n_orientations, max(2*self.n_orientations,5), self.n_scales)

        for n_scales in range(0, self.n_scales):
            firstBnum = (n_scales)*self.n_orientations + 1
            cousinSz = rpyr[firstBnum].shape[0]*rpyr[firstBnum].shape[1]

            if n_scales < self.n_scales-1:
                parents = torch.empty((cousinSz, self.n_orientations))
                rparents = torch.empty((cousinSz, self.n_orientations*2))
                
                for nor in range(0, self.n_orientations):
                    nband = (n_scales+1)*self.n_orientations + nor + 1
                    tmp = Portilla_Simoncelli.expand(pyr_coeffs[nband],2)/4.0

                    rtmp = tmp[:,:,0]
                    itmp = tmp[:,:,1]
                    # double phase

                    rtmp2 = (rtmp**2 + itmp**2)**.5 * torch.cos(2*torch.atan2(rtmp, itmp))
                    itmp2 = (rtmp**2 + itmp**2)**.5 * torch.sin(2*torch.atan2(rtmp, itmp))
                    
                    rparents[:, nor] = rtmp2.t().flatten()
                    rparents[:, nor+self.n_orientations] = itmp2.t().flatten()

                    tmp2 = (rtmp2**2+itmp2**2)**.5
                    parents[:,nor] = (tmp2 - tmp2.mean()).t().flatten()



            else:
                tmp = Portilla_Simoncelli.expand(rpyr[-1].squeeze(),2)/4.0
                tmp = tmp[:,:,0].t()
                rparents= torch.stack((tmp.flatten(), 
                    tmp.roll(1,0).flatten(),
                    tmp.roll(-1,0).flatten(),
                    tmp.roll(1,1).flatten(),
                    tmp.roll(-1,1).flatten()),1)

                parents=torch.empty((0))

            cousins = torch.stack(tuple([a.t() for a in apyr[n_scales * self.n_orientations+1:(n_scales+1) * self.n_orientations+1]])).view((self.n_orientations, cousinSz)).t()
            nc = cousins.shape[1]
            if parents.shape[0] > 0:
                np0 = parents.shape[1]
            else:
                np0 = 0
            C0[0:nc, 0:nc, n_scales] = (cousins.t()@cousins)/cousinSz
            if np0 > 0:
                Cx0[0:nc, 0:np0, n_scales] = (cousins.t()@parents)/cousinSz
                if n_scales==self.n_scales-1:
                    C0[0:np0, 0:np0, n_scales+1] = (parents.t()@parents)/(cousinSz/4.0)

            cousins = torch.stack(tuple([a[:,:,0].t() for a in pyr_coeffs[n_scales*self.n_orientations+1:(n_scales+1)*self.n_orientations+1]])).view((self.n_orientations,cousinSz)).t()
            nrc = cousins.shape[1]
            nrp = 0
            if rparents.shape[0]>0:
                nrp = rparents.shape[1]

            Cr0[0:nrc,0:nrc,n_scales]=(cousins.t()@cousins)/cousinSz
            if nrp>0:
                Crx0[0:nrc,0:nrp,n_scales] = (cousins.t()@rparents)/cousinSz
                
                if n_scales==self.n_scales-1:
                    
                    Cr0[0:nrp,0:nrp,n_scales+1]=(rparents.t()@rparents)/(cousinSz/4.0)

        # STATISTC: vHPR0 or the variance of the high-pass residual
        channel = pyr_coeffs[0]
        vHPR0 = channel.pow(2).mean()
        
        representation = torch.cat((pixel_statistics.flatten(),magMeans0.flatten(),ace.flatten(),
            skew0p.flatten(),kurt0p.flatten(),acr.flatten(), C0.flatten(), 
            Cx0.flatten(), Cr0.flatten(), Crx0.flatten(), vHPR0.unsqueeze(0))) 

        self.Crx0 = Crx0;

        if self.normalizationFactor is not None:
            representation = self.normalizationFactor @ representation
        return representation


    def get_stat_levels(self):
        """
            
            """
        # leg=['statsg0','magMeans0','ace','skew0p','kurt0','acr','C0','Cx0','Cr0','Crx0','vHPR0']
        # statg0
        statg0 = (self.n_scales+2)*torch.ones(6)
        
        # magMeans0
        t = torch.arange(1,self.n_scales+1)
        t =t.repeat(self.n_orientations,1).t().flatten()
        t= torch.cat((torch.tensor([0]),t,torch.tensor([self.n_scales+1])))
        magMeans0 = t.type(torch.float)
        
        # skew0p
        skew0p = torch.arange(1,self.n_scales+2,dtype=torch.float)
        
        # kurt0p
        kurt0p=torch.arange(1,self.n_scales+2,dtype=torch.float)
        
        acr = torch.empty([self.Na,self.Na,self.n_scales+1])
        C0 = torch.empty(self.n_orientations,self.n_orientations,self.n_scales+1)
        Cr0 = torch.empty(2*self.n_orientations,2*self.n_orientations,self.n_scales+1)
        ace = torch.empty([self.Na,self.Na,self.n_scales,self.n_orientations])
        Cx0 = torch.empty(self.n_orientations, self.n_orientations, self.n_scales)
        Crx0 = torch.empty(2*self.n_orientations, 2*self.n_orientations, self.n_scales)
        
        for i in range(0,self.n_scales+1):
            C0[:,:,i] = i+1
            Cr0[:,:,i] = i+1
            acr[:,:,i] = i+1
            if i<self.n_scales:
                Cx0[:,:,i]=i+1
                Crx0[:,:,i]=i+1
                ace[:,:,i,:]=i+1
        vHPR0 = torch.zeros(1)
        
        skew0p.flatten().shape
        
        levs = torch.cat((statg0.flatten(),magMeans0.flatten(),ace.flatten(),
                          skew0p.flatten(),kurt0p.flatten(),acr.flatten(), C0.flatten(),
                          Cx0.flatten(), Cr0.flatten(), Crx0.flatten(), vHPR0.flatten()))
        return levs


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
            skew0p = torch.mean(ch.pow(3))/(vari.pow(1.5))
            kurt0p = torch.mean(ch.pow(4))/(vari.pow(2))

        else:
            skew0p = 0
            kurt0p = 3
        return skew0p,kurt0p

    

    def skew(mtx):
        return torch.mean((mtx-mtx.mean()).pow(3))/(mtx.var().pow(1.5))

    def kurtosis(mtx):
        # implementation is only for real components
        return torch.mean(torch.abs(mtx-mtx.mean()).pow(4))/(mtx.var().pow(2))


