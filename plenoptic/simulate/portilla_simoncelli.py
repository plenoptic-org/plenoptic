import torch
import torch.nn as nn
import pyrtools
from .steerable_pyramid_freq import Steerable_Pyramid_Freq
import numpy as np


class PS(nn.Module):
    '''
    1) forcing the sample statistics of each steerable pyramid subband to match
    those of a reference texture image
    2) reconstructing an image from the pyramid, and forcing the sample statistics
    of the resulting pixels to match those of the reference image. We enforce the
    statistical constraints by moving in the direction of the gradient of the
    constraint function, until the constraint is satisfied. Note that apart from
    the choice of initial image, the algorithm is deterministic
    '''
    def __init__(self, shape, n_scales=4, n_orientations=4, Na=9):
        super(PS, self).__init__()


        self.image_shape = shape
        self.Na = Na
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.pyr = Steerable_Pyramid_Freq(self.image_shape,height=self.n_scales,
            order=self.n_orientations-1,is_complex=True,store_unoriented_bands=True)
        

        # weighted local statistics

    def forward(self, im0, height='auto', order=3):

        # pixel statistics
        mn0 = torch.min(im0)
        mx0 = torch.max(im0)
        mean0 = torch.mean(im0)
        var0 = torch.var(im0)
        skew0 = PS.skew(im0, mean0, var0)
        kurt0 = PS.kurtosis(im0, mean0, var0)
        statg0 = torch.stack((mean0, var0, skew0, kurt0, mn0, mx0)).view(6, 1)
        

        # get pyramid coefficients
        self.pyr.forward(im0)
        pyr0 = self.pyr.coeffout 

        ### measuring local statistics
        # subtract mean of lowBand
        nbands = len(pyr0)
        pyr0[-1] = (pyr0[-1])-torch.mean((pyr0[-1]))

        apyr0 = []
        rpyr0 = []

        for bb in range(0, nbands):

            if self.pyr.is_complex==True and pyr0[bb].shape[-1] == 2:  # if the pyramid complex
            # if self.complex_steerable_pyramid.complex and len(pyr0[bb].shape)==3:
                tmp = torch.unbind(pyr0[bb],-1)
                apyr0.append(((tmp[0]**2+tmp[1]**2)**.5).squeeze())
                rpyr0.append(tmp[0].squeeze())

            else:
                rpyr0.append(pyr0[bb].squeeze())
                apyr0.append(torch.abs(pyr0[bb]).squeeze())

            pyr0[bb] = pyr0[bb].squeeze()

        # subtract mean of magnitude
        magMeans0 = torch.empty([len(pyr0), 1])
        for bb in range(0, nbands):
            magMeans0[bb] = torch.mean(apyr0[bb])
            apyr0[bb] = apyr0[bb] - magMeans0[bb]

        # compute central autoCorr of lowband
        acr = torch.empty([self.Na, self.Na, self.n_scales+1])
        skew0p = torch.empty((self.n_scales+1,1))
        kurt0p = torch.empty((self.n_scales+1,1))

        ch = pyr0[-1]

        #######
        mpyrM = Steerable_Pyramid_Freq(ch.shape[-2:], height=0, order=0, return_list=True)
        mpyr = mpyrM.forward(ch.squeeze().unsqueeze(0).unsqueeze(0))

        im = mpyr[0].squeeze()
        #########################

        Sch = torch.min(torch.tensor(ch.shape[-2:]))
        la = int(np.floor([(self.Na-1)/2]))
        le = int(np.min((Sch/2-1,la)))
        cy = int(ch.shape[-1]/2)
        cx = int(ch.shape[-2]/2)

        ac_tmp = torch.fft(torch.stack((im.squeeze(), torch.zeros_like(im.squeeze())), -1), 2)
        ac_tmp2 = torch.stack((ac_tmp[:, :, 0].pow(2) + ac_tmp[:, :, 1].pow(2), torch.zeros_like(ac_tmp[:, :, 0])), -1)
        ac = torch.ifft(ac_tmp2, 2)
        ac= PS.fftshift(ac)/torch.numel(ch)
        ac = ac[cx-le:cx+le+1, cy-le:cy+le+1,0]
        acr[la-le:la+le+1, la-le:la+le+1, self.n_scales] = ac
        vari = ac[le,le]

        if vari/var0 > 1e-6:
            skew0p[self.n_scales] = torch.mean(im.pow(3))/(vari.pow(1.5))
            kurt0p[self.n_scales] = torch.mean(im.pow(4))/(vari.pow(2))

        else:
            skew0p[self.n_scales] = 0
            kurt0p[self.n_scales] = 3


        # compute central autoCorr of each Mag band, and the autoCorr of the combined (non-oriented) band
        ace = torch.empty([self.Na, self.Na, self.n_scales, self.n_orientations])
        for n_scales in range(self.n_scales-1, -1, -1):
            for nor in range(0, self.n_orientations):
                nband = n_scales*self.n_orientations + nor + 1
                ch = apyr0[nband]
                Sch = np.min((ch.shape[-1], ch.shape[-2]))
                le = int(np.min((Sch/2-1, la)))
                cx = int(ch.shape[-2]/2)
                cy = int(ch.shape[-1]/2)

                ac_tmp = torch.fft(torch.stack((ch.squeeze(), torch.zeros_like(ch.squeeze())), -1), 2)
                ac_tmp2 = torch.stack((ac_tmp[:, :, 0].pow(2) + ac_tmp[:, :, 1].pow(2), torch.zeros_like(ac_tmp[:, :, 0])), -1)
                ac = torch.ifft(ac_tmp2, 2)
                ac = PS.fftshift(ac)/torch.numel(ch)
                ac = ac[cx-le:cx+le+1, cy-le:cy+le+1,0]
                ace[la-le:la+le+1, la-le:la+le+1, n_scales, nor] = ac

            ######
            ch_tmp = self.pyr.unoriented_bands[n_scales].squeeze()
            ch = torch.stack((ch_tmp,torch.zeros_like(ch_tmp)),-1)
            ######

            ac_tmp = torch.fft(ch, 2)
            ac_tmp2 = torch.stack((ac_tmp[:, :, 0]**2 + ac_tmp[:, :, 1]**2, torch.zeros_like(ac_tmp[:, :, 0])), -1)
            ac = torch.ifft(ac_tmp2, 2)
            ac = PS.fftshift(ac)/torch.numel(ch)
            ac = ac[cx-le:cx+le+1, cy-le:cy+le+1,0]
            acr[la-le:la+le+1, la-le:la+le+1, n_scales] = ac


            vari = ac[le,le]
            if vari/var0 > 1e-6:
                skew0p[n_scales] = torch.mean(ch.pow(3))/(vari.pow(1.5))
                kurt0p[n_scales] = torch.mean(ch.pow(4))/(vari.pow(2))

            else:
                skew0p[n_scales] = 0
                kurt0p[n_scales] = 3

        # compute the cross-correlation matrices of the coefficient magnitudes pyramid at the different levels and orientations
        C0 = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales+1)
        Cx0 = torch.zeros(self.n_orientations, self.n_orientations, self.n_scales)

        Cr0 = torch.zeros(2*self.n_orientations, 2*self.n_orientations, self.n_scales+1)
        Crx0 = torch.zeros(2*self.n_orientations, 2*self.n_orientations, self.n_scales)

        for n_scales in range(0, self.n_scales):
            firstBnum = (n_scales)*self.n_orientations + 1
            cousinSz = rpyr0[firstBnum].shape[0]*rpyr0[firstBnum].shape[1]

            if n_scales < self.n_scales-1:
                parents = torch.empty((cousinSz, self.n_orientations))
                rparents = torch.empty((cousinSz, self.n_orientations*2))
                for nor in range(0, self.n_orientations):
                    nband = (n_scales+1)*self.n_orientations + nor + 1
                    tmp = PS.expand(pyr0[nband],2)/4

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
                tmp = PS.expand(rpyr0[-1].squeeze(),2)/4
                tmp = tmp[:,:,0]
                rparents= torch.stack((tmp.flatten(), tmp.roll((0,1),(0,1)).flatten(),
                    tmp.roll((0,-1),(0,1)).flatten(),tmp.roll((1,0),(0,1)).flatten(),
                    tmp.roll((-1,0),(0,1)).flatten()),1)

                parents=torch.empty((0))

            cousins = torch.stack(tuple([a.t() for a in apyr0[n_scales * self.n_orientations+1:(n_scales+1) * self.n_orientations+1]])).view((self.n_orientations, cousinSz)).t()
            nc = cousins.shape[1]
            if parents.shape[0] > 0:
                np0 = parents.shape[1]
            else:
                np0 = 0
            C0[0:nc, 0:nc, n_scales] = PS.innerprod(cousins)/cousinSz
            if np0 > 0:
                Cx0[0:nc, 0:np0, n_scales] = torch.mm(cousins.t(), parents)/cousinSz
                if n_scales==self.n_scales-1:
                    C0[0:np0, 0:np0, n_scales+1] = PS.innerprod(parents)/(cousinSz/4)

            cousins = torch.stack(tuple([a[:,:,0].t() for a in pyr0[n_scales*self.n_orientations+1:(n_scales+1)*self.n_orientations+1]])).view((self.n_orientations,cousinSz)).t()
            nrc = cousins.shape[1]
            nrp = 0
            if rparents.shape[0]>0:
                nrp = rparents.shape[1]
            Cr0[0:nrc,0:nrc,n_scales]=PS.innerprod(cousins)/cousinSz
            if nrp>0:
                Crx0[0:nrc,0:nrp,n_scales] = torch.mm(cousins.t(),rparents)/cousinSz
                if n_scales==self.n_scales-1:
                    Cr0[0:nrp,0:nrp,n_scales+1]=PS.innerprod(rparents)/(cousinSz/4)

        channel = pyr0[0]
        vHPR0 = channel.pow(2).mean()

        stats = torch.cat((statg0.flatten(),magMeans0.flatten(),ace.flatten(),
            skew0p.flatten(),kurt0p.flatten(),acr.flatten(), C0.flatten(), 
            Cx0.flatten(), Cr0.flatten(), Crx0.flatten(), vHPR0.unsqueeze(0))) 

        return stats

    def kurtosis(mtx, mn, v):
        # implementation is only for real components
        return torch.mean(torch.abs(mtx-mn).pow(4))/(v.pow(2))

    def skew(mtx, mn, v):
        return torch.mean((mtx-mn).pow(3))/(v.pow(1.5))

    def fftshift(X):
        real = X[:,:,0]
        imag = X[:,:,1]
        for dim in range(0, len(real.size())):
            real = PS.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = PS.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag),-1)

    def roll_n(X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None)
                      for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                      for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front],axis)

    def expand(t,f):

        t = t.squeeze()

        mx = t.shape[0]
        my = t.shape[1]
        my = f*my
        mx = f*mx

        Te = torch.zeros(my,mx,2)

        if len(t.shape)==2:
            t=torch.stack((t,torch.zeros_like(t)),-1)

        T =  f**2*PS.fftshift(torch.fft(t,2))

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

        Te = PS.fftshift(Te)

        # finish this
        te = torch.ifft(Te,2)


        return te

    def innerprod(mtx):
        return torch.mm(mtx.t(),mtx)
