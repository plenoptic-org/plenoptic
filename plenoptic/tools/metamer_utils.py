# TODO - reorganize this module as we finalize texture statistics

import torch
import abc
from .stats import skew, kurtosis


class Clamper(metaclass=abc.ABCMeta):
	"""
	Declare the interface for objects in the composition.
	Implement default behavior for the interface common to all classes,
	as appropriate.
	Declare an interface for accessing and managing its child
	components.
	Define an interface for accessing a component's parent in the
	recursive structure, and implement it if that's appropriate
	(optional).
	"""

	@abc.abstractmethod
	def clamp(self):
		pass



class RangeClamper(Clamper):

	def __init__(self,range):
		self.range = range

	def clamp(self,im):
		im = im.clamp(self.range[0],self.range[1])
		return im

class TwoMomentsClamper(Clamper):

    def __init__(self,targ):
        self.targ = targ

    def clamp(self,im):

        # mean and variance
        im = (im - im.mean())/im.std() * self.targ.std() + self.targ.mean()

        # range
        im = im.clamp(self.targ.min(),self.targ.max())

        return im

class FourMomentsClamper(Clamper):

    def __init__(self,targ):
        self.targ = targ

    def clamp(self,im):
        # kurtosis
        im = modkurt(im,kurtosis(self.targ))

        # skew
        im = modskew(im,skew(self.targ))

        # mean and variance
        im = (im - im.mean())/im.std() * self.targ.std() + self.targ.mean()

        # range
        im = im.clamp(self.targ.min(),self.targ.max())

        return im

def snr(s,n):

    # Compute the signal-to-noise ratio in dB
    #   X=SNR(signal,noise);
    # (it does not subtract the means).

    es=torch.sum(torch.sum(torch.abs(s).pow(2)));
    en=torch.sum(torch.sum(torch.abs(n).pow(2)));
    X=10*torch.log(es/en)

    return X


def roots(c):
    n = c.numel()
    inz = c.flatten().nonzero()

    # strip leading zeros and throw away
    # strip trailing zeros, but remember them as roots at zero
    nnz = inz.numel()
    c = c[inz[0]:inz[-1]+1]
    r = torch.zeros((n-inz[nnz-1]-1,2))

    # prevent relatively small leading coefficients from introducing Inf by removing them
    d = c[1:]/c[0]
    while torch.any(torch.isinf(d)):
        c = c[1:]
        d = c[1:]/c[0]

    # polynomial roots via a companion matrix
    n = c.numel()
    if n>1:
        a = torch.diag(torch.ones((n-2)),-1)
        a[0,:] = -d.flatten()
        r = torch.cat((r,torch.eig(a)[0]))

    return r

def polyval(c,x):
    s = 0
    for i, ci in enumerate(c):
        s += ci*x.pow(c.numel()-i-1)
    return s


def modkurt(ch,k,p=1):
    me = ch.mean()
    ch = ch-me
    m = torch.zeros(12)
    for n in range(1,12):
        m[n] = ch.pow(n+1).mean()

    k0 = m[3]/m[1].pow(2)
    snrk = snr(k,k-k0)

    if snrk >60:
        chm = ch+me
        return chm

    k=k0*(1-p) + k*p

    a = m[3]/m[1]

    # coefficients of the numerator
    A=m[11]-4*a*m[9]-4*m[2]*m[8]+6*a**2*m[7]+12*a*m[2]*m[6]+6*m[2]**2*m[5]-4*a**3*m[5]-12*a**2*m[2]*m[4]+a**4*m[3]-12*a*m[2]**2*m[3]+4*a**3*m[2]**2+6*a**2*m[2]**2*m[1]-3*m[2]**4
    B=4*(m[9]-3*a*m[7]-3*m[2]*m[6]+3*a**2*m[5]+6*a*m[2]*m[4]+3*m[2]**2*m[3]-a**3*m[3]-3*a**2*m[2]**2-3*m[3]*m[2]**2)
    C=6*(m[7]-2*a*m[5]-2*m[2]*m[4]+a**2*m[3]+2*a*m[2]**2+m[2]**2*m[1])
    D=4*(m[5]-a**2*m[1]-m[2]**2)
    E=m[3]

    # define the coefficients of the denominator
    F = D/4
    G = m[1]

    d = torch.empty(5)
    d[0] = B*F;
    d[1] = 2*C*F - 4*A*G;
    d[2] = 4*F*D -3*B*G - D*F;
    d[3] = 4*F*E - 2*C*G;
    d[4] = -D*G;

    mMlambda = roots(d)

    tg = mMlambda[:,1]/mMlambda[:,0]
    mMlambda = mMlambda[tg.abs()<1e-6]

    lNeg = mMlambda[mMlambda<0]
    if lNeg.numel() == 0:
        lNeg = torch.tensor([-1/(2.2204e-16)])

    lPos = mMlambda[mMlambda>=0]
    if lPos.numel() == 0:
        lPos = torch.tensor([1/(2.2204e-16)])

    lmi = lNeg.max()
    lma = lPos.min()
    lam = torch.Tensor([lmi,lma])

    mMnewKt = polyval(torch.tensor([A,B,C,D,E]),lam)/(polyval(torch.tensor([F,0,G]),lam).pow(2))
    kmin=min(mMnewKt)
    kmax = max(mMnewKt)

    # coefficients of the algebraic equation
    c0 = E-k*G**2
    c1 = D
    c2 = C-2*k*F*G
    c3 = B
    c4 = A - k*F**2

    # solves the equation
    r = roots(torch.tensor([c4, c3, c2, c1, c0]))

    # choose the real solution with minimum absolute value with the right sign

    tg = r[:,1]/r[:,0]
    lambd = r[tg.abs()==0,0]
    if lambd.numel()>0:
        lam = lambd[lambd.abs()==min(lambd.abs())]
    else:
        lam = torch.zeros(1)

    # print(lam)
    # modify the channel
    chm = ch+ lam*(ch**3 - a*ch-m[2])
    chm = chm*(m[1]/(ch**2).mean())**.5
    chm = chm+me

    return chm





def modskew(ch,sk,p=1):
    # Adjust the sample skewness of a vector/matrix, using gradient projection,
    # without affecting its sample mean and variance.
    #
    # This operation is not an orthogonal projection, but the projection angle is
    # near pi/2 when sk is close to the original skewness, which is a realistic
    # assumption when doing iterative projections in a pyramid, for example
    # (small corrections to the channels' statistics).
    #
    #   [xm, snrk] = modskew(x,sk,p);
    #       sk: new skweness
    #           p [OPTIONAL]:   mixing proportion between sk0 and sk
    #                           it imposes (1-p)*sk0 + p*sk,
    #                           being sk0 the current skewness.
    #                           DEFAULT: p = 1;
    #
    # converted from PortillaSimoncelli MATLAB implementation
    # KLB
    N = ch.numel()
    me = ch.mean()
    ch = ch-me

    m = torch.zeros(6,1)
    for n in range(2,7):
        m[n-1]=ch.pow(n).mean()

    sd = m[1].sqrt()
    s = m[2]/(sd.pow(3))
    snrk = snr(sk,sk-s)
    sk = s*(1-p) + sk*p

    A = m[5]-3*sd*s*m[4] + 3*(sd**2)*(s**2-1)*m[3] + sd**6 * (2+3*s**2-s**4)
    B = 3*(m[4]-2*sd*s*m[3] + sd**5*s**3)
    C = 3*(m[3]-sd**4*(1+s**2))
    D = s*sd**3

    a = torch.zeros(7,1)
    a[6] = A**2
    a[5] = 2*A*B
    a[4] = B**2 + 2*A*C
    a[3] = 2*(A*D + B*C)
    a[2] = C**2 + 2*B*D
    a[1] = 2*C*D
    a[0] = D**2

    A2 = sd**2
    B2 = m[3] - (1+s**2)*sd**4

    b = torch.zeros(7,1)
    b[6] = B2**3
    b[4] = 3*A2*B2**2
    b[2] = 3*A2**2*B2
    b[0] = A2**3

    d = torch.zeros(8,1)
    d[0] = B*b[6];
    d[1] = 2*C*b[6] - A*b[4];
    d[2] = 3*D*b[6];
    d[3] = C*b[4] - 2*A*b[2];
    d[4] = 2*D*b[4] - B*b[2];
    d[5] = -3*A*b[0];
    d[6] = D*b[2] - 2*B*b[0];
    d[7] = -C*b[0];

    mMlambda = roots(d)

    tg = mMlambda[:,1]/mMlambda[:,0]
    mMlambda = mMlambda[tg.abs()>1e-6,0]

    lNeg = mMlambda[mMlambda<0]
    if lNeg.numel() == 0:
        lNeg = torch.tensor([-1/(2.2204e-16)])

    lPos = mMlambda[mMlambda>=0]
    if lPos.numel() == 0:
        lPos = torch.tensor([1/(2.2204e-16)])

    lmi = lNeg.max()
    lma = lPos.min()

    lam = torch.Tensor([lmi,lma])
    mMnewSt = polyval(torch.tensor([A,B,C,D]),lam)/(polyval(b.flip(dims=[0]),lam).pow(.5))
    skmin=min(mMnewSt)
    skmax = max(mMnewSt)

    c = a-b*sk.pow(2)
    c = c.flip(dims=[0])
    r = roots(c)

    tg = r[:,1]/r[:,0]
    fi = tg.abs()<1e-6
    fi2 = r[:,0].sign()==(sk-s).sign().flatten()

    ti = torch.zeros_like(fi)
    for i,(fa,fb) in enumerate(zip(fi,fi2)):
        ti[i] = fa+fb

    if torch.any(ti==2):
        lam = r[fi,0]
    else:
        lam = torch.Tensor([0])

    p = torch.Tensor([A,B,C,D])
    if lam.numel()>1:
        foo = polyval(p,lam).sign()
        if torch.any(foo==0):
            lam = lam[foo==0]
        else:
            lam = lam[foo==sk.sign()] # rejects the symmetric solution

        if lam.numel()>0:
            lam = lam[lam.abs()==lam.abs().min()]
            lam = lam[0]
        else:
            lam=torch.Tensor([0])

    chm = ch+lam*(ch.pow(2)-sd.pow(2)-sd*s*ch) # adjust the skewness
    chm = chm + (m[1]/chm.pow(2).mean()).pow(.5) # adjust variance
    chm = chm + me

    return chm.data
