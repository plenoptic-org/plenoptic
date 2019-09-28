import torch
from torch import nn
from ..tools.signal import rescale
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product

"""
TODO see more of the spectrum
"""
# get more of the spectrum by implementing
# the power iteration method on deflated matrix F
# see: http://papers.nips.cc/paper/3575-deflation-methods-for-sparse-pca.pdf

# TODO compare with
# implicit_block_power_method
# Lanczos method

# def schur_complement_deflation(A, x):
#     """Schur complement matrix deflation
#     Eliminate the influence of a psuedo-eigenvector of A using the Schur complement
#     deflation technique from [1]::
#         A_new = A - \frac{A x x^T A}{x^T A x}
#     Parameters
#     ----------
#     A : np.ndarray, shape=(N, N)
#         A matrix
#     x : np.ndarray, shape=(N, )
#         A vector, ideally one that is "close to" an eigenvector of A
#     Returns
#     -------
#     A_new : np.ndarray, shape=(N, N)
#         A new matrix, determined from A by eliminating the influence of x
#     References
#     ----------
#     ... [1] Mackey, Lester. "Deflation Methods for Sparse PCA." NIPS. Vol.
#         21. 2008.
#     """
#     return A - np.outer(np.dot(A, x), np.dot(x, A)) / np.dot(np.dot(x, A), x)


def fisher_info_matrix_vector_produt(y, x, v):
    """Compute Fisher Information Matrix Vector Product: Fv
    Parameters
    ----------
    y: torch tensor with gradient function
        output
    x: torch tensor with gradient function
        input
    v: torch tensor
        direction
    Returns
    -------
    Fv: torch tensor
        vector, fvp
    Notes
    -----
    under white Gaussian noise assumption, F is matrix multiplication of
    Jacobian transpose and Jacobian: F = J.T J
    Hence:
    Fv = J.T (Jv)
       = ((jvp.T) J).T
    """
    Jv = jacobian_vector_product(y, x, v)
    Fv = vector_jacobian_product(y, x, Jv)
    return Fv.t()


def implicit_FIM_eigenvalue(y, x, v):
    """Implicitly compute the eigenvalue of the Fisher Information Matrix
    corresponding to eigenvector v
    lmbda = v.T F v
    """
    Fv = fisher_info_matrix_vector_produt(y, x, v)
    lmbda = torch.mm(Fv.t(), v)  # conjugate transpose
    return lmbda


def implicit_FIM_power_iteration(y, x, l=0, init='randn', seed=0, tol=1e-10, n_steps=100, verbose=False):
    """Apply the power method algorithm to approximate the extremal eigenvalue
    and eigenvector of the Fisher Information Matrix, without explicitely
    representing that matrix
    Parameters
    ----------
    y: torch tensor with gradient function
        output
    x: torch tensor
        input
    l: torch tensor, optional
        When l=0, this function estimates the leading eval evec pair.
        When l is set to the estimated maximum eigenvalue, this function
        will estimate the smallest eval evec pair (minor component).
    init: str, optional
        starting point for the power iteration
    seed: float, optional
        manual seed
    tol: float, optional
        tolerance value
    n_steps: integer, optional
        maximum number of steps
    verbose: boolean, optional
        flag to control amout of information printed out
    Returns
    -------
    lmbda: float
        eigenvalue
    v: torch tensor
        eigenvector
    Note
    ----
    - inverse power method (F - lmbda I)v
    - this function will most likely land on linear combinations of evecs
    TODO
    - check for division by zero
    - cleanup implementation of minor component
    - better stop criterion
    #         error = torch.sqrt(torch.sum(fvp(y, x, v_new) -
    #                                 (l + lmbda_new) * v_new)**2)
    #         error = torch.sqrt(torch.sum(v - v_new) ** 2)
    """
    n = x.shape[0]
    # m = y.shape[0]
    # assert (m >= n)

    torch.manual_seed(seed)

    if init == 'randn':
        v = torch.randn_like(x)
        v = v / torch.norm(v)
    elif init == 'ones':
        v = torch.ones_like(x) / torch.sqrt(torch.tensor(n).float())

    Fv = fisher_info_matrix_vector_produt(y, x, v)
    v = Fv / torch.norm(Fv)
    lmbda = implicit_FIM_eigenvalue(y, x, v)

    i = 0
    error = torch.ones(1)

    while i < n_steps and error > tol:

        Fv = fisher_info_matrix_vector_produt(y, x, v)
        Fv = Fv - l * v  # minor component
        v_new = Fv / torch.norm(Fv)

        lmbda_new = implicit_FIM_eigenvalue(y, x, v_new)

        error = torch.sqrt((lmbda - lmbda_new) ** 2)

        if verbose:
            print(i, error.detach().numpy()[0])

        v = v_new
        lmbda = lmbda_new
        i += 1

    return lmbda_new, v_new


def fisher_info_matrix_matrix_produt(y, x, A):

    n = x.shape[0]
    assert A.shape[0] == n
    r = A.shape[1]

    FA = torch.zeros((n, r))

    # TODO - vectorization spead-up?
    i = 0
    for v in A.t():
        v = v.unsqueeze(1)
        FA[:, i] = fisher_info_matrix_vector_produt(y, x, v).squeeze()
        i += 1

    return FA


def implicit_block_power_method(x, y, r, l=0, init='randn', seed=0, tol=1e-10, n_steps=100, verbose=False):
    """
    TODO
    """
    n = x.shape[0]

    # init
    Q = torch.qr(torch.randn(n, r))[0]

    i = 0
    while i < n_steps and error > tol:

        Z = fisher_info_matrix_matrix_produt(y, x, Q)
        Q, R = torch.qr(Z)
        i += 1

        if verbose:
            print(i, error.detach().numpy()[0])

    return Q


class Eigendistortion(nn.Module):
    r"""Synthesize the eigendistortions induced by a model on an image.
    Parameters
    -----------
    image: torch.Tensor
        image, (B x C x H x W)
    model: torch class
        torch model with defined forward and backward operations
    Notes
    -----
    This is a method for comparing image representations in terms of their
    ability to explain perceptual sensitivity in humans. It uses the
    power method to estimate largest and smallest eigenvectors of the FIM.
    Model implements y = f (x), a deterministic (and differentiable) mapping
    from the input pixels (R^n) to a mean output response vector (R^m),
    where we assume additive white Gaussian noise in the response space:
        f: R^n -> R^m
            x  ->  y
    The Jacobian matrix at x is:
        J(x) = dydx        [m x n] (ie. output_dim x input_dim)
    is the matrix of all first-order partial derivatives
    of the vector-valued function f.
    The Fisher Information Matrix (FIM) at x, under white Gaussian noise in
    the response space, is:
        F = J(x).T.dot(J(x))
    It is a quadratic approximation of the discriminability of
    distortions relative to x.
    Berardino, A., Laparra, V., Ballé, J. and Simoncelli, E., 2017.
    Eigen-distortions of hierarchical representations.
    In Advances in neural information processing systems (pp. 3530-3539).
    http://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf
    http://www.cns.nyu.edu/~lcv/eigendistortions/
    TODO
    ----
    control seed
    enforce bounding box during optimization (see other classes in this repo, stretch/squish)
    check for division by zero
    implement deflation
    handle batch input
    handle color image
    make sure that the class cashes learnt distortions, every couple iterations, to prevent from loosing things when crahses
    """

    def __init__(self, image, model):
        super().__init__()

        self.image = rescale(image, 0, 1)
        self.batch_size, self.n_channels, self.im_height, self.im_width = image.shape
        assert self.batch_size == 1

        if self.n_channels == 3:
            self.color_image = True

        im_flat = self.image.reshape(self.n_channels * self.im_height * self.im_width, 1)
        self.image_flattensor = im_flat.clone().detach().requires_grad_(True).type(torch.float32)
        self.model_input = self.image_flattensor.view((1, self.n_channels, self.im_height, self.im_width))

        self.model = model
        self.model_output = self.model(self.model_input)

        if len(self.model_output) > 1:
            self.out_flattensor = torch.cat([s.squeeze().view(-1) for s in self.model_output]).unsqueeze(1)
        else:
            self.out_flattensor = self.model_output.squeeze().view(-1).unsqueeze(1)
        self.distortions = dict()

    def solve_eigenproblem(self):
        J = jacobian(self.out_flattensor, self.image_flattensor)
        F = torch.mm(J.t(), J)
        evals, evecs = torch.symeig(F, eigenvectors=True)

        for i in list(range(evals.size(0)-1, -1, -1)):
            self.distortions[str(evals.size(0) - i)] = (evals[i], evecs[i])

    def synthesize(self, block=None, tol=1e-10, n_steps=100, jac=True, seed=0, verbose=True):
        '''Compute eigendistortion
        Parameters
        ----------
        tol: tolerance for error criterion in power iteration
        n_steps: total steps to run for power iteration in eigenvalue computation
        jac: boolean, optional (default True)
            Try to use the full jacobian method if the input and output sizes are small enough
        seed: control the random seed for reproducibility
        verbose: boolean, optional (default True)
            show progress during power iteration
        Returns
        -------
        distortions: dict of torch tensors
            dictionary containing the eigenvalues and eigen-distortions in decreasing order
        '''
        if verbose:
            print('out size', self.out_flattensor.size(), 'in size', self.image_flattensor.size())

        if jac and self.out_flattensor.size(0) * self.image_flattensor.size(0) < 10e6:
            solve_eigenproblem()

        elif block is not None:
            print('under construction')
            return implicit_block_power_method(self.image_flattensor, self.out_flattensor, r=block, n_steps=n_steps, verbose=verbose)

        else:
            if verbose:
                print('implicit power method, computing the maximum distortion')
            lmbda_max, v_max = implicit_FIM_power_iteration(self.out_flattensor, self.image_flattensor, l=0, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)
            self.distortions[str(0)] = (lmbda_max, v_max)

            if verbose:
                print('implicit power method, computing the minimum distortion')
            lmbda_min, v_min = implicit_FIM_power_iteration(self.out_flattensor, self.image_flattensor, l=lmbda_max, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)
            self.distortions[str(-1)] = (lmbda_min, v_min)

            # TODO deflation
            # for ind in distinds:
            #     if ind == 0:
            #         continue
            #     elif ind == -1:
            # else:
                # raise Exception('Error: Not implemented deflation for intermediate eigendistortions')
        return self.distortions

    def display(self, alpha=5, beta=10, **kwargs):

        try:
            import pyrtools as pt
            numpy = lambda x : x.detach().cpu().numpy().squeeze()

            image = numpy(self.image)
            maxdist = numpy(self.distortions['0'][1].reshape(self.image.shape))
            mindist = numpy(self.distortions['-1'][1].reshape(self.image.shape))

            pt.imshow([image, image + alpha * maxdist, beta * maxdist], title=['original', 'original + ' + str(alpha) + ' maxdist', str(beta) + ' * maximum eigendistortion'], **kwargs);
            pt.imshow([image, image + alpha * mindist, beta * mindist], title=['original', 'original + ' + str(alpha) + ' mindist', str(beta) + ' * minimum eigendistortion'], **kwargs);

        except:
            print("pyrtools unavailable")