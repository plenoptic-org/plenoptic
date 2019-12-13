import torch
from torch import nn
from ..tools.signal import rescale
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
from ..tools.data import to_numpy

# TODO compare with: implicit_block_power_method (incomplete, potentially numerically unstable), lanczos (Done)

# get more of the spectrum by implementing
# the power iteration method on deflated matrix F
# see: http://papers.nips.cc/paper/3575-deflation-methods-for-sparse-pca.pdf

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


def fisher_info_matrix_vector_product(y, x, v):
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
    Fv = vector_jacobian_product(y, x, Jv, retain_graph=True, create_graph=False).detach()
    return Fv.t()


def implicit_FIM_eigenvalue(y, x, v):
    """Implicitly compute the eigenvalue of the Fisher Information Matrix
    corresponding to eigenvector v
    lmbda = v.T F v
    """
    Fv = fisher_info_matrix_vector_product(y, x, v)
    lmbda = torch.mm(Fv.t(), v)  # conjugate transpose
    return lmbda


def implicit_power_method(y, x, l=0, init='randn', seed=0, tol=1e-10, n_steps=100, verbose=False):
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
    import numpy as np
    n = x.shape[0]
    # m = y.shape[0]
    # assert (m >= n)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if init == 'randn':
        v = torch.randn_like(x)
        v = v / torch.norm(v)
    elif init == 'ones':
        v = torch.ones_like(x) / torch.sqrt(torch.tensor(n).float())

    Fv = fisher_info_matrix_vector_product(y, x, v)
    v = Fv / torch.norm(Fv)
    lmbda = implicit_FIM_eigenvalue(y, x, v)
    i = 0
    error = torch.tensor(1)
    #TODO convergence criteria reworking


    while i < n_steps and error > tol:
        last_error = error
        Fv = fisher_info_matrix_vector_product(y, x, v)
        Fv = Fv - l * v  # minor component
        v_new = Fv / torch.norm(Fv)

        lmbda_new = implicit_FIM_eigenvalue(y, x, v_new)

        error = torch.sqrt((lmbda - lmbda_new) ** 2)
        delta2_lambda = (last_error - error).cpu().detach().numpy()[0][0]
        if verbose and i>=0:
            print("{:3d} -- deltaLambda: {:04.4f} -- delta^2Lambda: {:04.4f}".format(i, error.cpu().detach().numpy()[0][0], delta2_lambda))

        v = v_new
        lmbda = lmbda_new
        i += 1

    return lmbda_new, v_new


def fisher_info_matrix_matrix_product(y, x, A):

    device = x.device
    n = x.shape[0]
    assert A.shape[0] == n
    r = A.shape[1]

    FA = torch.zeros((n, r), device=device)

    # TODO - vectorization speed-up?
    i = 0
    for v in A.t():
        v = v.unsqueeze(1)
        FA[:, i] = fisher_info_matrix_vector_product(y, x, v).squeeze()
        i += 1

    return FA


def implicit_block_power_method(x, y, r, l=0, init='randn', seed=0, tol=1e-10, n_steps=100, verbose=False):
    """
    TODO: Block power method incomplete
    """
    device = x.device
    n = x.shape[0]

    # init
    Q = torch.qr(torch.randn(n, r, device=device))[0]
    error = tol+1
    i = 0
    while i < n_steps and error > tol:

        Z = fisher_info_matrix_matrix_product(y, x, Q)
        Q, R = torch.qr(Z)
        i += 1

        # TODO: Define error. calculation

        if verbose:
            print(i, error.detach().numpy()[0])

    return Q


def lanczos(y, x, n_steps=1000,  e_vecs=None, orthogonalize='full', verbose=False, debug_A=None):
    r""" Lanczos Algorithm with full reorthogonalization after each iteration.

    Computes approximate eigenvalues/vectors of FIM (i.e. the Ritz values and vectors). Each vector returned from
    power iteration is saved (i.e. v, Av, Avv, Avvv, etc.). These vectors span the Krylov subspace which is a good
    approximation of the matrix. Each Lanczos iteration orthogonalizes the current vector against all previously
    obtained vectors. Matrix A is the FIM which is computed implicitly. This orthogonalization is done via Gram-Schmidt
    orthogonalization of the current vector against all previous vectors at each iteration. Since this procedure is done
    by projecting the current vector onto all previous vectors (via matrix-vector multiplication), then removing the
    projected component at each iteration, later iterations to take longer than early iterations.

    We want to estimate :math:'k' eigenvalues and eigenvectors of our Fisher information matrix, :math:'A'. This is done
    by decomposing :math:'A' as

    .. math::
        \begin{align}
        A &\approx  Q_kT_kQ_K^T\\
        \text{where } T_{k}&=\left[\begin{array}{cccc}{\alpha_{1}} & {\beta_{2}} & {} & {} \\ {\beta_{2}} & {\ddots} & {\ddots} & {} \\ {} & {\ddots} & {\ddots} & {\beta_{k}} \\ {} & {} & {\beta_{k}} & {\alpha_{k}}\end{array}\right]\\
        Q_k &= [\bf{q_1}, \bf{q_2}, ..., \bf{q_k}].
        \end{align}

    Here :math:'T_k' is a :math:'(k \times k)' tri-diagonal matrix, and :math:'Q_k' is an :math:'(n\times k)' matrix of
    orthogonal vectors spanning the k-dimensional Krylov subspace of :math:'A'. The matrix :math:'T_k' is easier to
    diagonalize due to its reduced size and tridiagonal structure, :math:'T_k=V\Lambda V^T'. By subbing this into our
    earlier approximation for :math:'A', we find

    .. math::
        \begin{align}
        A &\approx Q_kT_kQ_k^T\\
          &= Q_kV\Lambda V^T Q_k^T
        \end{align}

    Thus, :math:'Q_kV' are our estimates of :math:'k' eigenvectors of :math:'A' and :math:'\text{diag}(\Lambda )' are
    their associated eigenvalues.

    Parameters
    ----------
    y: torch.tensor
        model output (flattened)
    x: torch.tensor
        model input (flattened)
    n_steps: int
        number of power iteration steps (i.e. eigenvalues/eigenvectors to compute and/or return)
    e_vecs: int or array-like
       Eigenvectors to return. If num_evecs is an int, it will return all eigenvectors in range(num_evecs).
       If an iterable, then it will return all vectors selected.
    orthogonalize: {'full', None}
    verbose: bool, optional
        Print progress to screen.
    debug_A: torch.tensor
        For debugging purposes. Explicit FIM for matrix vector multiplication, bypassing implicit implicit FIM.

    Returns
    -------
    eig_vals: torch.tensor
        (n_steps,) tensor of eigenvalues, sorted in descending order.
    eig_vecs: toch.tensor
        (n, len(e_vecs)) tensor of n_steps eigenvectors. If return_evecs=False, then returned  eig_vecs is an empty tensor.
    eig_vecs_ind: array-like
        indices of each returned eigenvector

    References
    __________
    Algorithm 7.2, Applied Numerical Linear Algebra - James W. Demmel

    Examples:
    Run Lanczos algorithm 5000 times, retain top and last 4 eigenvectors.
        >>> ee = Eigendistortion(img, model)
        >>> ee.synthesize(method='lanczos', n_steps=5000, e_vecs=[0,1,2,3,-4,-3,-2,-1], verbose=True)
    """

    n = x.shape[0]
    dtype = x.dtype
    device = x.device

    if n_steps > n:
        n_steps = n
    elif n_steps < 100:
        Warning("Consider increasing n_steps. Convergence of extremal eigenpairs is only guaranteed when n_steps is large.")

    if orthogonalize not in ['full', None]:
        raise Exception("orthogonalize must be {'full', None}, instead got %s" % orthogonalize)

    # T tridiagonal matrix, V orthogonalized Krylov vectors
    T = torch.zeros((n_steps, n_steps), device=device, dtype=dtype)
    Q = torch.zeros((n, n_steps), device=device, dtype=dtype)

    q0 = torch.zeros(n, device=device, dtype=dtype)
    q = torch.randn(n, device=device, dtype=dtype)
    q /= torch.norm(q)
    beta = torch.zeros(1, device=device, dtype=dtype)

    for i in range(n_steps):
        if verbose and i % 200 == 0:
            print('Step {:d}/{:d}'.format(i + 1, n_steps))

        # v = Aq where A is implicitly stored FIM operator
        if debug_A is None:
            v = fisher_info_matrix_vector_product(y, x, q.view(n, 1)).view(n)
        else:
            v = torch.mv(debug_A, q)

        alpha = q.dot(v) # alpha = q'Aq
        # print('alpha:{:f}'.format(alpha))

        if orthogonalize == 'full' and i > 0:  # orthogonalize using Gram-Schmidt TWICE to ensure orthogonality
            v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))
            v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))
        elif orthogonalize is None:  # Standard orthogonalization (against last 2 vecs)
            v += - (alpha * q) - (beta * q0)

        beta = torch.norm(v)
        # print('beta:{:f}'.format(beta))
        if beta == 0:
            print('Vector norm beta=0; Premature stoppage at iter {:d}/{:d}'.format(i, n_steps))
            break

        q0 = q
        q = v / beta  # normalize

        T[i, i] = alpha  # diagonal is alpha
        if i < n_steps - 1:  # off diagonals are beta
            T[i, i + 1] = beta
            T[i + 1, i] = beta
        Q[:, i] = q

    # only use T and Q that were successfully computed
    T = T[:i + 1, :i + 1]
    Q = Q[:, :i + 1]

    if e_vecs is not None:
        import numpy as np
        eig_vals, V = T.symeig(eigenvectors=True)  # expensive final step - diagonalize Tridiag matrix

        vecs_to_return = e_vecs
        eig_vecs_ind = e_vecs

        eig_vecs = Q.mm(V).flip(dims=(1,))[:, vecs_to_return]
    else:
        print("Returning empty Tensor of eigenvectors. Set e_vecs if you want eigvectors returned.")
        Lambda, _ = T.symeig(eigenvectors=False)  # expensive final step
        eig_vecs = torch.zeros(0)

    return eig_vals.flip(dims=(0,)), eig_vecs, eig_vecs_ind


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
    make sure that the class caches learnt distortions, every couple iterations, to prevent from loosing things when crashes
    """

    def __init__(self, image, model, dtype=torch.float32):
        super().__init__()

        self.image = rescale(image, 0, 1)
        self.batch_size, self.n_channels, self.im_height, self.im_width = image.shape
        assert self.batch_size == 1

        if self.n_channels == 3:
            self.color_image = True

        im_flat = self.image.reshape(self.n_channels * self.im_height * self.im_width, 1)
        self.image_flattensor = im_flat.clone().detach().requires_grad_(True).type(dtype)
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
        eig_vals, eig_vecs = torch.symeig(F, eigenvectors=True)

        self.J = J

        return eig_vals.flip(dims=(0,)), eig_vecs.flip(dims=(1,))

    def synthesize(self, method='jacobian', block=None, e_vecs=None, tol=1e-10, n_steps=100, seed=0, verbose=True, debug_A=None):
        '''Compute eigendistortion
        Parameters
        ----------
        method: Eigensolver method ('jacobian', 'block', 'power', 'lanczos'). Jacobian (default) tries to do
            eigendecomposition directly (not recommended for very large matrices). 'power' uses the power method to
            compute first and last eigendistortions, with maximum number of iterations dictated by n_steps. 'block' uses
            power method on a block of eigenvectors, representing the first block of eigendistortions with highest
            associated eigenvalues. 'lanczos' uses the Arnoldi iteration algorithm to estimate the _entire_
            eigenspectrum and eigendistortions (GPU recommended).
        tol: tolerance for error criterion in power iteration
        n_steps: total steps to run for power iteration in eigenvalue computation
        seed: control the random seed for reproducibility
        verbose: boolean, optional (default True)
            show progress during power iteration and Lanczos methods.
        Returns
        -------
        distortions: dict of torch tensors
            dictionary containing the eigenvalues and eigen-distortions in decreasing order
        '''

        if verbose:
            print('out size', self.out_flattensor.size(), 'in size', self.image_flattensor.size())

        if method == 'jacobian' and self.out_flattensor.size(0) * self.image_flattensor.size(0) < 10e7:
            eig_vals, eig_vecs  = self.solve_eigenproblem()

            self.distortions['eigenvectors'] = eig_vecs.cpu().detach()
            self.distortions['eigenvalues'] = eig_vals.cpu().detach()
            self.distortions['eigenvector_index'] = torch.arange(len(eig_vals))

        elif method == 'block' and block is not None:
            print('under construction')

            distortions = implicit_block_power_method(self.image_flattensor, self.out_flattensor, r=block, n_steps=n_steps, verbose=verbose)
            return distortions.detach()

        elif method == 'power':
            if verbose:
                print('implicit power method, computing the maximum distortion \n')
            lmbda_max, v_max = implicit_power_method(self.out_flattensor, self.image_flattensor, l=0, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

            if verbose:
                print('\nimplicit power method, computing the minimum distortion \n')
            lmbda_min, v_min = implicit_power_method(self.out_flattensor, self.image_flattensor, l=lmbda_max, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

            self.distortions['eigenvalues'] = torch.cat([lmbda_max, lmbda_min]).cpu().detach()
            self.distortions['eigenvectors'] = torch.cat((v_max, v_min), dim=1).cpu().detach()
            self.distortions['eigenvector_index'] = [0, len(self.image_flattensor)]

        elif method == 'lanczos' and n_steps is not None:
            eig_vals, eig_vecs, eig_vecs_ind = lanczos(self.out_flattensor, self.image_flattensor, n_steps=n_steps, orthogonalize='full', e_vecs=e_vecs, verbose=verbose, debug_A=debug_A)
            self.distortions['eigenvalues'] = eig_vals.type(torch.float32).cpu().detach()
            self.distortions['eigenvectors'] = eig_vecs.type(torch.float32).cpu().detach()
            self.distortions['eigenvector_index'] = eig_vecs_ind

        return self.distortions

    def display(self, alpha=5, beta=10, **kwargs):
        # change to plot_eigendistortion(), plot_eigenvalues()
        import pyrtools as pt

        image = to_numpy(self.image)
        maxdist = to_numpy(self.distortions['0'][1].reshape(self.image.shape))
        mindist = to_numpy(self.distortions['-1'][1].reshape(self.image.shape))

        pt.imshow([image, image + alpha * maxdist, beta * maxdist], title=['original', 'original + ' + str(alpha) + ' maxdist', str(beta) + ' * maximum eigendistortion'], **kwargs);
        pt.imshow([image, image + alpha * mindist, beta * mindist], title=['original', 'original + ' + str(alpha) + ' mindist', str(beta) + ' * minimum eigendistortion'], **kwargs);

