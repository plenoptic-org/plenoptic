import torch
from torch import nn
from ..tools.signal import rescale
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
from ..tools.data import to_numpy
import numpy as np
import pyrtools as pt


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

    # detach the following product or else graph history accumulates and memory explodes
    Fv = vector_jacobian_product(y, x, Jv, retain_graph=True, create_graph=False).detach()
    return Fv.t()


def implicit_FIM_eigenvalue(y, x, v):
    """Implicitly compute the eigenvalue of the Fisher Information Matrix corresponding to eigenvector v
    lmbda = v.T F v
    """
    Fv = fisher_info_matrix_vector_product(y, x, v)
    lmbda = torch.mm(Fv.t(), v)  # conjugate transpose
    return lmbda


def implicit_power_method(y, x, l=0, init='randn', seed=0, tol=1e-10, n_steps=100, verbose=False):
    """ Use power method to obtain largest (smallest) eigenvalue/vector pair.

    Apply the power method algorithm to approximate the extremal eigenvalue and eigenvector of the Fisher Information
    Matrix, without explicitly representing that matrix.

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
    init: {'randn', 'ones'}
        starting point for the power iteration. 'randn' is random normal noise vector, 'ones' is a ones vector.
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
        if verbose and i>=0:
            print("{:3d} -- deltaLambda: {:04.4f}".format(i, error.item()))

        v = v_new
        lmbda = lmbda_new
        i += 1

    return lmbda_new, v_new


def lanczos(y, x, n_steps=1000, e_vecs=None, orthogonalize='full', verbose=False, debug_A=None):
    r""" Lanczos Algorithm with full reorthogonalization after each iteration.

    Computes approximate eigenvalues/vectors of FIM (i.e. the Ritz values and vectors). Each vector returned from
    power iteration is saved (i.e. v, Av, Avv, Avvv, etc.). These vectors span the Krylov subspace which is a good
    approximation of the matrix. Each Lanczos iteration orthogonalizes the current vector against all previously
    obtained vectors. Matrix A is the FIM which is computed implicitly. This orthogonalization is done via Gram-Schmidt
    orthogonalization of the current vector against all previous vectors at each iteration. Since this procedure is done
    by projecting the current vector onto all previous vectors (via matrix-vector multiplication), then removing the
    projected component at each iteration, later iterations to take longer than early iterations.

    This method simultaneously approximates both ends of the eigenspectrum. This is analyzed in depth in the below
    references [1] and [2]

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
        number of power iteration steps (i.e. eigenvalues/eigenvectors to compute and/or return). Recommended to do many
        more than amount of requested eigenvalue/eigenvectors, N. Some say 2N steps but as many as possible is preferred.
    e_vecs: int or array-like
       Eigenvectors to return. If num_evecs is an int, it will return all eigenvectors in range(num_evecs).
       If an iterable, then it will return all vectors selected.
    orthogonalize: {'full', None}
        Ideally, each synthesized vector should be mutually orthogonal; due to numerical error however, vectors computed
        from much earlier iterations will 'leak' into current iterations. This generally causes newly generated
        eigenvectors to align with eigenvectors with highest magnitude eigenvalue. 'full' argument explicitly
        orthogonalizes the entire set at each iteration (twice!) to mitigate the effects of leakage. If 'None', then the
        current iteration will only orthogonalize against the previous two eigenvectors, as in the original Lanczos alg.
    verbose: bool, optional
        Print progress to screen.
    debug_A: torch.tensor
        For debugging purposes. Explicit FIM for matrix vector multiplication. Bypasses matrix-vector product with
        implicitly stored FIM of model.

    Returns
    -------
    eig_vals: torch.tensor
        (n_steps,) if e_vecs is None. tensor of eigenvalues, sorted in descending order.
        (len(e_vecs),) if e_vecs is not None. eigenvalues corresponding to the eigenvectors at that index.
        Note: this is most accurate for extremal eigenvalues, e.g. top (bottom) 10,
    eig_vecs: toch.tensor
        (n, len(e_vecs)) tensor of n_steps eigenvectors. If return_evecs=False, then returned  eig_vecs is an empty tensor.
    eig_vecs_ind: array-like
        indices of each returned eigenvector

    References
    __________
    [1] Algorithm 7.2, Applied Numerical Linear Algebra - James W. Demmel
    [2] https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method

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

    if e_vecs is not None and n_steps < 2*len(e_vecs):
        Warning("Consider increasing n_steps to at least 2*len(e_vecs), but preferably much much more."
                " Convergence of extremal eigenpairs is only guaranteed when n_steps is large.")

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

        if orthogonalize.lower() == 'full' and i > 0:  # orthogonalize using Gram-Schmidt TWICE to ensure orthogonality
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
        # normalize
        q = v / beta

        # diagonal is alpha
        T[i, i] = alpha

        # off diagonals are beta
        if i < n_steps - 1:
            T[i, i + 1] = beta
            T[i + 1, i] = beta

        # assign new vector to matrix
        Q[:, i] = q

    # only use T and Q that were successfully computed
    T = T[:i + 1, :i + 1]
    Q = Q[:, :i + 1]

    if e_vecs is not None:
        # expensive final step - diagonalize Tridiag matrix
        eig_vals, V = T.symeig(eigenvectors=True)

        vecs_to_return = e_vecs
        eig_vecs_ind = e_vecs

        eig_vecs = Q.mm(V).flip(dims=(1,))[:, vecs_to_return]
        eig_vals = eig_vals[vecs_to_return]
    else:
        print("Returning all computed eigenvals and 0 eigenvectors. Set e_vecs if you want eigvectors returned.")
        eig_vals, _ = T.symeig(eigenvectors=False)  # expensive final step
        eig_vecs = torch.zeros(0)
        eig_vecs_ind = torch.zeros(0)

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
        r""" Eigendecomposition of explicitly computed FIM.
        To be used when the input is small (e.g. less than 70x70 image on cluster or 30x30 on your own machine). This
        method obviates the power iteration and its related algorithms (e.g. Lanczos).
        """
        J = jacobian(self.out_flattensor, self.image_flattensor)
        F = torch.mm(J.t(), J)
        eig_vals, eig_vecs = torch.symeig(F, eigenvectors=True)

        self.J = J

        return eig_vals.flip(dims=(0,)), eig_vecs.flip(dims=(1,))

    def synthesize(self, method='jacobian', e_vecs=None, tol=1e-10, n_steps=100, orthogonalize='full', seed=0, verbose=True, debug_A=None):
        '''Compute eigendistortion
        Parameters
        ----------
        method: String
            Eigensolver method ('jacobian', 'block', 'power', 'lanczos'). Jacobian (default) tries to do
            eigendecomposition directly (not recommended for very large matrices). 'power' uses the power method to
            compute first and last eigendistortions, with maximum number of iterations dictated by n_steps. 'block' uses
            power method on a block of eigenvectors, representing the first block of eigendistortions with highest
            associated eigenvalues. 'lanczos' uses the Arnoldi iteration algorithm to estimate the _entire_
            eigenspectrum and eigendistortions (GPU recommended).
        e_vecs: iterable
            integer list of which eigenvectors to return
        tol: float
            tolerance for error criterion in power iteration
        n_steps: int
            total steps to run for power iteration in eigenvalue computation
        orthogonalize: {'full', None}
            For Lanzos method, full re-orthogonalization ('full') or standard algorithm (None). More details in method.
        seed: int
            control the random seed for reproducibility
        verbose: boolean, optional
            show progress during power iteration and Lanczos methods.
        debug_A: torch.tensor
            Explicit Fisher Information Matrix in the form of 2D tensor. Used to debug lanczos algorithm.

        Returns
        -------
        distortions: dict with keys {'eigenvalues', 'eigenvectors', 'eigenvector_index'}
            Dictionary containing the eigenvalues and eigen-distortions in decreasing order and their indices. This dict
            is also added as an attribute to the object.
        '''

        if verbose:
            print('out size', self.out_flattensor.size(), 'in size', self.image_flattensor.size())

        if method == 'jacobian' and self.out_flattensor.size(0) * self.image_flattensor.size(0) < 10e7:
            eig_vals, eig_vecs = self.solve_eigenproblem()

            self.distortions['eigenvalues'] = eig_vals.detach()
            self.distortions['eigenvectors'] = self.vector_to_image(eig_vecs.detach())
            self.distortions['eigenvector_index'] = torch.arange(len(eig_vals))

        elif method == 'power':
            if verbose:
                print('implicit power method, computing the maximum distortion \n')
            lmbda_max, v_max = implicit_power_method(self.out_flattensor, self.image_flattensor, l=0, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

            if verbose:
                print('\nimplicit power method, computing the minimum distortion \n')
            lmbda_min, v_min = implicit_power_method(self.out_flattensor, self.image_flattensor, l=lmbda_max, init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

            self.distortions['eigenvalues'] = torch.cat([lmbda_max, lmbda_min]).detach()
            self.distortions['eigenvectors'] = self.vector_to_image(torch.cat((v_max, v_min), dim=1).detach())
            self.distortions['eigenvector_index'] = [0, len(self.image_flattensor)]

        elif method == 'lanczos' and n_steps is not None:
            eig_vals, eig_vecs, eig_vecs_ind = lanczos(self.out_flattensor, self.image_flattensor, n_steps=n_steps, orthogonalize=orthogonalize, e_vecs=e_vecs, verbose=verbose, debug_A=debug_A)
            self.distortions['eigenvalues'] = eig_vals.type(torch.float32).detach()

            # reshape into image if not empty tensor
            if eig_vecs.ndim == 2 and eig_vecs.shape[0] == self.im_height*self.im_width:
                self.distortions['eigenvectors'] = self.vector_to_image(eig_vecs.type(torch.float32).detach())
            else:
                self.distortions['eigenvectors'] = eig_vecs

            self.distortions['eigenvector_index'] = eig_vecs_ind

        return self.distortions

    def vector_to_image(self, vecs):
        """ Reshapes eigenvectors back into correct image dimensions.
            Returns an list with N images for each column in vecs.
        """

        img = [vecs[:,i].reshape((self.im_height, self.im_width)) for i in range(vecs.shape[1])]

        return img

    def display(self, alpha=5, beta=10, **kwargs):
        """
        Will soon be moved to different script. Should work for now though.
        """
        image = to_numpy(self.image)
        maxdist = to_numpy(self.distortions['eigenvectors'][0])
        mindist = to_numpy(self.distortions['eigenvectors'][-1])

        pt.imshow([image, image + alpha * maxdist, beta * maxdist], title=['original', 'original + ' + str(alpha) + ' maxdist', str(beta) + ' * maximum eigendistortion'], **kwargs)
        pt.imshow([image, image + alpha * mindist, beta * mindist], title=['original', 'original + ' + str(alpha) + ' mindist', str(beta) + ' * minimum eigendistortion'], **kwargs);
