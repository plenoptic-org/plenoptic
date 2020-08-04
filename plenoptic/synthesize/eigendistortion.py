import torch
from torch import nn
from ..tools.signal import rescale
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
from ..tools.data import to_numpy
import numpy as np
import pyrtools as pt
import warnings


def fisher_info_matrix_vector_product(y, x, v):
    r"""Compute Fisher Information Matrix Vector Product: :math:`Fv`

    Parameters
    ----------
    y: torch.Tensor
        output tensor with gradient attached
    x: torch.Tensor
        input tensor with gradient attached
    v: torch.Tensor
        direction

    Returns
    -------
    Fv: torch.Tensor
        vector, fvp

    Notes
    -----
    under white Gaussian noise assumption, :math:`F` is matrix multiplication of
    Jacobian transpose and Jacobian: :math:`F = J^T J`
    Hence:
    :math:`Fv = J^T (Jv)`
    """

    Jv = jacobian_vector_product(y, x, v)
    Fv = vector_jacobian_product(y, x, Jv, detach=True)

    return Fv


def implicit_FIM_eigenvalue(y, x, v):
    r"""Implicitly compute the eigenvalue of the Fisher Information Matrix corresponding to eigenvector v
    :math:`\lambda= v^T F v`
    """
    Fv = fisher_info_matrix_vector_product(y, x, v)
    lmbda = torch.mm(Fv.t(), v)
    return lmbda


def implicit_power_method(y, x, l=0, init='randn', seed=0, tol=1e-10, n_steps=1000, verbose=False):
    r""" Use power method to obtain largest (smallest) eigenvalue/vector pair.
    Apply the power method algorithm to approximate the extremal eigenvalue and eigenvector of the Fisher Information
    Matrix, without explicitly representing that matrix.

    Parameters
    ----------
    y: torch.Tensor
        output tensor, with gradient
    x: torch.Tensor
        input
    l: torch.Tensor, optional
        Optional argument. When l=0, this function estimates the leading eval evec pair. When l is set to the
        estimated maximum eigenvalue, this function will estimate the smallest eval evec pair (minor component).
    init: {'randn', 'ones'}
        starting point for the power iteration. 'randn' is random normal noise vector, 'ones' is a ones vector. Both
        will be normalized.
    seed: float, optional
        manual seed
    tol: float, optional
        tolerance value
    n_steps: int, optional
        maximum number of steps
    verbose: bool, optional
        flag to control amout of information printed out

    Returns
    -------
    lmbda: float
        eigenvalue
    v: torch.Tensor
        eigenvector
    Notes
    -----
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

    if init == 'randn' or init != 'ones':
        v = torch.randn_like(x)
    elif init == 'ones':
        v = torch.ones_like(x)

    v = v / v.norm()

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
            print(f"{i:3d} -- deltaLambda: {error.item():04.4f}")

        v = v_new
        lmbda = lmbda_new
        i += 1

    return lmbda_new, v_new


def lanczos(y, x, n_steps=1000, e_vecs=[], orthogonalize=True, verbose=False, debug_A=None):
    r""" Lanczos Algorithm with full reorthogonalization after each iteration.

    Computes approximate eigenvalues/vectors of FIM (i.e. the Ritz values and vectors). Each vector returned from
    power iteration is saved (i.e. :math:`v, Av, Avv, Avvv`, etc.). These vectors span the Krylov subspace which is a good
    approximation of the matrix. Each Lanczos iteration orthogonalizes the current vector against all previously
    obtained vectors. Matrix :math:`A` is the FIM which is computed implicitly. This orthogonalization is done via
    Gram-Schmidt orthogonalization of the current vector against all previous vectors at each iteration. Since this
    procedure is done by projecting the current vector onto all previous vectors (via matrix-vector multiplication),
    then removing the projected component at each iteration, later iterations to take longer than early iterations.

    This method simultaneously approximates both ends of the eigenspectrum. This is analyzed in depth in the below
    references [1] and [2].

    We want to estimate :math:`k` eigenvalues and eigenvectors of our Fisher information matrix, :math:`A`. This is done
    by decomposing :math:`A` as

    .. math::
        \begin{align}
        A &\approx  Q_kT_kQ_K^T\\
        \text{where } T_{k}&=\left[\begin{array}{cccc}{\alpha_{1}} & {\beta_{2}} & {} & {} \\ {\beta_{2}} & {\ddots} & {\ddots} & {} \\ {} & {\ddots} & {\ddots} & {\beta_{k}} \\ {} & {} & {\beta_{k}} & {\alpha_{k}}\end{array}\right]\\
        Q_k &= [\bf{q_1}, \bf{q_2}, ..., \bf{q_k}].
        \end{align}

    Here :math:`T_k` is a :math:`(k \times k)` tri-diagonal matrix, and :math:`Q_k` is an :math:`(n\times k)` matrix of
    orthogonal vectors spanning the k-dimensional Krylov subspace of :math:`A`. The matrix :math:`T_k` is easier to
    diagonalize due to its reduced size and tridiagonal structure, :math:`T_k=V\Lambda V^T`. By subbing this into our
    earlier approximation for :math:`A`, we find

    .. math::
        \begin{align}
        A &\approx Q_kT_kQ_k^T\\
          &= Q_kV\Lambda V^T Q_k^T
        \end{align}

    Thus, :math:`Q_kV` are our estimates of :math:`k` eigenvectors of :math:`A` and :math:`\text{diag}(\Lambda )` are
    their associated eigenvalues.

    Parameters
    ----------
    y: torch.Tensor
        model output (flattened)
    x: torch.Tensor
        model input (flattened)
    n_steps: int
        number of power iteration steps (i.e. eigenvalues/eigenvectors to compute and/or return). Recommended to do many
        more than amount of requested eigenvalue/eigenvectors, N. Some say 2N steps but as many as possible is preferred.
    e_vecs: int or array-like
       Eigenvectors to return. If num_evecs is an int, it will return all eigenvectors in range(num_evecs).
       If an iterable, then it will return all vectors selected.
    orthogonalize: bool
        Ideally, each synthesized vector should be mutually orthogonal; due to numerical error however, vectors computed
        from much earlier iterations will 'leak' into current iterations. This generally causes newly generated
        eigenvectors to align with eigenvectors with highest magnitude eigenvalue. True argument explicitly
        orthogonalizes the entire set at each iteration (twice!) to mitigate the effects of leakage. If False, then the
        current iteration will only orthogonalize against the previous two eigenvectors, as in the original Lanczos alg.
    verbose: bool, optional
        Print progress to screen.
    debug_A: torch.Tensor
        For debugging purposes. Explicit FIM for matrix vector multiplication. Bypasses matrix-vector product with
        implicitly stored FIM of model.

    Returns
    -------
    eig_vals: torch.Tensor
        Tensor of eigenvalues, sorted in descending order. torch.Size(n_steps,) if e_vecs is None.
        torch.Size(len(e_vecs),) if e_vecs is not None. eigenvalues corresponding to the eigenvectors at that index.
        Note: this is most accurate for extremal eigenvalues, e.g. top (bottom) 10,
    eig_vecs: toch.Tensor
        torch.Size(n, len(e_vecs)) tensor of n_steps eigenvectors. If return_evecs=False, then returned  eig_vecs is an
        empty tensor.
    eig_vecs_ind: torch.Tensor
        indices of each returned eigenvector

    References
    ----------
    [1] Algorithm 7.2, Applied Numerical Linear Algebra - James W. Demmel
    [2] https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method

    Examples
    --------
    Run Lanczos algorithm 5000 times, retain top and last 4 eigenvectors.
        >>> ee = Eigendistortion(img, model)
        >>> ee.synthesize(method='lanczos', n_steps=5000, e_vecs=[0,1,2,3,-4,-3,-2,-1], verbose=True)
    """

    n = x.shape[0]
    dtype = x.dtype
    device = x.device

    if len(e_vecs) > n_steps:
        raise Exception("Lanczos method requires at least n_steps=len(e_vecs) (but should preferably be much more).")

    if n_steps > n:
        warnings.warn("Dim of Fisher matrix, n, is < n_steps. Setting n_steps = n")
        n_steps = n
    if n_steps < 2*len(e_vecs):
        warnings.warn("n_steps should be at least 2*len(e_vecs) but preferably even more for accuracy.")

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

        if orthogonalize and i > 0:  # orthogonalize using Gram-Schmidt TWICE to ensure orthogonality
            v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))
            v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))
        else:  # Standard orthogonalization (against last 2 vecs)
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

    if len(e_vecs) > 0:
        # expensive final step - diagonalize Tridiag matrix
        eig_vals, V = T.symeig(eigenvectors=True)

        vecs_to_return = e_vecs
        eig_vecs_ind = torch.as_tensor(e_vecs)

        eig_vecs = Q.mm(V).flip(dims=(1,))[:, vecs_to_return]
        eig_vals = eig_vals[vecs_to_return]
    else:
        print("Returning all computed eigenvals and 0 eigenvectors. Set e_vecs if you want eigvectors returned.")
        eig_vals, _ = T.symeig(eigenvectors=False)  # expensive final step
        eig_vecs = torch.zeros(0)
        eig_vecs_ind = torch.zeros(0)

    return eig_vals.flip(dims=(0,)), eig_vecs, eig_vecs_ind


class Eigendistortion(nn.Module):
    r"""Synthesis object to compute eigendistortions induced by a model on a given input image.

    Attributes
    ----------
    batch_size: int
    n_channels: int
    im_height: int
    im_width: int
    color_image: bool
    image_flattensor: torch.Tensor
    model_input: torch.Tensor
    model_output: torch.Tensor
    out_flattensor: torch.Tensor
    distortions: dict
        Dict whose keys are {'eigenvectors', 'eigenvalues', 'eigenvector_index'} after `synthesis()` is run.

    Parameters
    -----------
    image: torch.Tensor
        image, torch.Size(batch=1, channel, height, width). We currently do not support batches of images,
        as each image requires its own optimization.
    model: torch class
        torch model with defined forward and backward operations

    Notes
    -----
    This is a method for comparing image representations in terms of their ability to explain perceptual sensitivity
    in humans. It estimates eigenvectors of the FIM. A model, :math:`y = f(x)`, is a deterministic (and differentiable)
    mapping from the input pixels :math:`x \in \mathbb{R}^n` to a mean output response vector :math:`y\in \mathbb{
    R}^m`, where we assume additive white
    Gaussian noise in the response space:

    .. math::
        \begin{align}
        f: \mathbb{R}^n &\rightarrow \mathbb{R}^m\\
            x &\rightarrow y
        \end{align}

    The Jacobian matrix at x is:
        :math:`J(x) = J = dydx`,       :math:`J\in\mathbb{R}^{m \times n}` (ie. output_dim x input_dim)
    is the matrix of all first-order partial derivatives of the vector-valued function f.
    The Fisher Information Matrix (FIM) at x, under white Gaussian noise in the response space, is:
        :math:`F = J^T J`
    It is a quadratic approximation of the discriminability of distortions relative to :math:`x`.
    Berardino, A., Laparra, V., Ballé, J. and Simoncelli, E., 2017.
    Eigen-distortions of hierarchical representations.
    In Advances in neural information processing systems (pp. 3530-3539).
    http://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf
    http://www.cns.nyu.edu/~lcv/eigendistortions/

    TODO
    ----
    enforce bounding box during optimization (see other classes in this repo, stretch/squish)
    check for division by zero
    handle color image
    allow for caching learnt distortions every few iterations to prevent from loosing things when crashes
    """

    def __init__(self, image, model, dtype=torch.float32):
        super().__init__()

        self.image = rescale(image, 0, 1)
        self.batch_size, self.n_channels, self.im_height, self.im_width = image.shape
        assert self.batch_size == 1
        assert len(image.shape) == 4, "Input must be (1)"

        self.color_image = (self.n_channels == 3)

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

    def synthesize(self, method='power', e_vecs=[], tol=1e-10, n_steps=1000, orthogonalize=True, seed=0,
                   verbose=True, debug_A=None):
        r"""Compute eigendistortions of Fisher Information Matrix with given input image.

        Parameters
        ----------
        method: str
            Eigensolver method {'jacobian', 'block', 'power', 'lanczos'}. Jacobian tries to do
            eigendecomposition directly (not recommended for very large matrices). 'power' (default) uses the power
            method to compute first and last eigendistortions, with maximum number of iterations dictated by n_steps.
            'lanczos' uses the Arnoldi iteration algorithm to estimate the _entire_ eigenspectrum and thus more than
            just two eigendistortions, as opposed to the power method. Note: 'lanczos' method is experimental and may be
            numerically unstable. We recommend using the power method.
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
        verbose: bool
            show progress during power iteration and Lanczos methods.
        debug_A: torch.Tensor, optional
            Explicit Fisher Information Matrix in the form of 2D tensor. Used to debug lanczos algorithm.
            Dimensionality must be torch.Size(N, N)

        Returns
        -------
        distortions: dict
            Dictionary containing keys {'eigenvalues', 'eigenvectors', 'eigenvector_index'} the eigenvalues and
            eigen-distortions in decreasing order and their indices. This dict is also added as an attribute to the
            object.
        """

        if method == 'jacobian' and self.out_flattensor.size(0) * self.image_flattensor.size(0) > 1e6:
            warnings.warn("Jacobian > 1e6 elements and may cause out-of-memory. Use method =  {'power', 'lanczos'}.")

        if verbose:
            print(f'Output dim: {self.out_flattensor.size(0)}. Input dim: {self.image_flattensor.size(0)}')

        if method == 'jacobian':
            eig_vals, eig_vecs = self.solve_eigenproblem()

            self.distortions['eigenvalues'] = eig_vals.detach()
            self.distortions['eigenvectors'] = self.vector_to_image(eig_vecs.detach())
            self.distortions['eigenvector_index'] = torch.arange(len(eig_vals))

        elif method == 'power':
            if verbose:
                print('implicit power method, computing the maximum distortion \n')
            lmbda_max, v_max = implicit_power_method(self.out_flattensor, self.image_flattensor, l=0, init='randn',
                                                     seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

            if verbose:
                print('\nimplicit power method, computing the minimum distortion \n')
            lmbda_min, v_min = implicit_power_method(self.out_flattensor, self.image_flattensor, l=lmbda_max,
                                                     init='randn', seed=seed, tol=tol, n_steps=n_steps, verbose=verbose)

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
        r""" Reshapes eigenvectors back into correct image dimensions.

        Parameters
        ----------
        vecs: torch.Tensor
            Eigendistortion tensor with torch.Size(N, num_distortions). Each distortion will be reshaped into the
            original image shape and placed in a list.

        Returns
        -------
        imgs: list
            List of torch.Tensor images, each with torch.Size(img_height, im_width).
        """

        imgs = [vecs[:,i].reshape((self.im_height, self.im_width)) for i in range(vecs.shape[1])]

        return imgs

    def display(self, alpha=5., beta=10., **kwargs):
        r""" Displays the first and last synthesized eigendistortions alone, and added to the image.

        Parameters
        ----------
        alpha: float
            Amount by which to scale eigendistortion for image + (alpha * eigendistortion) for display.
        beta: float
            Amount by which to scale eigendistortion to be displayed alone.
        kwargs:
            Additional arguments for pt.imshow()
        """

        assert len(self.distortions['eigenvectors'])>1, "Assumes at least two eigendistortions were synthesized."

        image = to_numpy(self.image)
        max_dist = to_numpy(self.distortions['eigenvectors'][0])
        min_dist = to_numpy(self.distortions['eigenvectors'][-1])

        pt.imshow([image, image + alpha * max_dist, beta * max_dist],
                  title=['original', f'original + {alpha:.0f} * maxdist', f'{beta:.0f} * maxdist'], **kwargs)

        pt.imshow([image, image + alpha * min_dist, beta * min_dist],
                  title=['original', f'original + {alpha:.0f} * mindist', f'{beta:.0f} * mindist'], **kwargs);
