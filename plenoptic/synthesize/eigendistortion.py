import torch
from ..tools.signal import rescale
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
import numpy as np
import pyrtools as pt
import warnings
from .synthesis import Synthesis


def fisher_info_matrix_vector_product(y, x, v):
    r"""Compute Fisher Information Matrix Vector Product: :math:`Fv`

    Parameters
    ----------
    y: torch.Tensor
        Output tensor with gradient attached
    x: torch.Tensor
        Input tensor with gradient attached
    v: torch.Tensor
        The vectors with which to compute Fisher vector products.

    Returns
    -------
    Fv: torch.Tensor
        vector, fvp

    Notes
    -----
    Under white Gaussian noise assumption, :math:`F` is matrix multiplication of Jacobian transpose and Jacobian:
    :math:`F = J^T J`. Hence:
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


class Eigendistortion(Synthesis):
    r"""Synthesis object to compute eigendistortions induced by a model on a given input image.
    Attributes
    ----------
    batch_size: int
    n_channels: int
    im_height: int
    im_width: int
    color_image: bool
    model_input: torch.Tensor
        Image input tensor with ``torch.Size([batch_size=1, n_channels, im_height, im_width])``. No support for
        batch
        synthesis yet.
    model_output: torch.Tensor
        Model representation of input image. Size varies by model.
    image_flattensor: torch.Tensor
        Eigendistortions are computed using vectorized (flattened) inputs and outputs.
    out_flattensor: torch.Tensor
        See above.
    distortions: dict
        Dict whose keys are {'eigenvectors', 'eigenvalues', 'eigenvector_index'} after `synthesis()` is run.
    jacobian: torch.Tensor
        Is only set when :func:`synthesize` is run with ``method='jacobian'``. Default to ``None``.
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
    in humans. It estimates eigenvectors of the FIM. A model, :math:`y = f(x)`, is a deterministic (and
    differentiable)
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
    """

    def __init__(self, base_signal, model, **model_kwargs):
        base_signal = rescale(base_signal, 0, 1)
        # TODO: make note about rescaling
        assert len(base_signal.shape) == 4, "Input must be torch.Size([batch=1, n_channels, im_height, im_width])"

        self.batch_size, self.n_channels, self.im_height, self.im_width = base_signal.shape
        assert self.batch_size == 1, "Batch synthesis is not available"
        self.is_color_image = (self.n_channels == 3)

        # flatten and attach gradient and reshape to image
        self.input_flat = base_signal.flatten().unsqueeze(1).requires_grad_(True)

        super().__init__(self.input_flat.view(*base_signal.shape), model, loss_function=None)

        if len(self.base_representation) > 1:
            self.representation_flat = torch.cat([s.squeeze().view(-1) for s in self.base_representation]).unsqueeze(1)
        else:
            self.representation_flat = self.base_representation.squeeze().view(-1).unsqueeze(1)

        print(self.input_flat.shape, self.representation_flat.shape)

        self.distortions = dict()
        self.jacobian = None

    @staticmethod
    def _hidden_attrs():
        """Inherited attributes from parent `Synthesis` class that aren't necessary for `Eigendistortion`."""
        hidden_attrs = ['coarse_to_fine', 'gradient', 'loss_function', 'learning_rate',
                        'objective_function', 'plot_loss', 'plot_representation_error',
                         'representation_error', 'saved_representation', 'saved_representation_gradient',
                        'saved_signal_gradient', 'scales', 'scales_finished', 'scales_loss', 'scales_timing',
                        'synthesized_representation']
        return hidden_attrs

    def __dir__(self):
        """Hide unused parent attributes from user"""
        attrs = set(dir(super()) + list(self.__dict__.keys()) + list(Eigendistortion.__dict__.keys()))
        hidden_attrs = set(self._hidden_attrs())
        to_keep = list(attrs.difference(hidden_attrs))
        return to_keep

    def __getattribute__(self, attr):
        """Prevent access to certain attributes from parent class"""
        if hasattr(Synthesis, attr) and attr in self._hidden_attrs():
            raise AttributeError(f"`Eigendistortion` does not require `{attr}`, so it's restricted.")
        return Synthesis.__getattribute__(self, attr)

    @classmethod
    def load(file_path, model_attr_name='model', model_constructor=None, map_location='cpu',
             **state_dict_kwargs):
        # TODO
        super().load(file_path, model_attr_name, model_constructor, map_location, **state_dict_kwargs)

    def save(self, file_path, save_model_reduced=False, attrs=['model'], model_attr_names=['model']):
        # TODO
        super().save(file_path, save_model_reduced, attrs, model_attr_names)

    def to(self, *args, attrs=[], **kwargs):
        # TODO
        super().to(*args, attrs, **kwargs)

    def synthesize(self, method='power', e_vecs=None, tol=1e-10, n_steps=1000, seed=0, verbose=True, print_every=1,
                   debug_A=None):
        r"""Compute eigendistortions of Fisher Information Matrix with given input image.

        Parameters
        ----------
        method: {'jacobian', 'power', 'lanczos'}, optional
            Eigensolver method. Jacobian tries to do eigendecomposition directly (
            not recommended for very large matrices). 'power' (default) uses the power method to compute first and
            last eigendistortions, with maximum number of iterations dictated by n_steps. 'lanczos' uses the Arnoldi
            iteration algorithm to estimate the _entire_ eigenspectrum and thus more than just two eigendistortions,
            as opposed to the power method. Note: 'lanczos' method is experimental and may be numerically unstable.
            We recommend using the power method.
        e_vecs: iterable, optional
            Integer list of which eigenvectors to return for ``method='lanczos'``.
        tol: float, optional
            Tolerance for error criterion in power iteration.
        n_steps: int, optional
            Total steps to run for ``method='power'`` or ``method='lanczos'`` in eigenvalue computation.
        seed: int, optional
            Control the random seed for reproducibility.
        verbose: bool, optional
            Show progress during ``method='power'`` or ``method='lanczos'``.
        print_every: int, optional
            Prints progress of iterative method after every ``print_every`` steps.
        debug_A: torch.Tensor, optional
            Explicit Fisher Information Matrix in the form of 2D tensor. Used to debug lanczos algorithm.
            Dimensionality must be torch.Size([N, N]) where N is the flattened input size.

        Returns
        -------
        distortions: dict
            Dictionary containing keys {'eigenvalues', 'eigenvectors', 'eigenvector_index'} the eigenvalues and
            eigen-distortions in decreasing order and their indices. This dict is also added as an attribute to the
            object.
        """

        if e_vecs is None:
            e_vecs = []

        assert method in ['power', 'jacobian', 'lanczos'], "method must be in {'power', 'jacobian', 'lanczos'}"

        if method == 'jacobian' and self.representation_flat.size(0) * self.input_flat.size(0) > 1e6:
            warnings.warn("Jacobian > 1e6 elements and may cause out-of-memory. Use method =  {'power', 'lanczos'}.")

        if verbose:
            print(f'Output dim: {self.representation_flat.size(0)}. Input dim: {self.input_flat.size(0)}')

        if method == 'jacobian':
            eig_vals, eig_vecs = self._solve_eigenproblem()
            self.distortions['eigenvalues'] = eig_vals.detach()
            self.distortions['eigenvectors'] = self._vector_to_image(eig_vecs.detach())
            self.distortions['eigenvector_index'] = torch.arange(len(eig_vals))

        elif method == 'power':
            if verbose:
                print('Power method -- computing the maximum distortion \n')
            lmbda_max, v_max = self._implicit_power_method(l=0, init='randn', seed=seed, tol=tol, n_steps=n_steps,
                                                           verbose=verbose, print_every=print_every)

            if verbose:
                print('\nPower method -- computing the minimum distortion \n')
            lmbda_min, v_min = self._implicit_power_method(l=lmbda_max, init='randn', seed=seed, tol=tol,
                                                           n_steps=n_steps, verbose=verbose, print_every=print_every)

            self.distortions['eigenvalues'] = torch.cat([lmbda_max, lmbda_min]).squeeze()
            self.distortions['eigenvectors'] = self._vector_to_image(torch.cat((v_max, v_min), dim=1).detach())
            self.distortions['eigenvector_index'] = [0, len(self.input_flat) - 1]

        elif method == 'lanczos' and n_steps is not None:
            eig_vals, eig_vecs, eig_vecs_ind = self._lanczos(n_steps=n_steps, e_vecs=e_vecs, verbose=verbose,
                                                             print_every=print_every, debug_A=debug_A)
            self.distortions['eigenvalues'] = eig_vals.type(torch.float32).detach()

            # reshape into image if not empty tensor
            if eig_vecs.ndim == 2 and eig_vecs.shape[0] == self.im_height*self.im_width:
                self.distortions['eigenvectors'] = self._vector_to_image(eig_vecs.type(torch.float32).detach())
            else:
                self.distortions['eigenvectors'] = eig_vecs

            self.distortions['eigenvector_index'] = eig_vecs_ind

        return self.distortions

    def _vector_to_image(self, vecs):
        r""" Reshapes eigenvectors back into correct image dimensions.

        Parameters
        ----------
        vecs: torch.Tensor
            Eigendistortion tensor with ``torch.Size([N, num_distortions])``. Each distortion will be reshaped into the
            original image shape and placed in a list.

        Returns
        -------
        imgs: list
            List of torch.Tensor images, each with ``torch.Size(img_height, im_width)``.
        """

        imgs = [vecs[:, i].reshape(self.base_signal.shape).squeeze() for i in range(vecs.shape[1])]

        return imgs

    @staticmethod
    def _color_to_grayscale(img):
        """Takes weighted sum of RGB channels to return a grayscale image"""

        gray = torch.einsum('xy,...abc->...xbc', torch.tensor([.2989, .587, .114]).unsqueeze(0), img)
        return gray

    def display(self, alpha=5., beta=10., **kwargs):
        r""" Displays the first and last synthesized eigendistortions alone, and added to the image.

        If image or eigendistortions have 3 channels, then it is assumed to be a color image and it is converted to
        grayscale. This is merely for display convenience and may change in the future.

        Parameters
        ----------
        alpha: float, optional
            Amount by which to scale eigendistortion for image + (alpha * eigendistortion) for display.
        beta: float, optional
            Amount by which to scale eigendistortion to be displayed alone.
        kwargs:
            Additional arguments for :meth:`pt.imshow()`.
        """

        assert len(self.distortions['eigenvectors'])>1, "Assumes at least two eigendistortions were synthesized."

        if self.is_color_image:
            print('Collapsing color image to grayscale for display')
            image = self.base_signal.mean(dim=1).squeeze()
            max_dist = self._color_to_grayscale(self.distortions['eigenvectors'][0]).squeeze()
            min_dist = self._color_to_grayscale(self.distortions['eigenvectors'][-1]).squeeze()
        else:
            image = self.base_signal.squeeze()
            max_dist = self.distortions['eigenvectors'][0]
            min_dist = self.distortions['eigenvectors'][-1]

        def clamp(img):
            return torch.clamp(img, 0, 1)

        pt.imshow([clamp(image), clamp(image + alpha * max_dist), beta * max_dist],
                  title=['original', f'original + {alpha:.0f} * maxdist', f'{beta:.0f} * maxdist'], **kwargs)

        pt.imshow([clamp(image), clamp(image + alpha * min_dist), beta * min_dist],
                  title=['original', f'original + {alpha:.0f} * mindist', f'{beta:.0f} * mindist'], **kwargs);

    def compute_jacobian(self):
        """Calls autodiff.jacobian and returns jacobian. Will throw error if input too big."""
        if self.jacobian is None:
            J = jacobian(self.representation_flat, self.input_flat)
            self.jacobian = J
        else:
            print("Jacobian already computed, returning self.jacobian")
            J = self.jacobian

        return J

    def _solve_eigenproblem(self):
        r""" Eigendecomposition of explicitly computed FIM.
        To be used when the input is small (e.g. less than 70x70 image on cluster or 30x30 on your own machine). This
        method obviates the power iteration and its related algorithms (e.g. Lanczos).
        """
        J = self.compute_jacobian()
        F = J.T @ J
        eig_vals, eig_vecs = torch.symeig(F, eigenvectors=True)

        return eig_vals.flip(dims=(0,)), eig_vecs.flip(dims=(1,))

    def _implicit_power_method(self, l=0, init='randn', seed=0, tol=1e-10, n_steps=1000, verbose=False,
                               print_every=1):
        r""" Use power method to obtain largest (smallest) eigenvalue/vector pair.
        Apply the power method algorithm to approximate the extremal eigenvalue and eigenvector of the Fisher Information
        Matrix, without explicitly representing that matrix.

        Parameters
        ----------
        l: torch.Tensor, optional
            Optional argument. When l=0, this function estimates the leading eval evec pair. When l is set to the
            estimated maximum eigenvalue, this function will estimate the smallest eval evec pair (minor component).
        init: {'randn', 'ones'}
            Starting vector for the power iteration. 'randn' is random normal noise vector, 'ones' is a ones vector. Both
            will be normalized.
        seed: float, optional
            Manual seed
        tol: float, optional
            Tolerance value
        n_steps: int, optional
            Maximum number of steps
        verbose: bool, optional
            Flag for printing eigenval at this iter minus eigenval at last iter. Use this as a proxy for convergence --
            the closer to zero, the less the eigenvalue is changing upon each iteration, and the closer you are getting
            to the true eigendistortion.
        print_every: int
            Determines nth step to display convergence info. 1 (default) means it will print a message on every step.

        Returns
        -------
        lmbda: float
            Eigenvalue corresponding to final vector of power iteration.
        v: torch.Tensor
            Final eigenvector of power iteration procedure.
        """
        x, y = self.input_flat, self.representation_flat

        n = x.shape[0]

        np.random.seed(seed)
        torch.manual_seed(seed)

        if init not in ['randn', 'ones']:
            raise ValueError("init should be in ['randn', 'ones']")

        if init == 'randn':
            v = torch.randn_like(x)
        elif init == 'ones':
            v = torch.ones_like(x)

        v = v / v.norm()

        Fv = fisher_info_matrix_vector_product(y, x, v)
        v = Fv / torch.norm(Fv)
        lmbda = implicit_FIM_eigenvalue(y, x, v)
        i = 0
        error = torch.tensor(1)
        lmbda_new, v_new = None, None
        while i < n_steps and error > tol:
            Fv = fisher_info_matrix_vector_product(y, x, v)
            Fv = Fv - l * v  # minor component
            v_new = Fv / torch.norm(Fv)

            lmbda_new = implicit_FIM_eigenvalue(y, x, v_new)

            error = torch.sqrt((lmbda - lmbda_new) ** 2)
            if verbose and i >= 0 and i % print_every == 0:
                print(f"{i:3d} -- delta_eigenval: {error.item():04.4f}")

            v = v_new
            lmbda = lmbda_new
            i += 1

        return lmbda_new, v_new

    def _lanczos(self, n_steps=1000, e_vecs=None, verbose=True, print_every=1, debug_A=None):
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
        n_steps: int
            number of power iteration steps (i.e. eigenvalues/eigenvectors to compute and/or return). Recommended to do many
            more than amount of requested eigenvalue/eigenvectors, N. Some say 2N steps but as many as possible is preferred.
        e_vecs: int or iterable, optional
           Eigenvectors to return. If num_evecs is an int, it will return all eigenvectors in range(num_evecs).
           If an iterable, then it will return all vectors selected.
        verbose: bool, optional
            Print progress to screen.
        print_every: int
            Determines nth step to display convergence info. 1 (default) means it will print a message on every step.
        debug_A: torch.Tensor
            For debugging purposes. Explicit FIM for matrix vector multiplication. Bypasses matrix-vector product with
            implicitly stored FIM of model.

        Returns
        -------
        eig_vals: torch.Tensor
            Tensor of eigenvalues, sorted in descending order. torch.Size(n_steps,) if e_vecs is None.
            torch.Size(len(e_vecs),) if e_vecs is not None. eigenvalues corresponding to the eigenvectors at that index.
        eig_vecs: toch.Tensor
            Tensor of n_steps eigenvectors with torch.Size(n, len(e_vecs)) . If ``return_evecs=False``,
            then returned  eig_vecs is an empty tensor.
        eig_vecs_ind: torch.Tensor
            Indices of each returned eigenvector

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

        warnings.warn("Lanczos algo is currently experimental. It may be numerically unstable and give inaccurate results.")
        x, y = self.input_flat, self.representation_flat

        n = x.shape[0]
        dtype = x.dtype
        device = x.device

        if e_vecs is None:
            e_vecs = []

        if len(e_vecs) > n_steps:
            raise Exception("Lanczos method requires at least n_steps=len(e_vecs) (but should preferably be much more).")

        if n_steps > n:
            warnings.warn("Dim of Fisher matrix, n, is < n_steps. Setting n_steps = n")
            n_steps = n
        if n_steps < 2*len(e_vecs):
            warnings.warn("n_steps should be at least 2*len(e_vecs) but preferably even more for accuracy.", RuntimeWarning)

        # T tridiagonal matrix, V orthogonalized Krylov vectors
        T = torch.zeros((n_steps, n_steps), device=device, dtype=dtype)
        Q = torch.zeros((n, n_steps), device=device, dtype=dtype)

        q0 = torch.zeros(n, device=device, dtype=dtype)
        q = torch.randn(n, device=device, dtype=dtype)
        q /= torch.norm(q)
        beta = torch.zeros(1, device=device, dtype=dtype)

        for i in range(n_steps):
            if verbose and i % print_every == 0:
                print(f'Step {i+1:d}/{n_steps:d}')

            # v = Aq where A is implicitly stored FIM operator
            if debug_A is None:
                v = fisher_info_matrix_vector_product(y, x, q.view(n, 1)).view(n)
            else:
                v = torch.mv(debug_A, q)

            alpha = q.dot(v)  # alpha = q'Aq

            if i > 0:  # orthogonalize using Gram-Schmidt TWICE to ensure orthogonality
                v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))
                v -= Q[:, :i+1].mv(Q[:, :i + 1].t().mv(v))

            beta = torch.norm(v)
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
