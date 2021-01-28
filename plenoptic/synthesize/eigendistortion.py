import torch
from torch import Tensor
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
import numpy as np
import pyrtools as pt
import warnings
from .synthesis import Synthesis
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Tuple


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


def fisher_info_matrix_eigenvalue(y, x, v):
    r"""Implicitly compute the eigenvalue of the Fisher Information Matrix corresponding to eigenvector v
    :math:`\lambda= v^T F v`
    """
    Fv = fisher_info_matrix_vector_product(y, x, v)
    lmbda = Fv.T @ v
    if v.shape[1] > 1:
        lmbda = torch.diag(lmbda)

    return lmbda


class Eigendistortion(Synthesis):
    r"""Synthesis object to compute eigendistortions induced by a model on a given input image.

    Attributes
    ----------
    batch_size: int
    n_channels: int
    im_height: int
    im_width: int
    jacobian: torch.Tensor
        Is only set when :func:`synthesize` is run with ``method='exact'``. Default to ``None``.
    synthesized_signal: torch.Tensor
        Tensor of eigendistortions (eigenvectors of Fisher matrix) with Size((n_distortions, n_channels, h, w)).
    synthesized_eigenvalues: torch.Tensor
        Tensor of eigenvalues corresponding to each eigendistortion, listed in decreasing order.
    synthesized_eigenindex: listlike
        Index of each eigenvector/eigenvalue.

    Parameters
    -----------
    signal: torch.Tensor
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
    """

    def __init__(self, base_signal, model):
        assert len(base_signal.shape) == 4, "Input must be torch.Size([batch=1, n_channels, im_height, im_width])"

        self.batch_size, self.n_channels, self.im_height, self.im_width = base_signal.shape
        assert self.batch_size == 1, "Batch dim must be 1. Image batch synthesis is not available."

        # flatten and attach gradient and reshape to image
        self._input_flat = base_signal.flatten().unsqueeze(1).requires_grad_(True)

        super().__init__(self._input_flat.view(*base_signal.shape), model, loss_function=None)

        if len(self.base_representation) > 1:
            self._representation_flat = torch.cat([s.squeeze().view(-1) for s in self.base_representation]).unsqueeze(1)
        else:
            self._representation_flat = self.base_representation.squeeze().view(-1).unsqueeze(1)

        print(f"\n Initializing Eigendistortion -- "
              f"Input dim: {len(self._input_flat.squeeze())} | Output dim: {len(self._representation_flat.squeeze())}")

        self.jacobian = None
        self.synthesized_eigenvalues = None
        self.synthesized_eigenindex = None

        self._all_losses = []
        self._all_saved_signals = []
        self.saved_representation = None

    @staticmethod
    def _hidden_attrs():
        """Inherited attributes from parent `Synthesis` class that aren't necessary for `Eigendistortion`."""
        hidden_attrs = ['coarse_to_fine', 'gradient', 'loss_function', 'learning_rate',
                        'objective_function', 'plot_representation_error',
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
    def load(file_path, model_constructor=None, map_location='cpu', **state_dict_kwargs):
        # TODO attribute name *****
        super().load(file_path, 'model', model_constructor, map_location, **state_dict_kwargs)

    def save(self, file_path, save_model_reduced=False, attrs=['model'], model_attr_names=['model']):
        # TODO
        super().save(file_path, save_model_reduced, attrs, model_attr_names)

    def to(self, *args, attrs=[], **kwargs):
        """Send attrs to specified device. See docstring of Synthesis.to()"""
        super().to(*args, attrs, **kwargs)

    def synthesize(self, k=1, method='power', tol=1e-8, max_steps=1000, seed=0, store_progress=False):
        r"""Compute eigendistortions of Fisher Information Matrix with given input image.

        Parameters
        ----------
        method: {'exact', 'power', 'svd'}, optional
            Eigensolver method. Jacobian tries to do eigendecomposition directly (
            not recommended for very large matrices). 'power' (default) uses the power method to compute first and
            last eigendistortions, with maximum number of iterations dictated by n_steps. 'svd' uses randomized SVD
            to approximate the top k eigendistortions and their corresponding eigenvalues.
        k: int
            How many vectors to return using block power method or svd.
        tol: float, optional
            Tolerance for error criterion in power iteration.
        max_steps: int, optional
            Total steps to run for ``method='power'`` in eigenvalue computation.
        seed: int, optional
            Control the random seed for reproducibility.
        store_progress: bool
            Store loss after each iteration. Change in approximated eigenvalue is used as a proxy for loss and a
            measure of convergence. Can be displayed with `plot_loss()`.

        Returns
        -------
        eigendistortions: Union([torch.Tensor, List])
            Eigenvectors of the Fisher Information Matrix, ordered by eigenvalue. For Lanczos method, it is possible
            to not return any eigendistortions, in which case an empty list will be returned. This tensor points to
            the `synthesized_signal` attribute of the object. If not an empty list, then Size((num_distortions,
            num_channels, img_height, img_width)).
        eigenvalues: Tensor
            Eigenvalues corresponding to each eigendistortion, listed in decreasing order. This tensor points to the
            `synthesized_eigenvalue` attribute of the object.
        eigen_index: Listlike
            Index of each eigendistortion/eigenvalue. This points to the `synthesized_eigenindex` attribute of the
            object.
        """
        self._set_seed(seed)
        self.store_progress = store_progress

        assert method in ['power', 'exact', 'svd'], "method must be in {'power', 'exact', 'svd'}"

        if method == 'exact' and self._representation_flat.size(0) * self._input_flat.size(0) > 1e6:
            warnings.warn("Jacobian > 1e6 elements and may cause out-of-memory. Use method =  {'power', 'lanczos'}.")

        if method == 'exact':  # compute exact Jacobian
            eig_vals, eig_vecs = self._synthesize_exact()
            eig_vecs = self._vector_to_image(eig_vecs.detach())
            eig_vecs_ind = torch.arange(len(eig_vecs))

        elif method == 'power':

            lmbda_max, v_max = self._synthesize_power(k=k, l=0., tol=tol, max_steps=max_steps)
            lmbda_min, v_min = self._synthesize_power(k=k, l=lmbda_max[0], tol=tol, max_steps=max_steps)
            n = v_max.shape[0]

            eig_vecs = self._vector_to_image(torch.cat((v_max, v_min), dim=1).detach())
            eig_vals = torch.cat([lmbda_max, lmbda_min]).squeeze()
            eig_vecs_ind = torch.cat((torch.arange(k), torch.arange(n-k, n)))

        elif method == 'svd':

            lmbda_new, v_new, error_approx = self._synthesize_randomized_svd(k=k, p=5, q=2)
            eig_vecs = self._vector_to_image(v_new.detach())
            eig_vals = lmbda_new.squeeze()
            eig_vecs_ind = torch.arange(k)
            print(f'FIM range approximation error: {error_approx:.2f}')

        # reshape to (n x num_chans x h x w)
        self.synthesized_signal = torch.stack(eig_vecs, 0) if len(eig_vecs) != 0 else []

        self.synthesized_eigenvalues = eig_vals.detach()
        self.synthesized_eigenindex = eig_vecs_ind

        # return self.distortions
        return self.synthesized_signal, self.synthesized_eigenvalues, self.synthesized_eigenindex

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

        imgs = [vecs[:, i].reshape((self.n_channels, self.im_height, self.im_width)) for i in range(vecs.shape[1])]
        return imgs

    def _check_for_stabilization(self, i):
        r"""Check whether the loss has stabilized and, if so, return True, else return False.

        Parameters
        ----------
        i : int
            the current iteration (0-indexed)
        """
        if len(self.loss) > self.loss_change_iter:
            if abs(self.loss[-self.loss_change_iter] - self.loss[-1]) < self.loss_thresh:
                return True
        return False

    def compute_jacobian(self):
        r"""Calls autodiff.jacobian and returns jacobian. Will throw error if input too big.

        Returns
        -------
        J: torch.Tensor
            Jacobian of representation wrt input.
        """
        if self.jacobian is None:
            J = jacobian(self._representation_flat, self._input_flat)
            self.jacobian = J
        else:
            print("Jacobian already computed, returning self.jacobian")
            J = self.jacobian

        return J

    def _synthesize_exact(self):
        r""" Eigendecomposition of explicitly computed Fisher Information Matrix.
        To be used when the input is small (e.g. less than 70x70 image on cluster or 30x30 on your own machine). This
        method obviates the power iteration and its related algorithms (e.g. Lanczos). This method computes the
        Fisher Information Matrix by explicitly computing the Jacobian of the representation wrt the input.

        Returns
        -------
        eig_vals: torch.Tensor
            Eigenvalues in decreasing order.
        eig_vecs: torch.Tensor
            Eigenvectors in 2D tensor, whose cols are eigenvectors (i.e. eigendistortions) corresponding to eigenvalues.
        """

        J = self.compute_jacobian()
        F = J.T @ J
        eig_vals, eig_vecs = torch.symeig(F, eigenvectors=True)
        return eig_vals.flip(dims=(0,)), eig_vecs.flip(dims=(1,))

    def _clamp_and_store(self, idx, v, d_lambda):
        """Overwrite base class _clamp_and_store. We don't actually need to clamp the signal."""
        if self.store_progress:
            v = self._vector_to_image(v)[0]
            # self._all_saved_signals[idx].append(v)
            self._all_losses[idx].append(d_lambda.item())

    def _synthesize_power(self, k=1, l=0, tol=1e-10, max_steps=1000):
        r""" Use power method to obtain largest (smallest) eigenvalue/vector pair.
        Apply the power method algorithm to approximate the extremal eigenvalue and eigenvector of the Fisher
        Information Matrix, without explicitly representing that matrix.

        Parameters
        ----------
        l: Union([float, torch.Tensor]), optional
            Optional argument. When l=0, this function estimates the leading eval evec pair. When l is set to the
            estimated maximum eigenvalue, this function will estimate the smallest eval evec pair (minor component).
        tol: float, optional
            Tolerance value
        max_steps: int, optional
            Maximum number of steps

        Returns
        -------
        lmbda: float
            Eigenvalue corresponding to final vector of power iteration.
        v: torch.Tensor
            Final eigenvector (i.e. eigendistortion) of power iteration procedure.

        References
        ----------
        [1] Algorithm 8.2.8 Golub and Van Loan, Matrix Computations, 3rd Ed.
        """

        idx = len(self._all_losses)
        if self.store_progress:
            self._all_losses.append([])
            # self._all_saved_signals.append([])

        x, y = self._input_flat, self._representation_flat

        v = torch.randn(len(x), k).to(x.device)
        v = v / v.norm()

        Fv = fisher_info_matrix_vector_product(y, x, v)
        v = Fv / torch.norm(Fv)
        lmbda = fisher_info_matrix_eigenvalue(y, x, v)

        d_lambda = torch.tensor(1)
        lmbda_new, v_new = None, None
        pbar = tqdm(range(max_steps))
        postfix_dict = {'step': None, 'delta_eigenval': None}
        for i in pbar:
            postfix_dict.update(dict(step=f"{i+1:d}/{max_steps:d}", delta_eigenval=f"{d_lambda.item():04.4f}"))
            pbar.set_postfix(**postfix_dict)

            if d_lambda <= tol:
                print(f"Tolerance {tol:.2E} reached. Stopping early.")
                break

            Fv = fisher_info_matrix_vector_product(y, x, v)
            Fv = Fv - l * v  # minor component

            v_new, _ = torch.qr(Fv)

            lmbda_new = fisher_info_matrix_eigenvalue(y, x, v_new)

            d_lambda = torch.sqrt((lmbda.sum() - lmbda_new.sum()) ** 2)
            v = v_new
            lmbda = lmbda_new

            self._clamp_and_store(idx, v.clone(), d_lambda.clone())

        pbar.close()

        return lmbda_new, v_new

    def _synthesize_randomized_svd(self, k, p=0, q=0):
        r"""  Synthesize eigendistortions using randomized truncated SVD.
        Parameters
        ----------
        k: int
            Rank of factorization to be returned.
        p: int, Optional
            Oversampling parameter.
        q: int, Optional
            Matrix power iteration. Used to squeeze the eigen spectrum for more accurate approximation.

        Returns
        -------
        S: torch.Tensor
            Eigenvalues, Size((n, )).
        V: torch.Tensor
            Eigendistortions, Size((n, k)).

        References
        -----
        [1] Halko, Martinsson, Tropp, Finding structure with randomness: Probabilistic algorithms for constructing
        approximate matrix decompositions, SIAM Rev. 53:2, pp. 217-288 https://arxiv.org/abs/0909.4061 (2011)
        """

        x, y = self._input_flat, self._representation_flat
        n = len(x)

        P = torch.randn(n, k + p).to(x.device)
        P, _ = torch.qr(P)  # numerical stability
        Z = fisher_info_matrix_vector_product(y, x, P)

        for _ in range(q):  # optional power iteration to squeeze the spectrum for more accurate estimate
            Z = fisher_info_matrix_vector_product(y, x, Z)

        Q, _ = torch.qr(Z)
        B = Q.T @ fisher_info_matrix_vector_product(y, x, Q)  # B = Q.T @ A @ Q
        _, S, V = torch.svd(B, some=True)  # eigendecomp of small matrix
        V = Q @ V  # lift up to original dimensionality

        # estimate error in Q estimate of range space
        omega = fisher_info_matrix_vector_product(y, x, torch.randn(n, 20).to(x.device))
        error_approx = omega - (Q @ Q.T @ omega)
        error_approx = error_approx.norm(dim=0).mean()

        return S[:k], V[:, :k], error_approx  # truncate

    def _indexer(self, idx):
        """Maps eigenindex to arg index (0-indexed)"""
        if idx == 0 or idx == -1:
            return idx
        else:
            all_idx = self.synthesized_eigenindex
            assert idx in all_idx, "eigenindex must be the index of one of the vectors"
            assert all_idx is not None and len(all_idx) != 0, "No eigendistortions have been synthesized."
            return int(np.where(all_idx==idx))

    def plot_loss(self, eigenindex, iteration=None, figsize=(5, 5), ax=None,  **kwargs):
        """Wraps plot_loss of base class. Plots change in eigenvalue after each iteration."""
        idx = self._indexer(eigenindex)
        self.loss = self._all_losses[idx]
        # call super plot_loss
        title = f"Change in eigenval {eigenindex:d}"
        fig = super().plot_loss(iteration, figsize, ax, title, **kwargs)
        plt.show()
        return fig

    @staticmethod
    def _color_to_grayscale(img):
        """Takes weighted sum of RGB channels to return a grayscale image"""
        gray = torch.einsum('xy,...abc->...xbc', torch.tensor([.2989, .587, .114]).unsqueeze(0), img)
        return gray

    def display_first_and_last(self, alpha=5., beta=10., **kwargs):
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

        assert len(self.synthesized_signal) > 1, "Assumes at least two eigendistortions were synthesized."

        # reshape so channel dim is last
        im_shape = self.n_channels, self.im_height, self.im_width
        image = self.base_signal.detach().view(im_shape).permute((1, 2, 0)).squeeze()
        max_dist = self.synthesized_signal[0].permute((1, 2, 0)).squeeze()
        min_dist = self.synthesized_signal[-1].permute((1, 2, 0)).squeeze()

        def _preprocess(img):
            x = torch.clamp(img, 0, 1)
            return x.numpy()

        fig_max = pt.imshow([_preprocess(image), _preprocess(image + alpha * max_dist), alpha * max_dist.numpy()],
                  title=['original', f'original + {alpha:.0f} * maxdist', f'{alpha:.0f} * maxdist'], **kwargs);

        fig_min = pt.imshow([_preprocess(image), _preprocess(image + beta * min_dist), beta * min_dist.numpy()],
                  title=['original', f'original + {beta:.0f} * mindist', f'{beta:.0f} * mindist'], **kwargs);

        return fig_max, fig_min

    def plot_synthesized_image(self, eigenindex, add_base_image=True, scale=1., channel_idx=0, iteration=None,
                               title=None, figsize=(5, 5), ax=None, imshow_zoom=None, vrange=(0, 1)):
        """Wraps Synthesis.plot_synthesized_image.
        Parameters
        ---------
        eigenindex: int
            Index of eigendistortion to display. Must be an element of `synthesized_eigenindex` attribute of object,
            or 0 for first, and -1 for last. This is converted to an index with which to index the batch dim of tensor.
        """

        batch_idx = self._indexer(eigenindex)
        fig = super().plot_synthesized_image(batch_idx, channel_idx, iteration, title, figsize, ax, imshow_zoom, vrange)
        return fig
