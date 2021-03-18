import torch
from torch import Tensor
from .autodiff import jacobian, vector_jacobian_product, jacobian_vector_product
import numpy as np
import warnings
from tqdm import tqdm
from ..tools.display import imshow
from typing import Tuple, List, Callable, Union
import matplotlib.pyplot
from matplotlib.figure import Figure


def fisher_info_matrix_vector_product(y: Tensor, x: Tensor, v: Tensor, dummy_vec: Tensor) -> Tensor:
    r"""Compute Fisher Information Matrix Vector Product: :math:`Fv`

    Parameters
    ----------
    y: Tensor
        Output tensor with gradient attached
    x: Tensor
        Input tensor with gradient attached
    v: Tensor
        The vectors with which to compute Fisher vector products
    dummy_vec: Tensor
        Dummy vector for Jacobian vector product trick

    Returns
    -------
    Fv: Tensor
        Vector, Fisher vector product

    Notes
    -----
    Under white Gaussian noise assumption, :math:`F` is matrix multiplication of Jacobian transpose and Jacobian:
    :math:`F = J^T J`. Hence:
    :math:`Fv = J^T (Jv)`
    """
    Jv = jacobian_vector_product(y, x, v, dummy_vec)
    Fv = vector_jacobian_product(y, x, Jv, detach=True)

    return Fv


def fisher_info_matrix_eigenvalue(y: Tensor, x: Tensor, v: Tensor, dummy_vec: Tensor = None) -> Tensor:
    r"""Compute the eigenvalues of the Fisher Information Matrix corresponding to eigenvectors in v
    :math:`\lambda= v^T F v`
    """
    if dummy_vec is None:
        dummy_vec = torch.ones_like(y, requires_grad=True)

    Fv = fisher_info_matrix_vector_product(y, x, v, dummy_vec)

    # compute eigenvalues for all vectors in v
    lambduh = torch.stack([a.dot(b) for a, b in zip(v.T, Fv.T)])
    return lambduh


class Eigendistortion:
    r"""Synthesis object to compute eigendistortions induced by a model on a given input image.

    Attributes
    ----------
    batch_size: int
    n_channels: int
    im_height: int
    im_width: int
    jacobian: Tensor
        Is only set when :func:`synthesize` is run with ``method='exact'``. Default to ``None``.
    synthesized_signal: Tensor
        Tensor of eigendistortions (eigenvectors of Fisher matrix) with Size((n_distortions, n_channels, h, w)).
    synthesized_eigenvalues: Tensor
        Tensor of eigenvalues corresponding to each eigendistortion, listed in decreasing order.
    synthesized_eigenindex: listlike
        Index of each eigenvector/eigenvalue.

    Parameters
    -----------
    base_signal: Tensor
        image, torch.Size(batch=1, channel, height, width). We currently do not support batches of images,
        as each image requires its own optimization.
    model: torch class
        torch model with defined forward and backward operations
    Notes
    -----
    This is a method for comparing image representations in terms of their ability to explain perceptual sensitivity
    in humans. It estimates eigenvectors of the FIM. A model, :math:`y = f(x)`, is a deterministic (and differentiable)
    mapping from the input pixels :math:`x \in \mathbb{R}^n` to a mean output response vector :math:`y\in \mathbb{
    R}^m`, where we assume additive white Gaussian noise in the response space.
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

    def __init__(self, base_signal: Tensor, model: torch.nn.Module):
        assert len(base_signal.shape) == 4, "Input must be torch.Size([batch=1, n_channels, im_height, im_width])"
        assert base_signal.shape[0] == 1, "Batch dim must be 1. Image batch synthesis is not available."

        self.batch_size, self.n_channels, self.im_height, self.im_width = base_signal.shape

        self.model = model
        # flatten and attach gradient and reshape to image
        self._input_flat = base_signal.flatten().unsqueeze(1).requires_grad_(True)

        self.base_signal = self._input_flat.view(*base_signal.shape)
        self.base_representation = self.model(self.base_signal)

        if len(self.base_representation) > 1:
            self._representation_flat = torch.cat([s.squeeze().view(-1) for s in self.base_representation]).unsqueeze(1)
        else:
            self._representation_flat = self.base_representation.squeeze().view(-1).unsqueeze(1)

        print(f"\nInitializing Eigendistortion -- "
              f"Input dim: {len(self._input_flat.squeeze())} | Output dim: {len(self._representation_flat.squeeze())}")

        self.jacobian = None

        self.synthesized_signal = None  # eigendistortion
        self.synthesized_eigenvalues = None
        self.synthesized_eigenindex = None

    @classmethod
    def load(file_path, model_constructor=None, map_location='cpu', **state_dict_kwargs):
        # TODO attribute name *****
        raise NotImplementedError

    def save(self, file_path, save_model_reduced=False, attrs=['model'], model_attr_names=['model']):
        # TODO
        raise NotImplementedError

    def to(self, *args, attrs=[], **kwargs):
        """Send attrs to specified device. See docstring of Synthesis.to()"""
        # TODO
        raise NotImplementedError

    def synthesize(self,
                   method: str = 'power',
                   k: int = 1,
                   max_steps: int = 1000,
                   p: int = 5,
                   q: int = 2,
                   tol: float = 1e-8,
                   seed: int = None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Compute eigendistortions of Fisher Information Matrix with given input image.

        Parameters
        ----------
        method: {'exact', 'power', 'randomized_svd'}, optional
            Eigensolver method. Jacobian tries to do eigendecomposition directly (
            not recommended for very large inputs). 'power' (default) uses the power method to compute first and
            last eigendistortions, with maximum number of iterations dictated by n_steps. 'randomized_svd' uses
            randomized SVD to approximate the top k eigendistortions and their corresponding eigenvalues.
        k: int
            How many vectors to return using block power method or svd.
        max_steps: int, optional
            Maximum number of steps to run for ``method='power'`` in eigenvalue computation N/A for 'randomized_svd'.
        p: int, optional
            Oversampling parameter for randomized SVD. k+p vectors will be sampled, and k will be returned. See
            docstring of ``_synthesize_randomized_svd`` for more details including algorithm reference.
        q: int, optional
            Matrix power parameter for randomized SVD. This is an effective trick for the algorithm to converge to
            the correct eigenvectors when the eigenspectrum does not decay quickly. See
            ``_synthesize_randomized_svd`` for more details including algorithm reference.
        tol: float, optional
            Tolerance for error criterion in power iteration.
        seed: int, optional
            Control the random seed for reproducibility. Defaults to ``None``, with no seed being set.

        Returns
        -------
        eigendistortions: Tensor
            Eigenvectors of the Fisher Information Matrix, ordered by eigenvalue. This tensor points to
            the `synthesized_signal` attribute of the object. Tensor has Size((num_distortions,
            num_channels, img_height, img_width)).
        eigenvalues: Tensor
            Eigenvalues corresponding to each eigendistortion, listed in decreasing order. This tensor points to the
            `synthesized_eigenvalue` attribute of the object.
        eigen_index: Tensor
            Index of each eigendistortion/eigenvalue. This points to the `synthesized_eigenindex` attribute of the
            object.
        """
        if seed is not None:
            assert isinstance(seed, int), "random seed must be integer"
            torch.manual_seed(seed)

        assert method in ['power', 'exact', 'randomized_svd'], "method must be in {'power', 'exact', 'randomized_svd'}"

        if method == 'exact' and self._representation_flat.size(0) * self._input_flat.size(0) > 1e6:
            warnings.warn("Jacobian > 1e6 elements and may cause out-of-memory. Use method =  {'power', 'randomized_svd'}.")

        if method == 'exact':  # compute exact Jacobian
            print(f"Computing all eigendistortions")
            eig_vals, eig_vecs = self._synthesize_exact()
            eig_vecs = self._vector_to_image(eig_vecs.detach())
            eig_vecs_ind = torch.arange(len(eig_vecs))

        elif method == 'randomized_svd':
            print(f"Estimating top k={k} eigendistortions using randomized SVD")
            lmbda_new, v_new, error_approx = self._synthesize_randomized_svd(k=k, p=p, q=q)
            eig_vecs = self._vector_to_image(v_new.detach())
            eig_vals = lmbda_new.squeeze()
            eig_vecs_ind = torch.arange(k)

            # display the approximate estimation error of the range space
            print(f'Randomized SVD complete! Estimated spectral approximation error = {error_approx:.2f}')

        else:  # method == 'power'

            assert max_steps > 0, "max_steps must be greater than zero"

            lmbda_max, v_max = self._synthesize_power(k=k, shift=0., tol=tol, max_steps=max_steps)
            lmbda_min, v_min = self._synthesize_power(k=k, shift=lmbda_max[0], tol=tol, max_steps=max_steps)
            n = v_max.shape[0]

            eig_vecs = self._vector_to_image(torch.cat((v_max, v_min), dim=1).detach())
            eig_vals = torch.cat([lmbda_max, lmbda_min]).squeeze()
            eig_vecs_ind = torch.cat((torch.arange(k), torch.arange(n - k, n)))

        # reshape to (n x num_chans x h x w)
        self.synthesized_signal = torch.stack(eig_vecs, 0) if len(eig_vecs) != 0 else []

        self.synthesized_eigenvalues = eig_vals.detach()
        self.synthesized_eigenindex = eig_vecs_ind

        return self.synthesized_signal, self.synthesized_eigenvalues, self.synthesized_eigenindex

    def _vector_to_image(self, vecs: Tensor) -> List[Tensor]:
        r""" Reshapes eigenvectors back into correct image dimensions.

        Parameters
        ----------
        vecs: Tensor
            Eigendistortion tensor with ``torch.Size([N, num_distortions])``. Each distortion will be reshaped into the
            original image shape and placed in a list.

        Returns
        -------
        imgs: List
            List of Tensor images, each with ``torch.Size(img_height, im_width)``.
        """

        imgs = [vecs[:, i].reshape((self.n_channels, self.im_height, self.im_width)) for i in range(vecs.shape[1])]
        return imgs

    def compute_jacobian(self) -> Tensor:
        r"""Calls autodiff.jacobian and returns jacobian. Will throw error if input too big.

        Returns
        -------
        J: Tensor
            Jacobian of representation wrt input.
        """
        if self.jacobian is None:
            J = jacobian(self._representation_flat, self._input_flat)
            self.jacobian = J
        else:
            print("Jacobian already computed, returning self.jacobian")
            J = self.jacobian

        return J

    def _synthesize_exact(self) -> Tuple[Tensor, Tensor]:
        r""" Eigendecomposition of explicitly computed Fisher Information Matrix.
        To be used when the input is small (e.g. less than 70x70 image on cluster or 30x30 on your own machine). This
        method obviates the power iteration and its related algorithms (e.g. Lanczos). This method computes the
        Fisher Information Matrix by explicitly computing the Jacobian of the representation wrt the input.

        Returns
        -------
        eig_vals: Tensor
            Eigenvalues in decreasing order.
        eig_vecs: Tensor
            Eigenvectors in 2D tensor, whose cols are eigenvectors (i.e. eigendistortions) corresponding to eigenvalues.
        """

        J = self.compute_jacobian()
        F = J.T @ J
        eig_vals, eig_vecs = torch.symeig(F, eigenvectors=True)
        return eig_vals.flip(dims=(0,)), eig_vecs.flip(dims=(1,))

    def _synthesize_power(self,
                          k: int,
                          shift: Union[Tensor, float],
                          tol: float,
                          max_steps: int) -> Tuple[Tensor, Tensor]:
        r""" Use power method (or orthogonal iteration when k>1) to obtain largest (smallest) eigenvalue/vector pairs.
        Apply the algorithm to approximate the extremal eigenvalues and eigenvectors of the Fisher
        Information Matrix, without explicitly representing that matrix.

        This method repeatedly calls ``fisher_info_matrix_vector_product()`` with a single (when `k=1`), or multiple
        (when `k>1`) vectors.

        Parameters
        ----------
        k: int
            Number of top and bottom eigendistortions to synthesize; i.e. if k=2, then the top 2 and bottom 2 will
            be returned. When `k>1`, multiple eigendistortions are synthesized, and each power iteration step is
            followed by a QR orthogonalization step to ensure the vectors are orthonormal.
        shift: Union([float, Tensor])
            When `shift=0`, this function estimates the top `k` eigenvalue/vector pairs. When `shift` is set to the
            estimated top eigenvalue this function will estimate the smallest eigenval/eigenvector pairs.
        tol: float
            Tolerance value
        max_steps: int
            Maximum number of steps

        Returns
        -------
        lmbda: Tensor
            Eigenvalue corresponding to final vector of power iteration.
        v: Tensor
            Final eigenvector(s) (i.e. eigendistortions) of power (orthogonal) iteration procedure.

        References
        ----------
        [1] Orthogonal iteration; Algorithm 8.2.8 Golub and Van Loan, Matrix Computations, 3rd Ed.
        """

        x, y = self._input_flat, self._representation_flat

        # note: v is an n x k matrix where k is number of eigendists to be synthesized!
        v = torch.randn(len(x), k).to(x.device)
        v = v / v.norm()

        _dummy_vec = torch.ones_like(y, requires_grad=True)  # cache a dummy vec for jvp
        Fv = fisher_info_matrix_vector_product(y, x, v, _dummy_vec)
        v = Fv / torch.norm(Fv)
        lmbda = fisher_info_matrix_eigenvalue(y, x, v, _dummy_vec)

        d_lambda = torch.tensor(float('inf'))
        lmbda_new, v_new = None, None
        pbar = tqdm(range(max_steps), desc=("Top" if shift == 0 else "Bottom") + f" k={k} eigendists")
        postfix_dict = {'delta_eigenval': None}

        for _ in pbar:
            postfix_dict.update(dict(delta_eigenval=f"{d_lambda.item():.4E}"))
            pbar.set_postfix(**postfix_dict)

            if d_lambda <= tol:
                print(("Top" if shift == 0 else "Bottom")
                      + f" k={k} eigendists computed" + f" | Tolerance {tol:.2E} reached.")
                break

            Fv = fisher_info_matrix_vector_product(y, x, v, _dummy_vec)
            Fv = Fv - shift * v  # minor component

            v_new = torch.qr(Fv)[0] if k > 1 else Fv

            lmbda_new = fisher_info_matrix_eigenvalue(y, x, v_new, _dummy_vec)

            d_lambda = (lmbda.sum() - lmbda_new.sum()).norm()  # stability of eigenspace
            v = v_new
            lmbda = lmbda_new

        pbar.close()

        return lmbda_new, v_new

    def _synthesize_randomized_svd(self, k: int, p: int, q: int) -> Tuple[Tensor, Tensor, Tensor]:
        r"""  Synthesize eigendistortions using randomized truncated SVD.
        This method approximates the column space of the Fisher Info Matrix, projects the FIM into that column space,
        then computes its SVD.

        Parameters
        ----------
        k: int
            Number of eigenvecs (rank of factorization) to be returned.
        p: int
            Oversampling parameter, recommended to be 5.
        q: int
            Matrix power iteration. Used to squeeze the eigen spectrum for more accurate approximation.
            Recommended to be 2.

        Returns
        -------
        S: Tensor
            Eigenvalues, Size((n, )).
        V: Tensor
            Eigendistortions, Size((n, k)).
        error_approx: Tensor
            Estimate of the approximation error. Defined as the

        References
        -----
        [1] Halko, Martinsson, Tropp, Finding structure with randomness: Probabilistic algorithms for constructing
        approximate matrix decompositions, SIAM Rev. 53:2, pp. 217-288 https://arxiv.org/abs/0909.4061 (2011)
        """

        x, y = self._input_flat, self._representation_flat
        n = len(x)

        P = torch.randn(n, k + p).to(x.device)
        P, _ = torch.qr(P)  # orthogonalize first for numerical stability
        _dummy_vec = torch.ones_like(y, requires_grad=True)
        Z = fisher_info_matrix_vector_product(y, x, P, _dummy_vec)

        for _ in range(q):  # optional power iteration to squeeze the spectrum for more accurate estimate
            Z = fisher_info_matrix_vector_product(y, x, Z, _dummy_vec)

        Q, _ = torch.qr(Z)
        B = Q.T @ fisher_info_matrix_vector_product(y, x, Q, _dummy_vec)  # B = Q.T @ A @ Q
        _, S, V = torch.svd(B, some=True)  # eigendecomp of small matrix
        V = Q @ V  # lift up to original dimensionality

        # estimate error in Q estimate of range space
        omega = fisher_info_matrix_vector_product(y, x, torch.randn(n, 20).to(x.device), _dummy_vec)
        error_approx = omega - (Q @ Q.T @ omega)
        error_approx = error_approx.norm(dim=0).mean()

        return S[:k], V[:, :k], error_approx  # truncate

    def _indexer(self, idx: int) -> int:
        """Maps eigenindex to arg index (0-indexed)"""
        if idx == 0 or idx == -1:
            return idx
        else:
            all_idx = self.synthesized_eigenindex
            assert idx in all_idx, "eigenindex must be the index of one of the vectors"
            assert all_idx is not None and len(all_idx) != 0, "No eigendistortions have been synthesized."
            return int(np.where(all_idx == idx))

    def plot_distorted_image(self,
                             eigen_index: int = 0,
                             alpha: float = 5.,
                             process_image: Callable = None,
                             ax: matplotlib.pyplot.axis = None,
                             plot_complex: str = 'rectangular',
                             **kwargs) -> Figure:
        r""" Displays specified eigendistortions alone, and added to the image.

        If image or eigendistortions have 3 channels, then it is assumed to be a color image and it is converted to
        grayscale. This is merely for display convenience and may change in the future.

        Parameters
        ----------
        eigen_index: int
            Index of eigendistortion to plot. E.g. If there are 10 eigenvectors, 0 will index the first one, and
            -1 or 9 will index the last one.
        alpha: float, optional
            Amount by which to scale eigendistortion for image + (alpha * eigendistortion) for display.
        process_image: Callable
            A function to process the image+alpha*distortion before clamping between 0,1. E.g. multiplying by the
            stdev ImageNet then adding the mean of ImageNet to undo image preprocessing.
        ax: matplotlib.pyplot.axis, optional
            Axis handle on which to plot.
        plot_complex: str, optional
            Parameter for :meth:`plenoptic.imshow` determining how to handle complex values. Defaults to 'rectangular',
            which plots real and complex components as separate images. Can also be 'polar' or 'logpolar'; see that
            method's docstring for details.
        kwargs:
            Additional arguments for :meth:`po.imshow()`.

        Returns
        -------
        fig: Figure
            matplotlib Figure handle returned by plenoptic.imshow()
        """
        if process_image is None:  # identity transform
            def process_image(x): return x

        # reshape so channel dim is last
        im_shape = self.n_channels, self.im_height, self.im_width
        image = self.base_signal.detach().view(1, * im_shape).cpu()
        dist = self.synthesized_signal[self._indexer(eigen_index)].unsqueeze(0).cpu()

        img_processed = process_image(image + alpha * dist)
        to_plot = torch.clamp(img_processed, 0, 1)
        fig = imshow(to_plot, ax=ax, plot_complex=plot_complex, **kwargs)

        return fig
