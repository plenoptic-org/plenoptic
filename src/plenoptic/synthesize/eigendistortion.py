"""
Eigendistortions.

Classes to perform the synthesis of eigendistortions.
"""

import warnings
from collections.abc import Callable
from typing import Any, Literal

import matplotlib.pyplot
import torch
from matplotlib.figure import Figure
from torch import Tensor
from tqdm.auto import tqdm

from ..tools.display import imshow
from ..tools.validate import validate_input, validate_model
from .autodiff import (
    jacobian,
    jacobian_vector_product,
    vector_jacobian_product,
)
from .synthesis import Synthesis


def fisher_info_matrix_vector_product(
    y: Tensor, x: Tensor, v: Tensor, dummy_vec: Tensor
) -> Tensor:
    r"""
    Compute Fisher Information Matrix Vector Product: :math:`Fv`.

    Parameters
    ----------
    y
        Output tensor with gradient attached.
    x
        Input tensor with gradient attached.
    v
        The vectors with which to compute Fisher vector products.
    dummy_vec
        Dummy vector for Jacobian vector product trick.

    Returns
    -------
    Fv
        Vector, Fisher vector product.

    Notes
    -----
    Under white Gaussian noise assumption, :math:`F` is matrix multiplication
    of Jacobian transpose and Jacobian:
    :math:`F = J^T J`. Hence:
    :math:`Fv = J^T (Jv)`
    """  # numpydoc ignore=ES01
    Jv = jacobian_vector_product(y, x, v, dummy_vec)
    Fv = vector_jacobian_product(y, x, Jv, detach=True)

    return Fv


def fisher_info_matrix_eigenvalue(
    y: Tensor, x: Tensor, v: Tensor, dummy_vec: Tensor | None = None
) -> Tensor:
    r"""
    Compute eigenvalues of Fisher vector products.

    We compute the Fisher Information Matrix corresponding to eigenvectors in ``v``:
    :math:`\lambda= v^T F v`.

    Parameters
    ----------
    y
        Output tensor with gradient attached.
    x
        Input tensor with gradient attached.
    v
        The vectors with which to compute Fisher vector products.
    dummy_vec
        Dummy vector for Jacobian vector product trick.

    Returns
    -------
    lmbda
        The computed eigenvalues.
    """
    if dummy_vec is None:
        dummy_vec = torch.ones_like(y, requires_grad=True)

    Fv = fisher_info_matrix_vector_product(y, x, v, dummy_vec)

    # compute eigenvalues for all vectors in v
    lmbda = torch.stack([a.dot(b) for a, b in zip(v.T, Fv.T)])
    return lmbda


class Eigendistortion(Synthesis):
    r"""
    Synthesize eigendistortions induced by a model on a given input image.

    Following the basic idea in [1]_, this class synthesizes image perturbations that
    are considered the most and least noticeable for a model on a given image. Because
    these are perturbations on the input image, they are local in pixel space, i.e.,
    they do not change the pixels much.

    Parameters
    ----------
    image
        Image to perturb. We currently do not support batches of images, as each image
        requires its own optimization, so either ``image.ndimension()==1`` or
        ``image.shape[0]==1``.
    model
        Torch model with defined forward and backward operations.

    Notes
    -----
    This is a method for comparing image representations in terms of their ability to
    explain perceptual sensitivity in humans. It estimates eigenvectors of the Fisher
    Information Matrix. A model, :math:`y = f(x)`, is a deterministic (and
    differentiable) mapping from the input pixels :math:`x \in \mathbb{R}^n` to a mean
    output response vector :math:`y\in \mathbb{R}^m`, where we assume additive white
    Gaussian noise in the response space.
    The Jacobian matrix at :math:`x` is:
        :math:`J(x) = J = dydx`,
        :math:`J\in\mathbb{R}^{m \times n}` (ie. output_dim x input_dim)
    The matrix consists of all partial derivatives of the vector-valued function
    :math:`f`. The Fisher Information Matrix (FIM) at :math:`x`, under white Gaussian
    noise in the response space, is:
        :math:`F = J^T J`
    It is a quadratic approximation of the discriminability of distortions
    relative to :math:`x`.

    References
    ----------
    .. [1] Berardino, A., Laparra, V., Ballé, J. and Simoncelli, E., 2017.
           Eigen-distortions of hierarchical representations. In Advances in
           neural information processing systems (pp. 3530-3539).
           https://www.cns.nyu.edu/pub/lcv/berardino17c-final.pdf
           https://www.cns.nyu.edu/~lcv/eigendistortions/
    """

    def __init__(self, image: Tensor, model: torch.nn.Module):
        super().__init__()
        validate_input(image, no_batch=True)
        validate_model(
            model,
            image_shape=image.shape,
            image_dtype=image.dtype,
            device=image.device,
        )

        self._image_shape = image.shape
        if image.ndimension() != 1:
            self._image_shape = self._image_shape[1:]

        self._model = model
        # flatten and attach gradient and reshape to image
        self._image_flat = image.flatten().unsqueeze(1).requires_grad_(True)
        self._init_representation(image)

        self._jacobian = None
        self._eigendistortions = None
        self._eigenvalues = None
        self._eigenindex = None

    def _init_representation(self, image: torch.Tensor):
        """Set self._representation_flat, based on ``self.model`` and ``image``."""
        # numpydoc ignore=ES01,PR01
        self._image = self._image_flat.view(*image.shape)
        image_representation = self.model(self.image)

        if len(image_representation) > 1:
            self._representation_flat = torch.cat(
                [s.squeeze().view(-1) for s in image_representation]
            ).unsqueeze(1)
        else:
            self._representation_flat = (
                image_representation.squeeze().view(-1).unsqueeze(1)
            )

    def synthesize(
        self,
        method: Literal["exact", "power", "randomized_svd"] = "power",
        k: int = 1,
        max_iter: int = 1000,
        p: int = 5,
        q: int = 2,
        stop_criterion: float = 1e-7,
    ):
        r"""
        Compute eigendistortions of Fisher Information Matrix with given input image.

        There are three potential ways of computing the eigendistortion for a model;
        all have the same interpretation. See ``method`` argument for details.

        Parameters
        ----------
        method
            Eigensolver method. ``'exact'`` tries to do eigendecomposition directly
            (not recommended for very large inputs). ``'power'`` uses the power
            method to compute first and last eigendistortions, with maximum number of
            iterations dictated by n_steps. ``'randomized_svd'`` uses randomized SVD to
            approximate the top k eigendistortions and their corresponding eigenvalues.
        k
            How many vectors to return using block power method or svd.
        max_iter
            Maximum number of steps to run for ``method='power'`` in eigenvalue
            computation. Ignored for other methods.
        p
            Oversampling parameter for randomized SVD. k+p vectors will be sampled,
            and k will be returned. See docstring of :meth:`_synthesize_randomized_svd`
            for more details including algorithm reference.
        q
            Matrix power parameter for randomized SVD. This is an effective trick for
            the algorithm to converge to the correct eigenvectors when the
            eigenspectrum does not decay quickly. See :meth:`_synthesize_randomized_svd`
            for more details including algorithm reference.
        stop_criterion
            Used if ``method='power'`` to check for convergence. If the L2-norm
            of the eigenvalues has changed by less than this value from one
            iteration to the next, we terminate synthesis.

        Raises
        ------
        ValueError
            If ``method`` takes an illegal value.

        Warns
        -----
        UserWarning
            If ``method == "power"`` but the Jacobian size is greater than 1e6 (which
            depends on the number of elements in the model's representation and input
            image), in which case we're worried abour running out of memory.
        """
        allowed_methods = ["power", "exact", "randomized_svd"]
        if method not in allowed_methods:
            raise ValueError(f"method must be in {allowed_methods}")

        if (
            method == "exact"
            and self._representation_flat.size(0) * self._image_flat.size(0) > 1e6
        ):
            warnings.warn(
                "Jacobian > 1e6 elements and may cause out-of-memory. Use"
                " method =  {'power', 'randomized_svd'}."
            )

        if method == "exact":  # compute exact Jacobian
            eig_vals, eig_vecs = self._synthesize_exact()
            eig_vecs = self._vector_to_image(eig_vecs.detach())
            eig_vecs_ind = torch.arange(len(eig_vecs))

        elif method == "randomized_svd":
            lmbda_new, v_new, error_approx = self._synthesize_randomized_svd(
                k=k, p=p, q=q
            )
            eig_vecs = self._vector_to_image(v_new.detach())
            eig_vals = lmbda_new.squeeze()
            eig_vecs_ind = torch.arange(k)

            # display the approximate estimation error of the range space
            warnings.warn(
                "Randomized SVD complete! Estimated spectral approximation"
                f" error = {error_approx:.2f}"
            )

        else:  # method == 'power'
            assert max_iter > 0, "max_iter must be greater than zero"

            lmbda_max, v_max = self._synthesize_power(
                k=k, shift=0.0, tol=stop_criterion, max_iter=max_iter
            )
            lmbda_min, v_min = self._synthesize_power(
                k=k, shift=lmbda_max[0], tol=stop_criterion, max_iter=max_iter
            )
            n = v_max.shape[0]

            eig_vecs = self._vector_to_image(torch.cat((v_max, v_min), dim=1).detach())
            eig_vals = torch.cat([lmbda_max, lmbda_min]).squeeze()
            eig_vecs_ind = torch.cat((torch.arange(k), torch.arange(n - k, n)))

        # reshape to (n x num_chans x h x w)
        self._eigendistortions = torch.stack(eig_vecs, 0) if len(eig_vecs) != 0 else []
        self._eigenvalues = torch.abs(eig_vals.detach())
        self._eigenindex = eig_vecs_ind

    def _synthesize_exact(self) -> tuple[Tensor, Tensor]:
        r"""
        Eigendecomposition of explicitly computed Fisher Information Matrix.

        To be used when the input is small (e.g. less than 70x70 image on cluster or
        30x30 on your own machine). This method obviates the power iteration and its
        related algorithms (e.g. Lanczos). This method computes the Fisher Information
        Matrix by explicitly computing the Jacobian of the representation with respect
        to the input.

        Returns
        -------
        eig_vals
            Eigenvalues in decreasing order.
        eig_vecs
            Eigenvectors in 2D tensor, whose cols are eigenvectors (i.e.
            eigendistortions) corresponding to eigenvalues.
        """
        J = self.compute_jacobian()
        F = J.T @ J
        eig_vals, eig_vecs = torch.linalg.eigh(F, UPLO="U")
        eig_vecs = eig_vecs.flip(dims=(1,))
        eig_vals = eig_vals.flip(dims=(0,))
        return eig_vals, eig_vecs

    def compute_jacobian(self) -> Tensor:
        r"""
        Compute (via :func:`autodiff.jacobian`), cache, and return jacobian.

        Returns
        -------
        J
            Jacobian of representation with respect to input.

        Warns
        -----
        UserWarning
            If input dimensionality is greater than 1e4, in which case we believe that
            this calculation will take too long.
        """  # numpydoc ignore=ES01
        if self.jacobian is None:
            J = jacobian(self._representation_flat, self._image_flat)
            self._jacobian = J
        else:
            J = self.jacobian

        return J

    def _synthesize_power(
        self, k: int, shift: Tensor | float, tol: float, max_iter: int
    ) -> tuple[Tensor, Tensor]:
        r"""
        Use power method  to obtain largest (smallest) eigenvalue/vector pairs.

        When ``k>1``, uses orthogonal iteration, see [1]_.

        Apply the algorithm to approximate the extremal eigenvalues and eigenvectors
        of the Fisher Information Matrix, without explicitly representing that matrix.

        This method repeatedly calls :func:`fisher_info_matrix_vector_product` with a
        single (``k=1``), or multiple (``k>1``) vectors.

        Parameters
        ----------
        k
            Number of top and bottom eigendistortions to synthesize; i.e. if k=2,
            then the top 2 and bottom 2 will be returned. When `k>1`, multiple
            eigendistortions are synthesized, and each power iteration step is followed
            by a QR orthogonalization step to ensure the vectors are orthonormal.
        shift
            When `shift=0`, this function estimates the top `k` eigenvalue/vector
            pairs. When `shift` is set to the estimated top eigenvalue this function
            will estimate the smallest eigenval/eigenvector pairs.
        tol
            Tolerance value.
        max_iter
            Maximum number of steps.

        Returns
        -------
        lmbda
            Eigenvalue corresponding to final vector of power iteration.
        v
            Final eigenvector(s) (i.e. eigendistortions) of power (orthogonal)
            iteration procedure.

        References
        ----------
        .. [1] Orthogonal iteration; Algorithm 8.2.8 Golub and Van Loan, Matrix
               Computations, 3rd Ed.
        """
        x, y = self._image_flat, self._representation_flat

        # note: v is an n x k matrix where k is number of eigendists to be synthesized!
        v = torch.randn(len(x), k, device=x.device, dtype=x.dtype)
        v = v / torch.linalg.vector_norm(v, dim=0, keepdim=True, ord=2)

        _dummy_vec = torch.ones_like(y, requires_grad=True)  # cache a dummy vec for jvp
        Fv = fisher_info_matrix_vector_product(y, x, v, _dummy_vec)
        v = Fv / torch.linalg.vector_norm(Fv, dim=0, keepdim=True, ord=2)
        lmbda = fisher_info_matrix_eigenvalue(y, x, v, _dummy_vec)

        d_lambda = torch.as_tensor(float("inf"))
        lmbda_new, v_new = None, None
        desc = ("Top" if shift == 0 else "Bottom") + f" k={k} eigendists"
        pbar = tqdm(range(max_iter), desc=desc)
        postfix_dict = {"delta_eigenval": None}

        for _ in pbar:
            postfix_dict.update(dict(delta_eigenval=f"{d_lambda.item():.2E}"))
            pbar.set_postfix(**postfix_dict)

            if d_lambda <= tol:
                print(f"{desc} computed | Stop criterion {tol:.2E} reached.")
                break

            Fv = fisher_info_matrix_vector_product(y, x, v, _dummy_vec)
            Fv = Fv - shift * v  # optionally shift: (F - shift*I)v

            v_new, _ = torch.linalg.qr(Fv, "reduced")  # (ortho)normalize vector(s)

            lmbda_new = fisher_info_matrix_eigenvalue(y, x, v_new, _dummy_vec)

            d_lambda = torch.linalg.vector_norm(
                lmbda - lmbda_new, ord=2
            )  # stability of eigenspace
            v = v_new
            lmbda = lmbda_new

        pbar.close()

        return lmbda_new, v_new

    def _synthesize_randomized_svd(
        self, k: int, p: int, q: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Synthesize eigendistortions using randomized truncated SVD.

        This method approximates the column space of the Fisher Info Matrix, projects
        the FIM into that column space, then computes its SVD. See [1]_ for details.

        Parameters
        ----------
        k
            Number of eigenvecs (rank of factorization) to be returned.
        p
            Oversampling parameter, recommended to be 5.
        q
            Matrix power iteration. Used to squeeze the eigen spectrum for more
            accurate approximation. Recommended to be 2.

        Returns
        -------
        S
            Eigenvalues, Size((n, )).
        V
            Eigendistortions, Size((n, k)).
        error_approx
            Estimate of the approximation error. Defined as the expected error
            between the true subspace and approximated subspace.

        References
        ----------
        .. [1] Halko, Martinsson, Tropp, Finding structure with randomness:
               Probabilistic algorithms for constructing approximate matrix
               decompositions, SIAM Rev. 53:2, pp. 217-288
               https://arxiv.org/abs/0909.4061 (2011)
        """
        x, y = self._image_flat, self._representation_flat
        n = len(x)

        P = torch.randn(n, k + p).to(x.device)
        # orthogonalize first for numerical stability
        P, _ = torch.linalg.qr(P, "reduced")
        _dummy_vec = torch.ones_like(y, requires_grad=True)
        Z = fisher_info_matrix_vector_product(y, x, P, _dummy_vec)

        # optional power iteration to squeeze the spectrum for more accurate
        # estimate
        for _ in range(q):
            Z = fisher_info_matrix_vector_product(y, x, Z, _dummy_vec)

        Q, _ = torch.linalg.qr(Z, "reduced")
        # B = Q.T @ A @ Q
        B = Q.T @ fisher_info_matrix_vector_product(y, x, Q, _dummy_vec)
        _, S, Vh = torch.linalg.svd(B, False)  # eigendecomp of small matrix
        V = Vh.T
        V = Q @ V  # lift up to original dimensionality

        # estimate error in Q estimate of range space
        omega = fisher_info_matrix_vector_product(
            y, x, torch.randn(n, 20).to(x.device), _dummy_vec
        )
        error_approx = omega - (Q @ Q.T @ omega)
        error_approx = torch.linalg.vector_norm(error_approx, dim=0, ord=2).mean()

        return S[:k].clone(), V[:, :k].clone(), error_approx  # truncate

    def _vector_to_image(self, vecs: Tensor) -> list[Tensor]:
        r"""
        Reshape eigenvectors back into correct image dimensions.

        Parameters
        ----------
        vecs
            Eigendistortion tensor with ``torch.Size([N, num_distortions])``.
            Each distortion will be reshaped into the original image shape and
            placed in a list.

        Returns
        -------
        imgs
            List of Tensor images.
        """  # numpydoc ignore=ES01
        imgs = [vecs[:, i].reshape(self._image_shape) for i in range(vecs.shape[1])]
        return imgs

    def _indexer(self, idx: int) -> int:
        """
        Map eigenindex to arg index (0-indexed).

        Parameters
        ----------
        idx
            Eigenvalue number.

        Returns
        -------
        idx
            Index into eigenvalue tensor.

        Raises
        ------
        ValueError
            If ``eigenindex`` doesn't correspond to one of the synthesized
            eigendistortions.
        ValueError
            If no eigendistortions have been synthesized.
        """  # numpydoc ignore=ES01
        n = len(self._image_flat)
        idx_range = range(n)
        i = idx_range[idx]

        all_idx = self.eigenindex
        if i not in all_idx:
            raise ValueError("eigenindex must be the index of one of the vectors")
        if all_idx is None or len(all_idx) == 0:
            raise ValueError("No eigendistortions synthesized")
        return torch.where(all_idx == i)[0].item()

    def save(self, file_path: str):
        r"""
        Save all relevant variables in .pt file.

        See :meth:`load` docstring for an example of use.

        Parameters
        ----------
        file_path : str
            The path to save the Eigendistortion object to.
        """
        save_io_attrs = [("_model", ("_image",))]
        super().save(file_path, save_io_attrs)

    def to(self, *args: Any, **kwargs: Any):
        r"""
        Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See :meth:`torch.nn.Module.to` for examples.

        .. note::
            This method modifies the module in-place.

        Parameters
        ----------
        device : torch.device
            The desired device of the parameters and buffers in this module.
        dtype : torch.dtype
            The desired floating point type of the floating point parameters and
            buffers in this module.
        tensor : torch.Tensor
            Tensor whose dtype and device are the desired dtype and device for
            all parameters and buffers in this module.
        """  # numpydoc ignore=PR01,PR02
        attrs = [
            "_jacobian",
            "_eigendistortions",
            "_eigenvalues",
            "_eigenindex",
            "_image",
            "_image_flat",
            "_representation_flat",
        ]
        super().to(*args, attrs=attrs, **kwargs)
        # try to call .to() on model. this should work, but it might fail if e.g., this
        # a custom model that doesn't inherit torch.nn.Module
        try:
            self._model = self._model.to(*args, **kwargs)
        except AttributeError:
            warnings.warn("Unable to call model.to(), so we leave it as is.")
        # we need _representation_flat and _image_flat to be connected in the
        # computation graph for the autograd calls to work, so we reinitialize
        # it here
        self._init_representation(self.image)

    def load(
        self,
        file_path: str,
        map_location: str | None = None,
        tensor_equality_atol: float = 1e-8,
        tensor_equality_rtol: float = 1e-5,
        **pickle_load_args: Any,
    ):
        r"""
        Load all relevant stuff from a .pt file.

        This must be called by a ``Eigendistortion`` object initialized just like the
        saved object.

        Note this operates in place and so doesn't return anything.

        .. versionchanged:: 1.2
           load behavior changed in a backwards-incompatible manner in order to
           compatible with breaking changes in torch 2.6.

        Parameters
        ----------
        file_path
            The path to load the synthesis object from.
        map_location
            Argument to pass to :func:`torch.load` as ``map_location``. If you
            save stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to :class:`torch.device`.
        tensor_equality_atol
            Absolute tolerance to use when checking for tensor equality during load,
            passed to :func:`torch.allclose`. It may be necessary to increase if you are
            saving and loading on two machines with torch built by different cuda
            versions. Be careful when changing this! See
            :class:`torch.finfo<torch.torch.finfo>` for more details about floating
            point precision of different data types (especially, ``eps``); if you have
            to increase this by more than 1 or 2 decades, then you are probably not
            dealing with a numerical issue.
        tensor_equality_rtol
            Relative tolerance to use when checking for tensor equality during load,
            passed to :func:`torch.allclose`. It may be necessary to increase if you are
            saving and loading on two machines with torch built by different cuda
            versions. Be careful when changing this! See
            :class:`torch.finfo<torch.torch.finfo>` for more details about floating
            point precision of different data types (especially, ``eps``); if you have
            to increase this by more than 1 or 2 decades, then you are probably not
            dealing with a numerical issue.
        **pickle_load_args
            Any additional kwargs will be added to ``pickle_module.load`` via
            :func:`torch.load`, see that function's docstring for details.

        Raises
        ------
        ValueError
            If :func:`synthesize` has been called before this call to ``load``.
        ValueError
            If the object saved at ``file_path`` is not a ``Eigendistortion`` object.
        ValueError
            If the saved and loading ``Eigendistortion`` objects have a different value
            for :attr:`image`.
        ValueError
            If the behavior of :attr:`model` is different between the saved and loading
            objects.

        See Also
        --------
        :func:`~plenoptic.tools.io.examine_saved_synthesis`
            Examine metadata from saved object: pytorch and plenoptic versions, name of
            the synthesis object, shapes of tensors, etc.

        Examples
        --------
        >>> import plenoptic as po
        >>> img = po.data.einstein()
        >>> model = po.simul.Gaussian(30).eval()
        >>> po.tools.remove_grad(model)
        >>> eig = po.synth.Eigendistortion(img, model)
        >>> eig.synthesize(max_iter=5)
        >>> eig.save("eig.pt")
        >>> eig_copy = po.synth.Eigendistortion(img, model)
        >>> eig_copy.load("eig.pt")
        """
        check_attributes = ["_image"]
        check_io_attrs = [("_model", ("_image",))]
        super().load(
            file_path,
            "eigenindex",
            map_location=map_location,
            check_attributes=check_attributes,
            check_io_attributes=check_io_attrs,
            tensor_equality_atol=tensor_equality_atol,
            tensor_equality_rtol=tensor_equality_rtol,
            **pickle_load_args,
        )
        # make these require a grad again
        self._image_flat.requires_grad_()
        # we need _representation_flat and _image_flat to be connected in the
        # computation graph for the autograd calls to work, so we reinitialize
        # it here
        self._init_representation(self.image)

    @property
    def model(self) -> torch.nn.Module:
        """The model for which the eigendistortions are synthesized."""
        # numpydoc ignore=RT01,ES01
        return self._model

    @property
    def image(self) -> torch.Tensor:
        """Target image of eigendistortion synthesis."""
        # numpydoc ignore=RT01,ES01
        return self._image

    @property
    def jacobian(self) -> torch.Tensor:
        """
        Jacobian matrix of :attr:`model` with respect to :attr:`image`.

        Only set when :func:`synthesize` is run with ``method='exact'``.
        Else, ``None``.
        """  # numpydoc ignore=RT01
        return self._jacobian

    @property
    def eigendistortions(self) -> torch.Tensor:
        """
        Eigendistortions, ordered by eigenvalue.

        Eigendistortions are the eigenvectors of Fisher matrix, will have size
        ``Size((n_distortions, *image.shape[1:]))``.
        """  # numpydoc ignore=RT01
        return self._eigendistortions

    @property
    def eigenvalues(self) -> torch.Tensor:
        """Eigenvalues corresponding to each eigendistortion, in decreasing order."""
        # numpydoc ignore=RT01,ES01,SS05
        return self._eigenvalues

    @property
    def eigenindex(self) -> torch.Tensor:
        """Index of each eigenvector/eigenvalue."""
        # numpydoc ignore=RT01,ES01
        return self._eigenindex


def display_eigendistortion(
    eigendistortion: Eigendistortion,
    eigenindex: int = 0,
    alpha: float = 5.0,
    process_image: Callable[[Tensor], Tensor] = lambda x: x,
    ax: matplotlib.axes.Axes | None = None,
    plot_complex: str = "rectangular",
    **kwargs: Any,
) -> Figure:
    r"""
    Display specified eigendistortion added to the image.

    If image or eigendistortions have 3 channels, then it is assumed to be a color
    image and it is converted to grayscale. This is merely for display convenience
    and may change in the future.

    Parameters
    ----------
    eigendistortion
        Eigendistortion object whose synthesized eigendistortion we want to display.
    eigenindex
        Index of eigendistortion to plot. E.g. If there are 10 eigenvectors, 0 will
        index the first one, and -1 or 9 will index the last one.
    alpha
        Amount by which to scale eigendistortion for
        ``image + (alpha * eigendistortion)`` for display.
    process_image
        A function to process the image+alpha*distortion before clamping between 0,1.
        E.g. multiplying by the stdev ImageNet then adding the mean of ImageNet to undo
        image preprocessing.
    ax
        Axis handle on which to plot.
    plot_complex
        Parameter for :func:`~plenoptic.tools.display.imshow` determining how to handle
        complex values. Defaults to ``'rectangular'``, which plots real and complex
        components as separate images. See that method's docstring for details.
    **kwargs
        Additional arguments for :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    fig
        Figure containing the displayed images.

    Raises
    ------
    ValueError
        If ``eigenindex`` doesn't correspond to one of the synthesized eigendistortions.

    See Also
    --------
    display_eigendistortion_all
        Display base image and multiple eigendistortions, alone and added to image.
    """
    # reshape so channel dim is last
    im_shape = eigendistortion._image_shape
    image = eigendistortion.image.detach().view(1, *im_shape).cpu()
    dist = (
        eigendistortion.eigendistortions[eigendistortion._indexer(eigenindex)]
        .unsqueeze(0)
        .cpu()
    )

    img_processed = process_image(image + alpha * dist)
    to_plot = torch.clamp(img_processed, 0, 1)
    title = f"{alpha} * Eigendistortion[{eigenindex}]"
    fig = imshow(to_plot, ax=ax, plot_complex=plot_complex, title=title, **kwargs)

    return fig


def display_eigendistortion_all(
    eigendistortion: Eigendistortion,
    eigenindex: int | list[int] = [0, -1],
    alpha: float | list[float] = 5.0,
    process_image: Callable[[Tensor], Tensor] = lambda x: x,
    plot_complex: str = "rectangular",
    as_rgb: bool = False,
    suptitle: str = "Eigendistortions",
    suptitle_kwargs: dict | None = None,
    **kwargs: Any,
) -> Figure:
    r"""
    Display base image, eigendistortions alone, and eigendistortions added to the image.

    If image or eigendistortions have 3 channels, then it is assumed to be a color
    image and it is converted to grayscale. This is merely for display convenience
    and may change in the future.

    Parameters
    ----------
    eigendistortion
        Eigendistortion object whose synthesized eigendistortion we want to display.
    eigenindex
        Index of eigendistortion to plot. E.g. If there are 10 eigenvectors, 0 will
        index the first one, and -1 or 9 will index the last one.
    alpha
        Amount by which to scale eigendistortion for ``image + (alpha *
        eigendistortion)`` for display. If a list, must be the same length as
        ``eigenindex`` and will multiply each distortion by the corresponding
        ``alpha`` value.
    process_image
        A function to process all images before display. E.g. multiplying by the
        stdev ImageNet then adding the mean of ImageNet to undo image
        preprocessing. If ``None`` and ``as_rgb is True``, will add 0.5 to the
        distortion(s) (to avoid matplotlib clipping).
    plot_complex
        Parameter for :func:`~plenoptic.tools.display.imshow` determining how to handle
        complex values. Defaults to ``'rectangular'``, which plots real and complex
        components as separate images. See that method's docstring for details.
    as_rgb
        Whether to consider the channels as encoding RGB(A) values. If ``True``, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension. If ``False``,
        we plot each channel as a separate grayscale image.
    suptitle
        Super title to plot above all axes.
    suptitle_kwargs
        Additional arguments for :func:`matplotlib.pyplot.suptitle`.
    **kwargs
        Additional arguments for :func:`~plenoptic.tools.display.imshow`.

    Returns
    -------
    fig
        Figure containing the displayed images.

    Raises
    ------
    ValueError
        If ``len(alpha) != len(eigenindex)``.
    ValueError
        If a value of ``eigenindex`` doesn't correspond to one of the
        synthesized eigendistortions.

    Warns
    -----
    UserWarning
        If ``process_image=None`` and ``as_rgb=True``, because we are adding 0.5
        to the distortion.

    See Also
    --------
    display_eigendistortion
        Display single eigendistortion added to image.
    """
    # reshape so channel dim is last
    im_shape = eigendistortion._image_shape
    image = eigendistortion.image.detach().view(1, *im_shape).cpu()
    if not hasattr(eigenindex, "__iter__"):
        eigenindex = [eigenindex]
    if not hasattr(alpha, "__iter__"):
        alpha = [alpha] * len(eigenindex)
    if len(alpha) != len(eigenindex):
        raise ValueError(
            "If alpha is a list, it must have the same number of values as eigenindex! "
            f"{len(alpha)=} vs. {len(eigenindex)=}"
        )
    distortions = [torch.ones_like(image)]
    distortion_titles = [""]
    img_processed = [process_image(image)]
    img_titles = ["Original image"]
    dist_suffix = ""
    if process_image is None and as_rgb:

        def process_dist(x: Tensor) -> Tensor:
            return x + 0.5

        warnings.warn(
            "Adding 0.5 to distortion to plot as RGB image, else matplotlib"
            " clipping will result in a strange looking image..."
        )
        dist_suffix = " + 0.5"
    elif process_image is not None:
        process_dist = process_image

    for a, idx in zip(alpha, eigenindex):
        dist = (
            eigendistortion.eigendistortions[eigendistortion._indexer(idx)]
            .unsqueeze(0)
            .cpu()
        )
        img_processed.append(torch.clamp(process_image(image + a * dist), 0, 1))
        img_titles.append(f"{a} * Eigendistortion[{idx}]")
        distortion_titles.append(f"Eigendistortion[{idx}]{dist_suffix}")
        distortions.append(process_dist(dist))

    fig = imshow(
        distortions + img_processed,
        plot_complex=plot_complex,
        title=distortion_titles + img_titles,
        col_wrap=len(distortions),
        as_rgb=as_rgb,
        **kwargs,
    )
    fig.axes[0].set_visible(False)

    if suptitle_kwargs is None:
        suptitle_kwargs = {}
    va = suptitle_kwargs.pop("verticalalignment", None)
    va = suptitle_kwargs.pop("va", "bottom") if va is None else "bottom"
    y = suptitle_kwargs.pop("y", 1.05)
    fig.suptitle(suptitle, y=y, va=va, **suptitle_kwargs)

    return fig
