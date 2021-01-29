import torch
import warnings


def jacobian(y, x):
    """Explicitly compute the full Jacobian matrix.
    N.B. This is only recommended for small input sizes (e.g. <100x100 image)

    Parameters
    ----------
    y: torch.Tensor
        model output with gradient attached
    x: torch.Tensor with gradient function
        model input with gradient attached

    Returns
    -------
    J: torch.Tensor
        Jacobian matrix with ``torch.Size([len(y), len(x)])``
    """

    if x.numel() > 1E4:
        warnings.warn("Calculation of Jacobian with input dimensionality greater than 1E4 may take too long; consider"
                      "an iterative method (e.g. power method, randomized svd) instead.")

    J = torch.stack([torch.autograd.grad([y[i].sum()], [x], retain_graph=True, create_graph=True)[0] for i in range(
        y.size(0))], dim=-1).squeeze().t()

    if y.shape[0] == 1:  # need to return a 2D tensor even if y dimensionality is 1
        J = J.unsqueeze(0)

    return J.detach()


def vector_jacobian_product(y, x, U, retain_graph=True, create_graph=True, detach=False):
    r"""Compute vector Jacobian product: :math:`\text{vjp} = u^T(\partial y/\partial x)`

    Backward Mode Auto-Differentiation (`Lop` in Theano)

    Note on efficiency: When this function is used in the context of power iteration for computing eigenvectors, the
    vector output will be repeatedly fed back into :meth:`vector_jacobian_product()` and :meth:`jacobian_vector_product()`.
    To prevent the accumulation of gradient history in this vector (especially on GPU), we need to ensure the
    computation graph is not kept in memory after each iteration. We can do this by detaching the output, as well as
    carefully specifying where/when to retain the created graph.

    Parameters
    ----------
    y: torch.Tensor
        Output with gradient attached, ``torch.Size([m, 1])``.
    x: torch.Tensor
        Input with gradient attached, ``torch.Size([n, 1])``.
    U: torch.Tensor
        Direction, shape is ``torch.Size([m, k])``, i.e. same dim as output tensor.
    retain_graph: bool, optional
        Whether or not to keep graph after doing one :meth:`vector_jacobian_product`. Must be set to True if k>1.
    create_graph: bool, optional
        Whether or not to create computational graph. Usually should be set to True unless you're reusing the graph like
        in the second step of :meth:`jacobian_vector_product`.
    detach: bool, optional
        As with ``create_graph``, only necessary to be True when reusing the output like we do in the 2nd step of
        :meth:`jacobian_vector_product`.

    Returns
    -------
    vJ: torch.Tensor
        vector-Jacobian product, ``torch.Size([m, k])``.
    """

    assert y.shape[-1] == 1
    assert U.shape[0] == y.shape[0]

    k = U.shape[-1]
    product = []


    for i in range(k):
        grad_x, = torch.autograd.grad(y, x, U[:, i].unsqueeze(-1),
                                      retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
        product.append(grad_x.reshape(x.shape))

    vJ = torch.cat(product, dim=1)

    if detach:
        return vJ.detach()
    else:
        return vJ


def jacobian_vector_product(y, x, V, dummy_vec=None):
    r"""Compute Jacobian Vector Product: :math:`\text{jvp} = (\partial y/\partial x) v`

    Forward Mode Auto-Differentiation (``Rop`` in Theano). PyTorch does not natively support this operation; this
    function essentially calls backward mode autodiff twice, as described in [1].

    See :meth:`vector_jacobian_product()` docstring on why we and pass arguments for ``retain_graph`` and
    ``create_graph``.

    Parameters
    ----------
    y: torch.Tensor
        Model output with gradient attached, shape is torch.Size([m, 1])
    x: torch.Tensor
        Model input with gradient attached, shape is torch.Size([n, 1]), i.e. same dim as input tensor
    V: torch.Tensor
        Directions in which to compute product, shape is torch.Size([n, k]) where k is number of vectors to compute

    Returns
    -------
    Jv: torch.Tensor
        Jacobian-vector product, torch.Size([n, k])

    Notes
    -----
    [1] https://j-towns.github.io/2017/06/12/A-new-trick.html
    [2] https://pytorch.org/docs/stable/notes/faq.html ; First part talks about unintended out-of-memory errors.
    """
    assert y.shape[-1] == 1
    assert V.shape[0] == x.shape[0]

    if dummy_vec is None:
        dummy_vec = torch.ones_like(y, requires_grad=True)

    # do vjp twice to get jvp; set detach = False first; dummy_vec must be non-zero and is only there as a helper
    g = vector_jacobian_product(y, x, dummy_vec, retain_graph=True, create_graph=True, detach=False)
    Jv = vector_jacobian_product(g, dummy_vec, V, retain_graph=True, create_graph=False, detach=True)

    return Jv
