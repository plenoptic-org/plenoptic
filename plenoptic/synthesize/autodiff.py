import torch
import warnings

def jacobian(y, x):
    """Explicitly compute the full Jacobian matrix.
    N.B. This is only recommended for small input sizes (e.g. <100x100 image)

    Parameters
    ----------
    y: torch tensor with gradient function
        output
    x: torch tensor with gradient function
        input

    Returns
    -------
    J: torch tensor
        Jacobian matrix, y.size by x.size

    """

    if x.shape[0] > 1E6:
        warnings.warn("Calculation of Jacobian with input dimensionality greater than 1E6 may take too long; consider"
                      "an iterative method (e.g. power method, Lanczos) instead.")

    J = torch.stack([torch.autograd.grad([y[i].sum()], [x],
                                         retain_graph=True, create_graph=True)[0]
                     for i in range(y.size(0))], dim=-1).squeeze().t()

    return J.detach()


def vector_jacobian_product(y, x, U, retain_graph=True, create_graph=True, detach=False):
    """Compute vector Jacobian product: vjp = u^T(∂y/∂x)

    Backward Mode Auto-Differentiation (`Lop` in Theano)

    Note on efficiency: When this function is used in the context of power iteration for computing eigenvectors, the
    vector output will be repeatedly fed back into vjp() and jvp(). To prevent the accumulation of gradient history in
    this vector (especially on GPU), we need to ensure the computation graph is not kept in memory after each iteration.
    We can do this by detaching the output, as well as carefully specifying where/when to retain the created graph.

    Parameters
    ----------
    y: torch tensor with gradient function
        output, shape is (m,1)
    x: torch tensor with gradient function
        input, shape is (n,1)
    U: torch tensor
        direction, shape is (m,k), i.e. same dim as output tensor
    retain_graph: bool
        whether or not to keep graph after doing one vjp. Must be set to True if K>1.
    create_graph: bool
        whether or not to create computational graph. Usually should be set to True unless you're reusing the graph like
        in the second step of jacobian_vector_product
    detach: bool
        as with create_graph, only necessary to be True when reusing the output like we do in the 2nd step of jvp

    Returns
    -------
    vJ: torch tensor
        shape (m,k)

    """

    assert y.shape[-1] == 1
    assert U.shape[0] ==y.shape[0]

    k = U.shape[-1]
    product = []

    for i in range(k):
        grad_x, = torch.autograd.grad(y, x, U[:, i].unsqueeze(-1),
                                      retain_graph=retain_graph, create_graph=create_graph,
                                      allow_unused=True)
        product.append(grad_x.reshape(x.shape))

    vJ = torch.cat(product, dim=1)

    if detach:
        return vJ.detach()
    else:
        return vJ


def jacobian_vector_product(y, x, V):
    """Compute Jacobian Vector Product: jvp = (∂y/∂x) v

    Forward Mode Auto-Differentiation (`Rop` in Theano). PyTorch does not natively support this operation; this
    function essentially calls backward mode autodiff twice, as described in [1].

    See vector_jacobian_product() docstring on why we and pass arguments for retain_graph and create_graph.

    Parameters
    ----------
    y: torch tensor with gradient function
        output, shape is (m,1)
    x: torch tensor with gradient function
        input, shape is (n,1), i.e. same dim as input tensor
    V: torch tensor
        direction, shape is (n,k) where k is number of vectors to compute

    Returns
    -------
    Jv: torch tensor
        shape (n,k)

    Notes
    -----
    [1] https://j-towns.github.io/2017/06/12/A-new-trick.html
    [2] https://pytorch.org/docs/stable/notes/faq.html ; First part talks about unintended out-of-memory errors.

    """
    assert y.shape[-1] == 1
    assert V.shape[0] == x.shape[0]

    device = x.device
    dummy_vec = torch.ones_like(y, device=device).requires_grad_(True)

    # do vjp twice; set detach = False first using dummy_vec
    g = vector_jacobian_product(y, x, dummy_vec, retain_graph=True, create_graph=True, detach=False)
    Jv = vector_jacobian_product(g, dummy_vec, V, retain_graph=True, create_graph=False, detach=True)

    return Jv
