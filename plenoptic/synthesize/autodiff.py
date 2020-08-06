import torch


def jacobian(y, x):
    """Explicitly compute the full Jacobian matrix.
    N.B. This

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

    J = torch.stack([torch.autograd.grad([y[i].sum()], [x],
                                         retain_graph=True, create_graph=True)[0]
                     for i in range(y.size(0))], dim=-1).squeeze().t()

    return J


def vector_jacobian_product(y, x, u, retain_graph=True, create_graph=True):
    """Compute vector Jacobian product: vjp = u^T(∂y/∂x)

    Backward Mode Auto-Differentiation (`Lop` in Theano)

    Note on efficiency: When this function is used in the context of power iteration for computing eigenvectors, the
    vector output will be repeatedly fed back into vjp() and jvp(). To prevent the accumulation of gradient history in
    this vector (especially on GPU), we need to ensure the computation graph is not kept in memory after each iteration.
    We do this by detaching the output, as well as carefully specifying where/when to retain the created graph.

    Parameters
    ----------
    y: torch tensor with gradient function
        output
    x: torch tensor with gradient function
        input
    u: torch tensor
        direction, same shape as output y

    Returns
    -------
    uJ: torch tensor
        vector

    """

    assert (u.shape == y.shape)  # u is output shaped

    uJ = torch.autograd.grad([y], [x], u, retain_graph=retain_graph, create_graph=create_graph, allow_unused=False)[0].t()

    return uJ


def jacobian_vector_product(y, x, v):
    """Compute Jacobian Vector Product: jvp = (∂y/∂x) v

    Forward Mode Auto-Differentiation (`Rop` in Theano). PyTorch does not natively support this operation; this
    function essentially calls backward mode autodiff twice, as described in [1].

    See vector_jacobian_product() docstring on why we explicitly detach() and pass arguments for retain_graph and
    create_graph.

    Parameters
    ----------
    y: torch tensor with gradient function
        output
    x: torch tensor with gradient function
        input
    v: torch tensor
        direction, same shape as input x

    Returns
    -------
    Jv: torch tensor
        vector

    Notes
    -----
    [1] https://j-towns.github.io/2017/06/12/A-new-trick.html
    [2] https://pytorch.org/docs/stable/notes/faq.html ; First part talks about unintended out-of-memory errors.

    """
    device = x.device

    m = y.shape[0]  # output dimension
    assert (v.shape == x.shape)  # v is input shaped

    u = torch.ones(m, 1, requires_grad=True, device=device)  # dummy variable (should not matter)
    g = vector_jacobian_product(y, x, u, retain_graph=True, create_graph=True).t()
    Jv = vector_jacobian_product(g, u, v, retain_graph=False, create_graph=False).t()

    return Jv.detach()
