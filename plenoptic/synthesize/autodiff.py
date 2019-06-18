import torch


def jacobian(y, x):
    """Build the full Jacobian matrix (does not scale).

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


def vector_jacobian_product(y, x, u):
    """Compute vector Jacobian product:
    vjp = u^T(∂y/∂x)

    aka. Backward Mode Auto-Differentiation (`Lop` in theano)

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
    # m = y.shape[0]
    assert (u.shape == y.shape)  # u is output shaped

    uJ = torch.autograd.grad([y], [x], u, retain_graph=True, create_graph=True, allow_unused=True)[0].t()
    return uJ


def jacobian_vector_product(y, x, v):
    """Compute Jacobian Vector Product:
    jvp = (∂y/∂x) v

    aka. Forward Mode Auto-Differentiation (`Rop` in theano)

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
    using a trick described in https://j-towns.github.io/2017/06/12/A-new-trick.html

    """
    m = y.shape[0]  # output dimension
    assert (v.shape == x.shape)  # v is input shaped
    u = torch.ones(m, 1, requires_grad=True)  # dummy variable (should not matter)
    # u = torch.randn_like(y, requires_grad=True)
    g = vector_jacobian_product(y, x, u).t()
    Jv = vector_jacobian_product(g, u, v).t()
    return Jv
