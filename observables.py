"""
These functions make computations upon an MPS based upon its interpretation
as a quantum state.
"""
import os
import jax_vumps.contractions as ct

if os.environ["LINALG_BACKEND"] == "Jax":
    import jax_vumps.jax_backend.mps_linalg as mps_linalg
else:
    import jax_vumps.numpy_backend.mps_linalg as mps_linalg


def B2_variance(oldlist, newlist):
    """
    Given two MPS tensors in mixed canonical form, estimate the gradient
    variance.

    PARAMETERS
    ----------
    oldlist, newlist: Both lists [A_L, C, A_R] representing two MPS in
                      mixed canonical form.

    RETURNS
    ------
    B2 (float) : The gradient variance.
    """
    NL, NR = mps_linalg.mps_null_spaces(oldlist)
    AL, C, AR = newlist
    AC = ct.rightmult(AL, C)
    L = ct.XopL(AC, NL)
    R = ct.XopR(AR, NR)
    B2_tensor = L @ R.T
    B2 = mps_linalg.norm(B2_tensor)
    return B2


def twositeexpect(mpslist, H):
    """
    The expectation value of the operator H in the state represented
    by A_L, C, A_R in mpslist.

    RETURNS
    -------
    out: The expectation value.
    """
    A_L, C, A_R = mpslist
    A_CR = ct.leftmult(C, A_R)
    expect = ct.twositeexpect(A_L, A_CR, H)
    return expect


def norm(mpslist):
    A_L, C, A_R = mpslist
    A_CR = ct.leftmult(C, A_R)
    rho = ct.rholoc(A_L, A_CR)
    d = rho.shape[0]
    the_norm = mps_linalg.trace(rho.reshape(d*d, d*d))
    return the_norm.real
