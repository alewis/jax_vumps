"""
These functions make computations upon an MPS based upon its interpretation
as a quantum state.
"""
import numpy as np
import jax_vumps.numpy_impl.mps_linalg as mps_linalg
import jax_vumps.numpy_impl.contractions as ct


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
    L = ct.XopL(AC, B=np.conj(NL))
    R = ct.XopR(AR, B=np.conj(NR))
    B2_tensor = L @ R.T
    B2 = np.linalg.norm(B2_tensor)
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


def onesiteexpectL(mpslist, H):
    A_Lp = 
