"""
These functions make computations upon an MPS based upon its interpretation
as a quantum state.
"""
import os
import importlib
import jax_vumps.contractions as ct

try:
    environ_name = os.environ["LINALG_BACKEND"]
except KeyError:
    print("While importing observables.py,")
    print("os.environ[\"LINALG_BACKEND\"] was undeclared; using NumPy.")
    environ_name = "numpy"

if environ_name == "jax":
    mps_linalg_name = "jax_vumps.jax_backend.mps_linalg"
elif environ_name == "numpy":
    mps_linalg_name = "jax_vumps.numpy_backend.mps_linalg"
else:
    raise ValueError("Invalid LINALG_BACKEND ", environ_name)

mps_linalg = importlib.import_module(mps_linalg_name)


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
    L = ct.XopL(AC, B=NL)
    R = ct.XopR(AR, B=NR)
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
    the_norm = mps_linalg.trace(rho)
    return the_norm.real
