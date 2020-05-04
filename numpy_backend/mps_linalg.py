"""
Functions that operate upon MPS tensors as if they were matrices. 
These functions are not necessarily backend-agnostic.
"""
import numpy as np
import scipy as sp

import jax_vumps.contractions as ct


def fuse_left(A):
    """
    Joins the left bond with the physical index.
    """
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.reshape(d*chiL, chiR)
    return A


def unfuse_left(A, shp):
    """
    Reverses fuse_left.
    """
    return A.reshape(shp)


def fuse_right(A):
    """
    Joins the right bond with the physical index.
    """
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.transpose((1, 0, 2)).reshape((chiL, d*chiR)).conj()
    return A


def unfuse_right(A, shp):
    """
    Reverses fuse_right.
    """
    d, chiL, chiR = shp
    A = A.reshape((chiL, d, chiR)).transpose((1, 0, 2)).conj()
    return A


def norm(A):
    return np.norm(A)


def trace(A):
    return np.trace(A)


###############################################################################
# QR
###############################################################################
def qrpos(mps):
    """
    Reshapes the (d, chiL, chiR) MPS tensor into a (d*chiL, chiR) matrix,
    and computes its QR decomposition, with the phase of R fixed so as to
    have a non-negative main diagonal. A new left-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    R.

    In addition to being phase-adjusted, R is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (d, chiL, chiR) MPS tensor.

    RETURNS
    -------
    mps_L, R: A left-orthogonal (d, chiL, chiR) MPS tensor, and an upper
              triangular (chiR x chiR) matrix with a non-negative main
              diagonal such that mps = mps_L @ R.
    """
    d, chiL, chiR = mps.shape
    mps_mat = fuse_left(mps)
    Q, R = np.linalg.qr(mps_mat)
    phases = np.sign(np.diag(R))
    Q = Q*phases
    R = phases.conj()[:, None] * R
    R = R / norm(R)
    mps_L = unfuse_left(Q, mps.shape)
    return (mps_L, R)


def lqpos(mps):
    """
    Reshapes the (d, chiL, chiR) MPS tensor into a (chiL, d*chiR) matrix,
    and computes its LQ decomposition, with the phase of L fixed so as to
    have a non-negative main diagonal. A new right-orthogonal
    (d, chiL, chiR) MPS tensor (reshaped from Q) is returned along with
    L.
    In addition to being phase-adjusted, L is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (d, chiL, chiR) MPS tensor.

    RETURNS
    -------
    L, mps_R:  A lower-triangular (chiL x chiL) matrix with a non-negative
               main-diagonal, and a right-orthogonal (d, chiL, chiR) MPS
               tensor such that mps = L @ mps_R.
    """
    d, chiL, chiR = mps.shape
    mps_mat = fuse_right(mps)
    Qdag, Ldag = np.linalg.qr(mps_mat)
    Q = Qdag.T.conj()
    L = Ldag.T.conj()
    phases = np.sign(np.diag(L))
    L = L*phases
    L = L / norm(L)
    Q = phases.conj()[:, None] * Q
    mps_R = unfuse_right(Q, mps.shape)
    return (L, mps_R)


def mps_null_spaces(mpslist):
    """
    Return matrices spanning the null spaces of A_L and A_R, and
    the hermitian conjugates of these, reshaped into rank
    3 tensors.
    """
    AL, C, AR = mpslist
    d, chi, _ = AL.shape
    NLshp = (d, chi, (d-1)*chi)
    ALdag = fuse_left(AL).T.conj()
    NLm = sp.linalg.null_space(ALdag)
    NL = NLm.reshape(NLshp)

    ARmat = fuse_right(AR)
    NRm_dag = sp.linalg.null_space(ARmat)
    NRm = NRm_dag.conj()
    NR = NRm.reshape((d, chi, (d-1)*chi))
    NR = NR.transpose((0, 2, 1))
    return (NL, NR)


def gauge_match(A_C, C):
    QAC, RAC = qrpos(A_C)
    QC, RC = qrpos(C)
    A_L = ct.rightmult(QAC, np.conj(RC.T))
    errL = norm(RAC-RC)
    QAC, LAC = lqpos(A_C)
    QC, LC = lqpos(C)
    A_R = ct.leftmult(QC.T, QAC)
    errR = norm(LAC-LC)
    err = max(errL, errR)
    return (A_L, A_R, err)



