"""
Low level tensor network manipulations.

Conventions
      2                 3 4
      |                 | |
      O                  U
      |                 | |
      1                 1 2

  2---A---3            1
      |                |
      1             2--A--3
"""

import numpy as np
import scipy as sp

import tensornetwork as tn


def Bop(O, B):
    """
       1
       |
       O
       |
    2--B--3
    """
    return tn.ncon([O, B],
                   [(1, -1),
                    (1, -2, -3)])


def qrmat(A, mode="full"):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    Q, R = sp.linalg.qr(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, phases)
    R = np.dot(np.conj(phases), R)
    return (Q, R)


def qrpos(A):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal.

    If A is an MPS tensor (d, chiL, chiR), it is reshaped appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiR x chiR matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return qrmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = fuse_left(A)  # d*chiL, chiR
    Q, R = qrmat(A, mode="economic")
    Q = unfuse_left(Q, Ashp)
    return (Q, R)


def rqmat(A, mode="full"):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    R, Q = sp.linalg.rq(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(phases, Q)
    R = np.dot(R, np.conj(phases))
    return (Q, R)


def rqpos(A):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal.

    If A is an MPS tensor (d, chiL, chiR), it is reshaped and
    transposed appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiL x chiL matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return rqmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = fuse_right(A) #chiL, d*chiR
    R, Q = qrmat(A, mode="economic")
    Q = unfuse_right(Q, Ashp)
    return (Q, R)


def fuse_left(A):
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.reshape(d*chiL, chiR)
    return A


def unfuse_left(A, shp):
    return A.reshape(shp)


def fuse_right(A):
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.transpose((1, 0, 2)).reshape((chiL, d*chiR))
    return A


def unfuse_right(A, shp):
    d, chiL, chiR = shp
    A = A.reshape((chiL, d, chiR)).transpose((1, 0, 2))
    return A


def leftmult(lam, gam):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    If lam is None this is a no-op.
    lam can either be a vector of diagonal entries or a matrix.
    This function also works if gam is a matrix.
    """
    if lam is None:
        return gam
    ngam = len(gam.shape)
    nlam = len(lam.shape)
    if nlam == 1:
        return lam[:, None]*gam
    if nlam == 2:
        if ngam == 2:
            return np.dot(lam, gam)  # lambda is a matrix, note this assumes
                                     # lam[2] hits gam[2]
        if ngam == 3:
            idx = ([-2, 1], [-1, 1, -3])
            return tn.ncon([lam, gam], idx)
    raise IndexError("Invalid shapes. Gamma: ", gam.shape,
                     "Lambda: ", lam.shape)


def rightmult(gam, lam):
    """
    2--gam--lam--3
       |
       1
       |
    where lam is stored 1--lam--2
    If lam is None this is a no-op.
    lam can either be a vector of diagonal entries or a matrix.
    This function also works if gam is a matrix.
    """
    if lam is None:
        return gam
    nlam = len(lam.shape)
    if nlam == 1:
        return lam*gam
    if nlam == 2:
        return np.dot(gam, lam)
    raise IndexError("Invalid shapes. Gamma: ", gam.shape,
                     "Lambda: ", lam.shape)


def gauge_transform(gl, A, gr):
    """
            |
            1
     2--gl--A--gr--3
    """
    return rightmult(leftmult(gl, A), gr)


###############################################################################
# Chain contractors - MPS.
###############################################################################


def leftchain(As, Bs=None, X=None):
    """
      |---A1---A2---...-AN-2
      |   |    |         |
      X   |    |   ...   |
      |   |    |         |
      |---B1---B2---...-BN-1
      (note: AN and BN will be contracted into a single matrix)
    """
    if Bs is None:
        Bs = list(map(np.conj, As))
    for A, B in zip(As, Bs):
        X = XopL(A, B=B, X=X)
    return X


def chainnorm(As, Bs=None, X=None, Y=None):
    """
      |---A1---A2---...-AN---
      |   |    |        |   |
      X   |    |   ...  |   Y
      |   |    |        |   |
      |---B1---B2---...-BN---
    """
    X = leftchain(As, Bs=Bs, X=X)
    if Y is not None:
        X = np.dot(X, Y.T)
    return np.trace(X)


def proj(X, Y=None):
    """
    2   2
    |---|
    |   |
    X   Y
    |   |
    |---|
    1   1
    Be careful that Y especially is in fact stored this way!
    """
    if Y is not None:
        X = np.dot(X, Y.T)
    return np.trace(X)


# *****************************************************************************
# Single site to open legs.
# *****************************************************************************
def XopL(A, O=None, X=None, B=None):
    """
      |---A---2
      |   |
      X   O
      |   |
      |---B---1
    """
    if B is None:
        B = np.conj(A)
    A = leftmult(X, A)
    if O is not None:
        B = Bop(O, B)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx)


def XopR(A, O=None, X=None, B=None):
    """
      2---A---|
          |   |
          O   X
          |   |
      1---B---|
    """
    if B is None:
        B = np.conj(A)
    B = rightmult(B, X)
    if O is not None:
        B = Bop(O, B)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx)

# ***************************************************************************
# TWO SITE OPERATORS
# ***************************************************************************


def rholoc(A1, A2, B1=None, B2=None):
    """
    -----A1-----A2-----
    |    |(3)   |(4)   |
    |                  |
    |                  |
    |    |(1)   |(2)   |
    -----B1-----B2------
    returned as a (1:2)x(3:4) matrix.
    Assuming the appropriate Schmidt vectors have been contracted into the As,
    np.trace(np.dot(op, rholoc.T)) is the expectation value of the two-site
    operator op coupling A1 to A2.
    """
    if B1 is None:
        B1 = np.conj(A1)
    if B2 is None:
        B2 = np.conj(A2)
    d = A1.shape[0]
    to_contract = [A1, A2, B1, B2]
    idxs = [(-3, 1, 2),
            (-4, 2, 3),
            (-1, 1, 4),
            (-2, 4, 3)]
    rholoc = tn.ncon(to_contract, idxs).reshape((d**2, d**2))
    return rholoc


def twositecontract(left, right, U=None):
    """
       2--left-right--4
            |__|
            |U |
            ----
            |  |
            1  3
    """
    if U is None:
        return np.dot(left, right)
    else:
        to_contract = (left, right, U)
        idxs = [(2, -2, 1),
                (3, 1, -4),
                (-1, -3, 2, 3)]
        return tn.ncon(to_contract, idxs)
